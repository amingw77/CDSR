"""
A²GSTran-style asymmetric cross-attention fusion.
Depth features as Query, RGB edge features as Key/Value.
Uses pixel unshuffle to align high-res RGB features with lower-res depth features.

Includes A2GS-style stage-by-stage cross-attention blocks:
- CrossAttention with Conv2d projections, DWConv, cosine similarity
- CrossTransformerBlock: LN + CrossAttn + residual + LN + MLP(DWConv) + residual
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_even(x: torch.Tensor):
    """Ensure spatial dims are even for pixel unshuffle."""
    B, C, H, W = x.shape
    if H % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1))
    if W % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0))
    return x


class CrossAttention(nn.Module):
    """Cross-attention: Q from depth, K/V from RGB guide.

    Q (depth) drives the interaction — it queries RGB for structurally relevant info.
    Residual connection from Q (depth) preserves depth identity.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, depth_feat: torch.Tensor, rgb_feat: torch.Tensor):
        """
        depth_feat: (B, C, H, W) — query source
        rgb_feat:   (B, C, H, W) — key/value source
        Returns: (B, C, H, W) — fused features
        """
        B, C, H, W = depth_feat.shape
        assert rgb_feat.shape[2:] == (H, W), \
            f"Resolution mismatch: depth {depth_feat.shape}, rgb {rgb_feat.shape}"

        # Flatten spatial dims
        depth_tokens = depth_feat.flatten(2).transpose(1, 2)  # (B, N, C)
        rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)      # (B, N, C)

        shortcut = depth_tokens  # residual from Q source

        depth_tokens = self.norm_q(depth_tokens)
        rgb_tokens = self.norm_kv(rgb_tokens)

        q = self.to_q(depth_tokens)
        k = self.to_k(rgb_tokens)
        v = self.to_v(rgb_tokens)

        # Multi-head reshape
        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        out = self.proj_dropout(out)

        # Residual from Q source + feedforward
        out = out + shortcut  # depth identity preserved
        out = out.transpose(1, 2).contiguous().view(B, C, H, W)
        return out


class MultiScaleFusion(nn.Module):
    """Multi-scale A²GSTran fusion module.

    At each scale:
    1. Optionally pixel-unshuffle RGB features to match depth resolution
    2. Cross-attention: depth(Q) queries RGB(K/V)
    3. Channel mixing via convolution
    """

    def __init__(self, channels: list, num_heads: int = 8, unshuffle_ratios: list = None):
        """
        channels: list of channel dims at each scale [C0, C1, C2, C3]
        unshuffle_ratios: list of pixel-unshuffle ratios per scale (None = no unshuffle)
        """
        super().__init__()
        if unshuffle_ratios is None:
            unshuffle_ratios = [0, 0, 0, 0]  # if same resolution, no unshuffle needed

        self.fusions = nn.ModuleList()
        self.unshuffle_ratios = unshuffle_ratios

        for i, (c, ratio) in enumerate(zip(channels, unshuffle_ratios)):
            fusion_dim = c
            if ratio > 0:
                # after unshuffle, RGB channels are multiplied
                fusion_dim = c * (4 ** ratio)

            self.fusions.append(nn.ModuleDict({
                "cross_attn": CrossAttention(fusion_dim, num_heads),
                "mix": nn.Sequential(
                    nn.Conv2d(fusion_dim, c, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c, c, 3, padding=1),
                ) if fusion_dim != c else nn.Identity(),
                "post_norm": nn.GroupNorm(min(32, c), c),
            }))

    def forward(self, depth_features: list, rgb_features: list):
        """
        depth_features: list of (B, C, H, W)
        rgb_features:   list of (B, C, H, W)
        Returns: list of fused (B, C, H, W)
        """
        fused = []
        for i, (df, rf) in enumerate(zip(depth_features, rgb_features)):
            ratio = self.unshuffle_ratios[i]

            if ratio > 0:
                # Pixel unshuffle RGB to match depth resolution
                rf = make_even(rf)
                for _ in range(ratio):
                    rf = F.pixel_unshuffle(rf, 2)  # H/2, W/2, C*4

            # Cross-attention fusion
            out = self.fusions[i]["cross_attn"](df, rf)
            out = self.fusions[i]["mix"](out)
            out = self.fusions[i]["post_norm"](out)
            fused.append(out)

        return fused


# ---------------------------------------------------------------------------
# A2GS-style stage-by-stage cross-attention blocks
# ---------------------------------------------------------------------------

class A2GSMlp(nn.Module):
    """MLP with DWConv in the middle, matching A2GS FFN structure.

    Linear(in_dim → hidden_dim) → GELU → DWConv3×3 → GELU → Linear(hidden_dim → in_dim)
    DWConv operates in 2D space for local spatial mixing.
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                groups=hidden_features)
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, x_size: tuple):
        H, W = x_size
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act1(x)
        x = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act2(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class A2GSCrossAttention(nn.Module):
    """A2GS-style cross-attention: Q from depth, K/V from RGB.

    Key differences from standard cross-attention:
    - Conv2d projections instead of Linear (preserves spatial layout)
    - 3×3 depthwise conv on Q and K/V before attention (local spatial mixing)
    - Cosine similarity (L2-normalize Q, K) + learnable temperature
    """

    def __init__(self, dim: int, y_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim        # depth channels
        self.y_dim = y_dim    # RGB channels
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q branch: depth features
        self.x_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x_q_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                groups=dim, bias=bias)

        # K/V branch: RGB features
        self.y_kv = nn.Conv2d(y_dim, y_dim * 2, kernel_size=1, bias=bias)
        self.y_kv_dw = nn.Conv2d(y_dim * 2, y_dim * 2, kernel_size=3, stride=1,
                                 padding=1, groups=y_dim * 2, bias=bias)

        # Output
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_size: tuple):
        """
        x: (B, N, dim) — depth features in sequence format
        y: (B, N, y_dim) — RGB features in sequence format
        x_size: (H, W)
        Returns: (B, N, dim) — depth features updated with RGB structure
        """
        B, H, W = x.shape[0], x_size[0], x_size[1]

        # Sequence → 2D
        x_2d = x.transpose(1, 2).contiguous().view(B, self.dim, H, W)
        y_2d = y.transpose(1, 2).contiguous().view(B, self.y_dim, H, W)

        # Q from depth: Conv1×1 → DWConv3×3
        q = self.x_q_dw(self.x_q(x_2d))

        # K, V from RGB: Conv1×1 → DWConv3×3 → chunk
        kv = self.y_kv_dw(self.y_kv(y_2d))
        k, v = kv.chunk(2, dim=1)

        # Multi-head reshape: (B, head*C, H, W) → (B, head, C, H*W)
        q = q.reshape(B, self.num_heads, self.dim // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, self.y_dim // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, self.y_dim // self.num_heads, H * W)

        # Cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention with learnable temperature
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        # Reshape back: (B, head, C, H*W) → (B, head*C, H, W)
        out = out.reshape(B, self.dim, H, W)
        out = self.project_out(out)
        out = out.flatten(2).transpose(1, 2)  # → (B, N, dim)
        return out


class A2GSCrossTransformerBlock(nn.Module):
    """A2GS-style cross-attention Transformer block.

    For depth branch (queries RGB):
      LN(depth) + LN(RGB) → CrossAttn → +shortcut(depth) → LN → MLP(DWConv) → +shortcut
    """

    def __init__(self, dim: int, y_dim: int, num_heads: int,
                 mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)       # depth (Q source)
        self.norm_kv = nn.LayerNorm(y_dim)    # RGB (K/V source)
        self.norm2 = nn.LayerNorm(dim)        # post-attn

        self.attn = A2GSCrossAttention(dim, y_dim, num_heads, bias=True)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = A2GSMlp(dim, mlp_hidden, drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_size: tuple):
        """
        x: (B, N, dim) — depth features
        y: (B, N, y_dim) — RGB features
        x_size: (H, W)
        Returns: (B, N, dim) — updated depth features
        """
        shortcut = x
        x = self.attn(self.norm_q(x), self.norm_kv(y), x_size) + shortcut
        x = x + self.mlp(self.norm2(x), x_size)
        return x


# ---------------------------------------------------------------------------
# Original batch fusion module (kept for backward compatibility / ablation)
# ---------------------------------------------------------------------------

class A2GSTranFusion(nn.Module):
    """Full A²GSTran-style fusion pipeline.

    Fuses multi-scale Swin features from RGB edge branch and depth branch.
    Depth features serve as Q in cross-attention to selectively retrieve
    structurally relevant RGB edge information.
    """

    def __init__(self, feature_dims: list, num_heads: int = 8,
                 unshuffle_ratios: list = None):
        super().__init__()
        if unshuffle_ratios is None:
            unshuffle_ratios = [0, 0, 0, 0]

        self.ms_fusion = MultiScaleFusion(feature_dims, num_heads, unshuffle_ratios)

        # feature dimension equalization
        self.depth_adjust = nn.ModuleList([
            nn.Conv2d(feature_dims[i], feature_dims[i], 1)
            for i in range(len(feature_dims))
        ])
        self.rgb_adjust = nn.ModuleList([
            nn.Conv2d(feature_dims[i], feature_dims[i], 1)
            for i in range(len(feature_dims))
        ])

    def forward(self, depth_features: list, rgb_features: list):
        assert len(depth_features) == len(rgb_features)

        df_adjusted = [self.depth_adjust[i](depth_features[i])
                       for i in range(len(depth_features))]
        rf_adjusted = [self.rgb_adjust[i](rgb_features[i])
                       for i in range(len(rgb_features))]

        fused = self.ms_fusion(df_adjusted, rf_adjusted)
        return fused
