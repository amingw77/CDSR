"""
Mamba-based reconstruction decoder for depth map super-resolution.
Uses 2D selective scan (SS2D) blocks with progressive upsampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Mamba SS2D block with cross-directional scanning.

    Simplified but effective: uses 4-direction spatial scanning
    with depthwise conv + gating to approximate SS2D behavior.
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.expand = expand
        inner_dim = dim * expand

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)

        # Convolution for local context (scanned in 4 directions)
        self.conv_h = nn.Conv2d(inner_dim, inner_dim, (3, 1), padding=(1, 0), groups=inner_dim, bias=False)
        self.conv_w = nn.Conv2d(inner_dim, inner_dim, (1, 3), padding=(0, 1), groups=inner_dim, bias=False)
        self.conv_diag1 = nn.Conv2d(inner_dim, inner_dim, 3, padding=1, groups=inner_dim, bias=False)

        # Output
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        # Learnable skip weight
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        shortcut = x

        x_norm = self.norm(x.flatten(2).transpose(1, 2))
        x_norm = x_norm.transpose(1, 2).contiguous().view(B, C, H, W)

        xz = self.in_proj(x_norm.flatten(2).transpose(1, 2))
        xz = xz.transpose(1, 2).contiguous().view(B, -1, H, W)
        x_conv, z = xz.chunk(2, dim=1)

        # Multi-direction scanning
        out_h = self.conv_h(x_conv)
        out_w = self.conv_w(x_conv)
        out_diag = self.conv_diag1(x_conv)
        # Flipped scans
        out_hf = self.conv_h(torch.flip(x_conv, [2]))
        out_wf = self.conv_w(torch.flip(x_conv, [3]))

        out = (out_h + out_w + out_diag + torch.flip(out_hf, [2]) + torch.flip(out_wf, [3])) / 5

        out = F.silu(z) * out
        out = out.flatten(2).transpose(1, 2)
        out = self.out_proj(out)
        out = out.transpose(1, 2).contiguous().view(B, C, H, W)

        return shortcut + self.alpha * out


class UpBlock(nn.Module):
    """Upsampling block with Mamba + skip connection."""

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int,
                 d_state: int = 16, expand: int = 2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.skip_proj = nn.Conv2d(skip_dim, out_dim, 1, bias=False) if skip_dim != out_dim else nn.Identity()
        self.mamba = MambaBlock(out_dim, d_state, expand)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.upsample(x)
        skip = self.skip_proj(skip)

        # Align spatial dims
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = x + skip
        x = self.mamba(x)
        x = self.conv(x) + x
        return x


class MambaDecoder(nn.Module):
    """Mamba-based U-Net decoder with progressive upsampling for depth map reconstruction.

    Takes multi-scale fused features from A²GSTran fusion,
    progressively upsamples from coarsest to finest scale,
    with Mamba blocks at each level for global context.
    """

    def __init__(self, feature_dims: list, scale: int = 8,
                 d_state: int = 16, expand: int = 2):
        super().__init__()
        # feature_dims: [C0, C1, C2, C3] from coarsest to finest (Swin stages)
        self.scale = scale
        dims = feature_dims[::-1]  # reverse: coarsest first

        # Bottleneck
        self.bottleneck = nn.Sequential(
            MambaBlock(dims[0], d_state, expand),
            MambaBlock(dims[0], d_state, expand),
        )

        # Up blocks (3 stages: S4→S3→S2→S1, each ×2 upsample)
        self.up_blocks = nn.ModuleList()
        cur_dim = dims[0]
        for i in range(1, len(dims)):
            self.up_blocks.append(
                UpBlock(cur_dim, dims[i], dims[i], d_state, expand)
            )
            cur_dim = dims[i]

        # Final upsampling: S1 (H/4) → HR (H), needs ×4 upsampling
        # because PatchEmbed uses stride=4
        self.final_ups = nn.Sequential(
            nn.Conv2d(cur_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, features: list, encoder_depth_features: list):
        """
        features: list of fused features (B, C_i, H_i, W_i) from coarsest to finest
        encoder_depth_features: same order, for skip connections
        Returns: HR depth (B, 1, H, W)
        """
        rev_feats = features[::-1]
        rev_skips = encoder_depth_features[::-1]

        x = self.bottleneck(rev_feats[0])

        for i, up in enumerate(self.up_blocks):
            x = up(x, rev_skips[i + 1])

        x = self.final_ups(x)
        return x
