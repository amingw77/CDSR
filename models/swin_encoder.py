"""
Swin Transformer encoder for CDSR.
Dual-branch: one for RGB edge map, one for LR depth map (upsampled to HR).
Outputs multi-scale features at 4 stages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int):
    """Partition into non-overlapping windows.
    x: (B, H, W, C) -> (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Reverse window partition.
    windows: (B * num_windows, window_size, window_size, C) -> (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def generate_shift_mask(H: int, W: int, window_size: int, shift_size: int):
    """Generate attention mask for shifted window attention.

    After cyclic shift, windows at the boundary contain patches from
    non-adjacent regions. The mask prevents cross-region attention.
    """
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) for Swin Transformer blocks."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        if self.scale_by_keep:
            x = x / keep_prob
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer block: W-MSA or SW-MSA + MLP, with DropPath."""

    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, shift_size: int = 0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padded feature map dims (needed for window partition to work)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = H + pad_b, W + pad_r

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = generate_shift_mask(H_pad, W_pad, self.window_size, self.shift_size)
            attn_mask = attn_mask.to(x.device)
        else:
            shifted_x = x
            attn_mask = None

        # window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(x_windows, attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)

        # crop back to original size
        if pad_r > 0 or pad_b > 0:
            shifted_x = shifted_x[:, :H, :W, :]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP with DropPath
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Split image into patches and embed."""

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (B, C, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x, H, W


class PatchMerging(nn.Module):
    """Merge 2x2 patches: (B, H*W, C) -> (B, H/2 * W/2, 2C)"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # Pad to even H/W so 0::2 and 1::2 have matching sizes
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(self.norm(x))
        return x, H_pad // 2, W_pad // 2


class SwinStage(nn.Module):
    """One Swin stage: optional patch merging + Swin blocks."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, do_merge: bool = True,
                 drop_path_rates: list | None = None):
        super().__init__()
        if do_merge:
            self.patch_merging = PatchMerging(dim)
            dim = dim * 2
        else:
            self.patch_merging = None

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(SwinBlock(
                dim, num_heads, window_size, mlp_ratio, dropout,
                shift_size, drop_path_rates[i]
            ))

    def forward(self, x: torch.Tensor, H: int, W: int):
        if self.patch_merging is not None:
            x, H, W = self.patch_merging(x, H, W)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W


class SwinEncoder(nn.Module):
    """Swin Transformer encoder for feature extraction.

    Handles arbitrary input sizes by padding feature maps to be
    divisible by window_size at each stage.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96, depths: list = None,
                 num_heads: list = None, window_size: int = 7, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path_rate: float = 0.1):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.window_size = window_size
        self.patch_size = 4
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.stages = nn.ModuleList()
        self.num_stages = len(depths)
        dim = embed_dim

        # linear drop_path schedule
        total_blocks = sum(depths)
        dpr = [drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        for i in range(len(depths)):
            stage_dpr = dpr[block_idx:block_idx + depths[i]]
            block_idx += depths[i]
            stage = SwinStage(
                dim=dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                do_merge=(i > 0),
                drop_path_rates=stage_dpr,
            )
            self.stages.append(stage)
            dim = dim * 2 if i > 0 else dim

        self.out_dims = [embed_dim * (2 ** max(0, i)) for i in range(len(depths))]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        # Pad input to multiples of patch_size
        B, C, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x, Hp, Wp = self.patch_embed(x)

        features = []
        for stage in self.stages:
            x, Hp, Wp = stage(x, Hp, Wp)
            B = x.shape[0]
            feat = x.transpose(1, 2).contiguous().view(B, -1, Hp, Wp)
            features.append(feat)

        return features  # list of (B, C_i, H_i, W_i)
