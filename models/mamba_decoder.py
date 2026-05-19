"""
Mamba-based reconstruction decoder for depth map super-resolution.
Uses 2D selective scan (SS2D) blocks for global context refinement,
then PixelShuffle upsampling to HR resolution.
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


class MambaDecoder(nn.Module):
    """Mamba-based decoder for depth map reconstruction.

    Takes multi-scale fused features (all at same H/4 resolution),
    concatenates them, refines with Mamba blocks, and upsamples to HR.
    """

    def __init__(self, feature_dims: list, scale: int = 8,
                 d_state: int = 16, expand: int = 2):
        super().__init__()
        total_dim = sum(feature_dims)
        base_dim = feature_dims[0]

        # Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(total_dim, base_dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),
        )

        # Mamba refinement blocks
        self.mamba_blocks = nn.Sequential(
            MambaBlock(base_dim, d_state, expand),
            MambaBlock(base_dim, d_state, expand),
        )

        # Final upsampling: H/4 → H (×4 PixelShuffle)
        self.final_ups = nn.Sequential(
            nn.Conv2d(base_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, features: list, encoder_depth_features: list = None):
        """
        features: list of (B, C, H, W) at same spatial resolution
        Returns: HR depth (B, 1, H*4, W*4)
        """
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.mamba_blocks(x)
        x = self.final_ups(x)
        return x
