"""Channel Attention Fusion for CDSR-Net."""

import torch
import torch.nn as nn


class ChannelAttentionFusion(nn.Module):
    """Channel attention fusion: concat two branches → channel attention → conv.

    Fuses depth and RGB features by concatenating along the channel dimension,
    applying channel attention (SE-style: avgpool → FC → ReLU → FC → sigmoid),
    then a 3×3 convolution to project back to the output dimension.
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.fuse = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, (dim * 2) // reduction, 1),
            nn.ReLU(),
            nn.Conv2d((dim * 2) // reduction, dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuse two feature maps of the same spatial size.

        Args:
            a: (B, C, H, W) first branch (e.g. depth).
            b: (B, C, H, W) second branch (e.g. RGB).

        Returns:
            (B, C, H, W) fused features.
        """
        cat = torch.cat([a, b], dim=1)          # (B, 2C, H, W)
        weight = self.ca(cat)                   # (B, 2C, 1, 1)
        return self.fuse(cat * weight)          # (B, C, H, W)
