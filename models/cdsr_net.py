"""
CDSR-Net: Color-guided Depth Super-Resolution Network.
Swin Transformer encoder → A²GSTran asymmetric cross-attention fusion → Mamba decoder.

Architecture:
  RGB edge (HR) ──→ SwinEncoder ──→ multi-scale features ──┐
                                                             ├──→ A²GSTranFusion ──→ MambaDecoder ──→ HR Depth
  LR Depth (HR*) ──→ SwinEncoder ──→ multi-scale features ──┘
  (* upsampled to HR via bicubic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_encoder import SwinEncoder
from .fusion import A2GSTranFusion
from .mamba_decoder import MambaDecoder


class CDSRNet(nn.Module):
    """Color-guided Depth Super-Resolution Network.

    RGB branch processes edge map (1 or 3 channels).
    Depth branch processes upsampled LR depth (1 channel).
    Both Swin encoders share architecture but NOT weights.
    """

    def __init__(self,
                 # Swin encoder
                 embed_dim: int = 96,
                 depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1,
                 # Fusion
                 fusion_num_heads: int = 8,
                 # Mamba decoder
                 d_state: int = 16,
                 expand: int = 2,
                 # Super-resolution
                 scale: int = 8):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.scale = scale

        # RGB edge encoder (3-ch input; SwinEncoder has its own weight init)
        self.rgb_encoder = SwinEncoder(
            in_chans=3, embed_dim=embed_dim, depths=depths,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # Depth encoder (1-ch input)
        self.depth_encoder = SwinEncoder(
            in_chans=1, embed_dim=embed_dim, depths=depths,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        feature_dims = [embed_dim * (2 ** max(0, i)) for i in range(len(depths))]

        # A²GSTran fusion
        self.fusion = A2GSTranFusion(
            feature_dims=feature_dims,
            num_heads=fusion_num_heads,
            unshuffle_ratios=[0, 0, 0, 0]
        )

        # Mamba decoder
        self.decoder = MambaDecoder(
            feature_dims=feature_dims,
            scale=scale,
            d_state=d_state,
            expand=expand,
        )

    def forward(self, lr_depth: torch.Tensor, rgb: torch.Tensor):
        """
        lr_depth: (B, 1, H_lr, W_lr) — low-resolution depth
        rgb:      (B, 3, H_hr, W_hr) — RGB edge map (or raw RGB)
        Returns:  (B, 1, H_hr, W_hr) — super-resolved depth
        """
        # Upsample LR depth to HR size
        B, _, H_lr, W_lr = lr_depth.shape
        _, _, H_hr, W_hr = rgb.shape
        lr_depth_hr = F.interpolate(lr_depth, size=(H_hr, W_hr),
                                     mode="bilinear", align_corners=False)

        # Feature extraction
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(lr_depth_hr)

        # A²GSTran fusion (depth as Q, RGB as K/V)
        fused_features = self.fusion(depth_features, rgb_features)

        # Mamba reconstruction
        hr_depth = self.decoder(fused_features, depth_features)

        return hr_depth


def build_cdsr_net(scale: int = 8, **kwargs) -> CDSRNet:
    """Factory function with reasonable defaults."""
    defaults = dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        fusion_num_heads=8,
        d_state=16,
        expand=2,
        scale=scale,
    )
    defaults.update(kwargs)
    return CDSRNet(**defaults)
