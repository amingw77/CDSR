"""
CDSR-Net: Color-guided Depth Super-Resolution Network.
Swin Transformer encoder → stage-by-stage A2GS cross-attention → Mamba decoder.

Architecture (A2GS style, all stages at same resolution):
  RGB ──→ PatchEmbed ──→ Stage0 ──┐
                                   ├─→ CrossAttn_d0 → Stage1_d ──→ CrossAttn_c0 → Stage1_rgb
  LR Depth ──→ PatchEmbed ──→ Stage0 ──┘                                     │
       ┌──────────────────────────────────────────────────────────────────────┘
       │  ... → CrossAttn_d1 → Stage2_d → CrossAttn_c1 → Stage2_rgb → ...
       │
       └──→ [multi-scale depth features] → MambaDecoder → HR Depth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_encoder import SwinEncoder
from .fusion import A2GSCrossTransformerBlock
from .mamba_decoder import MambaDecoder


class CDSRNet(nn.Module):
    """Color-guided Depth Super-Resolution Network.

    Stage-by-stage cross-attention (A2GS style):
    Each Swin stage output is followed by interleaved bidirectional cross-attention —
    depth queries RGB (cross_d), RGB queries depth (cross_c).
    All stages operate at the same spatial resolution (no PatchMerging).
    Multi-scale depth features are collected and fed to the Mamba decoder.
    """

    def __init__(self,
                 # Swin encoder
                 embed_dim: int = 48,
                 depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 2.0,
                 drop_path_rate: float = 0.1,
                 # Fusion (cross-attention)
                 fusion_num_heads: int = 8,
                 cross_mlp_ratio: float = 2.0,
                 # Mamba decoder
                 d_state: int = 16,
                 expand: int = 2,
                 # Super-resolution
                 scale: int = 8):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.scale = scale
        self.num_stages = len(depths)
        self.patch_size = 4
        feature_dims = [embed_dim] * len(depths)

        # RGB encoder (3-ch input, 3 stages — g3 has no consumer)
        rgb_depths = depths[:-1]
        rgb_num_heads = num_heads[:len(rgb_depths)]
        self.rgb_encoder = SwinEncoder(
            in_chans=3, embed_dim=embed_dim, depths=rgb_depths,
            num_heads=rgb_num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # Depth encoder (1-ch input)
        self.depth_encoder = SwinEncoder(
            in_chans=1, embed_dim=embed_dim, depths=depths,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # cross_d: depth queries RGB (3 blocks, for stage pairs 0-1, 1-2, 2-3)
        # cross_c: RGB queries depth (2 blocks, for stage pairs 0-1, 1-2 only;
        #   no cross_c for stage 2-3 since g3 has no downstream consumer)
        self.cross_d_blocks = nn.ModuleList()
        self.cross_c_blocks = nn.ModuleList()
        for i in range(self.num_stages - 1):
            dim = feature_dims[i]
            self.cross_d_blocks.append(
                A2GSCrossTransformerBlock(dim, dim, fusion_num_heads, cross_mlp_ratio)
            )
            if i < self.num_stages - 2:
                self.cross_c_blocks.append(
                    A2GSCrossTransformerBlock(dim, dim, fusion_num_heads, cross_mlp_ratio)
                )

        # Mamba decoder: all features at same resolution, concat + Mamba + upsample
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

        # Pad input to multiple of patch_size (safety)
        pad_h = (self.patch_size - H_hr % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_hr % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            lr_depth_hr = F.pad(lr_depth_hr, (0, pad_w, 0, pad_h))
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h))

        # Patch embedding (separate for each branch)
        x, H, W = self.depth_encoder.patch_embed(lr_depth_hr)
        y, _, _ = self.rgb_encoder.patch_embed(rgb)
        x_size = (H, W)

        # Stage 0: both branches independently
        x, H, W = self.depth_encoder.stages[0](x, H, W)  # d0
        y, _, _ = self.rgb_encoder.stages[0](y, H, W)    # g0

        depth_features = []
        feat = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        depth_features.append(feat)  # d0 (raw depth features)

        # Interleaved stages 1-3 with cross-attention
        for i in range(self.num_stages - 1):
            # cross_d[i]: depth queries RGB → enhanced depth
            x = self.cross_d_blocks[i](x, y, x_size)

            # Next depth stage
            x, H, W = self.depth_encoder.stages[i + 1](x, H, W)  # d_{i+1}
            feat = x.transpose(1, 2).contiguous().view(B, -1, H, W)
            depth_features.append(feat)

            # cross_c[i]: RGB queries depth (skip last — no downstream consumer)
            if i < self.num_stages - 2:
                y = self.cross_c_blocks[i](y, x, x_size)
                y, _, _ = self.rgb_encoder.stages[i + 1](y, H, W)

        # Mamba decoder reconstruction
        hr_depth = self.decoder(depth_features, depth_features)
        return hr_depth


def build_cdsr_net(scale: int = 8, **kwargs) -> CDSRNet:
    """Factory function with reasonable defaults."""
    defaults = dict(
        embed_dim=48,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        fusion_num_heads=8,
        cross_mlp_ratio=2.0,
        d_state=16,
        expand=2,
        scale=scale,
    )
    defaults.update(kwargs)
    return CDSRNet(**defaults)
