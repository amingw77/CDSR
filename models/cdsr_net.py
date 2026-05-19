"""
CDSR-Net: Color-guided Depth Super-Resolution Network.
Swin Transformer encoder → stage-by-stage A2GS cross-attention → Mamba decoder.

Architecture (A2GS style):
  RGB edge (HR) ──→ PatchEmbed ──→ Stage0 ──→ CrossAttn ←── Stage0 ←── PatchEmbed ←── LR Depth (HR*)
                                     │  ↑                    ↑  │
                                  Stage1 │                  │ Stage1
                                     │  ↓                    ↓  │
                                  CrossAttn ──→ ... ←── CrossAttn
                                     │                         │
                                  [multi-scale depth features]  │
                                     │                         │
                                     └──────→ MambaDecoder → HR Depth
  (* upsampled to HR via bicubic)
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
    Each Swin stage is followed by bidirectional cross-attention —
    depth queries RGB (cross_d), RGB queries depth (cross_c).
    Multi-scale depth features are collected and fed to the Mamba decoder.
    """

    def __init__(self,
                 # Swin encoder
                 embed_dim: int = 96,
                 depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
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
        feature_dims = [embed_dim * (2 ** max(0, i)) for i in range(len(depths))]

        # RGB encoder (3-ch input)
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

        # Per-stage cross-attention blocks (A2GS style)
        # cross_d[i]: depth queries RGB → updated depth (used by decoder)
        # cross_c[i]: RGB queries depth → updated RGB (fed to next stage)
        # Last cross_c is omitted since its output has no downstream consumer
        self.cross_d_blocks = nn.ModuleList()
        self.cross_c_blocks = nn.ModuleList()
        for i in range(self.num_stages):
            dim = feature_dims[i]
            self.cross_d_blocks.append(
                A2GSCrossTransformerBlock(dim, dim, fusion_num_heads, cross_mlp_ratio)
            )
            if i < self.num_stages - 1:
                self.cross_c_blocks.append(
                    A2GSCrossTransformerBlock(dim, dim, fusion_num_heads, cross_mlp_ratio)
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

        # Pad input to multiple of patch_size (safety — dataset already aligns to 28)
        pad_h = (self.patch_size - H_hr % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_hr % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            lr_depth_hr = F.pad(lr_depth_hr, (0, pad_w, 0, pad_h))
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h))

        # Patch embedding (separate for each branch)
        x, H, W = self.depth_encoder.patch_embed(lr_depth_hr)  # (B, N, C), H, W
        y, _, _ = self.rgb_encoder.patch_embed(rgb)

        # Stage-by-stage processing with interleaved cross-attention
        x_size = (H, W)
        depth_features = []

        for i in range(self.num_stages):
            H_cur, W_cur = x_size

            # Swin stages (each includes optional PatchMerging + N SwinBlocks)
            x, H_new, W_new = self.depth_encoder.stages[i](x, H_cur, W_cur)
            y, _, _ = self.rgb_encoder.stages[i](y, H_cur, W_cur)
            x_size = (H_new, W_new)

            # Bidirectional cross-attention (depth↔RGB)
            x = self.cross_d_blocks[i](x, y, x_size)  # depth queries RGB
            if i < self.num_stages - 1:
                y = self.cross_c_blocks[i](y, x, x_size)  # RGB queries depth

            # Save depth feature AFTER cross-attn (RGB structure enriched)
            feat = x.transpose(1, 2).contiguous().view(B, -1, H_new, W_new)
            depth_features.append(feat)

        # Mamba decoder reconstruction
        hr_depth = self.decoder(depth_features, depth_features)
        return hr_depth


def build_cdsr_net(scale: int = 8, **kwargs) -> CDSRNet:
    """Factory function with reasonable defaults."""
    defaults = dict(
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        fusion_num_heads=8,
        cross_mlp_ratio=2.0,
        d_state=16,
        expand=2,
        scale=scale,
    )
    defaults.update(kwargs)
    return CDSRNet(**defaults)
