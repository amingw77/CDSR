"""
CDSR-Net v4.2: Color-guided Depth Super-Resolution Network.
A2GS-style interleaved cross-attention → dual Mamba → cross-attn fusion.

Architecture (20 SwinBlocks total):
  depth PatchEmbed ──┐                            RGB PatchEmbed ──┐
  depth_stage0       │                            rgb_stage0       │
       d0  ──────────┼─→ cross_d0 ─→ depth_stage1 ─→ d1            │
                      │       ↑                        │            │
                      │     g0                          │            │
                      │       └── cross_c0 ──→ rgb_stage1 ─→ g1    │
                      │              ...                           │
                      │       ┌── cross_d3 ─→ depth_stage4 ─→ d4   │
                      │       │                        │            │
                      │     g3                          │            │
                      └───────┼── cross_c3 ──→ rgb_stage4 ─→ g4   │
                              │                                     │
                          MambaBlock                         MambaBlock
                              │                                     │
                              └──→ CrossAttn(depth Q, RGB KV) ─→ Upsample → HR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_encoder import SwinStage, PatchEmbed
from .fusion import A2GSCrossTransformerBlock
from .mamba_decoder import MambaBlock


class BranchDecoder(nn.Module):
    """Decoder for one branch: sequence features → HR depth map."""

    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.mamba = MambaBlock(dim, d_state, expand)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.mamba(x)
        x = self.upsample(x)
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)
        self.ch_attn_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
        )
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.ch_attn_shared(F.adaptive_avg_pool2d(x, 1))
        max_out = self.ch_attn_shared(F.adaptive_max_pool2d(x, 1))
        ch_attn = torch.sigmoid(avg_out + max_out)
        x = x * ch_attn
        sp_avg = torch.mean(x, dim=1, keepdim=True)
        sp_max, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn = torch.sigmoid(self.spatial_conv(torch.cat([sp_avg, sp_max], dim=1)))
        x = x * sp_attn
        return x


class CDSRNet(nn.Module):
    """CDSR-Net v4.2: A2GS interleaved cross-attention + dual Mamba + cross-attn fusion.

    5 depth stages + 5 RGB stages (2 SwinBlocks each = 20 total),
    4 cross_d + 4 cross_c blocks in A2GS interleaved order.
    Dual MambaBlocks refine features, then cross-attention (depth Q, RGB KV)
    fuses the two branches before upsampling to HR.
    """

    def __init__(self,
                 embed_dim: int = 48,
                 block_depths: list = None,
                 num_heads: list = None,
                 window_size: int = 7,
                 mlp_ratio: float = 2.0,
                 drop_path_rate: float = 0.1,
                 fusion_num_heads: int = 8,
                 cross_mlp_ratio: float = 2.0,
                 d_state: int = 16,
                 expand: int = 2,
                 scale: int = 8):
        super().__init__()
        if block_depths is None:
            block_depths = [2, 2, 2, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_stages = len(block_depths)
        self.patch_size = 4

        # Expand to 5 stages: initial + 4 cross-attn groups
        if len(block_depths) == 4:
            block_depths = [2] + block_depths  # [2, 2, 2, 2, 2]
        if len(num_heads) == 4:
            num_heads = num_heads + [num_heads[-1]]  # [3, 6, 12, 24, 24]

        self.total_stages = len(block_depths)

        # Patch embedding
        self.depth_patch_embed = PatchEmbed(4, 1, embed_dim)
        self.rgb_patch_embed = PatchEmbed(4, 3, embed_dim)

        # Drop path rates: 20 SwinBlocks (5 stages × 2 branches × 2 blocks)
        total_blocks = sum(block_depths) * 2
        dpr = [drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)] if total_blocks > 1 else [0.0] * total_blocks

        # Build depth stages and RGB stages
        self.depth_stages = nn.ModuleList()
        self.rgb_stages = nn.ModuleList()
        self.cross_d_blocks = nn.ModuleList()
        self.cross_c_blocks = nn.ModuleList()

        idx = 0
        for i in range(self.total_stages):
            # Depth stage
            stage_dpr = dpr[idx:idx + block_depths[i]]
            idx += block_depths[i]
            self.depth_stages.append(
                SwinStage(dim=embed_dim, depth=block_depths[i],
                          num_heads=num_heads[i], window_size=window_size,
                          mlp_ratio=mlp_ratio, do_merge=False,
                          drop_path_rates=stage_dpr)
            )

            # RGB stage
            stage_dpr = dpr[idx:idx + block_depths[i]]
            idx += block_depths[i]
            self.rgb_stages.append(
                SwinStage(dim=embed_dim, depth=block_depths[i],
                          num_heads=num_heads[i], window_size=window_size,
                          mlp_ratio=mlp_ratio, do_merge=False,
                          drop_path_rates=stage_dpr)
            )

            # Cross-attention (4 pairs for 5 stages)
            if i < self.total_stages - 1:
                self.cross_d_blocks.append(
                    A2GSCrossTransformerBlock(embed_dim, embed_dim,
                                              fusion_num_heads, cross_mlp_ratio)
                )
                self.cross_c_blocks.append(
                    A2GSCrossTransformerBlock(embed_dim, embed_dim,
                                              fusion_num_heads, cross_mlp_ratio)
                )

        # Dual MambaBlocks (unshared, for depth and RGB branch refinement)
        self.depth_mamba = MambaBlock(embed_dim, d_state, expand)
        self.rgb_mamba = MambaBlock(embed_dim, d_state, expand)

        # Cross-attention fusion: depth features as Q, RGB features as K/V
        self.fusion_cross = A2GSCrossTransformerBlock(embed_dim, embed_dim,
                                                      fusion_num_heads, cross_mlp_ratio)

        # Upsampling: H/4 → H
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, lr_depth: torch.Tensor, rgb: torch.Tensor):
        B, _, H_lr, W_lr = lr_depth.shape
        _, _, H_hr, W_hr = rgb.shape

        lr_depth_hr = F.interpolate(lr_depth, size=(H_hr, W_hr),
                                     mode="bicubic", align_corners=False)

        pad_h = (self.patch_size - H_hr % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_hr % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            lr_depth_hr = F.pad(lr_depth_hr, (0, pad_w, 0, pad_h))
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h))

        # Patch embedding
        x, H, W = self.depth_patch_embed(lr_depth_hr)
        y, _, _ = self.rgb_patch_embed(rgb)
        x_size = (H, W)

        # Stage 0: initial transformer (before any cross-attention)
        x, H, W = self.depth_stages[0](x, H, W)  # d0
        y, _, _ = self.rgb_stages[0](y, H, W)    # g0

        # 4 groups: cross_d → depth_stage → cross_c → rgb_stage
        for i in range(self.total_stages - 1):
            x = self.cross_d_blocks[i](x, y, x_size)          # depth Q, RGB KV
            x, H, W = self.depth_stages[i + 1](x, H, W)       # d_{i+1}
            y = self.cross_c_blocks[i](y, x, x_size)          # RGB Q, depth KV
            y, _, _ = self.rgb_stages[i + 1](y, H, W)         # g_{i+1}

        # Mamba refinement on each branch
        x_2d = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        y_2d = y.transpose(1, 2).contiguous().view(B, -1, H, W)
        x_mamba = self.depth_mamba(x_2d)
        y_mamba = self.rgb_mamba(y_2d)

        # Back to sequence format for cross-attention
        x_seq = x_mamba.flatten(2).transpose(1, 2)
        y_seq = y_mamba.flatten(2).transpose(1, 2)

        # Cross-attention fusion: depth Q, RGB KV
        fused = self.fusion_cross(x_seq, y_seq, x_size)

        # Upsample to HR
        fused_2d = fused.transpose(1, 2).contiguous().view(B, -1, H, W)
        out = self.upsample(fused_2d)

        # Global residual: bicubic upsampled LR depth
        out = out + lr_depth_hr
        return out


def build_cdsr_net(scale: int = 8, **kwargs) -> CDSRNet:
    """Factory function with reasonable defaults."""
    defaults = dict(
        embed_dim=48,
        block_depths=[2, 2, 2, 2],
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
