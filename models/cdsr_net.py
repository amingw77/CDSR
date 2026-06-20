"""
CDSR-Net v5.7: Color-guided Depth Super-Resolution Network.
Preprocessing convs + Swin depth encoder + EchoSR CHRG RGB encoder +
A2GS interleaved cross-attention (3 pairs, matching A2GS) +
depth feature reuse + channel attention fusion.

Architecture:
  depth PreConv ─┐                            RGB PreConv ─┐
  depth PatchEmbed│                            RGB ConvEmbed │
  (seq)          │                            (2D)          │
  depth_stage0   │                            CHRG_stage0    │
       d0 (seq)  │                              g0 (2D)      │
       │         │                              │            │
       │         └─── cross_d0(depth Q, RGB KV)─┘ (flatten)  │
       ▼                    │                                 │
  depth_stage1 ─→ d1 (seq)  │                                 │
                 ┌── cross_c0(RGB Q, depth KV)───► CHRG_stage1│
                 │                                         g1 (2D)
                 │              (×3 total)                    │
                 │       ┌── cross_d2 ─→ depth_stage3 ─→ d3  │
                 │       │                        │         │
                 │     g2 (2D)                     │         │
                 └───────┼── cross_c2 ──→ CHRG_stage3 ─→ g3 │
                         │                      (2D)         │
                         │                                   │
                Concat[d0,d1,d2,d3] → project → ChannelAttn → Upsample → HR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_encoder import SwinStage, PatchEmbed
from .fusion import A2GSCrossTransformerBlock
from .echosr import CHRG


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
    """CDSR-Net v5.7: Swin depth encoder + EchoSR CHRG RGB encoder + cross-attn fusion.

    4 depth SwinStages (2 SwinBlocks each) + 4 RGB CHRGs (5 CHBs each).
    3 cross_d + 3 cross_c blocks in A2GS interleaved order (matching A2GS).
    Depth features from all 4 stages are concatenated (A2GS-style reuse),
    then fused with the final RGB features via channel attention.
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
                 scale: int = 8,
                 chrg_depths: list = None,
                 chrg_mlp_ratio: float = 1.5):
        super().__init__()
        if block_depths is None:
            block_depths = [2, 2, 2, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_stages = len(block_depths)  # 4
        self.total_stages = self.num_stages
        self.patch_size = 4

        # CHRG depths: 5 CHBs per group (EchoSR paper default)
        if chrg_depths is None:
            chrg_depths = [5] * self.total_stages  # [5, 5, 5, 5]

        # Preprocessing conv layers (before PatchEmbed)
        self.depth_pre = nn.Conv2d(1, 1, 3, padding=1)
        self.rgb_pre = nn.Conv2d(3, 3, 3, padding=1)

        # Patch embedding (depth) + strided conv embed (RGB, stays 2D for CHRG)
        self.depth_patch_embed = PatchEmbed(4, 1, embed_dim)
        self.rgb_embed = nn.Conv2d(3, embed_dim, 4, stride=4)

        # Drop path rates
        depth_total_blocks = sum(block_depths)
        dpr = [drop_path_rate * i / max(1, depth_total_blocks - 1)
               for i in range(depth_total_blocks)]

        # Build depth stages (Swin) and RGB stages (EchoSR CHRG)
        self.depth_stages = nn.ModuleList()
        self.rgb_stages = nn.ModuleList()
        self.cross_d_blocks = nn.ModuleList()
        self.cross_c_blocks = nn.ModuleList()

        idx = 0
        for i in range(self.total_stages):
            # Depth stage: SwinTransformer
            stage_dpr = dpr[idx:idx + block_depths[i]]
            idx += block_depths[i]
            self.depth_stages.append(
                SwinStage(dim=embed_dim, depth=block_depths[i],
                          num_heads=num_heads[i], window_size=window_size,
                          mlp_ratio=mlp_ratio, do_merge=False,
                          drop_path_rates=stage_dpr)
            )

            # RGB stage: EchoSR CHRG
            self.rgb_stages.append(
                CHRG(dim=embed_dim, depth=chrg_depths[i],
                     mlp_ratio=chrg_mlp_ratio, drop_path=drop_path_rate)
            )

            # Cross-attention (3 pairs for 4 stages, matching A2GS)
            if i < self.total_stages - 1:
                self.cross_d_blocks.append(
                    A2GSCrossTransformerBlock(embed_dim, embed_dim,
                                              fusion_num_heads, cross_mlp_ratio)
                )
                self.cross_c_blocks.append(
                    A2GSCrossTransformerBlock(embed_dim, embed_dim,
                                              fusion_num_heads, cross_mlp_ratio)
                )

        # Project concatenated depth features: 4×C → C
        self.depth_concat_proj = nn.Conv2d(embed_dim * self.total_stages, embed_dim, 1)

        # A2GS-style three-output supervision (each branch has its own feature head + output)
        feat_dim = 64
        self.feat_depth = nn.Sequential(
            nn.Conv2d(embed_dim, feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
        )
        self.feat_rgb = nn.Sequential(
            nn.Conv2d(embed_dim, feat_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
        )
        # Channel attention + conv for fusion
        self.fusion_ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim * 2, feat_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 8, feat_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.fusion_conv = nn.Conv2d(feat_dim * 2, feat_dim, 3, padding=1)
        self.feat_fused = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.GELU(),
        )
        # Shared PixelShuffle + output conv
        self.to_depth = nn.Sequential(
            nn.Conv2d(feat_dim, 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, lr_depth: torch.Tensor, rgb: torch.Tensor):
        B, _, H_lr, W_lr = lr_depth.shape
        _, _, H_hr, W_hr = rgb.shape

        lr_depth_hr = F.interpolate(lr_depth, size=(H_hr, W_hr),
                                     mode="bicubic", align_corners=False)
        lr_depth_res = lr_depth_hr

        pad_h = (self.patch_size - H_hr % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_hr % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            lr_depth_hr = F.pad(lr_depth_hr, (0, pad_w, 0, pad_h))
            lr_depth_res = F.pad(lr_depth_res, (0, pad_w, 0, pad_h))
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h))

        # Preprocessing conv layers
        lr_depth_hr = self.depth_pre(lr_depth_hr)
        rgb = self.rgb_pre(rgb)

        # Patch embedding (depth: seq) + conv embed (RGB: 2D)
        x, H, W = self.depth_patch_embed(lr_depth_hr)
        y = self.rgb_embed(rgb)
        x_size = (H, W)

        # Stage 0: initial transformer (before any cross-attention)
        x, H, W = self.depth_stages[0](x, H, W)                   # d0 (seq)
        y = self.rgb_stages[0](y)                                  # g0 (2D)

        # Collect depth features from all stages (A2GS-style reuse)
        depth_feats = [x]  # d0

        # 3 groups: cross_d → depth_stage → cross_c → rgb_stage
        for i in range(self.total_stages - 1):
            # cross_d: depth Q, RGB KV
            y_seq = y.flatten(2).transpose(1, 2)
            x = self.cross_d_blocks[i](x, y_seq, x_size)
            x, H, W = self.depth_stages[i + 1](x, H, W)            # d_{i+1}
            depth_feats.append(x)                                   # store

            # cross_c: RGB Q, depth KV
            y_seq = y.flatten(2).transpose(1, 2)
            y_seq = self.cross_c_blocks[i](y_seq, x, x_size)
            y = y_seq.transpose(1, 2).contiguous().view(B, -1, H, W)
            y = self.rgb_stages[i + 1](y)                           # g_{i+1} (2D)

        # Concat all depth features: (B, N, C) × 4 → (B, N, 4C) → 2D → project
        x_cat = torch.cat(depth_feats, dim=2)                       # (B, N, 4C)
        x_cat = x_cat.transpose(1, 2).contiguous().view(B, -1, H, W)  # (B, 4C, H, W)
        x_cat = self.depth_concat_proj(x_cat)                       # (B, C, H, W)

        # A2GS-style three-output supervision
        # Depth branch: concat features → feat → to_depth
        fd = self.feat_depth(x_cat)
        depth_sr = self.to_depth(fd) + lr_depth_res

        # RGB branch: final RGB features → feat → to_depth
        fr = self.feat_rgb(y)
        rgb_sr = self.to_depth(fr) + lr_depth_res

        # Fused: concat(fd, fr) → channel attention → conv → feat → to_depth
        cat = torch.cat([fd, fr], dim=1)             # (B, 128, H, W)
        ca = self.fusion_ca(cat)
        fused_feat = self.fusion_conv(cat * ca)       # (B, 64, H, W)
        fused_feat = self.feat_fused(fused_feat)
        fused_sr = self.to_depth(fused_feat) + lr_depth_res

        return depth_sr, rgb_sr, fused_sr


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
        scale=scale,
    )
    defaults.update(kwargs)
    return CDSRNet(**defaults)
