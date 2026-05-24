"""
EchoSR components for CDSR-Net RGB branch.
Based on: EchoSR: Efficient context harnessing for lightweight image SR.
Components: GCE, LA, MRFE, CAFFN, CHB, COFB, CHRG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ('GELU', 'ReLU', 'SiLU')
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    """Learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


# ===================== EchoSR modules =====================

class GCE(nn.Module):
    """Global Context Estimation (GCP in paper).
    AdaptiveMaxPool → DWConv → variance modulation → upsample → element-wise multiply.
    """

    def __init__(self, dim, down_scale=8):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.down_scale = down_scale
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        _, _, h, w = x.shape
        x_s = self.dw_conv(F.adaptive_max_pool2d(
            x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        enhanced = x_s * self.alpha + x_v * self.belt
        x_l = x * F.interpolate(self.gelu(self.conv_1(enhanced)),
                                size=(h, w), align_corners=False, mode='bilinear')
        return x_l


class InceptionStyleDWConv2d(nn.Module):
    """MRFE: Multi-Scale Receptive Field Expansion.
    Splits channels into 4 groups → [Identity, DWConv(k1), DWConv(k2), DWConv(k3)].
    """

    def __init__(self, in_channels, branch_ratio=4, kernel_sizes=[0, 5, 11, 17]):
        super().__init__()
        assert in_channels % branch_ratio == 0
        gc = in_channels // branch_ratio
        kernel1, kernel2, kernel3 = kernel_sizes[1], kernel_sizes[2], kernel_sizes[3]

        self.dwconv_hw = nn.Conv2d(gc, gc, kernel_size=kernel1,
                                   padding=kernel1 // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=kernel2,
                                  padding=kernel2 // 2, groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=kernel3,
                                  padding=kernel3 // 2, groups=gc)
        self.split_indexes = (gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_5, x_11, x_17 = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_5), self.dwconv_w(x_11), self.dwconv_h(x_17)),
            dim=1,
        )


class LA(nn.Sequential):
    """Local Aggregation: PWC → GroupConv(groups=c/6) → PWC."""

    def __init__(self, dim: int, mlp_ratio=1.5):
        hidden = int(dim * mlp_ratio)
        super().__init__(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden // 6),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class CAFFN(nn.Module):
    """Channel Aggregation FFN with feature decomposition."""

    def __init__(self, embed_dims, kernel_size=3, act_type='GELU',
                 mlp_ratio=1.5, ffn_drop=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * mlp_ratio)

        self.fc1 = nn.Conv2d(embed_dims, self.feedforward_channels, 1)
        self.dwconv = nn.Conv2d(
            self.feedforward_channels, self.feedforward_channels,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            bias=True, groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(self.feedforward_channels, embed_dims, 1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(self.feedforward_channels, 1, 1)
        self.sigma = ElementScale(self.feedforward_channels, init_value=1e-5)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        return x + self.sigma(x - self.decompose_act(self.decompose(x)))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CHB(nn.Module):
    """Context-Harnessing Block = LA + (MRFE + λ·GCP) + CAFFN with residuals."""

    def __init__(self, dim, drop_path=0., mlp_ratio=1.5,
                 kernel_sizes=[0, 5, 11, 17], down_scale=8):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.LA = LA(dim, mlp_ratio)
        self.MRFE = InceptionStyleDWConv2d(dim, branch_ratio=4, kernel_sizes=kernel_sizes)
        self.GCE = GCE(dim, down_scale)
        self.CAFFN = CAFFN(dim, mlp_ratio=mlp_ratio, ffn_drop=drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale = nn.Parameter(0.1 * torch.ones(dim))  # λ, init 0.1

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.LA(x)
        x = self.MRFE(x) + self.GCE(x) * self.skip_scale.view(1, -1, 1, 1)
        x = self.CAFFN(self.norm2(x))
        x = shortcut + self.drop_path(x)
        return x


class COFB(nn.Module):
    """Cross-Scale Overlapping Fusion Block.
    1×1+GELU → DWConv(7)→DWConv(15)→1×1(attention gate) → 1×1.
    """

    def __init__(self, n_feats):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())
        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 15, 1, 15 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


class CHRG(nn.Module):
    """Context-Harnessing Residual Group: L×CHB + COFB + residual.

    Wraps sequence↔2D conversion to match SwinStage interface (for CDSR-Net).
    """

    def __init__(self, dim, depth=2, mlp_ratio=1.5, drop_path=0.,
                 kernel_sizes=[0, 5, 11, 17], down_scale=8):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            CHB(dim=dim, drop_path=drop_path, mlp_ratio=mlp_ratio,
                kernel_sizes=kernel_sizes, down_scale=down_scale)
            for _ in range(depth)
        ])
        self.cofb = COFB(dim)

    def forward(self, x):
        shortcut = x
        for blk in self.blocks:
            x = blk(x)
        x = self.cofb(x) + shortcut
        return x

    def forward_seq(self, x, H, W):
        """Sequence format interface: (B,N,C) → 2D → CHRG → 2D → (B,N,C)"""
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x_2d = self.forward(x_2d)
        x_seq = x_2d.flatten(2).transpose(1, 2).contiguous()
        return x_seq, H, W
