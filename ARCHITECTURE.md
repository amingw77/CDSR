# CDSR-Net v5.2: Color-guided Depth Super-Resolution Network

Swin Transformer (depth) + EchoSR CHRG (RGB) → A2GS 交错交叉注意力 → 双 Mamba → 交叉注意力融合

---

## 一、网络架构总览

```
                    LR Depth (B,1,28,28)               RGB (B,3,H,W)
                         │                                  │
                         │ Bicubic Upsample                 │
                         ▼                                  │
                    (B,1,H,W)                               │
                         │                                  │
                    ┌────┴────┐                        ┌────┴────┐
                    │ PreConv  │                       │ PreConv  │
                    │ 1→1, 3×3 │                       │ 3→3, 3×3 │
                    └────┬────┘                        └────┬────┘
                         │                                  │
                    ┌────┴────┐                        ┌────┴────┐
                    │ PatchEmbed│                       │ ConvEmbed │
                    │ 1→48ch    │                       │ 3→48ch    │
                    │ stride=4  │                       │ stride=4  │
                    └────┬────┘                        └────┬────┘
                         │ (B,N,48) H/4×W/4               │ (B,48,H/4,W/4) 2D
                         │                                  │
     ╔═══════════════════╪══════════════════════════════════╪═══════════╗
     ║ STAGE 0           ▼                                  ▼           ║
     ║           ┌──────────────┐                   ┌──────────────┐    ║
     ║           │  SwinBlock   │                   │    CHRG 0    │    ║
     ║           │  ×2 (W/SW)   │                   │  5×CHB+COFB  │    ║
     ║           │  dim=48      │                   │  dim=48      │    ║
     ║           │  heads=3     │                   └──────┬──────┘    ║
     ║           └──────┬──────┘                          │ g0        ║
     ║                  │ d0                                         ║
     ╠══════════════════╪══════════════════════════════════╪═══════════╣
     ║ CROSS_D0         ▼                                  │           ║
     ║        ┌──────────────────────────────────────────┐ │           ║
     ║        │      CrossAttention (depth Q, RGB KV)     │◄┘           ║
     ║        │      Conv2d + DWConv + Cosine + Temp      │             ║
     ║        └──────────────────┬───────────────────────┘             ║
     ║                           │ d0'                                  ║
     ╠═══════════════════════════╪══════════════════════════════════════╣
     ║ STAGE 1                   ▼                                      ║
     ║                  ┌──────────────┐                                ║
     ║                  │  SwinBlock   │                                ║
     ║                  │  ×2 (W/SW)   │                                ║
     ║                  │  heads=6     │                                ║
     ║                  └──────┬──────┘                                ║
     ║                         │ d1                                     ║
     ╠═════════════════════════╪════════════════════════════════════════╣
     ║ CROSS_C0                ▼                                       ║
     ║               ┌──────────────────────────────────────────┐      ║
     ║               │     CrossAttention (RGB Q, depth KV)      │      ║
     ║               │     g0 queries d1                         │      ║
     ║               └──────────────────┬───────────────────────┘      ║
     ║                                  │ g0'                           ║
     ╠══════════════════════════════════╪═══════════════════════════════╣
     ║ STAGE 1 RGB                      ▼                              ║
     ║                       ┌──────────────┐                          ║
     ║                       │    CHRG 1    │                          ║
     ║                       │  5×CHB+COFB  │                          ║
     ║                       └──────┬──────┘                          ║
     ║                              │ g1                                ║
     ╚══════════════════════════════╪═══════════════════════════════════╝
                                    │
                          (重复 3 次，共 4 组)
                                    │
                    ┌───────────────┴───────────────┐
                    │ d4 (B,N,48)                   │ g4 (B,48,H/4,W/4)
                    │                               │
                    ▼                               ▼
             MambaBlock                       MambaBlock
             (SS2D, 5方向扫描)               (SS2D, 5方向扫描)
             (B,48,H/4,W/4)                  (B,48,H/4,W/4)
                    │                               │
                    └───────────┬───────────────────┘
                                ▼
                   CrossAttention (depth Q, RGB KV)
                   Conv2d + DWConv + Cosine + Temp
                                │
                           (B,N,48)
                                │
                   Upsample: Conv(48→64→64→16)
                             PixelShuffle(×4)
                             Conv(1→1,3×3)
                                │
                           (B,1,H,W)
                                │
                          + lr_depth_hr
                                ▼
                      Super-Resolved Depth
```

---

## 二、预处理与嵌入层

两个分支各加入一个 3×3 卷积进行局部特征预处理，之后分别用不同方式嵌入到 48 维。

| 分支 | 预处理 | 嵌入（下采样+升维） | 输出格式 |
|------|--------|-------------------|---------|
| 深度 | Conv2d(1→1, 3×3, pad=1) | PatchEmbed: Conv2d(1→48, 4, stride=4) → flatten | (B,N,48) 序列 |
| RGB | Conv2d(3→3, 3×3, pad=1) | Conv2d(3→48, 4, stride=4) | (B,48,H/4,W/4) 2D |

RGB 分支使用普通 stride-4 卷积替代 PatchEmbed，全程保持 2D 格式，CHRG 直接处理 2D 特征，仅在交叉注意力边界临时转为序列。

---

## 三、深度分支：Swin Transformer 编码器

### 结构参数

| 阶段 | SwinBlock 数 | 注意力头数 | 窗口 | 输出维度 | 输出分辨率 |
|------|-------------|-----------|------|---------|-----------|
| PatchEmbed | — | — | — | 48 | H/4 × W/4 |
| Stage 0 | 2 | 3 | 7 | 48 | H/4 × W/4 |
| Stage 1 | 2 | 6 | 7 | 48 | H/4 × W/4 |
| Stage 2 | 2 | 12 | 7 | 48 | H/4 × W/4 |
| Stage 3 | 2 | 24 | 7 | 48 | H/4 × W/4 |
| Stage 4 | 2 | 24 | 7 | 48 | H/4 × W/4 |

5 个 stage，共 10 个 SwinBlock。所有 stage 无 PatchMerging，同分辨率同维度。

### SwinBlock 内部

```
x → LN → W-MSA/SW-MSA (窗口7×7) → +shortcut → LN → MLP → +shortcut → out
                                                    │
                                                    ├─ Linear(dim → dim×mlp_ratio)
                                                    ├─ GELU
                                                    ├─ Dropout
                                                    ├─ Linear(dim×mlp_ratio → dim)
                                                    └─ Dropout
```

---

## 四、RGB 分支：EchoSR CHRG 编码器

基于 EchoSR 论文的 Context-Harnessing Residual Group，替换原有的 SwinStage。

### 结构参数

| 阶段 | CHRG | CHB 数 | COFB | 输出维度 | 输出分辨率 |
|------|------|--------|------|---------|-----------|
| ConvEmbed | — | — | — | 48 | H/4 × W/4 |
| CHRG 0 | ✓ | 5 | ✓ | 48 | H/4 × W/4 |
| CHRG 1 | ✓ | 5 | ✓ | 48 | H/4 × W/4 |
| CHRG 2 | ✓ | 5 | ✓ | 48 | H/4 × W/4 |
| CHRG 3 | ✓ | 5 | ✓ | 48 | H/4 × W/4 |
| CHRG 4 | ✓ | 5 | ✓ | 48 | H/4 × W/4 |

5 个 CHRG，共 25 个 CHB。特征在 2D 格式 `(B,48,H/4,W/4)` 上处理，与交叉注意力模块交互时转换为序列格式 `(B,N,48)`。

### CHB（Context-Harnessing Block）内部

```
x → BN → LA → MRFE + λ·GCE → BN → CAFFN → +残差 → out
            │                                    │
            │  LA: 1×1 Conv(c→1.5c)              │
            │      GroupConv(3×3, groups=c/6)     │
            │      1×1 Conv(1.5c→c)              │
            │                                    │
            │  MRFE: 通道4等分                    │
            │       Identity | DWConv(k5)         │
            │       DWConv(k11) | DWConv(k17)     │
            │                                    │
            │  GCE: AdaptiveMaxPool(×8下采样) →   │
            │       DWConv(3×3) → α·x+β·var →   │
            │       1×1 Conv+GELU → Bilinear上采样 → ×x
            │       λ: 可学习系数(初始0.1)         │
            │                                    │
            └── CAFFN: 1×1 Conv(c→1.5c)          │
                       DWConv(3×3)               │
                       GELU                      │
                       Feature Decomposition     │
                       1×1 Conv(1.5c→c)          │
```

### COFB（Cross-Scale Overlapping Fusion Block）

```
x → 1×1 Conv+GELU → 注意门控[DWConv(k=7)→DWConv(k=15)→1×1 Conv] → ×x → 1×1 Conv → out
```

---

## 五、A2GS 交错交叉注意力（编码器内）

### 模式

深度分支全程序列格式 `(B,N,48)`，RGB 分支全程 2D 格式 `(B,48,H/4,W/4)`。
交叉注意力时 RGB 临时 flatten 为序列，交叉完成后转回 2D 供 CHRG 处理。

```
d0 (seq), g0 (2D)
  → g0 flatten → cross_d0: depth(Q) ← RGB(K/V) → d0' (seq)
  → Stage1 depth → d1 (seq)
  → g0 flatten → cross_c0: RGB(Q) ← depth(K/V) → g0' (seq) → reshape 2D
  → CHRG 1 → g1 (2D)
  → g1 flatten → cross_d1: depth(Q) ← RGB(K/V) → d1' (seq)
  → Stage2 depth → d2 (seq)
  → g1 flatten → cross_c1: RGB(Q) ← depth(K/V) → g1' (seq) → reshape 2D
  → CHRG 2 → g2 (2D)
  → g2 flatten → cross_d2: depth(Q) ← RGB(K/V) → d2' (seq)
  → Stage3 depth → d3 (seq)
  → g2 flatten → cross_c2: RGB(Q) ← depth(K/V) → g2' (seq) → reshape 2D
  → CHRG 3 → g3 (2D)
  → g3 flatten → cross_d3: depth(Q) ← RGB(K/V) → d3' (seq)
  → Stage4 depth → d4 (seq)
  → g3 flatten → cross_c3: RGB(Q) ← depth(K/V) → g3' (seq) → reshape 2D
  → CHRG 4 → g4 (2D)
```

4 对 cross_d + 4 对 cross_c，交错执行。

### A2GSCrossAttention 细节

```
Q: Conv1×1(48→48) → DWConv3×3 → L2-normalize
K,V: Conv1×1(48→96) → DWConv3×3 → chunk2 → K: L2-normalize, V: 直接使用

Attn = softmax(Q @ K^T * temperature)
Out = Attn @ V → Conv1×1(48→48) → 输出序列格式
```

- temperature: 可学习参数 `(num_heads, 1, 1)`
- 余弦相似度通过 L2 归一化实现

### A2GSCrossTransformerBlock

```
x → LN_Q(x) + LN_KV(y) → CrossAttention → +shortcut(x) → LN → MLP(DWConv) → +shortcut → out
```

MLP 使用 A2GSMlp：Linear → GELU → DWConv3×3 → GELU → Dropout → Linear → Dropout

---

## 六、解码器：双 Mamba + 交叉注意力融合

### 流程

```
d4 (B,N,48)                            g4 (B,48,H/4,W/4) 2D
  │                                      │
  │ reshape → (B,48,H/4,W/4)             │ (already 2D)
  ▼                                      ▼
MambaBlock (depth)                   MambaBlock (rgb)
(5方向SS2D扫描)                      (5方向SS2D扫描)
  │                                      │
  │ flatten → (B,N,48)                   │ flatten → (B,N,48)
  └──────────┬───────────────────────────┘
             ▼
   CrossAttention (depth Q, RGB KV)
   Conv2d + DWConv + Cosine + Temp
             │
        (B,N,48)
             │
        reshape → (B,48,H/4,W/4)
             │
        Conv(48→64,3×3) → GELU
        Conv(64→64,3×3) → GELU
        Conv(64→16,3×3)
        PixelShuffle(×4)    16ch → 1ch
        Conv(1→1,3×3)
             │
        (B,1,H,W)
             │
        + lr_depth_hr (bicubic)
             ▼
        HR Depth
```

### MambaBlock（SS2D）

- 输入/输出: `(B, C, H, W)` 2D 特征图
- 5 方向扫描：水平（3×1）、垂直（1×3）、对角线（3×3）、水平翻转、垂直翻转
- 每方向：Depthwise Conv → SiLU 门控
- 5 方向输出取平均
- 残差连接 + 可学习缩放因子 α

### 融合交叉注意力

- 深度 Mamba 输出作为 Q，RGB Mamba 输出作为 K/V
- 结构与编码器内交叉注意力完全一致

---

## 七、参数量分布

| 模块 | 参数量 |
|------|--------|
| depth_preconv | 9 |
| rgb_preconv | 27 |
| depth_patch_embed | 816 |
| rgb_embed | 2,352 |
| depth_encoder (5 SwinStages × 2 blocks) | 208,514 |
| rgb_encoder (5 CHRGs × 5 CHBs + 5 COFBs) | ~590,000 |
| cross_d_blocks (×4) | 85,856 |
| cross_c_blocks (×4) | 85,856 |
| depth_mamba | 15,360 |
| rgb_mamba | 15,360 |
| fusion_cross | 21,464 |
| upsample | 84,750 |
| **Total** | **~1.30M** |

---

## 八、训练策略

| 参数 | 值 |
|------|-----|
| 超分倍率 | ×8 |
| 数据集 | NYU Depth v2 (1000 train / 449 test) |
| RGB 输入 | 原始 RGB 图像（3 通道，不做 Sobel 边缘提取） |
| HR 裁剪 | 224×224 (LR = 28×28) |
| Batch size | 8 |
| 优化器 | Adam (lr=1e-4, no weight_decay) |
| 学习率调度 | StepLR (step=100, gamma=0.5) |
| 梯度裁剪 | 禁用 (grad_clip=0) |
| 混合精度 | AMP (GradScaler) |
| 损失函数 | L1 + 0.5×GradientLoss (Sobel) |
| 数据增强 | 随机翻转、旋转、裁剪 |

---

## 九、版本演变

| 版本 | 关键变化 | 参数量 |
|------|---------|--------|
| v2.0 | Swin+Mamba, PatchMerging, 大维度 | 74.60M |
| v3.0 | 去 PatchMerging, dim=48, 单 MambaDecoder | 0.71M |
| v3.1 | CosineAnnealingLR → StepLR | 0.71M |
| v4.1 | 双分支解码器 + CBAM 融合 + 全局残差 | 0.78M |
| v4.2 | 双 Mamba + 交叉注意力融合替换 CBAM + 双解码器 | 0.73M |
| v5.0 | 预处理卷积 + RGB分支EchoSR CHRG(5×5CHB)替换SwinStage | 1.30M |
| v5.1 | RGB分支ConvEmbed替代PatchEmbed(全程2D) + AdamW→Adam + lr=1e-4 | 1.30M |
| v5.2 | 关闭梯度裁剪 + RGB输入改为原始图像(不做Sobel边缘提取) | 1.30M |
