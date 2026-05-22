# CDSR-Net v4.2: Color-guided Depth Super-Resolution Network

Swin Transformer → A2GS 交错交叉注意力 → 双 Mamba → 交叉注意力融合

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
                    │ PatchEmbed│                       │ PatchEmbed│
                    │ 1→48ch    │                       │ 3→48ch    │
                    │ stride=4  │                       │ stride=4  │
                    └────┬────┘                        └────┬────┘
                         │ (B,N,48) H/4×W/4               │ (B,N,48)
                         │                                  │
     ╔═══════════════════╪══════════════════════════════════╪═══════════╗
     ║ STAGE 0           ▼                                  ▼           ║
     ║           ┌──────────────┐                   ┌──────────────┐    ║
     ║           │  SwinBlock   │                   │  SwinBlock   │    ║
     ║           │  ×2 (W/SW)   │                   │  ×2 (W/SW)   │    ║
     ║           │  dim=48      │                   │  dim=48      │    ║
     ║           │  heads=3     │                   │  heads=3     │    ║
     ║           └──────┬──────┘                   └──────┬──────┘    ║
     ║                  │ d0                               │ g0        ║
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
     ║                       │  SwinBlock   │                          ║
     ║                       │  ×2 (W/SW)   │                          ║
     ║                       │  heads=6     │                          ║
     ║                       └──────┬──────┘                          ║
     ║                              │ g1                                ║
     ╚══════════════════════════════╪═══════════════════════════════════╝
                                    │
                          (重复 3 次，共 4 组)
                                    │
                    ┌───────────────┴───────────────┐
                    │ d4 (B,N,48)                   │ g4 (B,N,48)
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

## 二、Swin Transformer 编码器

### 结构参数

| 阶段 | SwinBlock 数 | 注意力头数 | 窗口 | 输出维度 | 输出分辨率 |
|------|-------------|-----------|------|---------|-----------|
| PatchEmbed | — | — | — | 48 | H/4 × W/4 |
| Stage 0 | 2 | 3 | 7 | 48 | H/4 × W/4 |
| Stage 1 | 2 | 6 | 7 | 48 | H/4 × W/4 |
| Stage 2 | 2 | 12 | 7 | 48 | H/4 × W/4 |
| Stage 3 | 2 | 24 | 7 | 48 | H/4 × W/4 |
| Stage 4 | 2 | 24 | 7 | 48 | H/4 × W/4 |

两个分支各有 5 个 stage，共 20 个 SwinBlock。所有 stage 无 PatchMerging，同分辨率同维度。

### SwinBlock 内部

```
x → LN → W-MSA/SW-MSA (窗口7×7) → +shortcut → LN → MLP → +shortcut → out
                                                    │
                                                    ├─ Linear(dim → dim*mlp_ratio)
                                                    ├─ GELU
                                                    ├─ Dropout
                                                    ├─ Linear(dim*mlp_ratio → dim)
                                                    └─ Dropout
```

---

## 三、A2GS 交错交叉注意力（编码器内）

### 模式

```
d0, g0 (Stage0 输出)
  → cross_d0: depth(Q) ← RGB(K/V) → d0'
  → Stage1 depth → d1
  → cross_c0: RGB(Q) ← depth(K/V) → g0'
  → Stage1 RGB → g1
  → cross_d1: depth(Q) ← RGB(K/V) → d1'
  → Stage2 depth → d2
  → cross_c1: RGB(Q) ← depth(K/V) → g1'
  → Stage2 RGB → g2
  → cross_d2: depth(Q) ← RGB(K/V) → d2'
  → Stage3 depth → d3
  → cross_c2: RGB(Q) ← depth(K/V) → g2'
  → Stage3 RGB → g3
  → cross_d3: depth(Q) ← RGB(K/V) → d3'
  → Stage4 depth → d4
  → cross_c3: RGB(Q) ← depth(K/V) → g3'
  → Stage4 RGB → g4
```

4 对 cross_d + 4 对 cross_c，交错执行，每对交叉注意力的 Q 源交替互换。

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

## 四、解码器：双 Mamba + 交叉注意力融合

### 流程

```
d4 (B,N,48)                            g4 (B,N,48)
  │                                      │
  │ reshape → (B,48,H/4,W/4)             │ reshape → (B,48,H/4,W/4)
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
- depth 主动查询 RGB 的结构引导信息
- 结构与编码器内交叉注意力完全一致

---

## 五、参数量分布

| 模块 | 参数量 |
|------|--------|
| depth_patch_embed | 816 |
| rgb_patch_embed | 2,352 |
| depth_encoder (5 stages × 2 blocks) | 208,514 |
| rgb_encoder (5 stages × 2 blocks) | 208,514 |
| cross_d_blocks (×4) | 85,856 |
| cross_c_blocks (×4) | 85,856 |
| depth_mamba | 15,360 |
| rgb_mamba | 15,360 |
| fusion_cross | 21,464 |
| upsample | 84,750 |
| **Total** | **~0.73M** |

---

## 六、训练策略

| 参数 | 值 |
|------|-----|
| 超分倍率 | ×8 |
| 数据集 | NYU Depth v2 (1000 train / 449 test) |
| HR 裁剪 | 224×224 (LR = 28×28) |
| Batch size | 16 |
| 优化器 | AdamW (lr=1e-4, wd=1e-4) |
| 学习率调度 | StepLR (step=100, gamma=0.5) |
| 梯度裁剪 | max_norm=1.0 |
| 混合精度 | AMP (GradScaler) |
| 损失函数 | L1 + 0.5×GradientLoss (Sobel) |
| 数据增强 | 随机翻转、旋转、裁剪 |

---

## 七、版本演变

| 版本 | 关键变化 | 参数量 |
|------|---------|--------|
| v2.0 | Swin+Mamba, PatchMerging, 大维度 | 74.60M |
| v3.0 | 去 PatchMerging, dim=48, 单 MambaDecoder | 0.71M |
| v3.1 | CosineAnnealingLR → StepLR | 0.71M |
| v4.1 | 双分支解码器 + CBAM 融合 + 全局残差 | 0.78M |
| v4.2 | 双 Mamba + 交叉注意力融合替换 CBAM + 双解码器 | 0.73M |
