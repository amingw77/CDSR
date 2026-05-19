# CDSR-Net v3.0: Lightweight Color-guided Depth Super-Resolution

Swin Transformer → A2GS 交错交叉注意力 → Mamba 解码器

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
                         │ (B,3136,48)  H/4               │ (B,3136,48)
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
          ┌─────────────┬───────────┼───────────┬─────────────┐
          │             │           │           │             │
          ▼ d0          ▼ d1        ▼           ▼ d2         ▼ d3
    (B,48,H/4,W/4) (B,48,H/4,W/4)  ...    (B,48,H/4,W/4) (B,48,H/4,W/4)
          │             │           │           │             │
          └─────────────┴───────────┴───────────┴─────────────┘
                                    │
                            (B,192,H/4,W/4)
                                    │
                         ┌──────────┴──────────┐
                         │  Mamba Decoder       │
                         │                      │
                         │  Conv(192→48)        │
                         │  MambaBlock ×2       │
                         │  Conv(48→64→64→16)   │
                         │  PixelShuffle(×4)    │
                         │  Conv(1→1)           │
                         └──────────┬──────────┘
                                    │
                              (B,1,H,W)
                          Super-Resolved Depth
```

### 关键变化 (v3.0 vs v2.0)

| 项目 | v2.0 | v3.0 |
|------|------|------|
| embed_dim | 96 | 48 |
| mlp_ratio | 4.0 | 2.0 |
| PatchMerging | 4级层级下采样 | 无，全阶段同分辨率 |
| 交叉注意力 | 4 cross_d + 3 cross_c | 3 cross_d + 2 cross_c（交错） |
| 解码器 | 多尺度 U-Net | 同分辨率 concat + Mamba |
| 参数量 | 74.60M | 0.71M |

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

所有阶段无 PatchMerging，同分辨率同维度。深度编码器 4 个阶段，RGB 编码器 3 个阶段（g3 无下游消费者）。

### SwinBlock 内部

```
x → LN → W-MSA/SW-MSA (窗口7×7) → +shortcut → LN → MLP(DWConv) → +shortcut → out
                                                    │
                                                    ├─ Linear(dim → dim*mlp_ratio)
                                                    ├─ GELU
                                                    ├─ DWConv3×3 (空间局部混合)
                                                    ├─ GELU
                                                    └─ Linear(dim*mlp_ratio → dim)
```

---

## 三、A2GS 交错交叉注意力

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
```

3 对 cross_d（深度查询 RGB）+ 2 对 cross_c（RGB 查询深度），最后一对 cross_c 省略（g3 无下游消费者）。

### A2GSCrossAttention 细节

```
Q (depth): Conv1×1(48→48) → DWConv3×3 → L2-normalize
K,V (RGB): Conv1×1(48→96) → DWConv3×3 → chunk2 → K: L2-normalize, V: 直接使用

Attn = softmax(Q @ K^T * temperature)
Out = Attn @ V → Conv1×1(48→48) → 输出序列格式
```

- temperature: 可学习参数 (num_heads, 1, 1)
- 余弦相似度通过 L2 归一化实现

### A2GSCrossTransformerBlock

```
x → LN_Q(x) + LN_KV(y) → CrossAttention → +shortcut(x) → LN → MLP(DWConv) → +shortcut → out
```

---

## 四、Mamba 解码器

```
Concat(d0,d1,d2,d3)  (B,192,H/4,W/4)
        │
   Conv(192→96,3×3) → GELU → Conv(96→48,3×3)
        │
   MambaBlock(dim=48, expand=2) ×2
        │
   Conv(48→64) → GELU → Conv(64→64) → GELU → Conv(64→16)
        │
   PixelShuffle(×4)    16ch → 1ch
        │
   Conv(1→1,3×3)
        │
   (B,1,H,W)  HR Depth
```

### MambaBlock（SS2D）

- 5 方向扫描：水平、垂直、对角线、水平翻转、垂直翻转
- 每方向：Depthwise Conv → SiLU 门控
- 5 方向输出取平均
- 残差连接 + 可学习缩放因子 α

---

## 五、参数量

| 模块 | 参数量 |
|------|--------|
| depth_encoder (4 stages) | 0.17M |
| rgb_encoder (3 stages) | 0.12M |
| cross_d_blocks (×3) | 0.06M |
| cross_c_blocks (×2) | 0.04M |
| MambaDecoder | 0.31M |
| **Total** | **0.71M** |

---

## 六、训练策略

| 参数 | 值 |
|------|-----|
| 超分倍率 | ×8 |
| 数据集 | NYU Depth v2 (1000 train / 449 test) |
| HR 裁剪 | 224×224 |
| Batch size | 16 |
| 优化器 | AdamW (lr=1e-4, wd=1e-4) |
| 学习率调度 | StepLR (step=100, gamma=0.5) |
| 梯度裁剪 | max_norm=1.0 |
| 混合精度 | AMP (GradScaler) |
| 损失函数 | L1 + 0.5×GradientLoss |
| 数据增强 | 随机翻转、旋转、裁剪 |
