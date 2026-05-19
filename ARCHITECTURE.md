# CDSR-Net: Color-guided Depth Super-Resolution Network

Swin Transformer → A²GSTran 非对称交叉注意力 → Mamba 解码器

---

## 一、网络架构总览

```
RGB边缘图 (HR, 3×224×224) ──→ SwinEncoder ──→ 4级多尺度特征 ──┐
                                                                   ├──→ A²GSTranFusion ──→ MambaDecoder ──→ HR深度图
LR深度图 (HR, 1×224×224) ──→ SwinEncoder ──→ 4级多尺度特征 ──┘
  (双三次上采样后)                            (深度做Q, RGB做K/V)
```

### 关键设计原则

- **RGB 输入端**：使用 Sobel 边缘图替代原始彩色图，从源头消除颜色/纹理干扰，只保留结构边缘信息
- **交叉注意力**：深度特征做 Query，RGB 边缘特征做 Key/Value，深度主动检索 RGB 中的结构信息，天然过滤纹理噪声
- **残差连接**：交叉注意力输出 = Q + CrossAttn(Q, K, V)，Q 来源的残差连接保证深度自身特征不丢失
- **逐样本归一化**：深度值做 per-sample min-max 归一化到 [0, 1]，训练 loss 在归一化值上计算（稳定梯度），评估指标反归一化到 cm 后计算（与 A2GS 一致）

---

## 二、阶段一：Swin Transformer 特征提取

### 输入

| 分支 | 输入 | 通道 | 分辨率 |
|------|------|------|--------|
| RGB 分支 | Sobel 边缘图 | 3 | 224×224 |
| 深度分支 | LR 深度图双三次上采样（归一化后） | 1 | 224×224 |

### Swin Encoder 结构

| 阶段 | 操作 | 输出分辨率 | 通道数 | Block 数 | 注意力头数 |
|------|------|-----------|--------|----------|-----------|
| Patch Embed | Conv 4×4, stride 4 | 56×56 | 96 | — | — |
| Stage 1 | 2× SwinBlock (W-MSA + SW-MSA) | 56×56 | 96 | 2 | 3 |
| Stage 2 | PatchMerging + 2× SwinBlock | 28×28 | 192 | 2 | 6 |
| Stage 3 | PatchMerging + 6× SwinBlock | 14×14 | 384 | 6 | 12 |
| Stage 4 | PatchMerging + 2× SwinBlock | 7×7 | 768 | 2 | 24 |

- 窗口大小：7×7
- MLP 扩展比：4.0
- 相对位置编码
- DropPath（Stochastic Depth）：0.1，线性递增
- RGB 编码器和深度编码器**结构相同、权重独立**（不共享）

### 输出

4 级多尺度特征列表：
```
[(B, 96,  56, 56),
 (B, 192, 28, 28),
 (B, 384, 14, 14),
 (B, 768, 7, 7)  ]
```

---

## 三、阶段二：A²GSTran 非对称交叉注意力融合

### 核心公式

$$\text{Fused}_i = \text{GroupNorm}(\text{Mix}(\text{CrossAttn}(Q_{\text{depth}}, K_{\text{rgb}}, V_{\text{rgb}})))$$

其中交叉注意力为标准形式：

$$\text{CrossAttn}(Q, K, V) = Q + \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q（深度）驱动交互**：深度"主动检索"RGB 中结构相关的区域
- **残差从 Q**：`Output = Q_source + Attn(Q, K, V)`，保证深度自身特征不丢失
- **K/V 来自 RGB 边缘图**：提供结构参考信息

### 多尺度融合

对 4 个尺度分别进行交叉注意力融合，每个尺度：

```
深度特征 (B, C, H, W) ──── Q ────┐
                                  ├── CrossAttn ──→ Mix (1×1 Conv) ──→ GroupNorm ──→ 融合特征
RGB边缘特征 (B, C, H, W) ─ K/V ──┘
```

### 像素反混洗（Pixel Unshuffle）预留

当 RGB 特征分辨率高于深度特征时，使用 pixel unshuffle 进行无损下采样：
- `PixelUnshuffle(scale=2)`: (H, W, C) → (H/2, W/2, C×4)
- 相比池化，不丢失任何结构信息

### 融合参数

| 参数 | 值 |
|------|-----|
| 融合尺度数 | 4 |
| 每尺度注意力头数 | 8 |
| 反混洗比例 | [0, 0, 0, 0]（同分辨率） |
| 归一化 | GroupNorm (32 groups) |

---

## 四、阶段三：Mamba 重建解码器

### 设计思路

从最粗尺度 (7×7) 逐步上采样到目标分辨率 (224×224)，每层用 Mamba SS2D 块进行长程依赖建模，跳跃连接来自深度编码器同尺度特征。

### 解码器结构

```
融合特征[3] (7×7, 768) ──→ Bottleneck (2× MambaBlock) ──→ UpBlock ──┐
                                                                      │ + 跳跃连接
融合特征[2] (14×14, 384) ── 深度编码器特征[2] ──────────────────────→ UpBlock ──┐
                                                                                 │ + 跳跃连接
融合特征[1] (28×28, 192) ── 深度编码器特征[1] ────────────────────────────────→ UpBlock ──┐
                                                                                            │ + 跳跃连接
融合特征[0] (56×56, 96)  ── 深度编码器特征[0] ─────────────────────────────────────────→ UpBlock
                                                                                            │
                                                                                    Final Upsample (×4)
                                                                                            │
                                                                                    HR 深度图 (1×224×224)
```

### UpBlock 结构

```
输入特征 ──→ PixelShuffle(×2) ──→ + Skip ──→ MambaBlock ──→ Conv3×3 → Conv3×3 (+残差) ──→ 输出
```

### MambaBlock（SS2D 简化实现）

- 5 方向扫描：水平、垂直、对角线、水平翻转、垂直翻转
- 每方向：Depthwise Conv 3×3 → SiLU 门控
- 5 方向输出取平均
- 残差连接 + 可学习缩放因子 α

### 解码器参数

| 参数 | 值 |
|------|-----|
| 瓶颈 Mamba 块数 | 2 |
| 每 UpBlock Mamba 块数 | 1 |
| 状态维度 d_state | 16 |
| 扩展比 expand | 2 |
| 上采样方式 | PixelShuffle（2× 每阶段） |
| 最终上采样 | PixelShuffle(×4) + Conv1×1 → 1 通道（×4 因 PatchEmbed stride=4） |

---

## 五、网络参数统计

| 模块 | 参数量 |
|------|--------|
| RGB SwinEncoder | 27.52M |
| Depth SwinEncoder | 27.51M |
| A²GSTran Fusion | 4.71M |
| Mamba Decoder | 25.83M |
| **总计** | **85.58M** |

---

## 六、训练策略

### 数据集

| 项目 | 设置 |
|------|------|
| 数据集 | NYU Depth v2 (labeled) |
| 训练样本 | 1000 对 × repeat=10 = 10000/epoch |
| 测试样本 | 449 对 |
| RGB 输入 | Sobel 边缘图（3 通道复制） |
| 深度范围 | 0 ~ 10m (float32)，逐样本 min-max 归一化到 [0,1] |
| 反归一化 | 评估时反归一化到 cm（min×100, max×100，与 A2GS 一致） |

### 超参数

| 参数 | 值 |
|------|-----|
| 超分倍率 | ×8 |
| HR 裁剪尺寸 | 224×224（28 的倍数，patch=4 × window=7） |
| Batch size | 4 |
| 训练轮数 | 200 |
| 每 epoch 样本数 | 10000（1000 张 × repeat=10，随机裁剪+增强） |
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| 权重衰减 | 1e-4 |
| 学习率调度 | CosineAnnealingLR (eta_min=1e-6) |
| 梯度裁剪 | max_norm=1.0 |
| 混合精度 | AMP (GradScaler) |
| DropPath | 0.1（线性递增） |

### 损失函数

$$\mathcal{L} = \lambda_1 \cdot \text{L1}(I_{pred}, I_{gt}) + \lambda_{edge} \cdot \text{GradientLoss}(I_{pred}, I_{gt})$$

| 损失项 | 权重 | 说明 |
|--------|------|------|
| L1 Loss | 1.0 | 逐像素绝对误差 |
| Gradient Loss | 0.5 | Sobel 梯度域的 L1 损失，强化边缘保真 |

### 数据增强（仅训练集）

- 随机水平翻转（p=0.5）
- 随机垂直翻转（p=0.5）
- 随机 90° 旋转（p=0.5）
- 随机裁剪 256×256

---

## 七、文件结构

```
CDSR_Net/
├── config.py                 # 配置文件（路径跨平台，--data_root 可覆盖）
├── train.py                  # 训练脚本（tqdm 进度条，AMP + 梯度裁剪）
├── test.py                   # 评估脚本（保存全部449张 pred/gt/error 可视化）
├── requirements.txt          # 依赖列表
├── ARCHITECTURE.md           # 本文档
├── models/
│   ├── __init__.py
│   ├── swin_encoder.py       # Swin Transformer 编码器
│   ├── fusion.py             # A²GSTran 交叉注意力融合
│   ├── mamba_decoder.py      # Mamba 重建解码器
│   └── cdsr_net.py           # 完整网络组装
├── data/
│   ├── __init__.py
│   └── dataset.py            # NYU 数据集加载器
├── utils/
│   ├── __init__.py
│   ├── edge_extraction.py    # Sobel 边缘提取
│   └── metrics.py            # 评估指标
├── checkpoints/              # 模型保存目录
└── logs/                     # 日志目录
```

---

## 八、使用方式

```bash
# 训练（指定数据集路径）
python train.py --data_root /path/to/nyu_labeled --batch_size 4 --epochs 200 --scale 8

# 评估 + 保存可视化（全部449张 pred/gt/error）
python test.py --data_root /path/to/nyu_labeled --checkpoint checkpoints/cdsr_net_best.pth --scale 8

# 恢复训练
python train.py --data_root /path/to/nyu_labeled --resume checkpoints/cdsr_net_final.pth
```

### 依赖

- PyTorch >= 2.0 (CUDA)
- torchvision
- numpy
- opencv-python (cv2)
- Pillow
