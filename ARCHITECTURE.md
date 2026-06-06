# CDSR-Net v5.6: Color-Guided Depth Super-Resolution

CDSR-Net is designed for 8x color-guided depth super-resolution. The model takes a low-resolution depth map and the corresponding high-resolution RGB image as input, and predicts a high-resolution depth map.

Version 5.6 keeps the raw RGB and bicubic degradation pipeline from v5.5, and updates training stability and residual handling:

- RGB branch input: raw RGB image, normalized to `[0, 1]`.
- LR depth generation: bicubic downsampling from normalized HR depth.
- Training crop: `256 x 256` remains compatible with 8x SR, producing `32 x 32` LR depth.
- The old Sobel-edge guide path is kept in the dataset for ablation only, but training and testing now use `pre_extract_edge=False`.
- The global residual now uses the raw bicubic-upsampled LR depth, not the feature after `depth_pre`.
- Training uses AdamW with configured weight decay.
- Training and validation skip non-finite predictions/losses instead of letting one bad batch poison the epoch metrics.
- Validation runs every 5 epochs through epoch 150, then every 10 epochs.

---

## 1. Task Definition

Input:

- `lr_depth`: low-resolution depth, shape `(B, 1, H/8, W/8)`.
- `rgb`: high-resolution RGB guidance image, shape `(B, 3, H, W)`.

Output:

- `sr_depth`: super-resolved depth, shape `(B, 1, H, W)`.

For the default setting:

- HR crop size: `256 x 256`.
- LR depth size: `32 x 32`.
- Scale factor: `8`.

Depth values are normalized per sample with min-max normalization. Metrics are computed after de-normalization.

---

## 2. Data Pipeline

The dataset loader is implemented in `data/dataset.py`.

### 2.1 HR Depth Normalization

For each depth map:

```text
depth_norm = (depth - depth_min) / (depth_max - depth_min)
```

`depth_min` and `depth_max` are stored for de-normalized evaluation.

### 2.2 LR Depth Generation

The LR depth is generated from normalized HR depth using bicubic downsampling:

```python
F.interpolate(
    depth_tensor.unsqueeze(0).unsqueeze(0),
    size=(h // scale, w // scale),
    mode="bicubic",
    align_corners=False,
)
```

For the default `scale=8` and `crop_size=256`, this produces:

```text
HR depth: (1, 256, 256)
LR depth: (1, 32, 32)
```

### 2.3 RGB Guidance

In v5.6, the model uses raw RGB guidance:

```python
rgb_input = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
```

The Sobel edge option remains available through `pre_extract_edge=True`, but it is not used by the default training or testing scripts.

### 2.4 Size Alignment

Spatial size is trimmed to be compatible with LR generation and patch embedding:

```python
align = max(scale, 4)
```

For the default 8x setting, this keeps a `256 x 256` training crop unchanged.

---

## 3. Network Overview

```text
LR Depth (B,1,H/8,W/8)          RGB (B,3,H,W)
          |                            |
 Bicubic upsample to HR                |
          |                            |
      depth_pre                    rgb_pre
          |                            |
   PatchEmbed, stride 4          ConvEmbed, stride 4
          |                            |
   Depth Swin branch             RGB EchoSR/CHRG branch
          |                            |
          +---- interleaved cross-attention ----+
                                               |
          Depth Mamba                   RGB Mamba
                  \                       /
                   final cross-attention
                            |
                       PixelShuffle x4
                            |
                   residual + upsampled LR
                            |
                  SR Depth (B,1,H,W)
```

The network keeps both branches at `H/4 x W/4` feature resolution after embedding. There is no patch merging in the current main model.

---

## 4. Main Modules

### 4.1 Depth Branch

The depth branch processes bicubic-upsampled LR depth.

Main components:

- `depth_pre`: `Conv2d(1, 1, 3, padding=1)`.
- `depth_patch_embed`: patch embedding with patch size `4`.
- 5 Swin stages.
- Each Swin stage contains 2 Swin blocks.
- Window size: `8`.
- Embedding dimension: `64`.
- Number of heads: `[2, 2, 2, 2, 2]`.

The configured `block_depths=[2, 2, 2, 2]` is expanded internally to 5 stages:

```text
[2] + [2, 2, 2, 2] = [2, 2, 2, 2, 2]
```

All stages keep the same spatial resolution and channel dimension.

### 4.2 RGB Branch

The RGB branch now receives the raw RGB image.

Main components:

- `rgb_pre`: `Conv2d(3, 3, 3, padding=1)`.
- `rgb_embed`: `Conv2d(3, 64, kernel_size=4, stride=4)`.
- 5 EchoSR-style CHRG stages.
- Each CHRG contains 5 CHB blocks and one COFB block.

This branch is the largest part of the current model. It is useful for rich RGB guidance, but should be included in ablation studies because it contributes most of the parameter count.

### 4.3 Interleaved Cross-Attention

The encoder uses 4 groups of bidirectional cross-attention:

```text
Stage 0:
  depth_stage0
  rgb_stage0

Group i = 0..3:
  cross_d_i: depth queries RGB
  depth_stage_{i+1}
  cross_c_i: RGB queries depth
  rgb_stage_{i+1}
```

The cross-attention block follows the A2GS-style design:

- LayerNorm on query and key/value tokens.
- Conv2d projection for Q/K/V.
- Depthwise convolution for local spatial mixing.
- Cosine attention with learnable temperature.
- MLP with depthwise convolution.

### 4.4 Mamba Refinement

After the final encoder stage:

- Depth features are refined by `depth_mamba`.
- RGB features are refined by `rgb_mamba`.

The current `MambaBlock` is a lightweight 2D approximation with directional depthwise convolutions and gating. It is not a full selective scan implementation.

### 4.5 Final Fusion and Upsampling

Final fusion uses cross-attention with depth features as query and RGB features as key/value.

Then the fused feature map is upsampled by a `PixelShuffle(4)` reconstruction head:

```text
Conv2d(64 -> 64)
GELU
Conv2d(64 -> 64)
GELU
Conv2d(64 -> 16)
PixelShuffle(4)
Conv2d(1 -> 1)
```

A global residual connection adds the raw bicubic-upsampled LR depth. This tensor is preserved separately before `depth_pre`, so the residual path remains a clean depth baseline:

```python
lr_depth_hr = F.interpolate(lr_depth, size=(H_hr, W_hr), mode="bicubic", align_corners=False)
lr_depth_res = lr_depth_hr
...
lr_depth_hr = self.depth_pre(lr_depth_hr)
...
out = out + lr_depth_res
```

---

## 5. Current Configuration

Defined in `config.py`.

| Item | Value |
|---|---:|
| Version | 5.6 |
| Scale | 8 |
| Train split | 1000 |
| Test split | Remaining samples |
| Crop size | 256 |
| Repeat | 20 |
| Batch size | 8 |
| Epochs | 300 |
| Learning rate | 5e-4 |
| LR scheduler | StepLR |
| LR step | 100 |
| LR gamma | 0.5 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Gradient clipping | Disabled |
| Loss | L1 + 0.1 * gradient loss |
| AMP | Enabled |
| RGB input | Raw RGB |
| LR degradation | Bicubic downsampling |
| Validation interval | Every 5 epochs through epoch 150, then every 10 epochs |

When resuming from an older checkpoint, the optimizer parameter groups are re-synchronized to the current `lr` and `weight_decay`.

---

## 6. Parameter Count

Measured with the current v5.6 configuration:

```text
Total trainable parameters: 2,064,303
```

Approximate top-level distribution:

| Module | Parameters |
|---|---:|
| depth_pre | 10 |
| rgb_pre | 84 |
| depth_patch_embed | 1,088 |
| rgb_embed | 3,136 |
| depth_stages | 352,020 |
| rgb_stages | 1,240,345 |
| cross_d_blocks | 147,232 |
| cross_c_blocks | 147,232 |
| depth_mamba | 26,625 |
| rgb_mamba | 26,625 |
| fusion_cross | 36,808 |
| upsample | 83,098 |
| Total | 2,064,303 |

The RGB branch accounts for roughly 60% of the model parameters.

---

## 7. Training and Evaluation

Training entry:

```bash
python train.py --data_root /path/to/nyu_labeled
```

Evaluation entry:

```bash
python test.py --data_root /path/to/nyu_labeled --checkpoint checkpoints/cdsr_net_best.pth
```

Bicubic baseline:

```bash
python test.py --data_root /path/to/nyu_labeled --baseline
```

Evaluation metrics:

- RMSE
- MAE
- REL
- Delta1

Metrics are computed on de-normalized depth values, with 6-pixel border clipping during evaluation.

Training stability guards:

- If a training batch produces a non-finite prediction or loss, that batch is skipped and a warning is logged.
- If a validation batch produces non-finite predictions, that batch is skipped and a warning is logged.
- If every batch in an epoch or validation pass is skipped, training raises an error instead of reporting invalid metrics.

---

## 8. Redundant or Legacy Components

The repository still contains several modules kept for compatibility or ablation:

- `BranchDecoder`
- `CBAM`
- `SwinEncoder`
- `MambaDecoder`
- `A2GSTranFusion`
- `CrossAttention`

They are not used by the current `CDSRNet` forward path. If the project is prepared for release or paper artifact submission, these should either be moved into an ablation/legacy namespace or removed from the main exports to reduce confusion.

---

## 9. Recommended Ablations

To validate whether the v5.6 structure is necessary, run:

1. Raw RGB vs Sobel edge guidance.
2. CHRG depth `[5,5,5,5,5]` vs lighter `[3,3,3,3,3]` or `[2,2,2,2,2]`.
3. With and without `cross_c_blocks`.
4. Edge loss weights: `0`, `0.05`, `0.1`, `0.2`.
5. AdamW weight decay: `0`, `1e-5`, `1e-4`.

These ablations are important because the current RGB branch is expressive, but relatively heavy compared with the rest of the network.

---

## 10. Version History

| Version | Main Changes |
|---|---|
| v3.0 | Removed patch merging, reduced dimensions, used a single Mamba decoder. |
| v4.1 | Added dual branch decoding, CBAM fusion, and global residual. |
| v4.2 | Replaced CBAM fusion with cross-attention fusion and dual Mamba refinement. |
| v5.0 | Added preprocessing convolutions and EchoSR CHRG RGB branch. |
| v5.1 | Replaced RGB PatchEmbed with ConvEmbed and kept RGB features in 2D format. |
| v5.2 | Adjusted training choices and RGB/edge guidance experiments. |
| v5.3 | Added DWConv inside Swin MLP, changed crop to 256 and window to 8. |
| v5.4 | Increased embed dimension to 64, unified heads to 2, and updated training/test config. |
| v5.5 | Switched default guidance to raw RGB, changed LR depth degradation to bicubic, and fixed crop alignment for 8x SR. |
| v5.6 | Fixed the global residual to use raw bicubic depth, switched training to AdamW, added non-finite guards, changed validation cadence, and updated current training hyperparameters. |
