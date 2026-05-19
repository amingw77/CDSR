"""
RGB edge map extraction for structural guidance.
- compute_sobel_numpy(): used by dataset.py preprocessing (OpenCV, CPU)
- SobelEdgeExtractor: PyTorch nn.Module for online/inference edge extraction
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sobel_numpy(rgb_uint8: np.ndarray) -> np.ndarray:
    """Extract Sobel edge map from RGB image (numpy uint8, HWC).

    Used by NYUDepthSR dataset for efficient CPU preprocessing.
    Returns: (H, W) float32 edge map in [0, 1].
    """
    import cv2
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(edge, 0, 1)


def get_sobel_kernels():
    """Sobel kernels for gradient computation."""
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
    return sobel_x, sobel_y


class SobelEdgeExtractor(nn.Module):
    """PyTorch Sobel edge extractor for tensor inputs (B, 3, H, W).

    Computes per-RGB-channel gradient magnitude, then takes max across channels
    to capture structure regardless of color channel.
    Output: (B, 1, H, W) edge map in [0, 1].
    """

    def __init__(self):
        super().__init__()
        sobel_x, sobel_y = get_sobel_kernels()
        # (3, 1, 3, 3) for grouped convolution per channel
        kernel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        kernel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        gx = F.conv2d(rgb, self.kernel_x, padding=1, groups=3)
        gy = F.conv2d(rgb, self.kernel_y, padding=1, groups=3)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        edge = grad_mag.max(dim=1, keepdim=True)[0]  # max over RGB channels
        return edge


class LearnableEdgeRefiner(nn.Module):
    """Lightweight edge refinement: enhances Sobel edges with learned residual."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, edge: torch.Tensor) -> torch.Tensor:
        return self.conv(edge) * edge + edge  # residual refinement
