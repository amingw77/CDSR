"""Evaluation metrics for depth super-resolution.
All metrics computed on de-normalized depth values (A2GS style).
RMSE uses 6-pixel border clipping during evaluation.
"""
import torch
import torch.nn.functional as F
import numpy as np


def denormalize(pred, target, dmin, dmax):
    """De-normalize per-sample min-max normalization (A2GS style).
    dmin, dmax are stored as depth*100.
    After de-normalization, values are in centimeters.
    """
    pred = pred * (dmax - dmin) + dmin
    target = target * (dmax - dmin) + dmin
    return pred, target


def border_clip(x, border=6):
    """Clip border pixels from spatial dims (A2GS style)."""
    if x.dim() == 4:
        return x[:, :, border:-border, border:-border]
    elif x.dim() == 3:
        return x[:, border:-border, border:-border]
    else:
        return x[border:-border, border:-border]


def compute_rmse(pred: torch.Tensor, target: torch.Tensor,
                 dmin=None, dmax=None, eval_mode=False) -> float:
    """Root Mean Square Error, on de-normalized values if dmin/dmax given."""
    if dmin is not None and dmax is not None:
        pred, target = denormalize(pred, target, dmin, dmax)
    if eval_mode:
        pred = border_clip(pred)
        target = border_clip(target)
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor,
                dmin=None, dmax=None, eval_mode=False) -> float:
    """Mean Absolute Error."""
    if dmin is not None and dmax is not None:
        pred, target = denormalize(pred, target, dmin, dmax)
    if eval_mode:
        pred = border_clip(pred)
        target = border_clip(target)
    return torch.mean(torch.abs(pred - target)).item()


def compute_rel(pred: torch.Tensor, target: torch.Tensor,
                dmin=None, dmax=None, eval_mode=False, eps: float = 1e-6) -> float:
    """Mean Absolute Relative Error."""
    if dmin is not None and dmax is not None:
        pred, target = denormalize(pred, target, dmin, dmax)
    if eval_mode:
        pred = border_clip(pred)
        target = border_clip(target)
    return torch.mean(torch.abs(pred - target) / (target + eps)).item()


def compute_delta(pred: torch.Tensor, target: torch.Tensor,
                  dmin=None, dmax=None, eval_mode=False,
                  threshold: float = 1.25):
    """Percentage of pixels with max(pred/target, target/pred) < threshold."""
    if dmin is not None and dmax is not None:
        pred, target = denormalize(pred, target, dmin, dmax)
    if eval_mode:
        pred = border_clip(pred)
        target = border_clip(target)
    ratio = torch.max(pred / (target + 1e-6), target / (pred + 1e-6))
    return (ratio < threshold).float().mean().item() * 100


def compute_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on spatial gradients (edge-aware loss).
    Computed on normalized values — used during training only.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    def grad(x):
        return F.conv2d(x, sobel_x, padding=1), F.conv2d(x, sobel_y, padding=1)

    px, py = grad(pred)
    tx, ty = grad(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)
