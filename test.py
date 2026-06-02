"""
Evaluation script for CDSR-Net: compute metrics and visualize results.
Saves LR depth, RGB guide, prediction, GT, and error maps for all test images.
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from models import build_cdsr_net
from data import NYUDepthSR
from utils.metrics import compute_rmse, compute_mae, compute_rel, compute_delta


@torch.no_grad()
def evaluate(model, loader, dataset, device, results_dir: str = None):
    if model is not None:
        model.eval()
    metrics = {"rmse": [], "mae": [], "rel": [], "delta1": []}

    if results_dir:
        pred_dir = os.path.join(results_dir, "pred")
        gt_dir = os.path.join(results_dir, "gt")
        error_dir = os.path.join(results_dir, "error")
        lr_dir = os.path.join(results_dir, "lr")
        rgb_dir = os.path.join(results_dir, "rgb")
        for d in [pred_dir, gt_dir, error_dir, lr_dir, rgb_dir]:
            os.makedirs(d, exist_ok=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Test")
    for i, (lr_depth, rgb, hr_depth, dmin, dmax) in pbar:
        lr_depth = lr_depth.to(device)
        rgb = rgb.to(device)
        hr_depth = hr_depth.to(device)

        if model is not None:
            with autocast():
                pred = model(lr_depth, rgb)
        else:
            pred = F.interpolate(lr_depth, size=hr_depth.shape[2:],
                                 mode="bilinear", align_corners=False)

        if pred.shape != hr_depth.shape:
            pred = F.interpolate(pred, size=hr_depth.shape[2:],
                                 mode="bilinear", align_corners=False)

        p = pred[0]
        t = hr_depth[0]
        dm = dmin[0].to(device)
        dx = dmax[0].to(device)

        # Metrics (de-normalized, border clipped)
        metrics["rmse"].append(compute_rmse(p, t, dm, dx, eval_mode=True))
        metrics["mae"].append(compute_mae(p, t, dm, dx, eval_mode=True))
        metrics["rel"].append(compute_rel(p, t, dm, dx, eval_mode=True))
        metrics["delta1"].append(compute_delta(p, t, dm, dx, eval_mode=True))

        # Save visualizations
        if results_dir:
            pred_np = p.squeeze().cpu().numpy()
            t_np = t.squeeze().cpu().float().numpy()
            dm_np = dm.cpu().float().numpy()
            dx_np = dx.cpu().float().numpy()

            # De-normalize to cm
            pred_cm = pred_np * (dx_np - dm_np) + dm_np
            gt_cm = t_np * (dx_np - dm_np) + dm_np
            error_cm = np.abs(pred_cm - gt_cm)

            # Global vmin/vmax from GT for consistent colormap
            vmin, vmax = gt_cm.min(), gt_cm.max()

            plt.imsave(os.path.join(pred_dir, f"{i:04d}.png"),
                       pred_cm, vmin=vmin, vmax=vmax, cmap="gray")
            plt.imsave(os.path.join(gt_dir, f"{i:04d}.png"),
                       gt_cm, vmin=vmin, vmax=vmax, cmap="gray")
            plt.imsave(os.path.join(error_dir, f"{i:04d}.png"),
                       error_cm, cmap="hot")

            # LR depth: upscale from 28×28 to HR, de-normalize to cm
            lr_np = lr_depth[0, 0].cpu().float().numpy()
            lr_cm = lr_np * (dx_np - dm_np) + dm_np
            lr_cm_hr = np.array(Image.fromarray(lr_cm).resize(
                (gt_cm.shape[1], gt_cm.shape[0]), Image.BICUBIC))
            plt.imsave(os.path.join(lr_dir, f"{i:04d}.png"),
                       lr_cm_hr, vmin=vmin, vmax=vmax, cmap="gray")

            # Original RGB: load from disk
            rgb_path, _ = dataset.samples[i]
            rgb_orig = np.array(Image.open(rgb_path).convert("RGB"))  # uint8 HWC
            plt.imsave(os.path.join(rgb_dir, f"{i:04d}.png"), rgb_orig)

    for k in metrics:
        v = np.array(metrics[k])
        print(f"  {k}: {v.mean():.4f} ± {v.std():.4f}")

    return {k: np.mean(v) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=cfg.data_root)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint path (not needed for --baseline)")
    parser.add_argument("--baseline", action="store_true",
                        help="run bicubic upsampling baseline (no model)")
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="directory to save visual results (default: checkpoints/results)")
    parser.add_argument("--no_save", action="store_true",
                        help="skip saving visual results")
    args = parser.parse_args()

    device = torch.device(cfg.device)
    print(f"[Device] {device}")

    # Data
    dataset = NYUDepthSR(
        root=args.data_root, scale=args.scale, train=False,
        crop_size=0, augment=False, pre_extract_edge=True
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    print(f"[Data] {len(dataset)} test samples")

    # Model or baseline
    if args.baseline:
        print("[Mode] Bicubic baseline (no model)")
        model = None
    else:
        if args.checkpoint is None:
            print("[Error] --checkpoint is required (or use --baseline)")
            sys.exit(1)
        model = build_cdsr_net(scale=args.scale)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        print(f"[Model] Loaded from {args.checkpoint} (epoch {ckpt['epoch'] + 1})")

    # Evaluate
    results_dir = None if args.no_save else (
        args.results_dir or os.path.join(os.path.dirname(args.checkpoint or ""), "results")
        if args.checkpoint else "results_baseline"
    )
    results = evaluate(model, loader, dataset, device, results_dir)

    print(f"\n[Results] RMSE: {results['rmse']:.4f}  MAE: {results['mae']:.4f}  "
          f"Rel: {results['rel']:.4f}  δ1: {results['delta1']:.1f}%")
    if results_dir:
        print(f"[Saved] {results_dir}/pred/  {results_dir}/gt/  {results_dir}/error/  "
              f"{results_dir}/lr/  {results_dir}/rgb/")


if __name__ == "__main__":
    main()
