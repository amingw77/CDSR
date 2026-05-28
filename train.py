"""
Training script for CDSR-Net: 8x Color-guided Depth Super-Resolution.
Swin Transformer → A²GSTran asymmetric cross-attention → Mamba decoder.
"""
import os
import sys
import argparse
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from models import build_cdsr_net
from data import NYUDepthSR
from utils.metrics import compute_rmse, compute_mae, compute_rel, compute_delta, compute_gradient_loss


class TeeLogger:
    """Duplicate stdout to a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_l1 = 0.0
    total_edge = 0.0
    count = 0

    pbar = tqdm(loader, desc=f"Train E{epoch:3d}/{total_epochs}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for lr_depth, rgb, hr_depth, dmin, dmax in pbar:
        lr_depth = lr_depth.to(device)
        rgb = rgb.to(device)
        hr_depth = hr_depth.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(lr_depth, rgb)
            l1_loss = F.l1_loss(pred, hr_depth)
            edge_loss = compute_gradient_loss(pred, hr_depth)
            loss = cfg.loss_l1_weight * l1_loss + cfg.loss_edge_weight * edge_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_l1 += l1_loss.item() * lr_depth.size(0)
        total_edge += edge_loss.item() * lr_depth.size(0)
        count += lr_depth.size(0)

        pbar.set_postfix({"L1": f"{total_l1/count:.4f}", "Edge": f"{total_edge/count:.4f}"})

    return total_l1 / count, total_edge / count


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    metrics = {"rmse": 0.0, "mae": 0.0, "rel": 0.0, "delta1": 0.0}
    count = 0

    pbar = tqdm(loader, desc="Validate", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
    for lr_depth, rgb, hr_depth, dmin, dmax in pbar:
        lr_depth = lr_depth.to(device)
        rgb = rgb.to(device)
        hr_depth = hr_depth.to(device)

        pred = model(lr_depth, rgb)

        # Align sizes (in case of edge effects)
        if pred.shape != hr_depth.shape:
            pred = F.interpolate(pred, size=hr_depth.shape[2:],
                                 mode="bilinear", align_corners=False)

        for i in range(pred.size(0)):
            p = pred[i]
            t = hr_depth[i]
            dm = dmin[i].to(device)
            dx = dmax[i].to(device)
            metrics["rmse"] += compute_rmse(p, t, dm, dx, eval_mode=True)
            metrics["mae"] += compute_mae(p, t, dm, dx, eval_mode=True)
            metrics["rel"] += compute_rel(p, t, dm, dx, eval_mode=True)
            metrics["delta1"] += compute_delta(p, t, dm, dx, eval_mode=True)
            count += 1

    for k in metrics:
        metrics[k] /= count
    return metrics


def save_checkpoint(save_path, epoch, model, optimizer, scheduler, best_rmse):
    torch.save({
        "version": cfg.model_version,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_rmse": best_rmse,
    }, save_path)
    print(f"  [Checkpoint] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=cfg.data_root)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--lr_step", type=int, default=cfg.lr_step)
    parser.add_argument("--lr_gamma", type=float, default=cfg.lr_gamma)
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--crop_size", type=int, default=cfg.crop_size)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Setup log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg.log_dir, f"train_{timestamp}.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"========================================================")
    print(f"  CDSR-Net v{cfg.model_version}")
    print(f"========================================================")
    print(f"[Log] {log_path}")
    print(f"[Checkpoint Dir] {cfg.checkpoint_dir}")
    print(f"[Data Root] {args.data_root}")
    print(f"[Batch Size] {args.batch_size}")
    print(f"[Epochs] {args.epochs}")
    print(f"[Scale] {args.scale}")

    device = torch.device(cfg.device)
    print(f"[Device] {device}")

    # Data
    print("[Data] Loading...")
    try:
        train_dataset = NYUDepthSR(
            root=args.data_root, scale=args.scale, train=True,
            crop_size=args.crop_size, augment=True, pre_extract_edge=True,
            repeat=cfg.repeat
        )
        test_dataset = NYUDepthSR(
            root=args.data_root, scale=args.scale, train=False,
            crop_size=0, augment=False, pre_extract_edge=True
        )
    except Exception as e:
        print(f"[ERROR] Dataset loading failed: {e}")
        traceback.print_exc()
        tee.close()
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    print(f"[Data] Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Model
    print("[Model] Building...")
    model = build_cdsr_net(scale=args.scale)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = GradScaler()

    start_epoch = 0
    best_rmse = float("inf")
    if args.resume:
        print(f"[Resume] Loading {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_version = ckpt.get("version", "unknown")
        print(f"[Resume] Checkpoint version: v{ckpt_version}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_rmse = ckpt.get("best_rmse", float("inf"))
        print(f"[Resume] Start epoch {start_epoch}, Best RMSE: {best_rmse:.4f}")

    try:
        for epoch in range(start_epoch, args.epochs):
            l1_loss, edge_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch + 1, args.epochs)
            scheduler.step()

            # Save last checkpoint every epoch (always resumable)
            save_checkpoint(
                os.path.join(cfg.checkpoint_dir, "cdsr_net_last.pth"),
                epoch, model, optimizer, scheduler, best_rmse
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_metrics = validate(model, test_loader, device)
                rmse = val_metrics["rmse"]
                print(f"[Epoch {epoch + 1:3d}] L1: {l1_loss:.4f}  Edge: {edge_loss:.4f}  "
                      f"RMSE: {rmse:.4f}  MAE: {val_metrics['mae']:.4f}  "
                      f"Rel: {val_metrics['rel']:.4f}  δ1: {val_metrics['delta1']:.1f}%")

                if rmse < best_rmse:
                    best_rmse = rmse
                    save_checkpoint(
                        os.path.join(cfg.checkpoint_dir, "cdsr_net_best.pth"),
                        epoch, model, optimizer, scheduler, best_rmse
                    )
            else:
                print(f"[Epoch {epoch + 1:3d}] L1: {l1_loss:.4f}  Edge: {edge_loss:.4f}")

        # Final save
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, "cdsr_net_final.pth"),
            args.epochs - 1, model, optimizer, scheduler, best_rmse
        )
        print(f"\n[Done] Best RMSE: {best_rmse:.4f}")

    except Exception as e:
        crash_epoch = locals().get("epoch", start_epoch)
        print(f"\n[CRASH] Training crashed near epoch {crash_epoch}")
        print(f"[CRASH] Error: {e}")
        traceback.print_exc()

        # Emergency save on crash
        crash_path = os.path.join(cfg.checkpoint_dir, "cdsr_net_crash.pth")
        print(f"[CRASH] Saving emergency checkpoint to {crash_path}")
        try:
            save_checkpoint(crash_path, crash_epoch,
                          model, optimizer, scheduler, best_rmse)
        except Exception as save_err:
            print(f"[CRASH] Failed to save emergency checkpoint: {save_err}")

    finally:
        tee.close()
        sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
