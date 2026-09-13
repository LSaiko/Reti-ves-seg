"""Train the U-Net on the DRIVE dataset.

Usage:
    python train.py --epochs 150 --batch-size 2 --lr 1e-3

Saves the best-validation checkpoint to ``checkpoints/unet_drive.pth``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    DriveDataset,
    DrivePatchDataset,
    build_splits,
    fov_paths_for,
    random_flip,
)
from losses import BCEDiceLoss, segmentation_metrics
from models import ARCHS, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a vessel-segmentation model on DRIVE")
    p.add_argument("--drive-root", default="DRIVE")
    p.add_argument("--arch", default="unet", choices=ARCHS,
                   help="Model: unet (from scratch, default/best) or "
                        "smp_resnet34 (pretrained encoder).")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch-size", type=int, default=64,
                   help="Patch side length for training (multiple of 32). "
                        "<=0 trains on full images.")
    p.add_argument("--patches-per-epoch", type=int, default=8000)
    p.add_argument("--clip-limit", type=float, default=2.0)
    p.add_argument("--out", default="checkpoints/model_drive.pth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for weight init, patch sampling, and augmentation "
                        "(does not affect the train/val split, which build_splits "
                        "fixes separately).")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Run validation and aggregate metrics over the full split.

    Models emit logits, so a sigmoid is applied before thresholding.
    """
    model.eval()
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = torch.sigmoid(model(images))
        m = segmentation_metrics(probs, masks)
        for k in agg:
            agg[k] += m[k]

    eps = 1e-7
    tp, fp, tn, fn = agg["tp"], agg["fp"], agg["tn"], agg["fn"]
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
    return {"f1": f1, "sensitivity": sensitivity, "specificity": specificity}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    tr_imgs, tr_lbls, va_imgs, va_lbls = build_splits(args.drive_root)

    if args.patch_size > 0:
        train_ds = DrivePatchDataset(
            tr_imgs, tr_lbls, fov_paths_for(tr_imgs, args.drive_root),
            patch_size=args.patch_size, patches_per_epoch=args.patches_per_epoch,
            clip_limit=args.clip_limit, augment=random_flip,
        )
        print(f"Training on {args.patch_size}x{args.patch_size} patches "
              f"({args.patches_per_epoch}/epoch)")
    else:
        train_ds = DriveDataset(tr_imgs, tr_lbls, args.clip_limit, augment=random_flip)
        print("Training on full images")

    # Validation always runs on full padded images for true F1/Se/Sp.
    val_ds = DriveDataset(va_imgs, va_lbls, args.clip_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"Train items: {len(train_ds)}  Val images: {len(val_ds)}")

    model = build_model(args.arch).to(device)
    print(f"Model: {args.arch}  (2-channel input: CLAHE green + Frangi)")
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_f1 = 0.0

    log_path = os.path.splitext(args.out)[0] + "_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_f1,val_sensitivity,val_specificity,seconds\n")

    import time

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for i, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
            print(f"\r  epoch {epoch:3d}  batch {i}/{len(train_loader)}  "
                  f"loss {loss.item():.4f}  {time.time()-t0:.0f}s", end="", flush=True)
        scheduler.step()

        # Validate every epoch so progress is always visible.
        metrics = evaluate(model, val_loader, device)
        epoch_seconds = time.time() - t0
        print(
            f"\rEpoch {epoch:3d} | loss {running/len(train_loader):.4f} "
            f"| F1 {metrics['f1']:.4f} | Se {metrics['sensitivity']:.4f} "
            f"| Sp {metrics['specificity']:.4f} | {epoch_seconds:.0f}s/epoch",
            flush=True,
        )
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{running/len(train_loader):.6f},{metrics['f1']:.6f},"
                f"{metrics['sensitivity']:.6f},{metrics['specificity']:.6f},{epoch_seconds:.1f}\n"
            )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {"model_state": model.state_dict(), "f1": best_f1, "arch": args.arch},
                args.out,
            )
            print(f"  saved checkpoint (F1={best_f1:.4f}) -> {args.out}", flush=True)

    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
