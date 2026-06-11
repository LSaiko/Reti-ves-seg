"""Evaluate a trained U-Net on the DRIVE validation split.

Reports F1 (Dice), sensitivity, specificity, and AUC — the standard clinical
benchmark metrics for DRIVE. The literature target is F1 > 0.81. Metrics are
computed strictly inside the field-of-view (FOV) mask, matching how published
DRIVE results are reported.

Usage:
    python evaluate.py --checkpoint checkpoints/unet_drive.pth
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import DriveDataset, _pad_to, _read_image, build_splits, fov_paths_for
from losses import segmentation_metrics
from models import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a vessel-segmentation model on DRIVE")
    p.add_argument("--drive-root", default="DRIVE")
    p.add_argument("--checkpoint", default="checkpoints/model_drive.pth")
    p.add_argument("--clip-limit", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _padded_fov(path: str) -> np.ndarray:
    """Load a FOV mask and pad it to match :class:`DriveDataset` output."""
    fov = (np.asarray(_read_image(path)) > 0).astype(np.float32)
    padded, _ = _pad_to(fov)
    return padded


def _metrics_from_counts(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
    """Derive F1, sensitivity, specificity from confusion-matrix counts."""
    eps = 1e-7
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
    return {"f1": f1, "sensitivity": sensitivity, "specificity": specificity}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device

    _, _, va_imgs, va_lbls = build_splits(args.drive_root)
    va_fovs = fov_paths_for(va_imgs, args.drive_root)

    val_ds = DriveDataset(va_imgs, va_lbls, args.clip_limit)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt.get("arch", "unet")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for (images, masks), fov_path in zip(val_loader, va_fovs):
        images = images.to(device)
        probs = torch.sigmoid(model(images)).cpu()
        fov = torch.from_numpy(_padded_fov(fov_path)).unsqueeze(0).unsqueeze(0)

        m = segmentation_metrics(probs, masks, fov_mask=fov, threshold=args.threshold)
        for k in agg:
            agg[k] += m[k]

        # Restrict AUC pixels to the FOV as well.
        keep = fov.bool().numpy().ravel()
        all_probs.append(probs.numpy().ravel()[keep])
        all_targets.append(masks.numpy().ravel()[keep])

    metrics = _metrics_from_counts(agg["tp"], agg["fp"], agg["tn"], agg["fn"])
    auc = roc_auc_score(np.concatenate(all_targets), np.concatenate(all_probs))

    print("=== DRIVE validation metrics (inside FOV) ===")
    print(f"F1 (Dice)    : {metrics['f1']:.4f}   (clinical target > 0.81)")
    print(f"Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(f"AUC          : {auc:.4f}")
    print(f"{'PASS' if metrics['f1'] > 0.81 else 'BELOW TARGET'} clinical F1 benchmark")


if __name__ == "__main__":
    main()
