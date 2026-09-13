"""One-off cross-dataset check: evaluate the DRIVE-trained checkpoint on STARE.

This is deliberately NOT training/eval infrastructure — dataset.py's classes
are DRIVE-specific and untouched. This script reuses make_input / _pad_to /
estimate_fov / segmentation_metrics directly (same pattern as evaluate.py) to
spot-check how a checkpoint trained only on DRIVE generalizes to a different
retinal vessel dataset, with no retraining or fine-tuning.

Data (not included in this repo — download separately, no account needed):
    https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar   (raw images)
    https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar     (hand-labelled vessel masks)
Extract both, then decompress the .ppm.gz files for whichever image ids you
want to check (`gunzip im0001.ppm.gz`, etc.) into <stare-dir>/images and
<stare-dir>/labels respectively.

STARE images are 700x605 — larger than DRIVE's 584x565 and larger than the
608x608 pad size dataset.py uses for training, so this script pads to a
larger size (720x720, still divisible by 16) instead of reusing PAD_TO. The
plain from-scratch U-Net (arch="unet") is fully convolutional and only needs
input dimensions divisible by 16, so this is a safe substitution.

Usage:
    python evaluate_stare.py --stare-dir /path/to/stare --out-dir docs/assets
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from dataset import _pad_to, estimate_fov, make_input
from losses import segmentation_metrics
from models import build_model

DEFAULT_IMAGE_IDS = ["0001", "0002", "0003", "0044", "0077"]
PAD_SIZE = 720  # >= max(605, 700), divisible by 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a DRIVE checkpoint on STARE")
    p.add_argument("--stare-dir", required=True,
                    help="Directory with images/imNNNN.ppm and labels/imNNNN.ah.ppm")
    p.add_argument("--checkpoint", default="checkpoints/model_drive.pth")
    p.add_argument("--out-dir", default="docs/assets")
    p.add_argument("--image-ids", nargs="*", default=DEFAULT_IMAGE_IDS)
    p.add_argument("--overlay-ids", nargs="*", default=["0001", "0077"],
                    help="Subset of --image-ids to save original/overlay PNGs for")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_rgb(stare_dir: str, image_id: str) -> np.ndarray:
    return np.array(Image.open(f"{stare_dir}/images/im{image_id}.ppm").convert("RGB"))


def load_label(stare_dir: str, image_id: str) -> np.ndarray:
    arr = np.array(Image.open(f"{stare_dir}/labels/im{image_id}.ah.ppm").convert("L"))
    return (arr > 0).astype(np.float32)


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask == 1] = [255, 0, 0]
    return (0.6 * rgb + 0.4 * out).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt.get("arch", "unet")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    all_probs, all_targets = [], []
    per_image_rows = []

    for image_id in args.image_ids:
        rgb = load_rgb(args.stare_dir, image_id)
        label = load_label(args.stare_dir, image_id)
        h, w = rgb.shape[:2]

        inp = make_input(rgb)
        padded, (top, left) = _pad_to(inp, size=PAD_SIZE)
        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(device)
        probs = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
        probs = probs[top : top + h, left : left + w]

        fov = estimate_fov(rgb)

        probs_t = torch.from_numpy(probs).unsqueeze(0).unsqueeze(0)
        label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        fov_t = torch.from_numpy(fov.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        m = segmentation_metrics(probs_t, label_t, fov_mask=fov_t, threshold=args.threshold)
        for k in agg:
            agg[k] += m[k]
        per_image_rows.append((image_id, m["f1"], m["sensitivity"], m["specificity"]))

        keep = fov.ravel()
        all_probs.append(probs.ravel()[keep])
        all_targets.append(label.ravel()[keep])

        if image_id in args.overlay_ids:
            mask = ((probs >= args.threshold) & fov).astype(np.uint8)
            Image.fromarray(rgb).save(f"{args.out_dir}/stare_{image_id}_original.png")
            Image.fromarray(overlay(rgb, mask)).save(f"{args.out_dir}/stare_{image_id}_overlay.png")

    eps = 1e-7
    tp, fp, tn, fn = agg["tp"], agg["fp"], agg["tn"], agg["fn"]
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
    auc = roc_auc_score(np.concatenate(all_targets), np.concatenate(all_probs))

    print("=== Per-image STARE metrics (inside estimated FOV) ===")
    for image_id, imf1, sens, spec in per_image_rows:
        print(f"  im{image_id}: F1={imf1:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

    print(f"\n=== Aggregate STARE metrics (inside estimated FOV), n={len(args.image_ids)} images ===")
    print(f"F1 (Dice)    : {f1:.4f}")
    print(f"Sensitivity  : {sensitivity:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"AUC          : {auc:.4f}")


if __name__ == "__main__":
    main()
