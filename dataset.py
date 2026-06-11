"""DRIVE dataset loading and CLAHE preprocessing.

DRIVE images are RGB fundus photographs. Retinal vessels exhibit the highest
contrast in the green channel, so the pipeline extracts that channel and
applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost local
vessel contrast before feeding a single-channel tensor to the network.

The DRIVE training set provides 20 images with manual vessel annotations
(``1st_manual``) and field-of-view (FOV) masks. The official ``test`` split
ships without public vessel labels, so this module splits the 20 labelled
training images into train/validation subsets.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Pad target so spatial dims are divisible by 2**4 (four poolings).
PAD_TO = 592


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Extract the green channel and apply CLAHE.

    Args:
        image_rgb: RGB image as a ``uint8`` array of shape ``(H, W, 3)``.
        clip_limit: CLAHE contrast clipping threshold.

    Returns:
        Single-channel ``uint8`` array of shape ``(H, W)`` with enhanced
        vessel contrast.
    """
    green = image_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(green)


def _read_image(path: str) -> np.ndarray:
    """Read an image (``.tif``/``.gif``) as an RGB or grayscale uint8 array."""
    img = Image.open(path)
    return np.array(img)


def _pad_to(arr: np.ndarray, size: int = PAD_TO) -> tuple[np.ndarray, tuple[int, int]]:
    """Symmetrically zero-pad an array's first two dims up to ``size``.

    Returns the padded array and the ``(top, left)`` offsets used, so the
    prediction can later be cropped back to the original frame.
    """
    h, w = arr.shape[:2]
    top = (size - h) // 2
    left = (size - w) // 2
    pad_width = [(top, size - h - top), (left, size - w - left)]
    if arr.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(arr, pad_width, mode="constant"), (top, left)


class DriveDataset(Dataset):
    """DRIVE retinal vessel segmentation dataset.

    Each item is a ``(image, mask)`` pair where ``image`` is the CLAHE-enhanced
    green channel normalised to ``[0, 1]`` and ``mask`` is the binary vessel
    annotation. Both are zero-padded to :data:`PAD_TO` so spatial dimensions
    are divisible by 16.

    Args:
        image_paths: Paths to the fundus ``.tif`` images.
        label_paths: Paths to the matching ``1st_manual`` vessel ``.gif`` masks.
        clip_limit: CLAHE clip limit.
        augment: Optional callable applied to ``(image, mask)`` numpy arrays for
            data augmentation (training only).
    """

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        clip_limit: float = 2.0,
        augment: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.clip_limit = clip_limit
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = _read_image(self.image_paths[idx])
        enhanced = apply_clahe(rgb, self.clip_limit).astype(np.float32) / 255.0

        label = _read_image(self.label_paths[idx])
        label = (np.asarray(label) > 0).astype(np.float32)

        if self.augment is not None:
            enhanced, label = self.augment(enhanced, label)

        enhanced, _ = _pad_to(enhanced)
        label, _ = _pad_to(label)

        image_t = torch.from_numpy(enhanced).unsqueeze(0)  # (1, H, W)
        label_t = torch.from_numpy(label).unsqueeze(0)      # (1, H, W)
        return image_t, label_t


class DrivePatchDataset(Dataset):
    """Random-patch sampler for DRIVE — the configuration that reaches F1 > 0.81.

    All images are CLAHE-enhanced and cached in host RAM once at construction
    (16 small images, a few MB total). Each ``__getitem__`` returns a random
    ``patch_size x patch_size`` crop whose centre lies inside the field of view,
    so patches always contain retina. This keeps per-item tensors tiny, which
    avoids the host-memory thrash of full-image training and trains far faster.

    Args:
        image_paths: Fundus ``.tif`` image paths.
        label_paths: Matching ``1st_manual`` vessel ``.gif`` paths.
        fov_paths: Matching FOV ``mask`` ``.gif`` paths.
        patch_size: Square patch side length in pixels.
        patches_per_epoch: Number of random patches drawn per epoch.
        clip_limit: CLAHE clip limit.
        augment: Optional ``(image, mask) -> (image, mask)`` augmentation.
    """

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        fov_paths: list[str],
        patch_size: int = 48,
        patches_per_epoch: int = 8000,
        clip_limit: float = 2.0,
        augment: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        assert len(image_paths) == len(label_paths) == len(fov_paths)
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        self.images: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.fovs: list[np.ndarray] = []
        for img_p, lbl_p, fov_p in zip(image_paths, label_paths, fov_paths):
            rgb = _read_image(img_p)
            self.images.append(apply_clahe(rgb, clip_limit).astype(np.float32) / 255.0)
            self.labels.append((np.asarray(_read_image(lbl_p)) > 0).astype(np.float32))
            self.fovs.append((np.asarray(_read_image(fov_p)) > 0).astype(np.uint8))

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self.patch_size
        n = len(self.images)
        img_idx = np.random.randint(n)
        image, label, fov = self.images[img_idx], self.labels[img_idx], self.fovs[img_idx]
        h, w = image.shape

        # Reject-sample a patch whose centre is inside the FOV.
        for _ in range(20):
            top = np.random.randint(0, h - ps + 1)
            left = np.random.randint(0, w - ps + 1)
            if fov[top + ps // 2, left + ps // 2]:
                break

        img_patch = image[top : top + ps, left : left + ps].copy()
        lbl_patch = label[top : top + ps, left : left + ps].copy()
        if self.augment is not None:
            img_patch, lbl_patch = self.augment(img_patch, lbl_patch)

        return (
            torch.from_numpy(img_patch).unsqueeze(0),
            torch.from_numpy(lbl_patch).unsqueeze(0),
        )


def fov_paths_for(image_paths: list[str], drive_root: str = "DRIVE") -> list[str]:
    """Resolve FOV ``mask`` paths matching a list of training image paths."""
    mask_dir = os.path.join(drive_root, "training", "mask")
    out = []
    for img in image_paths:
        num = os.path.basename(img).split("_")[0]
        out.append(os.path.join(mask_dir, f"{num}_training_mask.gif"))
    return out


def random_flip(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal/vertical flip augmentation."""
    if np.random.rand() < 0.5:
        image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
    if np.random.rand() < 0.5:
        image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
    return image, mask


def build_splits(
    drive_root: str = "DRIVE",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Build train/validation file lists from the labelled DRIVE training set.

    Args:
        drive_root: Path to the ``DRIVE`` directory.
        val_fraction: Fraction of the 20 labelled images held out for validation.
        seed: RNG seed for the shuffle.

    Returns:
        ``(train_images, train_labels, val_images, val_labels)`` path lists.
    """
    img_dir = os.path.join(drive_root, "training", "images")
    lbl_dir = os.path.join(drive_root, "training", "1st_manual")

    images = sorted(glob(os.path.join(img_dir, "*.tif")))
    labels = []
    for img in images:
        num = os.path.basename(img).split("_")[0]
        labels.append(os.path.join(lbl_dir, f"{num}_manual1.gif"))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    n_val = max(1, int(len(images) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_images, train_labels, val_images, val_labels
