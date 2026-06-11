"""DRIVE dataset loading and 2-channel preprocessing.

DRIVE images are RGB fundus photographs. Retinal vessels exhibit the highest
contrast in the green channel, so the pipeline extracts that channel and applies
CLAHE (Contrast Limited Adaptive Histogram Equalization). A second channel — a
Frangi *vesselness* response, a Hessian-based filter tuned for tubular
structures — is stacked on top, giving the network an explicit vessel prior.
The model therefore consumes a **2-channel** input ``[CLAHE green, Frangi]``.

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
from skimage.filters import frangi
from torch.utils.data import Dataset

# Pad target so spatial dims are divisible by 2**5 = 32 — required by the
# ResNet encoder (5 downsamples) and also fine for the plain U-Net (needs 16).
PAD_TO = 608

# Number of input channels produced by make_input ([CLAHE green, Frangi]).
IN_CHANNELS = 2


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


def frangi_vesselness(clahe_green: np.ndarray) -> np.ndarray:
    """Frangi vesselness response of a CLAHE green channel, normalised to [0, 1].

    Args:
        clahe_green: ``uint8`` CLAHE-enhanced green channel, shape ``(H, W)``.

    Returns:
        ``float32`` vesselness map in ``[0, 1]`` of shape ``(H, W)``; bright
        ridges (vessels) score high.
    """
    v = frangi(clahe_green.astype(np.float32) / 255.0, sigmas=range(1, 5),
               black_ridges=False)
    v = np.nan_to_num(v, nan=0.0)
    return (v / (v.max() + 1e-8)).astype(np.float32)


def make_input(image_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Build the 2-channel network input from an RGB fundus image.

    Channel 0 is the CLAHE-enhanced green channel in ``[0, 1]``; channel 1 is
    the Frangi vesselness map in ``[0, 1]``.

    Args:
        image_rgb: RGB image, ``uint8``, shape ``(H, W, 3)``.
        clip_limit: CLAHE clip limit.

    Returns:
        ``float32`` array of shape ``(H, W, 2)`` (HWC, ready for cropping /
        augmentation; permute to CHW before passing to the model).
    """
    clahe = apply_clahe(image_rgb, clip_limit)
    ch0 = clahe.astype(np.float32) / 255.0
    ch1 = frangi_vesselness(clahe)
    return np.stack([ch0, ch1], axis=-1)


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
        cache: If ``True`` (default), the (expensive) 2-channel input —
            including the Frangi filter — is computed once per image and reused.
            This avoids recomputing Frangi every epoch, which otherwise leaks
            host memory on RAM-constrained machines.
    """

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        clip_limit: float = 2.0,
        augment: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        cache: bool = True,
    ) -> None:
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.clip_limit = clip_limit
        self.augment = augment
        self.cache = cache
        self._input_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def _input_for(self, idx: int) -> np.ndarray:
        """Return the 2-channel input for ``idx``, computing it at most once."""
        if self.cache and idx in self._input_cache:
            return self._input_cache[idx]
        inp = make_input(_read_image(self.image_paths[idx]), self.clip_limit)
        if self.cache:
            self._input_cache[idx] = inp
        return inp

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self._input_for(idx).copy()  # (H, W, 2); copy so augment is safe

        label = _read_image(self.label_paths[idx])
        label = (np.asarray(label) > 0).astype(np.float32)

        if self.augment is not None:
            inp, label = self.augment(inp, label)

        inp, _ = _pad_to(inp)        # (H, W, 2)
        label, _ = _pad_to(label)    # (H, W)

        image_t = torch.from_numpy(inp).permute(2, 0, 1).contiguous()  # (2, H, W)
        label_t = torch.from_numpy(label).unsqueeze(0)                  # (1, H, W)
        return image_t, label_t


class DrivePatchDataset(Dataset):
    """Random-patch sampler for DRIVE — the configuration that reaches F1 > 0.81.

    All 2-channel inputs ([CLAHE green, Frangi]) are computed and cached in host
    RAM once at construction (16 small images, a few MB total). Each
    ``__getitem__`` returns a random ``patch_size x patch_size`` crop whose
    centre lies inside the field of view, so patches always contain retina. This
    keeps per-item tensors tiny, which avoids the host-memory thrash of
    full-image training and trains far faster.

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
        patch_size: int = 64,
        patches_per_epoch: int = 8000,
        clip_limit: float = 2.0,
        augment: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        assert len(image_paths) == len(label_paths) == len(fov_paths)
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        self.images: list[np.ndarray] = []   # each (H, W, 2)
        self.labels: list[np.ndarray] = []
        self.fovs: list[np.ndarray] = []
        for img_p, lbl_p, fov_p in zip(image_paths, label_paths, fov_paths):
            rgb = _read_image(img_p)
            self.images.append(make_input(rgb, clip_limit))
            self.labels.append((np.asarray(_read_image(lbl_p)) > 0).astype(np.float32))
            self.fovs.append((np.asarray(_read_image(fov_p)) > 0).astype(np.uint8))

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self.patch_size
        n = len(self.images)
        img_idx = np.random.randint(n)
        image, label, fov = self.images[img_idx], self.labels[img_idx], self.fovs[img_idx]
        h, w = image.shape[:2]

        # Reject-sample a patch whose centre is inside the FOV.
        for _ in range(20):
            top = np.random.randint(0, h - ps + 1)
            left = np.random.randint(0, w - ps + 1)
            if fov[top + ps // 2, left + ps // 2]:
                break

        img_patch = image[top : top + ps, left : left + ps, :].copy()  # (ps, ps, 2)
        lbl_patch = label[top : top + ps, left : left + ps].copy()
        if self.augment is not None:
            img_patch, lbl_patch = self.augment(img_patch, lbl_patch)

        return (
            torch.from_numpy(img_patch).permute(2, 0, 1).contiguous(),  # (2, ps, ps)
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
