"""Seg-Grad-CAM: Grad-CAM adapted to dense (per-pixel) segmentation output.

Classic Grad-CAM backprops from a single class logit, which a segmentation
network doesn't have — its output is a full spatial map of independent
per-pixel logits. Seg-Grad-CAM (Vinogradova et al. 2020) substitutes the sum
of the predicted-vessel logits inside the field of view as that scalar
target, then applies the usual Grad-CAM recipe (weight a chosen conv layer's
activations by their averaged gradient, ReLU, upsample) to explain *where*
the vessel prediction came from.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import _pad_to, estimate_fov, make_input


def _target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Pick the last decoder conv block to explain, for either architecture."""
    if hasattr(model, "dec1"):  # from-scratch UNet (unet.py)
        return model.dec1
    if hasattr(model, "decoder"):  # segmentation_models_pytorch Unet
        return model.decoder.blocks[-1]
    raise ValueError(f"Don't know a target layer for {type(model).__name__}")


def seg_grad_cam(
    model: torch.nn.Module, rgb: np.ndarray, device: str = "cpu", clip_limit: float = 2.0
) -> np.ndarray:
    """Compute a Seg-Grad-CAM heatmap for the model's predicted vessel mask.

    Args:
        model: A segmentation model returning ``(N, 1, H, W)`` logits.
        rgb: RGB fundus image, ``uint8``, shape ``(H, W, 3)``.
        device: Torch device string.
        clip_limit: CLAHE clip limit (must match training/inference).

    Returns:
        ``float32`` heatmap in ``[0, 1]``, shape ``(H, W)`` — high values mark
        regions that most drove the model's vessel predictions.
    """
    h, w = rgb.shape[:2]
    inp = make_input(rgb, clip_limit)
    padded, (top, left) = _pad_to(inp)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).to(device)

    layer = _target_layer(model)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    fwd_handle = layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bwd_handle = layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)  # (1, 1, 608, 608)
        fov = torch.from_numpy(estimate_fov(rgb)).to(device)  # (h, w)
        region = logits[0, 0, top : top + h, left : left + w]
        target = region[fov].sum()
        target.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    act = activations[0][0]  # (C, h', w')
    grad = gradients[0][0]  # (C, h', w')
    weights = grad.mean(dim=(1, 2))  # (C,)
    cam = F.relu((weights[:, None, None] * act).sum(0))  # (h', w')
    cam = cam.detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)

    cam = cv2.resize(cam, (padded.shape[1], padded.shape[0]))
    return cam[top : top + h, left : left + w]


if __name__ == "__main__":
    from models import build_model

    model = build_model("unet").eval()
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    cam = seg_grad_cam(model, rgb)
    assert cam.shape == (64, 64), cam.shape
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6, (cam.min(), cam.max())
    print(f"seg_grad_cam self-check passed: shape={cam.shape} range=[{cam.min():.3f}, {cam.max():.3f}]")
