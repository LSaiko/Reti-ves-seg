"""Loss functions and segmentation metrics for vessel segmentation.

The combined loss ``0.5 * BCE + 0.5 * Dice`` pairs the pixel-wise calibration
of binary cross-entropy with the overlap-based Dice term, which counteracts the
severe foreground/background imbalance (vessels are ~10% of pixels in DRIVE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss.

    Args:
        probs: Predicted probabilities in ``[0, 1]``, shape ``(N, 1, H, W)``.
        targets: Binary ground-truth, same shape as ``probs``.
        eps: Smoothing constant to avoid division by zero.

    Returns:
        Scalar Dice loss ``1 - dice``.
    """
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss: ``bce_weight * BCE + dice_weight * Dice``.

    Args:
        bce_weight: Weight on the BCE term.
        dice_weight: Weight on the Dice term.
        from_logits: If ``True``, inputs are raw logits and BCE is computed with
            the numerically stable ``binary_cross_entropy_with_logits``; a
            sigmoid is applied before the Dice term. If ``False``, inputs are
            already probabilities.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            pred: Logits or probabilities of shape ``(N, 1, H, W)``.
            target: Binary ground-truth of shape ``(N, 1, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, target)
            probs = torch.sigmoid(pred)
        else:
            bce = F.binary_cross_entropy(pred, target)
            probs = pred
        dice = dice_loss(probs, target)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def segmentation_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    fov_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute F1/Dice, sensitivity, and specificity for a batch.

    Args:
        probs: Predicted probabilities in ``[0, 1]``.
        targets: Binary ground-truth.
        fov_mask: Optional binary field-of-view mask; pixels outside it are
            excluded (DRIVE metrics are computed inside the FOV).
        threshold: Probability threshold for binarisation.

    Returns:
        Dict with ``f1``, ``sensitivity``, ``specificity``, and the raw
        ``tp``/``fp``/``tn``/``fn`` counts.
    """
    preds = (probs >= threshold).float()
    if fov_mask is not None:
        keep = fov_mask.bool()
        preds = preds[keep]
        targets = targets[keep]

    tp = float((preds * targets).sum())
    fp = float((preds * (1 - targets)).sum())
    fn = float(((1 - preds) * targets).sum())
    tn = float(((1 - preds) * (1 - targets)).sum())

    eps = 1e-7
    sensitivity = tp / (tp + fn + eps)        # recall on vessels
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + eps)
    return {
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
