"""Model factory: plain U-Net or an smp U-Net with a pretrained ResNet34.

Both architectures take a 2-channel input ([CLAHE green, Frangi]) and emit a
single-channel **logit** map (no activation) so training can use the numerically
stable ``BCEDiceLoss(from_logits=True)``; callers apply ``torch.sigmoid`` at
inference.
"""

from __future__ import annotations

import torch.nn as nn

from dataset import IN_CHANNELS
from unet import UNet

ARCHS = ("unet", "smp_resnet34")


def build_model(arch: str = "smp_resnet34", in_channels: int = IN_CHANNELS) -> nn.Module:
    """Construct a segmentation model that returns logits.

    Args:
        arch: ``"unet"`` for the from-scratch U-Net, or ``"smp_resnet34"`` for a
            ``segmentation_models_pytorch`` U-Net with an ImageNet-pretrained
            ResNet34 encoder.
        in_channels: Number of input channels (2 for the CLAHE+Frangi stack).

    Returns:
        An ``nn.Module`` whose forward returns ``(N, 1, H, W)`` logits.
    """
    if arch == "unet":
        return UNet(in_channels=in_channels, base_channels=64, return_logits=True)

    if arch == "smp_resnet34":
        import segmentation_models_pytorch as smp

        # smp.Unet returns raw logits (activation=None). For in_channels != 3 it
        # adapts the pretrained first conv automatically.
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
        )

    raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}.")
