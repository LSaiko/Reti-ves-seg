"""U-Net architecture for single-channel (binary) image segmentation.

The network follows the classic encoder-decoder design of Ronneberger et al.
(2015) with four downsampling blocks, a bottleneck, and four upsampling blocks
that recombine encoder features through skip connections. The final 1x1
convolution maps to a single channel, and a sigmoid produces per-pixel
probabilities in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive (Conv -> BatchNorm -> ReLU) layers.

    This is the fundamental building block used in every encoder and decoder
    stage of the U-Net.

    Args:
        in_channels: Number of channels in the input feature map.
        out_channels: Number of channels produced by both convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double-convolution block.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Output tensor of shape ``(N, out_channels, H, W)``.
        """
        return self.block(x)


class UNet(nn.Module):
    """U-Net with four encoder and four decoder blocks.

    The encoder progressively halves spatial resolution while doubling the
    channel count. The decoder mirrors this, using transposed convolutions to
    upsample and concatenating the matching encoder feature map (skip
    connection) before each decoder block. A final 1x1 convolution followed by
    a sigmoid yields a single-channel probability map.

    Args:
        in_channels: Number of channels in the input image (e.g. 3 for RGB,
            1 for grayscale).
        base_channels: Channel count of the first encoder block. Each
            subsequent block doubles this value.

    Shape:
        - Input: ``(N, in_channels, H, W)``
        - Output: ``(N, 1, H, W)`` with values in ``[0, 1]``.

    Note:
        Input spatial dimensions should ideally be divisible by 16 (2**4) so
        that the four pooling operations and four upsampling operations align.
        For other sizes, decoder features are resized to match the skip
        connection before concatenation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        return_logits: bool = False,
    ) -> None:
        super().__init__()
        self.return_logits = return_logits

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        c_bottleneck = base_channels * 16

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c_bottleneck)

        # Decoder: transposed conv (upsample) + double conv (after skip concat)
        self.up4 = nn.ConvTranspose2d(c_bottleneck, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c_bottleneck, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        # Output head
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _concat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate an upsampled feature map with its encoder skip map.

        Resizes ``upsampled`` to the skip's spatial size when they differ
        (e.g. due to odd input dimensions) before concatenating on the channel
        axis.

        Args:
            upsampled: Decoder feature map of shape ``(N, C, H', W')``.
            skip: Encoder feature map of shape ``(N, C, H, W)``.

        Returns:
            Concatenated tensor of shape ``(N, 2C, H, W)``.
        """
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Single-channel segmentation map of shape ``(N, 1, H, W)`` with
            values in ``[0, 1]``.
        """
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder with skip connections
        d4 = self.dec4(self._concat(self.up4(b), s4))
        d3 = self.dec3(self._concat(self.up3(d4), s3))
        d2 = self.dec2(self._concat(self.up2(d3), s2))
        d1 = self.dec1(self._concat(self.up1(d2), s1))

        logits = self.out_conv(d1)
        if self.return_logits:
            return logits
        return torch.sigmoid(logits)


if __name__ == "__main__":
    model = UNet(in_channels=1, base_channels=64)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
