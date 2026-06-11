# Improvement Options (opt-in)

These are concrete, drop-in upgrade paths. None are enabled by default — pick
what you want and tell me to integrate it. Each lists the expected payoff and
the rough cost.

---

## A. OpenCV preprocessing upgrades

The current pipeline is `green channel -> CLAHE(clip=2.0)`. Classical retinal
imaging adds a couple of cheap steps that consistently sharpen thin vessels.

### A1. Frangi vesselness as an extra input channel
A Hessian-based "vesselness" filter lights up tubular structures specifically.
Feed it to the U-Net as a second channel (`in_channels=2`).

```python
from skimage.filters import frangi  # pip install scikit-image
import numpy as np

def vesselness(green_clahe: np.ndarray) -> np.ndarray:
    """Frangi vesselness map in [0, 1] from a CLAHE green channel."""
    v = frangi(green_clahe, sigmas=range(1, 5), black_ridges=False)
    return (v / (v.max() + 1e-8)).astype(np.float32)
```
*Payoff:* better recall on capillaries. *Cost:* set `UNet(in_channels=2)` and
stack the two maps in the dataset. ~+0.5–1.5 F1 points typically.

### A2. Background normalization (illumination flattening)
Fundus images have uneven illumination. Divide out a large-kernel blur:

```python
def normalize_illumination(green: np.ndarray) -> np.ndarray:
    bg = cv2.medianBlur(green, 55)
    norm = cv2.normalize(green.astype(np.float32) - bg, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)
```
*Payoff:* steadier contrast across the disc; helps generalization to other
cameras. *Cost:* one line before CLAHE.

### A3. Morphological post-processing of the mask
Remove speckle and bridge 1-px gaps in the predicted mask:

```python
def clean_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)         # drop speckles
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)           # close hairline gaps
    return m
```
*Payoff:* cleaner overlays, slightly higher precision. *Cost:* one call in
`app.predict`. Pure inference-time, no retraining.

---

## B. Hugging Face model upgrades

Swap the from-scratch U-Net for a pretrained backbone. These reach the DRIVE
ceiling (~0.82–0.83 F1) faster and generalize better, at the cost of a heavier
dependency and more VRAM.

### B1. `segmentation-models-pytorch` — U-Net with a pretrained encoder
The lowest-friction upgrade: same U-Net shape, but the encoder is an
ImageNet-pretrained ResNet/EfficientNet.

```python
# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # or "efficientnet-b3"
    encoder_weights="imagenet",
    in_channels=3,                  # use the full RGB image
    classes=1,
)
# forward returns logits; keep BCEDiceLoss(from_logits=True)
```
*Payoff:* faster convergence, often +1–2 F1, better cross-dataset transfer.
*Cost:* one dependency; switch the input pipeline from 1-ch CLAHE to 3-ch RGB
(or keep CLAHE and set `in_channels=1`). Drop-in for `unet.py`.

### B2. SegFormer (transformer encoder) via `transformers`
A modern semantic-segmentation transformer; strong on fine structures.

```python
# pip install transformers
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=1, ignore_mismatched_sizes=True
)
# logits = model(pixel_values).logits  -> upsample to input size, then sigmoid
```
*Payoff:* state-of-the-art backbone; best ceiling if you scale to `mit-b2+`.
*Cost:* heavier (more VRAM, needs the patch pipeline on 8 GB), output is at
1/4 resolution so you must `F.interpolate` logits back up.

### B3. MONAI U-Net (medical-imaging native)
MONAI ships battle-tested medical segmentation blocks and losses (incl. Dice).

```python
# pip install monai
from monai.networks.nets import UNet as MonaiUNet

model = MonaiUNet(
    spatial_dims=2, in_channels=1, out_channels=1,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
)
```
*Payoff:* well-tuned defaults, residual units, easy 3D path later. *Cost:* one
dependency; otherwise drops into the existing train/eval loop.

---

## C. Training-recipe tweaks (no new dependencies)

- **Light rotation/elastic augmentation** in `random_flip` — DRIVE is small;
  more augmentation is the cheapest F1 gain.
- **Test-time augmentation (TTA):** average predictions over flips/rotations at
  inference. ~+0.5 F1 for 4× inference cost.
- **Threshold tuning:** sweep `--threshold` on validation; 0.5 is rarely optimal
  for thin-vessel recall (often 0.4–0.45 helps sensitivity).

---

### Recommended first move
If you want one change: **A1 (Frangi channel)** for accuracy with zero new heavy
deps, or **B1 (smp U-Net + ResNet34)** if you're willing to add one library for
the best effort-to-payoff ratio. Tell me which and I'll integrate + retrain.
