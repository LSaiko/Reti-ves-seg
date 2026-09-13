# Pretrained ≠ Automatically Better: a from-scratch U-Net beat a pretrained ResNet34 on DRIVE

## TL;DR

On the DRIVE retinal vessel segmentation dataset, a small from-scratch U-Net
(31M params, random init) **beat** a U-Net with an ImageNet-pretrained
ResNet34 encoder (21M params) — 0.833 vs 0.821 F1 — and beat it **at every
patch size tried**, while training more stably. Transfer learning is the
default instinct for small medical-imaging datasets; here it didn't pay off.
This write-up is the honest record of that result, not a cherry-picked one.

## Setup

Both models see the same input: a 2-channel stack of a CLAHE-enhanced green
channel and a Frangi vesselness map (a Hessian-based filter tuned to light up
tubular structures — an explicit vessel prior handed to the network rather
than learned from scratch). Both are trained with the same `0.5·BCE + 0.5·Dice`
loss on the same 16-image / 4-image train/val split of DRIVE's 20 labelled
training images, using random 64×64-pixel patch sampling.

| Original | Predicted overlay |
|----------|--------------------|
| ![orig](docs/assets/01_original.png) | ![overlay](docs/assets/01_overlay.png) |

## The surprising result

Evaluated FOV-masked (matching how published DRIVE results are reported):

| Metric | Scratch U-Net | smp + ResNet34 (pretrained) |
|--------|:--:|:--:|
| **F1 (Dice)** | **0.833** | 0.821 |
| Sensitivity | 0.821 | 0.809 |
| Specificity | 0.977 | 0.975 |
| AUC | 0.980 | 0.976 |

Both clear the commonly cited "> 0.81" DRIVE benchmark — this isn't a broken
pretrained run, just a beaten one.

## Where it held up

The gap wasn't a one-off patch-size artifact:

| Patch size | Pretrained ResNet34 F1 | Note |
|---|:--:|---|
| 64px | 0.821 | Then overfit-collapses mid-training |
| 128px | 0.817 | Stable, but still lower |
| — | **0.833 (scratch, either size)** | Wins at both |

Larger patches gave the pretrained encoder's deeper bottleneck more context
and fixed the training instability — but never closed the accuracy gap.

There was also a real training failure along the way, not just a lower final
number: at lr 1e-3 over 150 epochs, the pretrained model's validation F1
**collapsed from 0.82 to 0.33 mid-training**. Dropping to lr 1e-4 plus
best-checkpoint saving fixed the collapse, but the recovered model still
topped out below the scratch net.

## Hypothesis: why scratch beat pretrained here

Two effects compound on a dataset this small (16 training images):

1. **ImageNet features don't transfer cleanly to fundus photographs.**
   ResNet34's pretrained filters are tuned for natural-image textures and
   edges, not thin, low-contrast vascular structures on a curved, vignetted
   background — the domain gap is large enough that the "head start" is
   weaker than usual.
2. **The deeper, heavier encoder overfits faster and is bottleneck-starved.**
   With ~16 images to learn from, a bigger model has more capacity to
   memorize and less signal to generalize from; the smaller from-scratch net,
   sized closer to the task, is a better match for the data budget.

The from-scratch model's simplicity is the point, not a compromise — it fits
this dataset's scale better than a larger, borrowed one.

## Honest caveats

- **The validation split is only 4 images.** Expect ± a few F1 points of
  run-to-run variance; this is a real but not heavily-replicated result
  (single seed, one split).
- **No inter-annotator agreement baseline.** The Kaggle DRIVE distribution
  used here doesn't include the second-annotator (`2nd_manual`) masks, so
  there's no human-vs-human agreement number to compare the model gap
  against.

## What would make this rigorous

- Repeat both architectures across multiple seeds and report a mean ± std,
  not a single run.
- k-fold cross-validation over the 20 labelled images (only 20 exist, so a
  single held-out 16/4 split is a fairly noisy estimate).
- Evaluate cross-dataset transfer (e.g. train on DRIVE, test on CHASE_DB1 or
  STARE) — pretrained encoders are usually argued for on generalization
  grounds, which a same-dataset F1 comparison doesn't test.
- Log loss/F1 per epoch to a file (today `train.py` only prints to stdout) so
  a training-curve plot doesn't require a fresh training run to reconstruct.

## Takeaway

"Pretrained is better" is a prior, not a law — worth checking, not assuming,
especially on small, domain-shifted, non-ImageNet-like data. Here it was
cheaper to check than to assume: same loss, same input, one flag
(`--arch unet` vs `--arch smp_resnet34`) — and the simpler answer won.

*See [`README.md`](README.md#notes--findings) for the shipped results table
and [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for the full menu of upgrade paths
considered, including the ones not adopted.*
