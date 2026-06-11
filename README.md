# Retinal Vessel Segmentation (DRIVE) — U-Net + FastAPI

Segment blood vessels in retinal fundus images with a from-scratch PyTorch
U-Net, trained and evaluated on the **DRIVE** dataset, and served through a
FastAPI endpoint that returns the predicted vessel mask overlaid on the original
image — together with a label-free "goodness" score.

![overlay example](docs/assets/01_overlay.png)

---

## Results

Evaluated on a held-out validation split, **strictly inside the field-of-view
(FOV) mask** (the way published DRIVE results are reported):

| Metric | Value | Notes |
|--------|-------|-------|
| **F1 (Dice)** | **0.834** | clinical benchmark target is > 0.81 ✅ |
| Sensitivity | 0.845 | true-vessel recall |
| Specificity | 0.972 | background correctness |
| AUC | 0.980 | threshold-independent separability |

The ~0.81 figure is the commonly cited DRIVE benchmark; this model clears it.

---

## Method

```
RGB fundus image
   └─ green channel (vessels have highest contrast here)
        └─ CLAHE (cv2.createCLAHE, clipLimit=2.0)   ← contrast enhancement
             └─ U-Net (4 encoder + 4 decoder blocks, skip connections)
                  └─ 1×1 conv → sigmoid → per-pixel vessel probability
```

- **Preprocessing** ([`dataset.py`](dataset.py)): green-channel extraction +
  CLAHE; images padded to 592×592 (divisible by 2⁴ for four poolings).
- **Model** ([`unet.py`](unet.py)): standard U-Net. Each block is
  `Conv → BatchNorm → ReLU` ×2; encoder downsamples with MaxPool, decoder
  upsamples with transposed convolutions and concatenates the matching encoder
  feature map (skip connection). Single-channel sigmoid output.
- **Loss** ([`losses.py`](losses.py)): `0.5 · BCE + 0.5 · Dice`. The Dice term
  counteracts the severe class imbalance — vessels are only ~8–13% of pixels.

---

## Training

- **Data:** DRIVE training set (20 annotated images), split 16 train / 4 val.
- **Strategy:** random **48×48 patch** sampling (8000 patches/epoch) whose
  centres lie inside the FOV. Patch training keeps memory tiny and is the
  configuration that reaches the benchmark — full-image training on an 8 GB GPU
  is both slower (~530 s/epoch) and memory-bound.
- **Optimizer:** Adam, lr 1e-3, cosine annealing, 150 epochs (~40 min on an
  RTX 5060). Horizontal/vertical flip augmentation. Best-validation-F1
  checkpoint is saved.

```bash
python train.py --epochs 150 --base-channels 64 --batch-size 32 \
                --patch-size 48 --patches-per-epoch 8000
python evaluate.py --checkpoint checkpoints/unet_drive.pth
```

---

## Serving (FastAPI)

```bash
uvicorn app:app --port 8000
```

- Open **http://localhost:8000/** for a browser upload form, or POST an image:

```bash
curl.exe -F "file=@DRIVE/test/images/01_test.tif" \
         http://localhost:8000/segment --output overlay.png
```

- `POST /segment` → overlay PNG (vessels in red) with the scores burned in and
  also returned as `X-Vessel-Coverage-Pct` / `X-Confidence-Pct` headers.
- `POST /segment.json` → just the numeric scores.
- `GET /health` → liveness + device + checkpoint status.

### "Goodness" scores (no ground truth needed)
- **Vessel coverage %** — share of the retina marked as vessel. Healthy fundus
  images sit around 8–13%; far outside that hints at over/under-segmentation.
- **Confidence %** — mean decisiveness `max(p, 1−p)` of the model inside the
  FOV, rescaled so 0.5 → 0% and 1.0 → 100%. High means the model rarely sits on
  the fence. (Typical: ~95%.)

---

## Examples

| Original | Segmentation overlay |
|----------|----------------------|
| ![orig](docs/assets/01_original.png) | ![overlay](docs/assets/01_overlay.png) |

More overlays in [`examples/`](examples/).

---

## Dataset

**DRIVE — Digital Retinal Images for Vessel Extraction.** This project expects
the Kaggle distribution placed in `DRIVE/`:

```
DRIVE/
├── training/{images, 1st_manual, mask}   # 20 images + vessel labels + FOV
└── test/{images, mask}                    # 20 images + FOV (no public labels)
```

- Kaggle: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
- Original: https://drive.grand-challenge.org/

> Note: the Kaggle DRIVE distribution does **not** include the second-annotator
> (`2nd_manual`) masks, so inter-annotator agreement is not computed here.

---

## Purpose & use case

Retinal vessel morphology is a biomarker for diabetic retinopathy,
hypertension, and other vascular disease. Automated segmentation supports
screening, vessel-density quantification, and as a preprocessing step for
downstream diagnosis. This repo is a compact, reproducible reference
implementation: classical preprocessing + a clean U-Net + a deployable API.

---

## What this project demonstrates

- **Medical image segmentation** end-to-end: preprocessing → model → loss →
  evaluation → deployment.
- **From-scratch U-Net** in PyTorch with skip connections (no black-box
  library).
- **Class-imbalance handling** via a combined BCE+Dice objective.
- **Correct evaluation:** FOV-masked F1/sensitivity/specificity/AUC, matching
  how the literature reports DRIVE.
- **Memory-aware training** (patch sampling) that fits an 8 GB consumer GPU.
- **Productionization:** a FastAPI service with a browser UI, JSON scores, and
  self-describing overlays.

See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for opt-in OpenCV / Hugging Face upgrade
paths.

---

## Repository layout

| File | Role |
|------|------|
| [`dataset.py`](dataset.py) | DRIVE loading, CLAHE, patch sampling, splits |
| [`unet.py`](unet.py) | U-Net architecture |
| [`losses.py`](losses.py) | BCE+Dice loss, segmentation metrics |
| [`train.py`](train.py) | Training loop (patch-based, GPU-aware) |
| [`evaluate.py`](evaluate.py) | FOV-masked F1 / Se / Sp / AUC |
| [`app.py`](app.py) | FastAPI service + overlay + scores |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | Optional upgrade paths |

## Install

```bash
pip install -r requirements.txt
# GPU (Blackwell/RTX 50-series needs CUDA 12.8 wheels):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## License

MIT — see [`LICENSE`](LICENSE).
