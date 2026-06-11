"""FastAPI service for retinal vessel segmentation.

Accepts a retinal fundus image upload and returns the predicted vessel mask
overlaid (in red) on the original image as a PNG.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Then POST an image:
    curl -F "file=@DRIVE/test/images/01_test.tif" \
         http://localhost:8000/segment --output overlay.png
"""

from __future__ import annotations

import io
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

from dataset import _pad_to, apply_clahe
from unet import UNet

CHECKPOINT = os.environ.get("UNET_CHECKPOINT", "checkpoints/unet_drive.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = float(os.environ.get("VESSEL_THRESHOLD", "0.5"))

app = FastAPI(title="Retinal Vessel Segmentation", version="1.0.0")
_model: UNet | None = None


def get_model() -> UNet:
    """Lazily load and cache the trained U-Net."""
    global _model
    if _model is None:
        model = UNet(in_channels=1, base_channels=64).to(DEVICE)
        if os.path.exists(CHECKPOINT):
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
        else:
            # Allow the service to start without weights (returns noise);
            # useful for wiring/smoke tests before training completes.
            print(f"WARNING: checkpoint '{CHECKPOINT}' not found — using random weights.")
        model.eval()
        _model = model
    return _model


def estimate_fov(rgb: np.ndarray) -> np.ndarray:
    """Estimate the circular field-of-view mask of a fundus image.

    The retina is the bright disc on a near-black background, so a simple
    luminance threshold recovers the FOV well enough for scoring.

    Args:
        rgb: RGB image of shape ``(H, W, 3)``, ``uint8``.

    Returns:
        Boolean FOV mask of shape ``(H, W)``.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray > 15


@torch.no_grad()
def predict(rgb: np.ndarray, clip_limit: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict vessel probabilities and a binary mask for an RGB fundus image.

    Args:
        rgb: RGB image of shape ``(H, W, 3)``, ``uint8``.
        clip_limit: CLAHE clip limit (must match training).

    Returns:
        Tuple ``(mask, probs, fov)`` where ``mask`` is a ``uint8`` ``{0, 1}``
        array, ``probs`` is the float probability map in ``[0, 1]``, and ``fov``
        is the boolean field-of-view mask — all of shape ``(H, W)``.
    """
    h, w = rgb.shape[:2]
    enhanced = apply_clahe(rgb, clip_limit).astype(np.float32) / 255.0
    padded, (top, left) = _pad_to(enhanced)

    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(DEVICE)
    probs = get_model()(tensor).squeeze().cpu().numpy()

    # Crop back to the original frame.
    probs = probs[top : top + h, left : left + w]
    fov = estimate_fov(rgb)
    mask = ((probs >= THRESHOLD) & fov).astype(np.uint8)
    return mask, probs, fov


def quality_scores(mask: np.ndarray, probs: np.ndarray, fov: np.ndarray) -> dict[str, float]:
    """Compute label-free "goodness" scores for a segmentation.

    Without ground truth we report two intuitive proxies, both as percentages:

    * ``coverage_pct`` — share of the retina (FOV) marked as vessel. Healthy
      fundus images sit around 8-13%; values far outside hint at over/under
      segmentation.
    * ``confidence_pct`` — mean decisiveness ``max(p, 1-p)`` of the model inside
      the FOV, rescaled so 0.5 (pure uncertainty) maps to 0% and 1.0 (fully
      committed) maps to 100%. Higher means the model is rarely "on the fence".

    Args:
        mask: Binary vessel mask, ``{0, 1}``.
        probs: Probability map in ``[0, 1]``.
        fov: Boolean field-of-view mask.

    Returns:
        Dict with ``coverage_pct`` and ``confidence_pct``.
    """
    fov_px = int(fov.sum()) or 1
    coverage_pct = 100.0 * float(mask.sum()) / fov_px

    p = probs[fov]
    decisiveness = np.maximum(p, 1.0 - p)           # in [0.5, 1.0]
    confidence_pct = 100.0 * float((decisiveness.mean() - 0.5) / 0.5)
    return {"coverage_pct": coverage_pct, "confidence_pct": confidence_pct}


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, scores: dict[str, float] | None = None) -> np.ndarray:
    """Overlay a binary mask in red on top of the original RGB image.

    If ``scores`` is given, the coverage and confidence percentages are burned
    into the top-left corner so the overlay is self-describing.
    """
    overlay = rgb.copy()
    overlay[mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    if scores is not None:
        lines = [
            f"Vessel coverage: {scores['coverage_pct']:.1f}%",
            f"Confidence: {scores['confidence_pct']:.1f}%",
        ]
        for i, text in enumerate(lines):
            y = 22 + i * 22
            cv2.putText(blended, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(blended, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 1, cv2.LINE_AA)
    return blended


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Minimal browser upload form for the segmentation endpoint."""
    return """
    <!doctype html><html><head><title>Retinal Vessel Segmentation</title></head>
    <body style="font-family:sans-serif;max-width:640px;margin:40px auto">
      <h2>Retinal Vessel Segmentation</h2>
      <p>Upload a fundus image; the predicted vessel mask is overlaid in red.</p>
      <form action="/segment" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Segment</button>
      </form>
    </body></html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "device": DEVICE, "checkpoint_loaded": str(os.path.exists(CHECKPOINT))}


async def _read_rgb(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an RGB uint8 array or raise HTTP 400."""
    try:
        raw = await file.read()
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@app.post("/segment")
async def segment(file: UploadFile = File(...)) -> StreamingResponse:
    """Segment vessels and return the overlay PNG with scores burned in.

    The coverage and confidence percentages are also returned in the
    ``X-Vessel-Coverage-Pct`` and ``X-Confidence-Pct`` response headers.
    """
    rgb = await _read_rgb(file)
    mask, probs, fov = predict(rgb)
    scores = quality_scores(mask, probs, fov)
    blended = overlay_mask(rgb, mask, scores)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Vessel-Coverage-Pct": f"{scores['coverage_pct']:.2f}",
            "X-Confidence-Pct": f"{scores['confidence_pct']:.2f}",
        },
    )


@app.post("/segment.json")
async def segment_json(file: UploadFile = File(...)) -> dict[str, float]:
    """Segment vessels and return only the numeric quality scores as JSON."""
    rgb = await _read_rgb(file)
    mask, probs, fov = predict(rgb)
    return quality_scores(mask, probs, fov)
