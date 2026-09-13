FROM python:3.11-slim

WORKDIR /app

# opencv-python needs libGL/libglib at runtime; slim doesn't ship them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Inference is CPU-only here — pull CPU torch wheels first so the plain
# `pip install -r requirements.txt` below doesn't grab multi-GB CUDA builds.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

# Only what app.py imports at runtime — training/eval scripts aren't needed.
COPY app.py models.py unet.py dataset.py losses.py explain.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
