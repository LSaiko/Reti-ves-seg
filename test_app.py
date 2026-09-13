import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import app

client = TestClient(app)


def _fake_image_bytes() -> io.BytesIO:
    img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_segment_smoke():
    r = client.post("/segment", files={"file": ("t.png", _fake_image_bytes(), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "X-Vessel-Coverage-Pct" in r.headers
