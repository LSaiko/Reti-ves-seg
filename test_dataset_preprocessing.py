import numpy as np

from dataset import _pad_to, make_input


def test_make_input_shape_and_range():
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    inp = make_input(rgb)

    assert inp.shape == (64, 64, 2)
    assert inp.dtype == np.float32
    assert inp.min() >= 0.0 and inp.max() <= 1.0


def test_pad_to_roundtrip():
    arr = np.random.rand(10, 12, 2).astype(np.float32)

    padded, (top, left) = _pad_to(arr, size=32)

    assert padded.shape == (32, 32, 2)
    cropped = padded[top : top + 10, left : left + 12]
    assert np.array_equal(cropped, arr)
