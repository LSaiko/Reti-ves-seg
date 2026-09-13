import torch

from models import build_model

# smp_resnet34 downloads ImageNet weights over the network — too flaky/slow
# for CI, so only the from-scratch architecture is exercised here.


def test_build_model_unet_output_shape():
    model = build_model("unet").eval()
    dummy = torch.randn(1, 2, 32, 32)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 1, 32, 32)
