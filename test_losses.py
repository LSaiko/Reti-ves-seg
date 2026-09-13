import torch

from losses import BCEDiceLoss, dice_loss


def test_bce_dice_loss_shapes_and_range():
    logits = torch.randn(2, 1, 8, 8)
    targets = (torch.rand(2, 1, 8, 8) > 0.5).float()

    loss = BCEDiceLoss()(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_dice_loss_perfect_match_and_disjoint():
    targets = (torch.rand(1, 1, 8, 8) > 0.5).float()

    perfect = dice_loss(targets, targets)
    assert perfect.item() < 1e-5

    disjoint = dice_loss(1.0 - targets, targets)
    assert disjoint.item() > 0.99
