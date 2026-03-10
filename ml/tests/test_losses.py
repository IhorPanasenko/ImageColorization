import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.losses import GANLoss


@pytest.fixture
def gan_loss():
    return GANLoss()


def test_gan_loss_real(gan_loss):
    """GANLoss with target_is_real=True must return a positive scalar tensor."""
    prediction = torch.randn(2, 1, 30, 30)
    loss = gan_loss(prediction, target_is_real=True)

    assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
    assert loss.ndim == 0, "Loss must be a scalar (0-dim tensor)"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"


def test_gan_loss_fake(gan_loss):
    """GANLoss with target_is_real=False must return a positive scalar tensor."""
    prediction = torch.randn(2, 1, 30, 30)
    loss = gan_loss(prediction, target_is_real=False)

    assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
    assert loss.ndim == 0, "Loss must be a scalar (0-dim tensor)"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"


def test_gan_loss_real_higher_than_fake_on_positive_logits():
    """With strong positive logits (confident 'real'), real-target loss should be
    lower than fake-target loss, since the discriminator predicts 'real' correctly."""
    loss_fn = GANLoss()
    # Large positive logits → discriminator confidently says 'real'
    prediction = torch.ones(2, 1, 30, 30) * 5.0

    loss_real = loss_fn(prediction, target_is_real=True).item()
    loss_fake = loss_fn(prediction, target_is_real=False).item()

    assert loss_real < loss_fake, (
        f"For positive logits, real-target loss ({loss_real:.4f}) should be "
        f"lower than fake-target loss ({loss_fake:.4f})"
    )


# ── Label smoothing tests ──────────────────────────────────────────────────────

def test_gan_loss_label_smoothing():
    """With label_smoothing=0.1, real label should be 0.9 instead of 1.0."""
    loss_smoothed = GANLoss(label_smoothing=0.1)
    prediction = torch.zeros(1, 1, 4, 4)
    target = loss_smoothed.get_target_tensor(prediction, target_is_real=True)
    assert torch.allclose(target, torch.full_like(prediction, 0.9)), \
        f"Expected real label 0.9, got {target.unique().tolist()}"

    target_fake = loss_smoothed.get_target_tensor(prediction, target_is_real=False)
    assert torch.allclose(target_fake, torch.full_like(prediction, 0.0)), \
        f"Expected fake label 0.0, got {target_fake.unique().tolist()}"


# ── Perceptual loss tests ───────────────────────────────────────────────────────

from src.losses import PerceptualLoss, ab_to_pseudo_rgb


def test_perceptual_loss_shape():
    """PerceptualLoss should return a scalar positive loss."""
    loss_fn = PerceptualLoss()
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss must be scalar"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"


def test_perceptual_loss_identical():
    """PerceptualLoss on identical inputs should be ~0."""
    loss_fn = PerceptualLoss()
    img = torch.rand(1, 3, 64, 64)
    loss = loss_fn(img, img)
    assert loss.item() < 1e-5, f"Expected near-zero loss for identical inputs, got {loss.item()}"


def test_ab_to_pseudo_rgb_shape():
    """ab_to_pseudo_rgb should produce (B, 3, H, W) in roughly [0, 1]."""
    L = torch.rand(2, 1, 32, 32)
    ab = torch.rand(2, 2, 32, 32) * 2 - 1  # [-1, 1]
    out = ab_to_pseudo_rgb(L, ab)
    assert out.shape == (2, 3, 32, 32), f"Expected (2,3,32,32), got {out.shape}"
    assert out.min() >= -0.01, f"Expected values >= ~0, got min {out.min()}"
    assert out.max() <= 1.01, f"Expected values <= ~1, got max {out.max()}"
