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
