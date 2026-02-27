import math
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Skip the entire module if any required package is missing
lpips = pytest.importorskip("lpips", reason="lpips not installed — skipping metric tests")

from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips, evaluate_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random float32 RGB image in [0, 1] at a small resolution."""
    return RNG.random((h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def test_psnr_identical_images():
    """PSNR of an image compared to itself must be infinite."""
    img = _random_image()
    psnr = compute_psnr(img, img)
    assert math.isinf(psnr) or psnr > 100, (
        f"Expected PSNR → inf for identical images, got {psnr}"
    )


def test_psnr_different_images():
    """PSNR between two different images must be a finite positive value."""
    img_a = _random_image()
    img_b = _random_image()
    psnr = compute_psnr(img_a, img_b)
    assert math.isfinite(psnr) and psnr > 0, (
        f"Expected finite positive PSNR, got {psnr}"
    )


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def test_ssim_identical_images():
    """SSIM of an image compared to itself must equal 1.0."""
    img = _random_image()
    ssim = compute_ssim(img, img)
    assert abs(ssim - 1.0) < 1e-4, f"Expected SSIM ≈ 1.0 for identical images, got {ssim}"


def test_ssim_different_images():
    """SSIM between two different images must be in (-1, 1)."""
    img_a = _random_image()
    img_b = _random_image()
    ssim = compute_ssim(img_a, img_b)
    assert -1.0 <= ssim < 1.0, f"Expected SSIM in [-1, 1), got {ssim}"


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def test_lpips_identical_images():
    """LPIPS of an image compared to itself must be ≈ 0."""
    img = _random_image()
    score = compute_lpips(img, img, device="cpu")
    assert score < 0.05, f"Expected LPIPS ≈ 0 for identical images, got {score}"


def test_lpips_different_images():
    """LPIPS between two random images must be a positive finite value."""
    img_a = _random_image()
    img_b = _random_image()
    score = compute_lpips(img_a, img_b, device="cpu")
    assert math.isfinite(score) and score > 0, (
        f"Expected positive finite LPIPS, got {score}"
    )


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------

def test_evaluate_batch_returns_all_keys():
    """evaluate_batch must return a dict with psnr, ssim, and lpips keys."""
    imgs_a = [_random_image() for _ in range(3)]
    imgs_b = [_random_image() for _ in range(3)]
    result = evaluate_batch(imgs_a, imgs_b, device="cpu")
    assert set(result.keys()) == {"psnr", "ssim", "lpips"}, (
        f"Unexpected keys: {result.keys()}"
    )
    for key, val in result.items():
        assert math.isfinite(val), f"Metric '{key}' is not finite: {val}"
