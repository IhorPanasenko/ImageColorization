import math
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Skip the entire module if any required package is missing
lpips = pytest.importorskip("lpips", reason="lpips not installed — skipping metric tests")

from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips, evaluate_batch, time_inference


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


# ---------------------------------------------------------------------------
# time_inference
# ---------------------------------------------------------------------------

def test_time_inference_returns_result_and_positive_ms():
    """time_inference must return (callable result, positive elapsed ms)."""
    def _add(a, b):
        return a + b

    result, elapsed_ms = time_inference(_add, 3, 4, device="cpu")
    assert result == 7, f"Expected 7, got {result}"
    assert elapsed_ms > 0, f"Expected positive elapsed time, got {elapsed_ms}"


def test_time_inference_elapsed_is_float():
    """elapsed_ms returned by time_inference must be a plain float."""
    _, elapsed_ms = time_inference(lambda: None, device="cpu")
    assert isinstance(elapsed_ms, float), (
        f"Expected float, got {type(elapsed_ms)}"
    )


def test_time_inference_slow_call_reflects_duration():
    """A deliberate sleep must be reflected in the measured elapsed time."""
    import time

    def _sleep_10ms():
        time.sleep(0.01)

    _, elapsed_ms = time_inference(_sleep_10ms, device="cpu")
    # Allow generous bounds: ≥ 8 ms (timing resolution jitter) and < 500 ms
    assert elapsed_ms >= 8, f"Expected ≥ 8 ms for a 10 ms sleep, got {elapsed_ms:.2f} ms"
    assert elapsed_ms < 500, f"Unexpectedly high elapsed time: {elapsed_ms:.2f} ms"
