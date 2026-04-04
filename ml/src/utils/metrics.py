"""
Quantitative evaluation metrics for colorization quality.

All functions expect RGB images as numpy arrays with shape (H, W, 3)
and values in the range [0, 1].
"""

import time
from typing import Any, Callable

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Module-level LPIPS model cache keyed by device string — initialized once
# per device on first use to avoid reloading weights on every request.
# LPIPS is used only for metric computation, so CPU is strongly preferred:
# lpips 0.1.4 has known instability on MPS with torch >=2.6, and CPU is
# fast enough for a single 256×256 image pair.
_lpips_models: dict = {}


def _get_lpips_model(device: str = "cpu"):
    # Lazy-import lpips and torch so that PSNR / SSIM remain available
    # even when lpips or its transitive dependencies (sympy, etc.) are broken.
    import lpips as _lpips
    import torch  # noqa: F811

    # Always run LPIPS on CPU to avoid MPS/CUDA compatibility issues in
    # the lpips library (the metric calculation is not a bottleneck).
    _device = "cpu"
    if _device not in _lpips_models:
        model = _lpips.LPIPS(net="alex").to(_device)
        model.eval()
        _lpips_models[_device] = model
    return _lpips_models[_device]


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio between predicted and target RGB images.

    Args:
        pred:   Predicted RGB image, float32 in [0, 1], shape (H, W, 3).
        target: Ground-truth RGB image, float32 in [0, 1], shape (H, W, 3).

    Returns:
        PSNR value in dB. Returns float('inf') for identical images.
    """
    pred = np.clip(pred, 0.0, 1.0).astype(np.float64)
    target = np.clip(target, 0.0, 1.0).astype(np.float64)
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Structural Similarity Index between predicted and target RGB images.

    Args:
        pred:   Predicted RGB image, float32 in [0, 1], shape (H, W, 3).
        target: Ground-truth RGB image, float32 in [0, 1], shape (H, W, 3).

    Returns:
        SSIM value in [-1, 1]; 1.0 means identical images.
    """
    pred = np.clip(pred, 0.0, 1.0).astype(np.float64)
    target = np.clip(target, 0.0, 1.0).astype(np.float64)
    return float(
        structural_similarity(target, pred, data_range=1.0, channel_axis=2)
    )


def compute_lpips(pred: np.ndarray, target: np.ndarray, device: str = "cpu") -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) using AlexNet features.
    Lower is better (0 = identical).

    Args:
        pred:   Predicted RGB image, float32 in [0, 1], shape (H, W, 3).
        target: Ground-truth RGB image, float32 in [0, 1], shape (H, W, 3).
        device: Ignored — LPIPS always runs on CPU to avoid lpips library
                incompatibilities with MPS/CUDA in newer torch versions.

    Returns:
        LPIPS distance as a float.
    """
    import torch as _torch

    model = _get_lpips_model("cpu")

    def _to_tensor(img: np.ndarray) -> _torch.Tensor:
        # LPIPS expects (1, 3, H, W) float tensors normalized to [-1, 1]
        t = _torch.from_numpy(img.transpose(2, 0, 1)).float()  # (3, H, W)
        t = t * 2.0 - 1.0  # [0,1] -> [-1,1]
        return t.unsqueeze(0).to("cpu")  # (1, 3, H, W)

    with _torch.no_grad():
        dist = model(_to_tensor(pred), _to_tensor(target))

    return float(dist.item())


def time_inference(
    fn: Callable,
    *args: Any,
    device: str = "cpu",
    **kwargs: Any,
) -> tuple[Any, float]:
    """
    Measure wall-clock time for a single forward-pass call in milliseconds.

    For CUDA devices the GPU is synchronised before and after the call so that
    asynchronous kernel execution does not skew the measurement.  MPS
    synchronisation is applied when available.

    Args:
        fn:     Callable to time (e.g. a lambda wrapping ``model(tensor)``).
        *args:  Positional arguments forwarded to *fn*.
        device: Torch device string — used to decide whether to synchronise.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        ``(result, elapsed_ms)`` — *fn*'s return value and elapsed wall-clock
        time in milliseconds.
    """
    import torch as _torch

    def _sync() -> None:
        if "cuda" in device:
            _torch.cuda.synchronize()
        elif "mps" in device:
            try:
                _torch.mps.synchronize()
            except Exception:
                pass

    _sync()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


def evaluate_batch(
    pred_images: list[np.ndarray],
    target_images: list[np.ndarray],
    device: str = "cpu",
) -> dict:
    """
    Compute average PSNR, SSIM, and LPIPS over a list of image pairs.

    Args:
        pred_images:   List of predicted RGB images, each (H, W, 3) in [0, 1].
        target_images: List of ground-truth RGB images, each (H, W, 3) in [0, 1].
        device:        Torch device string for LPIPS computation.

    Returns:
        Dict with keys 'psnr', 'ssim', 'lpips' containing averaged float values.
    """
    assert len(pred_images) == len(target_images), (
        f"Mismatched list lengths: {len(pred_images)} preds vs {len(target_images)} targets"
    )

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for pred, target in zip(pred_images, target_images):
        psnr_vals.append(compute_psnr(pred, target))
        ssim_vals.append(compute_ssim(pred, target))
        lpips_vals.append(compute_lpips(pred, target, device=device))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
    }
