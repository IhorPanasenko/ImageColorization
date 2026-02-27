"""
Quantitative evaluation metrics for colorization quality.

All functions expect RGB images as numpy arrays with shape (H, W, 3)
and values in the range [0, 1].
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips


# Module-level LPIPS model — initialized once on first use to avoid
# reloading weights for every image during batch evaluation.
_lpips_model = None


def _get_lpips_model(device: str = "cpu") -> lpips.LPIPS:
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


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
        device: Torch device string — 'cpu', 'cuda', or 'mps'.

    Returns:
        LPIPS distance as a float.
    """
    model = _get_lpips_model(device)

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        # LPIPS expects (1, 3, H, W) float tensors normalized to [-1, 1]
        t = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (3, H, W)
        t = t * 2.0 - 1.0  # [0,1] -> [-1,1]
        return t.unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        dist = model(_to_tensor(pred), _to_tensor(target))

    return float(dist.item())


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
