"""Classical colorization service.

Wraps the Welsh 2002 and Levin 2004 algorithms so that the inference route
can call ``ClassicalColorizer.colorize(...)`` and receive the same response
dict shape as ``Colorizer.colorize()``.

Response dict keys:
    colorized    — base64 PNG of the predicted colour image
    grayscale    — base64 PNG of the luminance channel (greyscale display)
    original     — base64 PNG of the uploaded target as-is
    ground_truth — base64 PNG (only in color_photo mode)
    metrics      — {"psnr": ..., "ssim": ..., "lpips": ...}
                   (populated only in color_photo mode)
"""

from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_SRC = os.path.join(ROOT, 'ml', 'src')
if ML_SRC not in sys.path:
    sys.path.insert(0, ML_SRC)

ML_PATH = os.path.join(ROOT, 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)


_RESAMPLE = (
    Image.Resampling.BICUBIC
    if hasattr(Image, "Resampling")
    else getattr(Image, "BICUBIC", 3)
)

TARGET_SIZE = 256


def _img_to_b64(arr: np.ndarray) -> str:
    """Convert float32 H×W×3 numpy array (0–1) to base64-encoded PNG string."""
    uint8 = (arr.clip(0, 1) * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(uint8).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize float32 RGB (0–1) to ``(w, h)``."""
    uint8 = (arr.clip(0, 1) * 255).astype(np.uint8)
    return np.array(Image.fromarray(uint8).resize(size, _RESAMPLE)).astype(np.float32) / 255.0


def _to_gray_rgb(path: str, target_size: int) -> np.ndarray:
    """Return the greyscale (L-only) representation as float32 RGB (H,W,3)."""
    from skimage import color
    from skimage.transform import resize as sk_resize

    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = sk_resize(arr, (target_size, target_size), anti_aliasing=True)
    lab = color.rgb2lab(arr)
    L   = lab[:, :, 0] / 100.0          # normalise to [0, 1]
    return np.stack([L, L, L], axis=2).astype(np.float32)


class ClassicalColorizer:
    """Stateless wrapper for the two classical colorization algorithms."""

    def colorize(
        self,
        target_path: str,
        reference_path: str,
        method: str = 'welsh',
        mode: str = 'grayscale',
        n_hints: int = 150,
        window: int = 5,
    ) -> dict[str, Any]:
        """Run classical colorization and return a result dict.

        Args:
            target_path:    Path to the (greyscale or colour) input image.
            reference_path: Path to the colour reference image.
            method:         ``"welsh"`` or ``"levin"``.
            mode:           ``"grayscale"`` — input is B&W; no GT metrics.
                            ``"color_photo"`` — input is colour; compute metrics.
            n_hints:        Number of hint pixels for Levin (ignored for Welsh).
            window:         Local sampling window size.

        Returns:
            Same dict shape that ``Colorizer.colorize()`` returns.
        """
        from algorithms.welsh import colorize_welsh
        from algorithms.levin import colorize_levin
        from utils.metrics import time_inference

        # -- Run the algorithm -------------------------------------------------
        def _run_algorithm():
            if method == 'welsh':
                return colorize_welsh(
                    target_path, reference_path,
                    target_size=TARGET_SIZE,
                    window=window,
                )
            elif method == 'levin':
                return colorize_levin(
                    target_path, reference_path,
                    target_size=TARGET_SIZE,
                    n_hints=n_hints,
                    window=3,
                )
            else:
                raise ValueError(f'Unknown classical method: {method!r}')

        pred_rgb, elapsed_ms = time_inference(_run_algorithm, device='cpu')

        # -- Build display images ----------------------------------------------
        original_np = (
            np.array(Image.open(target_path).convert('RGB'), dtype=np.float32) / 255.0
        )
        gray_rgb = _to_gray_rgb(target_path, TARGET_SIZE)

        # Resize original back to its natural resolution for display
        orig_h, orig_w = original_np.shape[:2]
        pred_disp = _resize_rgb(pred_rgb, (orig_w, orig_h))
        gray_disp = _resize_rgb(gray_rgb, (orig_w, orig_h))

        result: dict[str, Any] = {
            'colorized': _img_to_b64(pred_disp),
            'grayscale': _img_to_b64(gray_disp),
            'original':  _img_to_b64(original_np),
            'metrics':   {'psnr': None, 'ssim': None, 'lpips': None,
                         'inference_time_ms': round(elapsed_ms, 2)},
        }

        # -- Metrics (colour_photo mode only) ----------------------------------
        if mode == 'color_photo':
            try:
                from utils.metrics import compute_psnr, compute_ssim, compute_lpips
                from utils.common import get_device

                # Ground truth: treat colour original resized to TARGET_SIZE
                from skimage.transform import resize as sk_resize
                gt_rgb = sk_resize(original_np, (TARGET_SIZE, TARGET_SIZE), anti_aliasing=True)
                gt_rgb = gt_rgb.astype(np.float32)

                result['ground_truth'] = _img_to_b64(_resize_rgb(gt_rgb, (orig_w, orig_h)))
                device = str(get_device())
                result['metrics']['psnr']  = float(compute_psnr(pred_rgb, gt_rgb))
                result['metrics']['ssim']  = float(compute_ssim(pred_rgb, gt_rgb))
                result['metrics']['lpips'] = float(compute_lpips(pred_rgb, gt_rgb, device=device))
            except Exception:
                pass  # metrics are best-effort

        return result
