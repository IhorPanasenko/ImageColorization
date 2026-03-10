"""Inference wrapper — loads models and runs colorization.

Mirrors the logic in ml/scripts/evaluate.py so that the API produces
identical results to the standalone evaluation script.
"""

from __future__ import annotations

import base64
import os
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_PATH = os.path.join(ROOT, 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

_MAX_CACHE = 4
_RESAMPLE = (
    Image.Resampling.BICUBIC
    if hasattr(Image, "Resampling")
    else getattr(Image, "BICUBIC", 3)
)


def _img_to_b64(arr: np.ndarray) -> str:
    """Convert a float32 HxWxC numpy array (0-1 range) to a base64 PNG string."""
    uint8 = (arr.clip(0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(uint8)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a float32 RGB image (0-1 range) to the given (width, height)."""
    uint8 = (arr.clip(0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(uint8).resize(size, _RESAMPLE)
    return (np.array(img).astype(np.float32) / 255.0)


class Colorizer:
    """Wraps model loading and inference with an LRU model cache.

    Cache key is "model_type::checkpoint_path".  Up to _MAX_CACHE model pairs
    (model, hint_net) are kept in memory to avoid reloading ~100 MB weights
    on every request.
    """

    def __init__(self) -> None:
        # Maps cache_key -> (model, hint_net | None)
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def colorize(
        self,
        image_path: str,
        model_type: str,
        checkpoint_path: str,
        mode: str = 'grayscale',
    ) -> dict[str, Any]:
        """Colorize *image_path* and return base64-encoded result images.

        Args:
            image_path: Path to source image.
            model_type: One of "baseline", "unet", "gan", "fusion".
            checkpoint_path: Path to the .pth checkpoint file.
            mode: "grayscale" (input is B&W, no ground truth for metrics)
                  or "color_photo" (input is color, compute metrics vs original).

        Returns:
            Dict with keys:
                "colorized"    — base64 PNG of the predicted color image
                "grayscale"    — base64 PNG of the L-only (grayscale) input
                "original"     — base64 PNG of the source file as uploaded
                "ground_truth" — base64 PNG of GT (only in color_photo mode)
                "metrics"      — {"psnr", "ssim", "lpips"} (only in color_photo mode)
        """
        import torch
        # All image-processing helpers live in src.utils.common
        from src.utils.common import get_device, prepare_grayscale_input, lab_to_rgb
        from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips

        # Prefer MPS (Apple Silicon), then CUDA, else CPU
        device = get_device()
        model, hint_net = self._load_model(model_type, checkpoint_path, device)

        # Original image for display (before any conversion)
        img_pil = Image.open(image_path).convert('RGB')
        original_np = np.array(img_pil).astype(np.float32) / 255.0

        # prepare_grayscale_input returns (L_tensor (1,1,H,W), gt_rgb (H,W,3))
        L_tensor, gt_rgb, meta = prepare_grayscale_input(
            image_path,
            target_size=256,
            return_meta=True,
        )
        L_tensor = L_tensor.to(device)

        with torch.no_grad():
            if hint_net is not None:
                global_hint = hint_net(L_tensor)          # (1, 512)
                pred_ab = model(L_tensor, global_hint)    # (1, 2, H, W)
            else:
                pred_ab = model(L_tensor)                 # (1, 2, H, W)

        pred_rgb = lab_to_rgb(L_tensor[0], pred_ab[0])   # (H, W, 3) float32

        # Build 3-channel grayscale image for display
        L_np = L_tensor[0].cpu().squeeze().numpy()        # (H, W) in [0, 1]
        gray_rgb = np.stack([L_np] * 3, axis=2)           # (H, W, 3)

        # Recover original aspect ratio if padding was applied
        pad_left = meta['pad_left']
        pad_top = meta['pad_top']
        resized_w = meta['resized_w']
        resized_h = meta['resized_h']
        orig_w = meta['orig_w']
        orig_h = meta['orig_h']

        pred_crop = pred_rgb[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]
        gray_crop = gray_rgb[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]
        gt_crop = gt_rgb[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]

        pred_disp = _resize_rgb(pred_crop, (orig_w, orig_h))
        gray_disp = _resize_rgb(gray_crop, (orig_w, orig_h))

        result: dict[str, Any] = {
            'colorized': _img_to_b64(pred_disp),
            'grayscale': _img_to_b64(gray_disp),
            'original':  _img_to_b64(original_np),
            'metrics':   {'psnr': None, 'ssim': None, 'lpips': None},
        }

        # Compute metrics only in color_photo mode
        # (grayscale mode has no ground truth, metrics would be meaningless)
        if mode == 'color_photo':
            try:
                result['ground_truth'] = _img_to_b64(_resize_rgb(gt_crop, (orig_w, orig_h)))
                result['metrics']['psnr']  = float(compute_psnr(pred_crop, gt_crop))
                result['metrics']['ssim']  = float(compute_ssim(pred_crop, gt_crop))
                result['metrics']['lpips'] = float(
                    compute_lpips(pred_crop, gt_crop, device=str(device))
                )
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(
        self,
        model_type: str,
        checkpoint_path: str,
        device: Any,
    ) -> tuple[Any, Any]:
        """Return a (model, hint_net | None) pair, using LRU cache."""
        cache_key = f'{model_type}::{checkpoint_path}'
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        model, hint_net = self._build_model(model_type, device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            import torch
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state)

        model.to(device).eval()
        if hint_net is not None:
            hint_net.eval()

        pair = (model, hint_net)
        if len(self._cache) >= _MAX_CACHE:
            self._cache.popitem(last=False)
        self._cache[cache_key] = pair
        return pair

    @staticmethod
    def _build_model(model_type: str, device: Any) -> tuple[Any, Any]:
        """Instantiate model (and hint_net for fusion) following evaluate.py."""
        hint_net = None
        if model_type == 'baseline':
            from src.models.baseline_cnn import BaselineCNN
            model = BaselineCNN().to(device)
        elif model_type in ('unet', 'gan'):
            from src.models.u_net import UNet
            model = UNet().to(device)
        elif model_type == 'fusion':
            from src.models.unet_fusion import UNetFusion
            from src.models.global_hints import GlobalHintNet
            model = UNetFusion().to(device)
            hint_net = GlobalHintNet().to(device)
        else:
            raise ValueError(f'Unknown model type: {model_type!r}')
        return model, hint_net
