"""Evaluation and model comparison logic."""

from __future__ import annotations

import os
import sys
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_PATH = os.path.join(ROOT, 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)


class MetricsService:
    """Wraps metric computation, single image evaluation and model comparison."""

    def __init__(self) -> None:
        from api.services.colorizer import Colorizer
        # Reuse a single colorizer to keep models cached across requests.
        self._colorizer = Colorizer()

    def evaluate_single(
        self,
        image_path: str,
        model_type: str,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Colorize *image_path* and return PSNR / SSIM / LPIPS metrics.

        Delegates colorization to Colorizer to avoid code duplication.
        The colorizer already computes all three metrics against the GT.
        """
        result = self._colorizer.colorize(
            image_path,
            model_type,
            checkpoint_path,
            mode='color_photo',
        )
        return {
            'model': model_type,
            'checkpoint': checkpoint_path,
            'metrics': result.get('metrics', {'psnr': None, 'ssim': None, 'lpips': None}),
        }

    def evaluate_samples(
        self,
        model_type: str,
        checkpoint_path: str,
        sample_dir: str = os.path.join(ROOT, 'data', 'test_samples'),
    ) -> dict[str, Any]:
        """Run all images in *sample_dir* through the model and return averaged metrics.

        Returns:
            {
                "model": str,
                "checkpoint": str,
                "per_image": [{"filename": str, "psnr": float, "ssim": float}, ...],
                "avg_psnr": float,
                "avg_ssim": float,
            }
        """
        import glob
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        images = []
        for ext in exts:
            images.extend(glob.glob(os.path.join(sample_dir, ext)))

        per_image = []
        for img_path in sorted(images):
            try:
                result = self._colorizer.colorize(
                    img_path,
                    model_type,
                    checkpoint_path,
                    mode='color_photo',
                )
                metrics = result.get('metrics', {})
                per_image.append({
                    'filename': os.path.basename(img_path),
                    'psnr':  metrics.get('psnr'),
                    'ssim':  metrics.get('ssim'),
                    'lpips': metrics.get('lpips'),
                })
            except Exception as exc:
                per_image.append({
                    'filename': os.path.basename(img_path),
                    'error': str(exc),
                })

        valid_psnr = [r['psnr'] for r in per_image if r.get('psnr') is not None]
        valid_ssim = [r['ssim'] for r in per_image if r.get('ssim') is not None]
        valid_lpips = [r['lpips'] for r in per_image if r.get('lpips') is not None]

        avg_psnr = sum(valid_psnr) / len(valid_psnr) if valid_psnr else None
        avg_ssim = sum(valid_ssim) / len(valid_ssim) if valid_ssim else None
        avg_lpips = sum(valid_lpips) / len(valid_lpips) if valid_lpips else None

        return {
            'model': model_type,
            'checkpoint': checkpoint_path,
            'per_image': per_image,
            'avg_psnr':  avg_psnr,
            'avg_ssim':  avg_ssim,
            'avg_lpips': avg_lpips,
            'num_images': len(images),
        }

    def compare_models(
        self,
        image_path: str,
        model_configs: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Colorize *image_path* with each model config and return metrics + images.

        Args:
            image_path: Path to source image.
            model_configs: List of dicts with keys "model", "checkpoint", "label".

        Returns:
            List of result dicts, one per model config.
        """
        results = []
        for cfg in model_configs:
            try:
                result = self._colorizer.colorize(
                    image_path,
                    cfg.get('model', 'unet'),
                    cfg.get('checkpoint', ''),
                    mode='color_photo',
                )
                results.append({
                    'label': cfg.get('label', cfg.get('model', '?')),
                    **result,
                })
            except Exception as exc:
                results.append({
                    'label': cfg.get('label', cfg.get('model', '?')),
                    'error': str(exc),
                })
        return results
