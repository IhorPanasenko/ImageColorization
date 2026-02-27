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
        from api.services.colorizer import Colorizer
        colorizer = Colorizer()
        result = colorizer.colorize(image_path, model_type, checkpoint_path)
        return {
            'model': model_type,
            'checkpoint': checkpoint_path,
            'metrics': result.get('metrics', {'psnr': None, 'ssim': None, 'lpips': None}),
        }

    def evaluate_samples(
        self,
        model_type: str,
        checkpoint_path: str,
        sample_dir: str = 'data/test_samples',
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
        from api.services.colorizer import Colorizer
        colorizer = Colorizer()

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        images = []
        for ext in exts:
            images.extend(glob.glob(os.path.join(sample_dir, ext)))

        per_image = []
        for img_path in sorted(images):
            try:
                result = colorizer.colorize(img_path, model_type, checkpoint_path)
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

        valid = [r for r in per_image if r.get('psnr') is not None]
        avg_psnr  = sum(r['psnr']  for r in valid) / len(valid) if valid else None
        avg_ssim  = sum(r['ssim']  for r in valid) / len(valid) if valid else None
        avg_lpips = sum(r['lpips'] for r in valid) / len(valid) if valid else None

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
        from api.services.colorizer import Colorizer
        colorizer = Colorizer()
        results = []
        for cfg in model_configs:
            try:
                result = colorizer.colorize(
                    image_path,
                    cfg.get('model', 'unet'),
                    cfg.get('checkpoint', ''),
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
