"""Evaluation and model comparison logic."""

from __future__ import annotations

import glob
import os
import sys
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_PATH = os.path.join(ROOT, 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)

CLASSICAL_METHOD_MAP = {'classical_welsh': 'welsh', 'classical_levin': 'levin'}
_IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')


class MetricsService:
    """Wraps metric computation, single image evaluation and model comparison."""

    def __init__(self) -> None:
        from api.services.colorizer import Colorizer
        from api.services.classical_colorizer import ClassicalColorizer
        # Reuse a single colorizer to keep models cached across requests.
        self._colorizer = Colorizer()
        self._classical_colorizer = ClassicalColorizer()

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
        metrics = result.get('metrics', {})
        return {
            'model': model_type,
            'checkpoint': checkpoint_path,
            'metrics': {
                'psnr':  metrics.get('psnr'),
                'ssim':  metrics.get('ssim'),
                'lpips': metrics.get('lpips'),
                'inference_time_ms': metrics.get('inference_time_ms'),
            },
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
        images = []
        for ext in _IMAGE_EXTS:
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
                    'inference_time_ms': metrics.get('inference_time_ms'),
                })
            except Exception as exc:
                per_image.append({
                    'filename': os.path.basename(img_path),
                    'error': str(exc),
                })

        valid_psnr = [r['psnr'] for r in per_image if r.get('psnr') is not None]
        valid_ssim = [r['ssim'] for r in per_image if r.get('ssim') is not None]
        valid_lpips = [r['lpips'] for r in per_image if r.get('lpips') is not None]
        valid_time = [r['inference_time_ms'] for r in per_image if r.get('inference_time_ms') is not None]

        avg_psnr = sum(valid_psnr) / len(valid_psnr) if valid_psnr else None
        avg_ssim = sum(valid_ssim) / len(valid_ssim) if valid_ssim else None
        avg_lpips = sum(valid_lpips) / len(valid_lpips) if valid_lpips else None
        avg_inference_time_ms = sum(valid_time) / len(valid_time) if valid_time else None

        return {
            'model': model_type,
            'checkpoint': checkpoint_path,
            'per_image': per_image,
            'avg_psnr':  avg_psnr,
            'avg_ssim':  avg_ssim,
            'avg_lpips': avg_lpips,
            'avg_inference_time_ms': avg_inference_time_ms,
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

    # ------------------------------------------------------------------
    # Bulk benchmark — run every model on the test set
    # ------------------------------------------------------------------

    def benchmark(
        self,
        model_configs: list[dict[str, str]],
        sample_dir: str = os.path.join(ROOT, 'data', 'test_samples'),
        max_images: int | None = None,
        image_dir: str | None = None,
    ) -> dict[str, Any]:
        """Run every model config on the test set and return per-image + averaged metrics.

        For classical models the first image is used as the colour reference
        and the remaining images are evaluated.  Neural models evaluate all
        images.

        Args:
            image_dir: Optional server-side directory to use instead of *sample_dir*.
                       Must be an existing directory; if invalid, an error dict is returned.

        Returns:
            ``{"results": [BenchmarkModelResult, ...]}``
        """
        effective_dir = sample_dir
        if image_dir:
            if not os.path.isdir(image_dir):
                return {'error': f'Directory not found: {image_dir}', 'results': []}
            effective_dir = image_dir

        images: list[str] = []
        for ext in _IMAGE_EXTS:
            images.extend(glob.glob(os.path.join(effective_dir, ext)))
        images.sort()

        if max_images and max_images > 0:
            images = images[:max_images]

        if not images:
            return {'results': []}

        # Classical algorithms need a colour reference — use the first image.
        reference_path = images[0]

        all_results: list[dict[str, Any]] = []
        for cfg in model_configs:
            model_id: str = cfg.get('model', 'unet')
            is_classical = model_id in CLASSICAL_METHOD_MAP
            eval_images = images[1:] if is_classical else images

            per_image: list[dict[str, Any]] = []
            for img_path in eval_images:
                try:
                    if is_classical:
                        r = self._classical_colorizer.colorize(
                            img_path,
                            reference_path,
                            method=CLASSICAL_METHOD_MAP[model_id],
                            mode='color_photo',
                        )
                    else:
                        r = self._colorizer.colorize(
                            img_path,
                            model_id,
                            cfg.get('checkpoint', ''),
                            mode='color_photo',
                        )
                    m = r.get('metrics', {})
                    per_image.append({
                        'filename': os.path.basename(img_path),
                        'psnr': m.get('psnr'),
                        'ssim': m.get('ssim'),
                        'lpips': m.get('lpips'),
                        'inference_time_ms': m.get('inference_time_ms'),
                    })
                except Exception as exc:
                    per_image.append({
                        'filename': os.path.basename(img_path),
                        'error': str(exc),
                    })

            valid = [r for r in per_image if 'error' not in r]

            def _avg(key: str) -> float | None:
                vals = [r[key] for r in valid if r.get(key) is not None]
                return sum(vals) / len(vals) if vals else None

            all_results.append({
                'label': cfg.get('label', model_id),
                'model': model_id,
                'per_image': per_image,
                'avg_psnr': _avg('psnr'),
                'avg_ssim': _avg('ssim'),
                'avg_lpips': _avg('lpips'),
                'avg_inference_time_ms': _avg('inference_time_ms'),
                'num_images': len(eval_images),
            })

        return {'results': all_results}
