"""Metrics and evaluation endpoints.

Routes:
    POST  /api/metrics/evaluate        — Evaluate a single image with one model
    POST  /api/metrics/compare         — Run one image through multiple models
    POST  /api/metrics/batch_evaluate  — Evaluate all test samples for a model
"""

from flask import Blueprint, request, jsonify, current_app
from api.services.metrics_service import MetricsService

bp = Blueprint('metrics', __name__)
_svc = MetricsService()


@bp.route('/evaluate', methods=['POST'])
def evaluate_image():
    """Compute PSNR, SSIM, LPIPS for a single prediction.

    JSON body:
        {
            "image_path": str,    # path to source image
            "model": str,
            "checkpoint": str
        }
    """
    data = request.get_json(force=True) or {}
    try:
        result = _svc.evaluate_single(
            data['image_path'],
            data.get('model', 'unet'),
            data.get('checkpoint', ''),
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify(result)


@bp.route('/compare', methods=['POST'])
def compare_models():
    """Run one image through multiple model configurations and return metrics.

    JSON body:
        {
            "image_path": str,
            "models": [
                {"model": str, "checkpoint": str, "label": str},
                ...
            ]
        }
    """
    data = request.get_json(force=True) or {}
    try:
        result = _svc.compare_models(
            data['image_path'],
            data.get('models', []),
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify(result)


@bp.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """Evaluate all images in data/test_samples/ for a given model.

    JSON body:
        {
            "model": str,
            "checkpoint": str
        }
    """
    data = request.get_json(force=True) or {}
    try:
        result = _svc.evaluate_samples(
            data.get('model', 'unet'),
            data.get('checkpoint', ''),
            sample_dir=current_app.config.get('SAMPLE_DIR', 'data/test_samples'),
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify(result)
