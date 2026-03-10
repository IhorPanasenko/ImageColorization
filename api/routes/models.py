"""Model and checkpoint discovery endpoints.

Routes:
    GET  /api/models             — List available model types
    GET  /api/models/checkpoints — List available checkpoint files
"""

from flask import Blueprint, jsonify, current_app
from api.services.checkpoint_service import discover_checkpoints

bp = Blueprint('models', __name__)

MODEL_TYPES = [
    {'id': 'baseline',        'name': 'Baseline CNN',             'description': 'Encoder-decoder CNN',                          'classical': False},
    {'id': 'unet',            'name': 'U-Net',                    'description': 'U-Net with skip connections',                   'classical': False},
    {'id': 'gan',             'name': 'Pix2Pix GAN',              'description': 'Conditional GAN (generator + discriminator)',    'classical': False},
    {'id': 'fusion',          'name': 'Fusion GAN',               'description': 'U-Net + global hint fusion + GAN training',     'classical': False},
    {'id': 'classical_welsh', 'name': 'Welsh 2002 (Colour Transfer)', 'description': 'Texture-based colour transfer via Lab + KD-Tree matching (no checkpoint needed — upload a reference image)', 'classical': True},
    {'id': 'classical_levin', 'name': 'Levin 2004 (Optimisation)', 'description': 'Sparse optimisation with random colour hints sampled from a reference image (no checkpoint needed)', 'classical': True},
]


@bp.route('', methods=['GET'])
def list_models():
    """Return all supported model types."""
    return jsonify(MODEL_TYPES)


@bp.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """Discover all .pth checkpoint files under outputs/checkpoints/."""
    outputs_dir = current_app.config.get('OUTPUTS_DIR', 'outputs')
    return jsonify(discover_checkpoints(outputs_dir))
