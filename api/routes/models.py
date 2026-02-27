"""Model and checkpoint discovery endpoints.

Routes:
    GET  /api/models             — List available model types
    GET  /api/models/checkpoints — List available checkpoint files
"""

import os
import glob
from flask import Blueprint, jsonify, current_app

bp = Blueprint('models', __name__)

MODEL_TYPES = [
    {'id': 'baseline', 'name': 'Baseline CNN', 'description': 'Encoder-decoder CNN'},
    {'id': 'unet',     'name': 'U-Net',        'description': 'U-Net with skip connections'},
    {'id': 'gan',      'name': 'Pix2Pix GAN',  'description': 'Conditional GAN (generator + discriminator)'},
    {'id': 'fusion',   'name': 'Fusion GAN',   'description': 'U-Net + global hint fusion + GAN training'},
]


@bp.route('', methods=['GET'])
def list_models():
    """Return all supported model types."""
    return jsonify(MODEL_TYPES)


@bp.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """Discover all .pth checkpoint files under outputs/checkpoints/."""
    outputs_dir = current_app.config.get('OUTPUTS_DIR', 'outputs')
    pattern = os.path.join(outputs_dir, 'checkpoints', '**', '*.pth')
    paths = sorted(glob.glob(pattern, recursive=True))
    checkpoints = [
        {
            'path': p,
            'filename': os.path.basename(p),
            'size_mb': round(os.path.getsize(p) / 1e6, 1),
        }
        for p in paths
    ]
    return jsonify(checkpoints)
