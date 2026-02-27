"""Colorization / inference endpoints.

Routes:
    POST  /api/colorize              — Single image colorization
    POST  /api/colorize/batch        — Multi-image colorization
    GET   /api/colorize/result/<fn>  — Serve a saved result image
"""

import os
import uuid
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from api.services.colorizer import Colorizer

bp = Blueprint('inference', __name__)
_colorizer = Colorizer()


@bp.route('', methods=['POST'])
def colorize_single():
    """Colorize one uploaded image.

    Form fields:
        file        — image file
        model       — "baseline" | "unet" | "gan" | "fusion"
        checkpoint  — path to checkpoint .pth  (relative to outputs/)
        mode        — "grayscale" | "color_photo"

    Returns JSON with base64-encoded result images.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    model = request.form.get('model', 'unet')
    checkpoint = request.form.get('checkpoint', '')
    mode = request.form.get('mode', 'grayscale')

    upload_dir = current_app.config['UPLOAD_FOLDER']
    filename = f'{uuid.uuid4().hex}_{file.filename}'
    save_path = os.path.join(upload_dir, filename)
    file.save(save_path)

    try:
        result = _colorizer.colorize(save_path, model, checkpoint, mode)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify(result)


@bp.route('/batch', methods=['POST'])
def colorize_batch():
    """Colorize multiple uploaded images."""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    model = request.form.get('model', 'unet')
    checkpoint = request.form.get('checkpoint', '')
    mode = request.form.get('mode', 'grayscale')

    upload_dir = current_app.config['UPLOAD_FOLDER']
    results = []
    for file in files:
        filename = f'{uuid.uuid4().hex}_{file.filename}'
        save_path = os.path.join(upload_dir, filename)
        file.save(save_path)
        try:
            result = _colorizer.colorize(save_path, model, checkpoint, mode)
            results.append({'filename': file.filename, **result})
        except Exception as exc:
            results.append({'filename': file.filename, 'error': str(exc)})

    return jsonify(results)


@bp.route('/result/<filename>', methods=['GET'])
def get_result(filename: str):
    """Serve a previously saved result image."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
