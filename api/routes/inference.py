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
from api.services.classical_colorizer import ClassicalColorizer

bp = Blueprint('inference', __name__)
_colorizer = Colorizer()
_classical_colorizer = ClassicalColorizer()

# Classical method IDs — these require a reference image instead of a checkpoint.
CLASSICAL_IDS = {'classical_welsh', 'classical_levin'}
CLASSICAL_METHOD_MAP = {
    'classical_welsh': 'welsh',
    'classical_levin': 'levin',
}


@bp.route('', methods=['POST'])
def colorize_single():
    """Colorize one uploaded image.

    Form fields:
        file           — image file (greyscale or colour)
        model          — "baseline" | "unet" | "gan" | "fusion"
                         | "classical_welsh" | "classical_levin"
        checkpoint     — path to checkpoint .pth (ignored for classical models)
        mode           — "grayscale" | "color_photo"
        reference_file — colour reference image (required for classical models)

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
        if model in CLASSICAL_IDS:
            # Classical methods need a reference colour image
            ref_file = request.files.get('reference_file')
            if ref_file is None:
                return jsonify({'error': 'reference_file is required for classical models'}), 400

            ref_filename = f'{uuid.uuid4().hex}_ref_{ref_file.filename}'
            ref_path = os.path.join(upload_dir, ref_filename)
            ref_file.save(ref_path)

            try:
                result = _classical_colorizer.colorize(
                    save_path,
                    ref_path,
                    method=CLASSICAL_METHOD_MAP[model],
                    mode=mode,
                )
            finally:
                # Clean up reference file (best-effort)
                try:
                    os.remove(ref_path)
                except OSError:
                    pass
        else:
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
    ref_path: str | None = None

    if model in CLASSICAL_IDS:
        ref_file = request.files.get('reference_file')
        if ref_file is None:
            return jsonify({'error': 'reference_file is required for classical models'}), 400
        ref_filename = f'{uuid.uuid4().hex}_ref_{ref_file.filename}'
        ref_path = os.path.join(upload_dir, ref_filename)
        ref_file.save(ref_path)

    results = []
    try:
        for file in files:
            filename = f'{uuid.uuid4().hex}_{file.filename}'
            save_path = os.path.join(upload_dir, filename)
            file.save(save_path)
            try:
                if model in CLASSICAL_IDS:
                    result = _classical_colorizer.colorize(
                        save_path,
                        ref_path,  # type: ignore[arg-type]
                        method=CLASSICAL_METHOD_MAP[model],
                        mode=mode,
                    )
                else:
                    result = _colorizer.colorize(save_path, model, checkpoint, mode)
                results.append({'filename': file.filename, **result})
            except Exception as exc:
                results.append({'filename': file.filename, 'error': str(exc)})
    finally:
        if ref_path:
            try:
                os.remove(ref_path)
            except OSError:
                pass

    return jsonify(results)


@bp.route('/result/<filename>', methods=['GET'])
def get_result(filename: str):
    """Serve a previously saved result image."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
