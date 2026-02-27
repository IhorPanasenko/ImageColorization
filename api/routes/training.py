"""Training management endpoints.

Routes:
    POST   /api/training/start              — Launch a training subprocess
    GET    /api/training/status/<run_id>    — Poll run status
    GET    /api/training/stream/<run_id>    — SSE live log stream
    POST   /api/training/stop/<run_id>      — Terminate a run
    GET    /api/training/runs               — List all runs
"""

from flask import Blueprint, request, jsonify, Response, current_app
from api.services.train_runner import TrainRunner

bp = Blueprint('training', __name__)
_runner = TrainRunner()


@bp.route('/start', methods=['POST'])
def start_training():
    """Start a new training run.

    Expected JSON body:
        {
            "model": "baseline" | "unet" | "gan" | "fusion",
            "epochs": int,
            "batch_size": int,
            "lr": float,
            "lambda_l1": float,       # GAN / fusion only
            "data_path": str,
            "resume_g": str | null,   # GAN / fusion only
            "resume_d": str | null    # GAN / fusion only
        }

    Returns:
        {"run_id": str}
    """
    params = request.get_json(force=True) or {}
    run_id = _runner.start(params, outputs_dir=current_app.config['OUTPUTS_DIR'])
    return jsonify({'run_id': run_id}), 202


@bp.route('/status/<run_id>', methods=['GET'])
def get_status(run_id: str):
    """Return current status of a training run."""
    status = _runner.status(run_id)
    if status is None:
        return jsonify({'error': 'run not found'}), 404
    return jsonify(status)


@bp.route('/stream/<run_id>', methods=['GET'])
def stream_log(run_id: str):
    """Server-Sent Events endpoint — streams parsed training progress."""

    def generate():
        for event in _runner.stream(run_id):
            yield f'data: {event}\n\n'

    return Response(generate(), mimetype='text/event-stream')


@bp.route('/stop/<run_id>', methods=['POST'])
def stop_training(run_id: str):
    """Terminate a running training job."""
    ok = _runner.stop(run_id)
    if not ok:
        return jsonify({'error': 'run not found or already finished'}), 404
    return jsonify({'status': 'stopped'})


@bp.route('/runs', methods=['GET'])
def list_runs():
    """List all training runs (active and historical)."""
    return jsonify(_runner.list_runs())
