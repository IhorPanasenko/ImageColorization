"""Training history endpoints.

Routes:
    GET    /api/history/runs                         — List all historical runs
    GET    /api/history/logs/<run_id>                — Full parsed log for one run
    GET    /api/history/tensorboard-data/<model_type> — Scalar data from TensorBoard events
    DELETE /api/history/<run_id>                     — Remove a run record
"""

import os
import json
import re
from flask import Blueprint, jsonify, current_app

bp = Blueprint('history', __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runs_file(outputs_dir: str) -> str:
    return os.path.join(outputs_dir, 'runs', 'runs.json')


def _load_runs(outputs_dir: str) -> dict:
    path = _runs_file(outputs_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


_EPOCH_RE  = re.compile(r'Epoch\s+(\d+)/(\d+)', re.IGNORECASE)
_LOSS_RE   = re.compile(r'loss(?:_[GgDd])?[:\s=]+([0-9.eE+\-]+)', re.IGNORECASE)
_LR_RE     = re.compile(r'lr[:\s=]+([0-9.eE+\-]+)', re.IGNORECASE)


def _parse_log(log_path: str) -> dict:
    """Parse a train.log file into epoch-indexed arrays for epochs, loss, lr."""
    epochs: list[int]   = []
    losses: list[float] = []
    lrs:    list[float] = []

    if not os.path.exists(log_path):
        return {'epochs': epochs, 'losses': losses, 'lrs': lrs, 'lines': []}

    lines: list[str] = []
    with open(log_path) as f:
        for line in f:
            lines.append(line.rstrip())
            m_epoch = _EPOCH_RE.search(line)
            m_loss  = _LOSS_RE.search(line)
            m_lr    = _LR_RE.search(line)
            if m_epoch and m_loss:
                epochs.append(int(m_epoch.group(1)))
                try:
                    losses.append(float(m_loss.group(1)))
                except ValueError:
                    pass
            if m_lr:
                try:
                    lrs.append(float(m_lr.group(1)))
                except ValueError:
                    pass

    return {'epochs': epochs, 'losses': losses, 'lrs': lrs, 'lines': lines}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route('/runs', methods=['GET'])
def list_history():
    """Return all past training run records."""
    runs = _load_runs(current_app.config['OUTPUTS_DIR'])
    return jsonify(list(runs.values()))


@bp.route('/logs/<run_id>', methods=['GET'])
def get_logs(run_id: str):
    """Return parsed training log for a single run.

    Response:
        {
            "run_id": str,
            "epochs": [int, ...],
            "losses": [float, ...],
            "lrs":    [float, ...],
            "lines":  [str, ...]   # last 200 raw log lines
        }
    """
    outputs_dir = current_app.config['OUTPUTS_DIR']
    log_path = os.path.join(outputs_dir, 'runs', run_id, 'train.log')
    parsed = _parse_log(log_path)
    # Trim raw lines to avoid huge payloads
    parsed['lines'] = parsed['lines'][-200:]
    return jsonify({'run_id': run_id, **parsed})


@bp.route('/tensorboard-data/<model_type>', methods=['GET'])
def tensorboard_data(model_type: str):
    """Parse TensorBoard event files and return scalar arrays.

    Scans outputs/runs/{model_type}/ for event files produced by
    SummaryWriter.  Falls back gracefully if tensorboard is not installed
    or no event files are found.

    Response:
        {
            "model_type": str,
            "tags": {
                "Loss/train": [{"step": int, "value": float}, ...],
                ...
            }
        }
    """
    outputs_dir = current_app.config['OUTPUTS_DIR']
    run_dir = os.path.join(outputs_dir, 'runs', model_type)

    if not os.path.isdir(run_dir):
        return jsonify({'model_type': model_type, 'tags': {}})

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        ea = EventAccumulator(run_dir)
        ea.Reload()
        tags: dict[str, list] = {}
        for tag in ea.Tags().get('scalars', []):
            tags[tag] = [
                {'step': e.step, 'value': e.value}
                for e in ea.Scalars(tag)
            ]
        return jsonify({'model_type': model_type, 'tags': tags})
    except ImportError:
        return jsonify({
            'model_type': model_type,
            'tags': {},
            'warning': 'tensorboard package not installed',
        })
    except Exception as exc:
        return jsonify({'model_type': model_type, 'tags': {}, 'error': str(exc)})


@bp.route('/<run_id>', methods=['DELETE'])
def delete_run(run_id: str):
    """Remove a run record from runs.json (does not delete log or checkpoint files)."""
    outputs_dir = current_app.config['OUTPUTS_DIR']
    runs = _load_runs(outputs_dir)
    if run_id not in runs:
        return jsonify({'error': 'not found'}), 404
    del runs[run_id]
    path = _runs_file(outputs_dir)
    with open(path, 'w') as f:
        json.dump(runs, f, indent=2)
    return jsonify({'status': 'deleted'})

