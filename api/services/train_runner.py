"""Background training subprocess manager."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from typing import Any, Generator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_PATH = os.path.join(ROOT, 'ml')

# Regex patterns for parsing tqdm / print output
_EPOCH_RE = re.compile(r'Epoch\s+(\d+)/(\d+)', re.IGNORECASE)
_LOSS_RE = re.compile(r'(?:loss[_\s]?(?:G|D)?)[:\s=]+([0-9.eE+\-]+)', re.IGNORECASE)


class TrainRunner:
    """Manages training subprocesses and persists run state."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, params: dict[str, Any], outputs_dir: str) -> str:
        """Launch a training subprocess and return its run_id."""
        run_id = uuid.uuid4().hex
        model = params.get('model', 'unet')
        script = self._resolve_script(model)

        log_dir = os.path.join(outputs_dir, 'runs', run_id)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'train.log')

        cmd = self._build_cmd(script, params)

        env = os.environ.copy()
        env['PYTHONPATH'] = ML_PATH + os.pathsep + env.get('PYTHONPATH', '')

        with open(log_path, 'w') as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=ROOT,
            )

        record: dict[str, Any] = {
            'run_id': run_id,
            'model': model,
            'params': params,
            'status': 'running',
            'epoch': 0,
            'total_epochs': params.get('epochs', 20),
            'loss': None,
            'log_path': log_path,
            'started_at': time.time(),
            'finished_at': None,
        }

        with self._lock:
            self._runs[run_id] = record
            self._procs[run_id] = proc

        self._save_runs(outputs_dir)
        threading.Thread(target=self._monitor, args=(run_id, proc, outputs_dir), daemon=True).start()
        return run_id

    def status(self, run_id: str) -> dict[str, Any] | None:
        """Return a snapshot of a run's current state."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return None
            # Read last few log lines
            log_tail: list[str] = []
            if os.path.exists(record['log_path']):
                with open(record['log_path']) as f:
                    log_tail = f.readlines()[-20:]
            return {**record, 'log_tail': log_tail}

    def stream(self, run_id: str) -> Generator[str, None, None]:
        """Yield JSON-encoded progress events by tailing the log file."""
        import json as _json
        with self._lock:
            record = self._runs.get(run_id)
        if record is None:
            yield _json.dumps({'error': 'not found'})
            return

        log_path = record['log_path']
        # Wait for log to appear
        for _ in range(30):
            if os.path.exists(log_path):
                break
            time.sleep(0.5)

        with open(log_path) as f:
            f.seek(0, 2)  # seek to end
            while True:
                with self._lock:
                    current = self._runs.get(run_id, {})
                line = f.readline()
                if line:
                    epoch, total, loss = self._parse_line(line)
                    event = {
                        'epoch': epoch or current.get('epoch'),
                        'total_epochs': total or current.get('total_epochs'),
                        'loss': loss or current.get('loss'),
                        'status': current.get('status'),
                        'line': line.rstrip(),
                    }
                    yield _json.dumps(event)
                else:
                    if current.get('status') in ('finished', 'failed', 'stopped'):
                        break
                    time.sleep(2)

    def stop(self, run_id: str) -> bool:
        """Terminate a running subprocess."""
        with self._lock:
            proc = self._procs.get(run_id)
            if proc is None:
                return False
            proc.terminate()
            self._runs[run_id]['status'] = 'stopped'
        return True

    def list_runs(self) -> list[dict[str, Any]]:
        """Return all known runs."""
        with self._lock:
            return list(self._runs.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_script(self, model: str) -> str:
        scripts = {
            'baseline': os.path.join(ML_PATH, 'scripts', 'trains', 'train_baseline.py'),
            'unet':     os.path.join(ML_PATH, 'scripts', 'trains', 'train_unet.py'),
            'gan':      os.path.join(ML_PATH, 'scripts', 'trains', 'train_gan.py'),
            'fusion':   os.path.join(ML_PATH, 'scripts', 'trains', 'train_fusion.py'),
        }
        script = scripts.get(model)
        if script is None:
            raise ValueError(f'Unknown model type: {model!r}')
        return script

    @staticmethod
    def _build_cmd(script: str, params: dict[str, Any]) -> list[str]:
        cmd = ['python', script]
        if 'epochs' in params:
            cmd += ['--epochs', str(params['epochs'])]
        if 'batch_size' in params:
            cmd += ['--batch_size', str(params['batch_size'])]
        if 'lr' in params:
            cmd += ['--lr', str(params['lr'])]
        if 'lambda_l1' in params:
            cmd += ['--lambda_l1', str(params['lambda_l1'])]
        if 'data_path' in params:
            cmd += ['--data_path', str(params['data_path'])]
        if params.get('resume_g'):
            cmd += ['--resume_g', str(params['resume_g'])]
        if params.get('resume_d'):
            cmd += ['--resume_d', str(params['resume_d'])]
        return cmd

    def _monitor(self, run_id: str, proc: subprocess.Popen, outputs_dir: str) -> None:
        """Watch the process and update status when it exits."""
        proc.wait()
        with self._lock:
            record = self._runs.get(run_id)
            if record and record['status'] == 'running':
                record['status'] = 'finished' if proc.returncode == 0 else 'failed'
                record['finished_at'] = time.time()
        self._save_runs(outputs_dir)

    def _save_runs(self, outputs_dir: str) -> None:
        path = os.path.join(outputs_dir, 'runs', 'runs.json')
        with self._lock:
            serializable = {
                k: {kk: vv for kk, vv in v.items() if kk != 'log_path'}
                for k, v in self._runs.items()
            }
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def _parse_line(line: str) -> tuple[int | None, int | None, float | None]:
        epoch = total = loss = None
        m = _EPOCH_RE.search(line)
        if m:
            epoch, total = int(m.group(1)), int(m.group(2))
        m = _LOSS_RE.search(line)
        if m:
            try:
                loss = float(m.group(1))
            except ValueError:
                pass
        return epoch, total, loss
