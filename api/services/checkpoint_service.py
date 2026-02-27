"""Checkpoint discovery and metadata helpers."""

from __future__ import annotations

import glob
import os
from typing import Any


def discover_checkpoints(outputs_dir: str) -> list[dict[str, Any]]:
    """Scan *outputs_dir*/checkpoints/ for .pth files.

    Returns:
        Sorted list of dicts: {"path", "filename", "size_mb", "model_hint"}.
    """
    pattern = os.path.join(outputs_dir, 'checkpoints', '**', '*.pth')
    paths = sorted(glob.glob(pattern, recursive=True))
    results = []
    for p in paths:
        name = os.path.basename(p)
        results.append({
            'path': p,
            'filename': name,
            'size_mb': round(os.path.getsize(p) / 1e6, 1),
            'model_hint': _guess_model(name),
        })
    return results


def _guess_model(filename: str) -> str:
    """Heuristically identify model type from filename."""
    fn = filename.lower()
    if 'fusion' in fn:
        return 'fusion'
    if 'gan' in fn or 'discriminator' in fn or 'generator' in fn:
        return 'gan'
    if 'unet' in fn or 'u_net' in fn:
        return 'unet'
    if 'baseline' in fn:
        return 'baseline'
    return 'unknown'
