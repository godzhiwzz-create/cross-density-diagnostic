#!/usr/bin/env python3
"""Evaluation and seeding utilities shared across mechanism and gate research."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_map50(results_json_path: Path | str) -> float | None:
    """Read mAP50 from a results.json written by train_m13_dense or train_faster_rcnn."""
    path = Path(results_json_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # m13_dense format: {'metrics': {'map50': ...}}
    if 'metrics' in data:
        return float(data['metrics'].get('map50', 0.0))
    # faster_rcnn format: {'mAP50': ...}
    if 'mAP50' in data:
        return float(data['mAP50'])
    return None
