#!/usr/bin/env python3
"""Dataset and data-path utilities for Cityscapes-based KD experiments."""
from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path('/root/autodl-tmp/shared_datasets/low_visibility_kd/cityscapes_yolo')
CLEAR_ROOT = DATA_ROOT / 'clear'
FOGGY_ROOT = DATA_ROOT / 'foggy_all'

CLASS_NAMES = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'
}


def level_to_source(degradation_level: float) -> tuple[str, float | None, float, float]:
    """Map a scalar degradation level to (source_type, beta, contrast, haze_bias)."""
    level = float(degradation_level)
    if level <= 0.0:
        return 'clear', None, 1.0, 0.0
    if level <= 0.3:
        return 'foggy', 0.005, 1.0, 0.0
    if level <= 0.5:
        return 'foggy', 0.01, 1.0, 0.0
    if level <= 0.7:
        return 'foggy', 0.02, 0.95, 8.0
    return 'foggy', 0.02, 0.88, 18.0


def find_image_and_label(split: str, stem: str, degradation_level: float) -> tuple[Path, Path]:
    """Locate the source image and label file for a given split / stem / degradation level."""
    source_type, beta, _, _ = level_to_source(degradation_level)
    if source_type == 'clear':
        img = CLEAR_ROOT / 'images' / split / f'{stem}_leftImg8bit.png'
        lbl = CLEAR_ROOT / 'labels' / split / f'{stem}_leftImg8bit.txt'
    else:
        beta_str = f'{beta:.3f}'.rstrip('0').rstrip('.')
        img = FOGGY_ROOT / 'images' / split / f'{stem}_leftImg8bit_foggy_beta_{beta_str}.png'
        lbl = FOGGY_ROOT / 'labels' / split / f'{stem}_leftImg8bit_foggy_beta_{beta_str}.txt'
        if not lbl.exists():
            lbl = CLEAR_ROOT / 'labels' / split / f'{stem}_leftImg8bit.txt'
    return img, lbl
