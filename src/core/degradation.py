#!/usr/bin/env python3
"""Image degradation utilities shared across mechanism and gate research."""
from __future__ import annotations

import random
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageFile


def apply_global_degradation(image: np.ndarray, contrast: float, haze_bias: float) -> np.ndarray:
    out = image.astype(np.float32) * contrast + haze_bias
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_bbox_occlusion(
    image: np.ndarray,
    boxes: Iterable[tuple[int, int, int, int, int]],
    occlusion_ratio: float,
    rng: random.Random,
) -> np.ndarray:
    if occlusion_ratio <= 0:
        return image
    out = image.copy()
    for _, x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < 100:
            continue
        occ_w = max(1, int(bw * np.sqrt(occlusion_ratio)))
        occ_h = max(1, int(bh * np.sqrt(occlusion_ratio)))
        max_x = max(0, bw - occ_w)
        max_y = max(0, bh - occ_h)
        ox = x1 + (rng.randint(0, max_x) if max_x > 0 else 0)
        oy = y1 + (rng.randint(0, max_y) if max_y > 0 else 0)
        fill = int(rng.uniform(96, 160))
        out[oy:min(y2, oy + occ_h), ox:min(x2, ox + occ_w)] = fill
    return out


def read_image_resilient(path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as pil_img:
        pil_img.load()
        rgb = pil_img.convert('RGB')
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
