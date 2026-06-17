"""Parallel DCP dehazing for YOLO image directories.

This script preserves relative filenames under ``--src`` and writes dehazed
images under ``--dst``. Labels are intentionally not touched; use
``prepare_dcp_yolo_dataset.py`` to build YOLO dataset roots/yamls around the
generated image cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gate.analysis.physics_priors import (  # noqa: E402
    compute_dark_channel,
    estimate_atmospheric_light,
    estimate_transmission_dcp,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], src: Path) -> None:
        self.paths = paths
        self.src = src

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return str(path), str(path.relative_to(self.src)), tensor


def collate_image_list(batch):
    paths, rels, tensors = zip(*batch)
    return list(paths), list(rels), list(tensors)


def collect_images(src: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if limit is not None:
        paths = paths[:limit]
    return paths


@torch.no_grad()
def dehaze_batch(x: torch.Tensor, patch_size: int = 15, t0: float = 0.1) -> torch.Tensor:
    """DCP dehazing on ``(B, 3, H, W)`` tensors in [0, 1]."""
    dark = compute_dark_channel(x, patch_size=patch_size)
    atmospheric = estimate_atmospheric_light(x, dark).view(-1, 3, 1, 1)
    transmission = estimate_transmission_dcp(x, patch_size=patch_size).clamp(min=t0)
    restored = (x - atmospheric) / transmission + atmospheric
    return restored.clamp(0.0, 1.0)


def save_image(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        im.save(out_path, quality=95)
    else:
        im.save(out_path)


def run(args: argparse.Namespace) -> dict:
    src = args.src.resolve()
    dst = args.dst.resolve()
    paths = collect_images(src, limit=args.limit)
    dst.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        ImagePathDataset(paths, src),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_image_list,
    )

    start = time.time()
    written = 0
    skipped = 0
    seen = 0
    print(f"[dcp] src={src}")
    print(f"[dcp] dst={dst}")
    print(f"[dcp] images={len(paths)} batch={args.batch} workers={args.workers} device={device}")

    for paths_str, rels, tensors in loader:
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, tensor in enumerate(tensors):
            groups[tuple(tensor.shape)].append(i)

        for _, idxs in groups.items():
            rel_group = [rels[i] for i in idxs]
            out_paths = [dst / rel for rel in rel_group]
            keep = [i for i, out in zip(idxs, out_paths) if args.overwrite or not out.exists()]
            skipped += len(idxs) - len(keep)
            if not keep:
                continue

            x = torch.stack([tensors[i] for i in keep], dim=0).to(device, non_blocking=True)
            y = dehaze_batch(x, patch_size=args.patch_size, t0=args.t0)
            y_np = (y.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)

            for i, arr in zip(keep, y_np):
                save_image(arr, dst / rels[i])
                written += 1

        seen += len(tensors)
        if seen % max(args.batch * 20, 1) == 0 or seen == len(paths):
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"[dcp] {seen}/{len(paths)} seen, written={written}, "
                f"skipped={skipped}, rate={seen / elapsed:.2f} img/s",
                flush=True,
            )

    elapsed = time.time() - start
    summary = {
        "src": str(src),
        "dst": str(dst),
        "images": len(paths),
        "written": written,
        "skipped": skipped,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(device),
        "patch_size": args.patch_size,
        "t0": args.t0,
        "elapsed_sec": elapsed,
        "rate_img_per_sec": len(paths) / elapsed if elapsed > 0 else None,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--t0", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
