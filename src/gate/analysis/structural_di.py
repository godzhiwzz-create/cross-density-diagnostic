"""Structural domain-invariance probes for TSP.

This script tests whether TSP remains competitive when DI uses spatial
structure descriptors rather than only marginal map statistics.

Outputs:
  - paired_cross_beta_consistency.csv/json
  - spatial_descriptor_di.csv/json

The denominator of DI is synthetic cross-beta variation, not a random
synthetic half split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.stats import ks_2samp, rankdata, spearmanr


DATA_ROOT = Path("./数据集_prepared/low_visibility_kd")
DEFAULT_YAMLS = {
    "beta_0.005": DATA_ROOT / "cityscapes_yolo/dataset_foggy_beta_0_005_dlhost.yaml",
    "beta_0.01": DATA_ROOT / "cityscapes_yolo/dataset_foggy_beta_0_01_dlhost.yaml",
    "beta_0.02": DATA_ROOT / "cityscapes_yolo/dataset_foggy_beta_0_02_dlhost.yaml",
    "rtts": DATA_ROOT / "external_eval/rtts_yolo/dataset_dlhost.yaml",
    "foggy_driving": DATA_ROOT / "external_eval/foggy_driving_yolo/dataset_dlhost.yaml",
}
SYNTH_TAGS = ("beta_0.005", "beta_0.01", "beta_0.02")
REAL_TAGS = ("rtts", "foggy_driving")
SIGNALS = ("rgb_grad", "raw_transmission", "dark_channel", "tsp_grad")


def resolve_images(yaml_path: Path, split: str = "val") -> list[Path]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "")).expanduser()
    item = cfg.get(split)
    if item is None:
        raise KeyError(f"{split} not found in {yaml_path}")
    p = Path(item)
    if not p.is_absolute():
        p = root / p
    if p.is_file() and p.suffix == ".txt":
        return [Path(line.strip()) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in exts)
    raise FileNotFoundError(p)


def sample_paths(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    if limit is None or len(paths) <= limit:
        return paths
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(paths), size=limit, replace=False))
    return [paths[int(i)] for i in idx]


def read_rgb(path: Path, imgsz: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def min_filter(x: np.ndarray, k: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(x.astype(np.float32), kernel)


def dark_channel(rgb: np.ndarray, patch: int = 15) -> np.ndarray:
    return min_filter(rgb.min(axis=2), patch)


def estimate_atmospheric_light(rgb: np.ndarray, dark: np.ndarray) -> np.ndarray:
    flat_dark = dark.reshape(-1)
    flat_rgb = rgb.reshape(-1, 3)
    k = max(1, int(0.001 * flat_dark.size))
    idx = np.argpartition(flat_dark, -k)[-k:]
    return flat_rgb[idx].mean(axis=0).clip(1e-3, 1.0)


def raw_transmission(rgb: np.ndarray, patch: int = 15, omega: float = 0.95) -> np.ndarray:
    dark = dark_channel(rgb, patch)
    a = estimate_atmospheric_light(rgb, dark)
    norm = rgb / a.reshape(1, 1, 3)
    dark_norm = dark_channel(norm.clip(0.0, 5.0), patch)
    return (1.0 - omega * dark_norm).clip(0.05, 1.0)


def sobel_xy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def signal_maps(rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ggx, ggy = sobel_xy(gray)
    rgb_grad = np.sqrt(ggx * ggx + ggy * ggy)
    t = raw_transmission(rgb)
    dc = dark_channel(rgb)
    tgx, tgy = sobel_xy(t)
    tsp = np.sqrt(tgx * tgx + tgy * tgy)
    return {
        "rgb_grad": norm01(rgb_grad),
        "raw_transmission": norm01(t),
        "dark_channel": norm01(dc),
        "tsp_grad": norm01(tsp),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_spatial(a: np.ndarray, b: np.ndarray, size: int = 96) -> float:
    # Full-resolution rank correlation is unnecessarily slow for this probe.
    aa = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).reshape(-1)
    bb = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA).reshape(-1)
    corr = spearmanr(aa, bb).correlation
    return float(corr) if corr is not None and np.isfinite(corr) else 0.0


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    vxy = ((x - mux) * (y - muy)).mean()
    den = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    if abs(den) < 1e-12:
        return 0.0
    return float(((2 * mux * muy + c1) * (2 * vxy + c2)) / den)


def edge_iou_topk(a: np.ndarray, b: np.ndarray, q: float = 0.9) -> float:
    ta, tb = np.quantile(a, q), np.quantile(b, q)
    ma, mb = a >= ta, b >= tb
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(ma, mb).sum() / union)


def orientation_histogram(x: np.ndarray, bins: int = 12) -> np.ndarray:
    gx, gy = sobel_xy(x)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + math.pi) % math.pi
    hist, _ = np.histogram(ang.reshape(-1), bins=bins, range=(0, math.pi), weights=mag.reshape(-1))
    hist = hist.astype(np.float64)
    return hist / max(hist.sum(), 1e-12)


def hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def paired_scene_key(path: Path) -> str:
    name = path.name
    name = re.sub(r"_foggy_beta_0(?:\.005|\.01|\.02)", "", name)
    name = re.sub(r"_foggy_beta_0_(?:005|01|02)", "", name)
    return name


def paired_cross_beta(paths_by_tag: dict[str, list[Path]], imgsz: int, limit: int | None, seed: int) -> list[dict]:
    maps_by_tag: dict[str, dict[str, Path]] = {
        tag: {paired_scene_key(p): p for p in paths_by_tag[tag]} for tag in SYNTH_TAGS
    }
    common = sorted(set.intersection(*(set(m.keys()) for m in maps_by_tag.values())))
    common = sample_paths([Path(k) for k in common], limit, seed)
    keys = [p.name for p in common]

    rows = []
    pairs = [("beta_0.005", "beta_0.01"), ("beta_0.005", "beta_0.02"), ("beta_0.01", "beta_0.02")]
    for key in keys:
        sig_cache = {}
        for tag in SYNTH_TAGS:
            sig_cache[tag] = signal_maps(read_rgb(maps_by_tag[tag][key], imgsz))
        for a_tag, b_tag in pairs:
            for sig in SIGNALS:
                a, b = sig_cache[a_tag][sig], sig_cache[b_tag][sig]
                rows.append({
                    "scene_key": key,
                    "pair": f"{a_tag}_vs_{b_tag}",
                    "signal": sig,
                    "ssim": ssim_global(a, b),
                    "pearson": pearson(a, b),
                    "spearman": spearman_spatial(a, b),
                    "edge_iou_top10": edge_iou_topk(a, b, 0.9),
                    "orient_intersection": hist_intersection(orientation_histogram(a), orientation_histogram(b)),
                })
    return rows


def map_descriptor(x: np.ndarray) -> np.ndarray:
    x = norm01(x)
    gx, gy = sobel_xy(x)
    mag = np.sqrt(gx * gx + gy * gy)
    desc: list[float] = []
    desc.extend([
        float(x.mean()),
        float(x.std()),
        float(np.quantile(x, 0.9)),
        float((x >= np.quantile(x, 0.9)).mean()),
        float((x >= np.quantile(x, 0.95)).mean()),
        float(mag.mean()),
        float(mag.std()),
    ])

    mask = (x >= np.quantile(x, 0.9)).astype(np.uint8)
    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if ncomp > 1 else np.array([], dtype=np.float32)
    desc.extend([
        float(max(ncomp - 1, 0)),
        float(areas.mean() if areas.size else 0.0),
        float(areas.max() if areas.size else 0.0),
    ])

    desc.extend(orientation_histogram(x, bins=12).tolist())

    small = cv2.resize(x, (64, 64), interpolation=cv2.INTER_AREA)
    fft = np.fft.fftshift(np.fft.fft2(small))
    power = np.log1p(np.abs(fft) ** 2)
    yy, xx = np.indices(power.shape)
    rr = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
    for lo, hi in [(0, 4), (4, 8), (8, 16), (16, 32), (32, 46)]:
        m = (rr >= lo) & (rr < hi)
        desc.append(float(power[m].mean()))

    z = small - small.mean()
    denom = float((z * z).sum()) + 1e-12
    for dy, dx in [(1, 0), (0, 1), (2, 0), (0, 2), (4, 0), (0, 4)]:
        desc.append(float((z[:-dy or None, :-dx or None] * z[dy:, dx:]).sum() / denom))

    patch = cv2.resize(mag, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    patch = patch / max(float(patch.sum()), 1e-12)
    desc.extend(patch.tolist())
    return np.asarray(desc, dtype=np.float32)


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    z = np.concatenate([x, y], axis=0)
    # Median heuristic on a bounded subset for speed.
    rng = np.random.default_rng(0)
    if len(z) > 300:
        z_med = z[rng.choice(len(z), size=300, replace=False)]
    else:
        z_med = z
    d = ((z_med[:, None, :] - z_med[None, :, :]) ** 2).sum(axis=2)
    sigma2 = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(2.0 * sigma2, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * dist)

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def ks_mean(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([ks_2samp(x[:, i], y[:, i]).statistic for i in range(x.shape[1])]))


def extract_descriptors(paths_by_tag: dict[str, list[Path]], imgsz: int, limit: int | None, seed: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, list[np.ndarray]]] = {sig: {} for sig in SIGNALS}
    for tag, paths in paths_by_tag.items():
        chosen = sample_paths(paths, limit, seed)
        for sig in SIGNALS:
            out[sig][tag] = []
        for i, path in enumerate(chosen, 1):
            maps = signal_maps(read_rgb(path, imgsz))
            for sig, m in maps.items():
                out[sig][tag].append(map_descriptor(m))
            if i % 100 == 0 or i == len(chosen):
                print(f"[desc] {tag}: {i}/{len(chosen)}", flush=True)
    return {sig: {tag: np.stack(v, axis=0) for tag, v in tags.items()} for sig, tags in out.items()}


def structural_di(desc: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    synth_pairs = [("beta_0.005", "beta_0.01"), ("beta_0.005", "beta_0.02"), ("beta_0.01", "beta_0.02")]
    for sig in SIGNALS:
        inter_mmd, inter_ks, intra_mmd, intra_ks = [], [], [], []
        for rt in REAL_TAGS:
            for st in SYNTH_TAGS:
                inter_mmd.append(mmd_rbf(desc[sig][rt], desc[sig][st]))
                inter_ks.append(ks_mean(desc[sig][rt], desc[sig][st]))
        for a, b in synth_pairs:
            intra_mmd.append(mmd_rbf(desc[sig][a], desc[sig][b]))
            intra_ks.append(ks_mean(desc[sig][a], desc[sig][b]))
        rows.append({
            "signal": sig,
            "di_mmd": float(np.mean(inter_mmd) / max(np.mean(intra_mmd), 1e-10)),
            "di_ks": float(np.mean(inter_ks) / max(np.mean(intra_ks), 1e-10)),
            "inter_mmd": float(np.mean(inter_mmd)),
            "intra_mmd": float(np.mean(intra_mmd)),
            "inter_ks": float(np.mean(inter_ks)),
            "intra_ks": float(np.mean(intra_ks)),
        })
    return rows


def summarize_paired(rows: list[dict]) -> list[dict]:
    metrics = ["ssim", "pearson", "spearman", "edge_iou_top10", "orient_intersection"]
    out = []
    for sig in SIGNALS:
        rs = [r for r in rows if r["signal"] == sig]
        item = {"signal": sig, "n_pairs": len(rs)}
        for metric in metrics:
            vals = np.asarray([r[metric] for r in rs], dtype=np.float64)
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("gate/experiments_dlhost/summaries/structural_di"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--paired-max-scenes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths_by_tag = {}
    for tag, yaml_path in DEFAULT_YAMLS.items():
        paths = resolve_images(yaml_path, split=args.split)
        paths_by_tag[tag] = paths
        print(f"[data] {tag}: {len(paths)} images from {yaml_path}")

    paired_rows = paired_cross_beta(paths_by_tag, args.imgsz, args.paired_max_scenes, args.seed)
    paired_summary = summarize_paired(paired_rows)
    desc = extract_descriptors(paths_by_tag, args.imgsz, args.max_images, args.seed)
    di_rows = structural_di(desc)

    write_csv(args.out / "paired_cross_beta_consistency_raw.csv", paired_rows)
    write_csv(args.out / "paired_cross_beta_consistency.csv", paired_summary)
    write_csv(args.out / "spatial_descriptor_di.csv", di_rows)
    (args.out / "paired_cross_beta_consistency.json").write_text(json.dumps(paired_summary, indent=2), encoding="utf-8")
    (args.out / "spatial_descriptor_di.json").write_text(json.dumps(di_rows, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("\n=== Paired cross-beta structural consistency ===")
    for row in paired_summary:
        print(row)
    print("\n=== Spatial-descriptor DI ===")
    for row in di_rows:
        print(row)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
