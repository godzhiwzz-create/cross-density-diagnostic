"""E032 DI signal ladder gate-shortcut probe.

This script evaluates whether trained DADG gates conditioned on intermediate
DI candidates still separate real fog from synthetic fog. The primary probe is
distributional and mirrors DI: real-vs-synthetic gate-output MMD/KS over the
three synthetic fog densities, normalized by synthetic cross-density MMD/KS.
The legacy beta-matched L2 ratio is retained as a compatibility diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

REPO = Path(".")
PROJECT_ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from gate.analysis.physics_priors import build_gate_input  # noqa: E402
from gate.models.dadg import DADGGate, build_gate  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "数据集_prepared/low_visibility_kd"
EXP_ROOT = REPO / "gate/experiments_dlhost"

DATASETS = {
    "beta_0.005": DATA_ROOT / "cityscapes_yolo/dataset_foggy_beta_0_005_dlhost.yaml",
    "beta_0.01": DATA_ROOT / "cityscapes_yolo/dataset_foggy_beta_0_01_dlhost.yaml",
    "beta_0.02": DATA_ROOT / "cityscapes_yolo/dataset_foggy_beta_0_02_dlhost.yaml",
    "rtts": DATA_ROOT / "external_eval/rtts_yolo/dataset_dlhost.yaml",
}
SYNTH_TAGS = ["beta_0.005", "beta_0.01", "beta_0.02"]


class ImageOnlyDataset(Dataset):
    def __init__(self, paths: list[Path], imgsz: int = 640):
        self.paths = paths
        self.tf = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        return self.tf(Image.open(path).convert("RGB")), str(path)


def resolve_images(yaml_path: Path, split: str = "val") -> list[Path]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "")).expanduser()
    item = Path(cfg[split])
    if not item.is_absolute():
        item = root / item
    if item.is_file() and item.suffix == ".txt":
        return [Path(line.strip()) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]
    if item.is_dir():
        return sorted(p for p in item.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    raise FileNotFoundError(f"Cannot resolve {split} images from {yaml_path}: {item}")


def make_loader(paths: list[Path], batch: int, workers: int, imgsz: int) -> DataLoader:
    return DataLoader(
        ImageOnlyDataset(paths, imgsz=imgsz),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


@torch.no_grad()
def estimate_beta_dcp(img: torch.Tensor, omega: float = 0.95, patch: int = 15) -> torch.Tensor:
    b = img.size(0)
    flat = img.view(b, 3, -1)
    k = max(1, int(0.001 * flat.size(-1)))
    airlight = flat.topk(k, dim=-1).values.mean(dim=-1)
    min_c = img.min(dim=1, keepdim=True).values
    dark = -F.max_pool2d(-min_c, kernel_size=patch, stride=1, padding=patch // 2)
    a_scalar = airlight.mean(dim=1).view(b, 1, 1, 1).clamp_min(1e-3)
    transmission = (1.0 - omega * (dark / a_scalar)).clamp(0.05, 1.0)
    return -transmission.log().mean(dim=(1, 2, 3)).cpu()


@torch.no_grad()
def collect_beta(
    dataset: str,
    *,
    max_images: int,
    batch: int,
    workers: int,
    imgsz: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    paths = resolve_images(DATASETS[dataset])
    if max_images > 0 and len(paths) > max_images:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(paths), size=max_images, replace=False).tolist())
        paths = [paths[i] for i in idx]
    betas: list[np.ndarray] = []
    out_paths: list[str] = []
    for imgs, batch_paths in make_loader(paths, batch=batch, workers=workers, imgsz=imgsz):
        betas.append(estimate_beta_dcp(imgs.to(device)).numpy())
        out_paths.extend(batch_paths)
    return out_paths, np.concatenate(betas)


def match_pairs(real_beta: np.ndarray, synth_beta: np.ndarray, tol: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, rb in enumerate(real_beta):
        diff = np.abs(synth_beta - rb) / (rb + 1e-6)
        j = int(diff.argmin())
        if diff[j] <= tol:
            pairs.append((i, j))
    return pairs


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)


def median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.concatenate([x, y], axis=0).astype(np.float64)
    if len(z) < 2:
        return 1.0
    d = _sq_dists(z, z)
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return 1.0
    return 1.0 / (2.0 * float(np.median(vals)))


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gamma = median_gamma(x, y)
    kxx = np.exp(-gamma * _sq_dists(x, x)).mean()
    kyy = np.exp(-gamma * _sq_dists(y, y)).mean()
    kxy = np.exp(-gamma * _sq_dists(x, y)).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0) ** 0.5)


def ks_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    vals = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, vals, side="right") / len(x)
    cdf_y = np.searchsorted(y, vals, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def mean_ks(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.mean([ks_1d(x[:, dim], y[:, dim]) for dim in range(x.shape[1])]))


def mean_or_none(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def ratio_or_none(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 1e-12:
        return None
    return float(num / den)


def load_gate(run_dir: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    gate_path = run_dir / "dadg_gate.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing DADG gate checkpoint: {gate_path}")
    payload = torch.load(gate_path, map_location=device)
    gate_cfg = dict(payload.get("config", {}))
    cfg = dict(gate_cfg)
    cfg["input_mode"] = payload.get("input_mode", {})
    # E041: MLP-gate checkpoints carry an "arch" key; old conv checkpoints do not.
    arch = str(gate_cfg.pop("arch", "conv"))
    gate = build_gate(arch, **gate_cfg).to(device)
    gate.load_state_dict(payload["state_dict"])
    gate.eval()
    return gate, cfg


@torch.no_grad()
def gate_outputs(
    gate: DADGGate,
    cfg: dict[str, Any],
    paths: list[str],
    *,
    batch: int,
    workers: int,
    imgsz: int,
    device: str,
) -> np.ndarray:
    loader = make_loader([Path(p) for p in paths], batch=batch, workers=workers, imgsz=imgsz)
    outs: list[np.ndarray] = []
    mode = str(cfg.get("input_mode", {}).get("gate_input", cfg.get("gate_input", "rgb")))
    use_dark_channel = bool(cfg.get("input_mode", {}).get("use_dark_channel", False))
    for imgs, _ in loader:
        imgs = imgs.to(device)
        gate_in = build_gate_input(imgs, mode=mode, use_dark_channel=use_dark_channel)
        outs.append(gate(gate_in).detach().cpu().numpy())
    return np.concatenate(outs)


def summarize_variant_seed(
    *,
    variant: str,
    seed: int,
    real_paths: list[str],
    real_beta: np.ndarray,
    synth_paths: list[str],
    synth_beta: np.ndarray,
    synth_paths_by_tag: dict[str, list[str]],
    pairs: list[tuple[int, int]],
    batch: int,
    workers: int,
    imgsz: int,
    device: str,
    run_suffix: str,
    tol: float,
) -> dict[str, Any]:
    run_dir = EXP_ROOT / f"tsp_dadg_{variant}{run_suffix}" / f"seed{seed}"
    gate, cfg = load_gate(run_dir, device=device)
    real_full_out = gate_outputs(gate, cfg, real_paths, batch=batch, workers=workers, imgsz=imgsz, device=device)
    synth_out_by_tag = {
        tag: gate_outputs(gate, cfg, paths, batch=batch, workers=workers, imgsz=imgsz, device=device)
        for tag, paths in synth_paths_by_tag.items()
    }

    mmd_real_synth = {tag: mmd_rbf(real_full_out, outs) for tag, outs in synth_out_by_tag.items()}
    ks_real_synth = {tag: mean_ks(real_full_out, outs) for tag, outs in synth_out_by_tag.items()}
    mmd_synth_synth: dict[str, float] = {}
    ks_synth_synth: dict[str, float] = {}
    for i, tag_i in enumerate(SYNTH_TAGS):
        for tag_j in SYNTH_TAGS[i + 1:]:
            key = f"{tag_i}_vs_{tag_j}"
            mmd_synth_synth[key] = mmd_rbf(synth_out_by_tag[tag_i], synth_out_by_tag[tag_j])
            ks_synth_synth[key] = mean_ks(synth_out_by_tag[tag_i], synth_out_by_tag[tag_j])
    mmd_real_mean = mean_or_none(list(mmd_real_synth.values()))
    mmd_synth_mean = mean_or_none(list(mmd_synth_synth.values()))
    ks_real_mean = mean_or_none(list(ks_real_synth.values()))
    ks_synth_mean = mean_or_none(list(ks_synth_synth.values()))

    real_matched = [real_paths[i] for i, _ in pairs]
    synth_matched = [synth_paths[j] for _, j in pairs]
    real_out = gate_outputs(gate, cfg, real_matched, batch=batch, workers=workers, imgsz=imgsz, device=device)
    synth_out = gate_outputs(gate, cfg, synth_matched, batch=batch, workers=workers, imgsz=imgsz, device=device)
    l2_rs = np.linalg.norm(real_out - synth_out, axis=1)

    rng = np.random.default_rng(seed)
    synth_pairs: list[tuple[int, int]] = []
    tries = 0
    while len(synth_pairs) < len(pairs) and tries < len(pairs) * 20:
        a, b = rng.choice(len(synth_paths), 2, replace=False)
        if abs(synth_beta[a] - synth_beta[b]) / (synth_beta[a] + 1e-6) <= tol:
            synth_pairs.append((int(a), int(b)))
        tries += 1
    if synth_pairs:
        synth_a = gate_outputs(gate, cfg, [synth_paths[a] for a, _ in synth_pairs], batch=batch, workers=workers, imgsz=imgsz, device=device)
        synth_b = gate_outputs(gate, cfg, [synth_paths[b] for _, b in synth_pairs], batch=batch, workers=workers, imgsz=imgsz, device=device)
        l2_ss = np.linalg.norm(synth_a - synth_b, axis=1)
        synth_mean = float(l2_ss.mean())
    else:
        l2_ss = np.array([])
        synth_mean = None
    real_mean = float(l2_rs.mean()) if len(l2_rs) else None
    ratio = real_mean / synth_mean if real_mean is not None and synth_mean and synth_mean > 1e-8 else None
    return {
        "variant": variant,
        "seed": seed,
        "gate_input": cfg.get("input_mode", {}).get("gate_input", variant),
        "n_real_synth_pairs": len(pairs),
        "n_synth_synth_pairs": len(synth_pairs),
        "l2_real_synth_mean": real_mean,
        "l2_real_synth_std": float(l2_rs.std()) if len(l2_rs) else None,
        "l2_synth_synth_mean": synth_mean,
        "l2_synth_synth_std": float(l2_ss.std()) if len(l2_ss) else None,
        "l2_ratio_real_synth_over_synth_synth": ratio,
        "mmd_real_synth_mean": mmd_real_mean,
        "mmd_synth_synth_mean": mmd_synth_mean,
        "mmd_ratio_real_synth_over_synth_synth": ratio_or_none(mmd_real_mean, mmd_synth_mean),
        "mmd_real_synth_by_beta": json.dumps(mmd_real_synth, ensure_ascii=False),
        "mmd_synth_synth_by_beta_pair": json.dumps(mmd_synth_synth, ensure_ascii=False),
        "ks_real_synth_mean": ks_real_mean,
        "ks_synth_synth_mean": ks_synth_mean,
        "ks_ratio_real_synth_over_synth_synth": ratio_or_none(ks_real_mean, ks_synth_mean),
        "ks_real_synth_by_beta": json.dumps(ks_real_synth, ensure_ascii=False),
        "ks_synth_synth_by_beta_pair": json.dumps(ks_synth_synth, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["rgb", "raw_dark_channel", "raw_transmission", "tsp_grad_mag"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--max-images", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", type=Path, default=EXP_ROOT / "summaries/e032_di_signal_ladder")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    real_paths, real_beta = collect_beta(
        "rtts", max_images=args.max_images, batch=args.batch, workers=args.workers, imgsz=args.imgsz, device=device
    )
    synth_paths: list[str] = []
    synth_paths_by_tag: dict[str, list[str]] = {}
    synth_beta_parts: list[np.ndarray] = []
    for tag in SYNTH_TAGS:
        paths, beta = collect_beta(
            tag, max_images=args.max_images, batch=args.batch, workers=args.workers, imgsz=args.imgsz, device=device
        )
        synth_paths.extend(paths)
        synth_paths_by_tag[tag] = paths
        synth_beta_parts.append(beta)
    synth_beta = np.concatenate(synth_beta_parts)
    pairs = match_pairs(real_beta, synth_beta, tol=args.tol)
    if not pairs:
        raise RuntimeError("No beta-matched RTTS/synthetic pairs were found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    beta_path = args.out_dir / "e032_beta_estimates.csv"
    with beta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "dataset", "beta"])
        for p, b in zip(real_paths, real_beta):
            writer.writerow([p, "rtts", f"{b:.8f}"])
        start = 0
        for tag, beta in zip(SYNTH_TAGS, synth_beta_parts):
            for p, b in zip(synth_paths[start:start + len(beta)], beta):
                writer.writerow([p, tag, f"{b:.8f}"])
            start += len(beta)

    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            try:
                rows.append(
                    summarize_variant_seed(
                        variant=variant,
                        seed=seed,
                        real_paths=real_paths,
                        real_beta=real_beta,
                        synth_paths=synth_paths,
                        synth_beta=synth_beta,
                        synth_paths_by_tag=synth_paths_by_tag,
                        pairs=pairs,
                        batch=args.batch,
                        workers=args.workers,
                        imgsz=args.imgsz,
                        device=device,
                        run_suffix=args.run_suffix,
                        tol=args.tol,
                    )
                )
            except FileNotFoundError as exc:
                rows.append({"variant": variant, "seed": seed, "missing": str(exc)})
                print(f"[missing] {exc}", flush=True)

    raw_path = args.out_dir / "e032_gate_probe_raw.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "mmd": "mmd_ratio_real_synth_over_synth_synth",
        "ks": "ks_ratio_real_synth_over_synth_synth",
        "l2_legacy": "l2_ratio_real_synth_over_synth_synth",
        "mmd_real_synth": "mmd_real_synth_mean",
        "mmd_synth_synth": "mmd_synth_synth_mean",
        "ks_real_synth": "ks_real_synth_mean",
        "ks_synth_synth": "ks_synth_synth_mean",
    }
    by_variant: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if "variant" not in row:
            continue
        bucket = by_variant.setdefault(str(row["variant"]), {name: [] for name in metrics})
        for name, key in metrics.items():
            value = row.get(key)
            if value is not None:
                bucket[name].append(float(value))
    summary_rows = []
    for variant, metric_vals in by_variant.items():
        row: dict[str, Any] = {"variant": variant}
        for name, vals in metric_vals.items():
            row[f"{name}_n_seeds"] = len(vals)
            if vals:
                arr = np.asarray(vals, dtype=float)
                row[f"{name}_mean"] = float(arr.mean())
                row[f"{name}_std"] = float(arr.std())
            else:
                row[f"{name}_mean"] = None
                row[f"{name}_std"] = None
        summary_rows.append(row)
    summary_path = args.out_dir / "e032_gate_probe_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["variant"]
        for name in metrics:
            fieldnames.extend([f"{name}_n_seeds", f"{name}_mean", f"{name}_std"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.out_dir / "e032_gate_probe_summary.json").write_text(
        json.dumps({"rows": summary_rows, "raw": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary_rows, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
