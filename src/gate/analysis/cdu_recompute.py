"""CDU recompute with a SHARED (pooled) kernel bandwidth + corrected estimator.

Reviewer point #1: the canonical probe (di_signal_ladder_probe.py) chooses a
median-heuristic bandwidth *per comparison*, so the numerator and the three
denominator MMDs live in different RKHS scales; it also uses a *biased*
estimator and returns sqrt(rectified MMD^2) (a distance), while the paper text
calls M a *squared* MMD. This script recomputes, per (variant, seed), with ONE
pooled bandwidth frozen across all five domains, and reports several internally
consistent index variants side by side, plus the truncation frequency.

For each (variant, seed) we compute gate outputs on real fog (RTTS) and the
three synthetic densities, then:
  - cdu_old      : EXACT replica of the canonical recipe (per-comparison gamma,
                   biased, sqrt-distance ratio) -> must reproduce the published
                   ~2.50 for E032 RGB (pipeline self-validation, project rule 0.B).
  - cdu_sq       : shared gamma, UNBIASED rectified MMD^2, ratio of squared discrepancies.
  - cdu_dist     : shared gamma, sqrt(rectified MMD^2_u), ratio of distances ("k x as far apart").
  - cdu_energy   : bandwidth-free energy distance ratio.
  - cdu_ks       : bandwidth-free mean-KS ratio.
  - trunc_frac   : fraction of the 6 raw MMD^2_u values that were < 0 (rectified to 0).
N = mean over the 3 real-vs-synth-density comparisons; D = mean over the 3
synth-vs-synth cross-density pairs; CDU = N / D (per-seed ratio).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(".")  # run from src/ ; produces results/cdu_recompute/*.csv (needs gate checkpoints + Cityscapes-Foggy/RTTS data; see README)
sys.path.insert(0, str(REPO))

# reuse the canonical data/gate machinery so we forward EXACTLY the same images/gates
from gate.analysis.di_signal_ladder_probe import (  # noqa: E402
    SYNTH_TAGS, EXP_ROOT, collect_beta, load_gate, gate_outputs,
    _sq_dists, median_gamma, mean_ks,
)


def _kern(sqd: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * sqd)


def pooled_gamma(sets: list[np.ndarray], rng: np.random.Generator, cap: int = 1200) -> float:
    """Median-heuristic bandwidth computed ONCE on the pooled five-domain sample, then frozen."""
    pooled = np.concatenate([np.asarray(s, dtype=np.float64) for s in sets], axis=0)
    if len(pooled) > cap:
        idx = rng.choice(len(pooled), size=cap, replace=False)
        pooled = pooled[idx]
    d = _sq_dists(pooled, pooled)
    vals = d[np.triu_indices_from(d, k=1)]
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return 1.0
    return 1.0 / (2.0 * float(np.median(vals)))


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD^2 estimator (diagonal excluded). Can be negative; NOT rectified here."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    Kxx = _kern(_sq_dists(x, x), gamma)
    Kyy = _kern(_sq_dists(y, y), gamma)
    Kxy = _kern(_sq_dists(x, y), gamma)
    kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))  # diag of Kxx is exp(0)=1
    kyy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    kxy = Kxy.mean()
    return float(kxx + kyy - 2.0 * kxy)


def mmd_old(x: np.ndarray, y: np.ndarray) -> float:
    """EXACT replica of canonical mmd_rbf: per-comparison median gamma, biased, sqrt(rectified)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    g = median_gamma(x, y)
    kxx = _kern(_sq_dists(x, x), g).mean()
    kyy = _kern(_sq_dists(y, y), g).mean()
    kxy = _kern(_sq_dists(x, y), g).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0) ** 0.5)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Bandwidth-free energy distance, U-statistic form; >= 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        return float("nan")
    dxy = np.sqrt(np.maximum(_sq_dists(x, y), 0.0)).mean()
    dxx = np.sqrt(np.maximum(_sq_dists(x, x), 0.0)).sum() / (m * (m - 1))  # diag = 0
    dyy = np.sqrt(np.maximum(_sq_dists(y, y), 0.0)).sum() / (n * (n - 1))
    return float(max(2.0 * dxy - dxx - dyy, 0.0))


def _mean(vals: list[float]) -> float | None:
    f = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(f)) if f else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 1e-12:
        return None
    return float(num / den)


def summarize_seed(variant: str, seed: int, run_suffix: str, real_paths, synth_paths_by_tag,
                   *, batch, workers, imgsz, device) -> dict:
    run_dir = EXP_ROOT / f"tsp_dadg_{variant}{run_suffix}" / f"seed{seed}"
    gate, cfg = load_gate(run_dir, device=device)
    real = gate_outputs(gate, cfg, real_paths, batch=batch, workers=workers, imgsz=imgsz, device=device)
    synth = {tag: gate_outputs(gate, cfg, paths, batch=batch, workers=workers, imgsz=imgsz, device=device)
             for tag, paths in synth_paths_by_tag.items()}

    rng = np.random.default_rng(0)  # fixed: pooled-gamma subsample is deterministic per call
    gamma = pooled_gamma([real, *synth.values()], rng)

    # N: real vs each synthetic density ; D: each synth-synth cross-density pair
    pairs_D = [(SYNTH_TAGS[i], SYNTH_TAGS[j]) for i in range(len(SYNTH_TAGS)) for j in range(i + 1, len(SYNTH_TAGS))]
    N_raw = [mmd2_unbiased(real, synth[t], gamma) for t in SYNTH_TAGS]
    D_raw = [mmd2_unbiased(synth[a], synth[b], gamma) for a, b in pairs_D]
    all_raw = N_raw + D_raw
    trunc_frac = float(np.mean([1.0 if (v is not None and np.isfinite(v) and v < 0) else 0.0 for v in all_raw]))

    N_rect = [max(v, 0.0) for v in N_raw]
    D_rect = [max(v, 0.0) for v in D_raw]
    cdu_sq = _ratio(_mean(N_rect), _mean(D_rect))
    cdu_dist = _ratio(_mean([np.sqrt(v) for v in N_rect]), _mean([np.sqrt(v) for v in D_rect]))

    N_e = [energy_distance(real, synth[t]) for t in SYNTH_TAGS]
    D_e = [energy_distance(synth[a], synth[b]) for a, b in pairs_D]
    cdu_energy = _ratio(_mean(N_e), _mean(D_e))

    N_ks = [mean_ks(real, synth[t]) for t in SYNTH_TAGS]
    D_ks = [mean_ks(synth[a], synth[b]) for a, b in pairs_D]
    cdu_ks = _ratio(_mean(N_ks), _mean(D_ks))

    N_old = [mmd_old(real, synth[t]) for t in SYNTH_TAGS]
    D_old = [mmd_old(synth[a], synth[b]) for a, b in pairs_D]
    cdu_old = _ratio(_mean(N_old), _mean(D_old))

    return {
        "variant": variant, "seed": seed, "run_suffix": run_suffix,
        "n_real": int(len(real)), "n_synth_per_density": int(min(len(v) for v in synth.values())),
        "gate_dim": int(real.shape[1]),
        "pooled_gamma": float(gamma),
        "cdu_old_percomp": cdu_old,
        "cdu_sq_shared": cdu_sq,
        "cdu_dist_shared": cdu_dist,
        "cdu_energy": cdu_energy,
        "cdu_ks": cdu_ks,
        "N_sq_mean": _mean(N_rect), "D_sq_mean": _mean(D_rect),
        "N_dist_mean": _mean([np.sqrt(v) for v in N_rect]), "D_dist_mean": _mean([np.sqrt(v) for v in D_rect]),
        "N_energy_mean": _mean(N_e), "D_energy_mean": _mean(D_e),
        "trunc_frac": trunc_frac,
        "N_raw_mmd2u": json.dumps([None if not np.isfinite(v) else round(v, 8) for v in N_raw]),
        "D_raw_mmd2u": json.dumps([None if not np.isfinite(v) else round(v, 8) for v in D_raw]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["rgb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    real_paths, _ = collect_beta("rtts", max_images=args.max_images, batch=args.batch,
                                 workers=args.workers, imgsz=args.imgsz, device=device)
    synth_paths_by_tag = {}
    for tag in SYNTH_TAGS:
        paths, _ = collect_beta(tag, max_images=args.max_images, batch=args.batch,
                                workers=args.workers, imgsz=args.imgsz, device=device)
        synth_paths_by_tag[tag] = paths

    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            try:
                row = summarize_seed(variant, seed, args.run_suffix, real_paths, synth_paths_by_tag,
                                     batch=args.batch, workers=args.workers, imgsz=args.imgsz, device=device)
                rows.append(row)
                print(f"[ok] {variant}{args.run_suffix} seed{seed}: "
                      f"old={row['cdu_old_percomp']} sq={row['cdu_sq_shared']} "
                      f"dist={row['cdu_dist_shared']} energy={row['cdu_energy']} ks={row['cdu_ks']} "
                      f"trunc={row['trunc_frac']}", flush=True)
            except FileNotFoundError as exc:
                rows.append({"variant": variant, "seed": seed, "run_suffix": args.run_suffix, "missing": str(exc)})
                print(f"[missing] {variant}{args.run_suffix} seed{seed}: {exc}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"out": str(args.out), "n_rows": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
