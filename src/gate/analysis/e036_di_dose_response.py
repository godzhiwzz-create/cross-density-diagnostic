"""E036 DI dose-response: known-positive calibration of the diagnostic index.

项目: 可见度识别研究 / 诊断框架论文证据补强 wave1（与 p3_method/p3_residual 无关）

Question
--------
Does the DI (trained-gate real-vs-synthetic MMD/KS, normalized by synthetic
cross-density MMD/KS) respond to a KNOWN injected shortcut, and does the
response track dose? This is the positive-control that the manuscript's DI is
sensitive to detectable shortcuts at all.

Design
------
For each trained gate condition arm (rgb / tsp_grad_mag / raw_dark_channel /
raw_transmission by default), for each fingerprint family (colorbias / noise /
jpeg) and each dose (0..4):

  real  = RTTS  (FIXED across all conditions and doses; never spiked)
  synth = the three beta densities, SPIKED with this family+dose
          (produced by e036_fingerprint_gen.py)

  DI_mmd(dose) = mean_beta MMD(gate(real), gate(synth_spiked_beta))
                 ----------------------------------------------------
                 mean_{beta_i<beta_j} MMD(gate(synth_i), gate(synth_j))

The numerator is the real-vs-synthetic separation; the denominator is the
synthetic cross-density separation. Because the SAME family+dose is applied to
all three beta splits (see e036_fingerprint_gen determinism contract), the
denominator is protected from spike contamination -- the spike cancels in the
cross-density comparison, so dose moves the numerator only.

Pre-registered response signs (write-once, before looking at numbers)
---------------------------------------------------------------------
  colorbias : DI numerator INCREASES with dose  -> sign +
  noise     : DI numerator INCREASES with dose  -> sign +
  jpeg      : DI numerator DECREASES / mixed     -> sign -/mixed
              (RTTS already carries JPEG compression; spiking synthetic with
               JPEG can move it TOWARD real, shrinking the gap).

Success  : observed dose-response sign matches pre-registration for >=2/3
           families AND each detectable family has a finite detection dose.
Failure  : no family is detectable (flat DI across doses for all 3) -> the
           DI-sensitivity claim is downgraded.

Outputs (summaries/<exp-name>/)
-------------------------------
  e036_di_dose_response_raw.csv      one row per (condition, seed, family, dose)
  e036_di_dose_response_summary.csv  seed-aggregated, plus numerator/denominator
                                     decomposition and per-family monotonic slope
  e036_di_dose_response.json         machine-readable bundle incl. preregistration

Reuses gate loading + MMD/KS from di_signal_ladder_probe.py. No training. GPU is
used only for the forward pass of the frozen gate.
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

REPO = Path(".")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gate/analysis"))

from gate.analysis.di_signal_ladder_probe import (  # noqa: E402
    DATASETS,
    EXP_ROOT,
    SYNTH_TAGS,
    collect_beta,
    gate_outputs,
    load_gate,
    mean_ks,
    mean_or_none,
    mmd_rbf,
    ratio_or_none,
    resolve_images,
)

# Where e036_fingerprint_gen.py wrote the spiked val splits.
DEFAULT_SPIKE_ROOT = EXP_ROOT / "e036_fingerprint_dose_response/spiked"
FAMILIES = ["colorbias", "noise", "jpeg"]
N_DOSES = 5
PREREG_SIGN = {"colorbias": "+", "noise": "+", "jpeg": "-/mixed"}


def spiked_dir(spike_root: Path, family: str, dose: int, beta_tag: str, split: str) -> Path:
    return spike_root / family / f"dose{dose}" / beta_tag / "images" / split


def list_images(d: Path) -> list[str]:
    if not d.is_dir():
        raise FileNotFoundError(f"Missing spiked dir (run e036_fingerprint_gen first): {d}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(str(p) for p in d.rglob("*") if p.suffix.lower() in exts)


def di_for_condition_seed(
    *,
    variant: str,
    seed: int,
    run_suffix: str,
    real_paths: list[str],
    spike_root: Path,
    family: str,
    dose: int,
    split: str,
    batch: int,
    workers: int,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    """Compute DI numerator/denominator for one (condition, seed, family, dose)."""
    run_dir = EXP_ROOT / f"tsp_dadg_{variant}{run_suffix}" / f"seed{seed}"
    gate, cfg = load_gate(run_dir, device=device)

    real_out = gate_outputs(gate, cfg, real_paths, batch=batch, workers=workers,
                            imgsz=imgsz, device=device)

    synth_out_by_beta: dict[str, np.ndarray] = {}
    for beta_tag in SYNTH_TAGS:
        paths = list_images(spiked_dir(spike_root, family, dose, beta_tag, split))
        synth_out_by_beta[beta_tag] = gate_outputs(
            gate, cfg, paths, batch=batch, workers=workers, imgsz=imgsz, device=device
        )

    mmd_real = {t: mmd_rbf(real_out, o) for t, o in synth_out_by_beta.items()}
    ks_real = {t: mean_ks(real_out, o) for t, o in synth_out_by_beta.items()}
    mmd_cross: dict[str, float] = {}
    ks_cross: dict[str, float] = {}
    for i, ti in enumerate(SYNTH_TAGS):
        for tj in SYNTH_TAGS[i + 1:]:
            mmd_cross[f"{ti}_vs_{tj}"] = mmd_rbf(synth_out_by_beta[ti], synth_out_by_beta[tj])
            ks_cross[f"{ti}_vs_{tj}"] = mean_ks(synth_out_by_beta[ti], synth_out_by_beta[tj])

    mmd_num = mean_or_none(list(mmd_real.values()))
    mmd_den = mean_or_none(list(mmd_cross.values()))
    ks_num = mean_or_none(list(ks_real.values()))
    ks_den = mean_or_none(list(ks_cross.values()))
    return {
        "variant": variant,
        "seed": seed,
        "family": family,
        "dose": dose,
        "gate_input": cfg.get("input_mode", {}).get("gate_input", variant),
        "di_mmd": ratio_or_none(mmd_num, mmd_den),
        "di_ks": ratio_or_none(ks_num, ks_den),
        "mmd_numerator_real_synth": mmd_num,
        "mmd_denominator_synth_cross": mmd_den,
        "ks_numerator_real_synth": ks_num,
        "ks_denominator_synth_cross": ks_den,
        "mmd_real_by_beta": json.dumps(mmd_real, ensure_ascii=False),
        "mmd_cross_by_beta_pair": json.dumps(mmd_cross, ensure_ascii=False),
    }


def detection_dose(doses: list[int], values: list[float | None], baseline: float | None,
                   rel_thresh: float) -> int | None:
    """First dose>0 whose DI deviates from dose-0 baseline by >= rel_thresh (abs rel)."""
    if baseline is None or baseline <= 1e-12:
        return None
    for d, v in zip(doses, values):
        if d == 0 or v is None:
            continue
        if abs(v - baseline) / baseline >= rel_thresh:
            return d
    return None


def monotonic_slope(doses: list[int], values: list[float | None]) -> float | None:
    """OLS slope of DI vs dose index over finite points (>=2 needed)."""
    pts = [(float(d), float(v)) for d, v in zip(doses, values) if v is not None and np.isfinite(v)]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if xs.std() < 1e-12:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="E036 DI dose-response (known-positive calibration)")
    ap.add_argument("--variants", nargs="+",
                    default=["rgb", "tsp_grad_mag", "raw_dark_channel", "raw_transmission"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--spike-root", type=Path, default=DEFAULT_SPIKE_ROOT)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--max-images", type=int, default=600,
                    help="Cap on the RTTS real side (synthetic side uses the generated subset).")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--detect-rel-thresh", type=float, default=0.10,
                    help="Relative DI deviation from dose-0 counted as 'detected'.")
    ap.add_argument("--out-dir", type=Path,
                    default=EXP_ROOT / "summaries/e036_di_dose_response")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # RTTS real side: fixed for every (condition, family, dose). Collect once.
    real_paths, _ = collect_beta(
        "rtts", max_images=args.max_images, batch=args.batch, workers=args.workers,
        imgsz=args.imgsz, device=device,
    )

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        for seed in args.seeds:
            for family in args.families:
                for dose in range(N_DOSES):
                    try:
                        rows.append(di_for_condition_seed(
                            variant=variant, seed=seed, run_suffix=args.run_suffix,
                            real_paths=real_paths, spike_root=args.spike_root,
                            family=family, dose=dose, split=args.split,
                            batch=args.batch, workers=args.workers, imgsz=args.imgsz,
                            device=device,
                        ))
                        r = rows[-1]
                        print(f"[di] {variant} seed{seed} {family} dose{dose}: "
                              f"di_mmd={r['di_mmd']}", flush=True)
                    except FileNotFoundError as exc:
                        rows.append({"variant": variant, "seed": seed, "family": family,
                                     "dose": dose, "missing": str(exc)})
                        print(f"[missing] {exc}", flush=True)

    raw_path = args.out_dir / "e036_di_dose_response_raw.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Seed-aggregate + per (condition, family) dose curve, slope, detection dose.
    summary_rows: list[dict[str, Any]] = []
    for variant in args.variants:
        for family in args.families:
            doses = list(range(N_DOSES))
            di_mmd_mean: list[float | None] = []
            di_mmd_std: list[float | None] = []
            num_mean: list[float | None] = []
            den_mean: list[float | None] = []
            for dose in doses:
                vals = [r.get("di_mmd") for r in rows
                        if r.get("variant") == variant and r.get("family") == family
                        and r.get("dose") == dose and r.get("di_mmd") is not None]
                nums = [r.get("mmd_numerator_real_synth") for r in rows
                        if r.get("variant") == variant and r.get("family") == family
                        and r.get("dose") == dose and r.get("mmd_numerator_real_synth") is not None]
                dens = [r.get("mmd_denominator_synth_cross") for r in rows
                        if r.get("variant") == variant and r.get("family") == family
                        and r.get("dose") == dose and r.get("mmd_denominator_synth_cross") is not None]
                di_mmd_mean.append(float(np.mean(vals)) if vals else None)
                di_mmd_std.append(float(np.std(vals)) if vals else None)
                num_mean.append(float(np.mean(nums)) if nums else None)
                den_mean.append(float(np.mean(dens)) if dens else None)
            baseline = di_mmd_mean[0]
            slope = monotonic_slope(doses, di_mmd_mean)
            det = detection_dose(doses, di_mmd_mean, baseline, args.detect_rel_thresh)
            observed_sign = ("+" if slope is not None and slope > 0
                             else "-" if slope is not None and slope < 0 else "0")
            prereg = PREREG_SIGN[family]
            sign_match = (prereg.startswith(observed_sign)
                          or (prereg == "-/mixed" and observed_sign in {"-", "0"}))
            summary_rows.append({
                "variant": variant,
                "family": family,
                "preregistered_sign": prereg,
                "observed_slope_di_mmd": slope,
                "observed_sign": observed_sign,
                "sign_matches_prereg": sign_match,
                "detection_dose": det,
                "di_mmd_by_dose": json.dumps(di_mmd_mean),
                "di_mmd_std_by_dose": json.dumps(di_mmd_std),
                "numerator_by_dose": json.dumps(num_mean),
                "denominator_by_dose": json.dumps(den_mean),
            })

    summary_path = args.out_dir / "e036_di_dose_response_summary.csv"
    sfields = ["variant", "family", "preregistered_sign", "observed_slope_di_mmd",
               "observed_sign", "sign_matches_prereg", "detection_dose",
               "di_mmd_by_dose", "di_mmd_std_by_dose", "numerator_by_dose", "denominator_by_dose"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary_rows)

    (args.out_dir / "e036_di_dose_response.json").write_text(
        json.dumps({
            "preregistration": PREREG_SIGN,
            "detect_rel_thresh": args.detect_rel_thresh,
            "summary": summary_rows,
            "raw": rows,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary_rows, "out_dir": str(args.out_dir)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
