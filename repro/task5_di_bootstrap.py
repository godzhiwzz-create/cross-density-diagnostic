"""Task 5: DI(MMD) bootstrap CI for RGB + 3 physical conditions.
Reuses di_evaluation feature extraction + mmd_rbf. Resamples image indices
(per dataset) B times, recomputes DI = mean_inter_mmd / mean_intra_mmd.
Does NOT modify any paper file. Output to reviewer_hardening scratch dir.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path(".")
sys.path.insert(0, str(REPO))
from gate.analysis.common import DATASETS, resolve_image_list, make_loader
from gate.analysis.physics_priors import PRIOR_REGISTRY
from gate.analysis import di_evaluation as die

SYNTH = ["beta_0.005", "beta_0.01", "beta_0.02"]
REAL = ["rtts"]  # primary real domain (foggy_driving available as secondary if wanted)

def rgb_prior(imgs: torch.Tensor) -> torch.Tensor:
    # produce a spatial map so spatial_to_descriptor applies (luminance map),
    # matching the 32-D descriptor pipeline used for physical priors.
    r, g, b = imgs[:, 0:1], imgs[:, 1:2], imgs[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return lum  # (B,1,H,W)

PRIORS = {
    "rgb": rgb_prior,
    "raw_dark_channel": PRIOR_REGISTRY["raw_dark_channel"],
    "raw_transmission": PRIOR_REGISTRY["raw_transmission"],
    "tsp_grad_mag": PRIOR_REGISTRY["tsp_grad_mag"],
}

def extract(prior_fn, tag, max_images, batch, workers, device):
    paths = resolve_image_list(DATASETS[tag], split="val")
    if len(paths) > max_images:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(paths), size=max_images, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    loader = make_loader([Path(p) if isinstance(p, str) else p for p in paths],
                         batch=batch, workers=workers)
    return die.extract_features(prior_fn, loader, device=device, max_images=max_images)

def di_from_feats(feats, rng=None):
    # feats: dict tag -> (N,D). subsample with replacement if rng given.
    def get(tag):
        f = feats[tag]
        if rng is None:
            return f
        idx = rng.integers(0, len(f), size=len(f))
        return f[idx]
    inter, intra = [], []
    for rt in REAL:
        fr = get(rt)
        for st in SYNTH:
            inter.append(die.mmd_rbf(fr, get(st)))
    sf = {st: get(st) for st in SYNTH}
    for i, a in enumerate(SYNTH):
        for b in SYNTH[i + 1:]:
            intra.append(die.mmd_rbf(sf[a], sf[b]))
    mi, ma = float(np.mean(inter)), float(np.mean(intra))
    return mi / max(ma, 1e-10), mi, ma

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO / "gate/experiments_dlhost/reviewer_hardening_20260617/task5_bootstrap/di_bootstrap.json"))
    args = ap.parse_args()
    device = "cuda:%s" % args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    results = {}
    rank_samples = []  # list of dict prior->DI per bootstrap, for rank stability
    for name, fn in PRIORS.items():
        feats = {tag: extract(fn, tag, args.max_images, args.batch, args.workers, device)
                 for tag in SYNTH + REAL}
        point, mi, ma = di_from_feats(feats, rng=None)
        rng = np.random.default_rng(20260617)
        boots = []
        for _ in range(args.n_boot):
            di, _, _ = di_from_feats(feats, rng=rng)
            boots.append(di)
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[name] = {
            "di_point": point, "inter_mmd": mi, "intra_mmd": ma,
            "boot_mean": float(boots.mean()), "boot_std": float(boots.std()),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "n_boot": args.n_boot, "max_images": args.max_images,
            "n_real": int(len(feats[REAL[0]])), "n_synth_each": {t: int(len(feats[t])) for t in SYNTH},
        }
        print("%-18s DI=%.3f  boot=%.3f [%.3f, %.3f]  (inter=%.5f intra=%.5f)"
              % (name, point, boots.mean(), lo, hi, mi, ma), flush=True)
        results[name]["_boots"] = boots.tolist()

    # rank stability: per bootstrap, rank priors by DI (recompute jointly with shared resample seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # strip heavy boots arrays from saved summary but keep a compact ci
    save = {k: {kk: vv for kk, vv in v.items() if kk != "_boots"} for k, v in results.items()}
    Path(args.out).write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print("\nSaved -> %s" % args.out)

if __name__ == "__main__":
    main()
