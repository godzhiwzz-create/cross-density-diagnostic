"""Domain Invariance (DI) evaluation for physical priors.

Computes DI ratio for each physical quantity:
  DI(P) = D(P_real, P_synth) / D(P_synth_a, P_synth_b)

Where D is MMD (Maximum Mean Discrepancy) with RBF kernel.

DI ≈ 1.0 → domain-invariant (good candidate prior)
DI >> 1.0 → domain-specific (like raw RGB features, known ratio=2.38)

Usage:
  python di_evaluation.py --max-images 500 --batch 32
  python di_evaluation.py --priors tsp_rank,fdgp --max-images 1000

Output:
  gate/experiments/di_evaluation/
    - di_results.json  (DI ratios per prior)
    - di_results.csv   (tabular)
    - distributions/   (per-prior numpy arrays for visualization)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gate.analysis.common import DATASETS, resolve_image_list, make_loader  # noqa: E402
from gate.analysis.physics_priors import PRIOR_REGISTRY  # noqa: E402

OUT_DIR = Path("./gate/experiments_dlhost/di_evaluation")


# ─── Distance metrics ────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """MMD^2 with RBF kernel between two sets of feature vectors.

    Args:
        X: (N, D) features from distribution 1
        Y: (M, D) features from distribution 2
        gamma: RBF bandwidth (if None, use median heuristic)
    Returns:
        MMD^2 value (>= 0, higher = more different)
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        XY = np.vstack([X[:200], Y[:200]])
        from scipy.spatial.distance import pdist
        dists = pdist(XY, 'sqeuclidean')
        gamma = 1.0 / np.median(dists) if np.median(dists) > 0 else 1.0

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = len(X)
    m = len(Y)
    mmd2 = (K_XX.sum() / (n * n) - 2 * K_XY.sum() / (n * m) + K_YY.sum() / (m * m))
    return max(0.0, mmd2)


def ks_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    """KS statistic for 1D distributions (averaged over dimensions if multi-dim)."""
    from scipy.stats import ks_2samp
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
    D = X.shape[1]
    stats = []
    for d in range(min(D, 50)):
        stat, _ = ks_2samp(X[:, d], Y[:, d])
        stats.append(stat)
    return float(np.mean(stats))


# ─── Feature extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    prior_fn: Callable,
    loader: DataLoader,
    device: str = "cuda",
    max_images: int | None = None,
) -> np.ndarray:
    """Extract prior features from all images in loader.

    Returns:
        (N, D) feature matrix. For spatial maps (B,1,H,W), we compute
        a compact descriptor: histogram (20 bins) + spatial statistics.
        For curves (B, num_bands), use directly.
    """
    all_features = []
    seen = 0

    for imgs, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = prior_fn(imgs)

        if out.dim() == 4:  # spatial map (B, 1, H, W)
            feats = spatial_to_descriptor(out)
        elif out.dim() == 2:  # curve (B, num_bands)
            feats = out.cpu().numpy()
        else:
            raise ValueError(f"Unexpected output dim: {out.dim()}")

        all_features.append(feats)
        seen += imgs.size(0)
        if max_images and seen >= max_images:
            break

    return np.concatenate(all_features, axis=0)


def spatial_to_descriptor(spatial_map: torch.Tensor, n_bins: int = 20) -> np.ndarray:
    """Convert spatial map to compact feature descriptor.

    Descriptor includes:
    - Histogram (n_bins): distribution of values
    - Spatial statistics (8): mean/std of top/bottom/left/right halves
    - Global statistics (4): mean, std, skewness, kurtosis

    Total: n_bins + 8 + 4 = 32 features per image.
    """
    B = spatial_map.size(0)
    s = spatial_map[:, 0]  # (B, H, W)
    feats = []

    for b in range(B):
        m = s[b].cpu().numpy()
        f = []

        # Histogram
        valid = m[np.isfinite(m)]
        if len(valid) == 0:
            f.extend([0.0] * n_bins)
        else:
            lo, hi = np.percentile(valid, [1, 99])
            if lo == hi:
                hi = lo + 1e-6
            hist, _ = np.histogram(valid, bins=n_bins, range=(lo, hi), density=True)
            f.extend(hist.tolist())

        # Spatial statistics (top/bottom/left/right halves)
        H, W = m.shape
        for region in [m[:H // 2], m[H // 2:], m[:, :W // 2], m[:, W // 2:]]:
            r = region[np.isfinite(region)]
            f.append(float(r.mean()) if len(r) > 0 else 0.0)
            f.append(float(r.std()) if len(r) > 0 else 0.0)

        # Global statistics
        valid = m[np.isfinite(m)]
        if len(valid) > 1:
            f.append(float(valid.mean()))
            f.append(float(valid.std()))
            from scipy.stats import skew, kurtosis
            f.append(float(skew(valid.ravel())))
            f.append(float(kurtosis(valid.ravel())))
        else:
            f.extend([0.0, 0.0, 0.0, 0.0])

        feats.append(f)

    return np.array(feats, dtype=np.float64)


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_di(
    prior_name: str,
    prior_fn: Callable,
    synth_tags: list[str],
    real_tags: list[str],
    max_images: int,
    batch: int,
    workers: int,
    device: str,
) -> dict:
    """Compute DI ratio for a single prior.

    DI = D(real, synth) / D(synth_a, synth_b)
    where synth_a and synth_b are two different β subsets of Cityscapes-Foggy.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating DI for: {prior_name}")
    print(f"{'='*60}")

    # Extract features for each dataset
    features = {}
    for tag in synth_tags + real_tags:
        print(f"  Extracting {prior_name} from {tag}...", end=" ", flush=True)
        paths = resolve_image_list(DATASETS[tag], split="val")
        if len(paths) > max_images:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(paths), size=max_images, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        loader = make_loader([Path(p) if isinstance(p, str) else p for p in paths],
                            batch=batch, workers=workers)
        features[tag] = extract_features(prior_fn, loader, device=device, max_images=max_images)
        print(f"shape={features[tag].shape}")

    # Compute inter-domain distances (real vs synth)
    inter_distances = {}
    for rt in real_tags:
        for st in synth_tags:
            key = f"{rt}_vs_{st}"
            d_mmd = mmd_rbf(features[rt], features[st])
            d_ks = ks_statistic(features[rt], features[st])
            inter_distances[key] = {"mmd": d_mmd, "ks": d_ks}
            print(f"  D({rt}, {st}): MMD={d_mmd:.6f}, KS={d_ks:.4f}")

    # Compute intra-domain distances (synth vs synth)
    intra_distances = {}
    for i, st_a in enumerate(synth_tags):
        for st_b in synth_tags[i + 1:]:
            key = f"{st_a}_vs_{st_b}"
            d_mmd = mmd_rbf(features[st_a], features[st_b])
            d_ks = ks_statistic(features[st_a], features[st_b])
            intra_distances[key] = {"mmd": d_mmd, "ks": d_ks}
            print(f"  D({st_a}, {st_b}): MMD={d_mmd:.6f}, KS={d_ks:.4f}")

    # Compute DI ratios
    mean_inter_mmd = np.mean([v["mmd"] for v in inter_distances.values()])
    mean_intra_mmd = np.mean([v["mmd"] for v in intra_distances.values()])
    mean_inter_ks = np.mean([v["ks"] for v in inter_distances.values()])
    mean_intra_ks = np.mean([v["ks"] for v in intra_distances.values()])

    di_mmd = mean_inter_mmd / max(mean_intra_mmd, 1e-10)
    di_ks = mean_inter_ks / max(mean_intra_ks, 1e-10)

    result = {
        "prior": prior_name,
        "di_mmd": float(di_mmd),
        "di_ks": float(di_ks),
        "mean_inter_mmd": float(mean_inter_mmd),
        "mean_intra_mmd": float(mean_intra_mmd),
        "mean_inter_ks": float(mean_inter_ks),
        "mean_intra_ks": float(mean_intra_ks),
        "inter_distances": {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in inter_distances.items()},
        "intra_distances": {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in intra_distances.items()},
    }

    print(f"\n  >>> DI({prior_name}) = {di_mmd:.4f} (MMD), {di_ks:.4f} (KS)")
    return result


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist_dir = OUT_DIR / "distributions"
    dist_dir.mkdir(exist_ok=True)

    synth_tags = ["beta_0.005", "beta_0.01", "beta_0.02"]
    real_tags = ["rtts", "foggy_driving"]

    if args.priors:
        prior_names = args.priors.split(",")
    else:
        prior_names = list(PRIOR_REGISTRY.keys())

    all_results = []
    for name in prior_names:
        if name not in PRIOR_REGISTRY:
            print(f"WARNING: unknown prior '{name}', skipping")
            continue

        result = evaluate_di(
            prior_name=name,
            prior_fn=PRIOR_REGISTRY[name],
            synth_tags=synth_tags,
            real_tags=real_tags,
            max_images=args.max_images,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )
        all_results.append(result)

    # Save results
    with open(OUT_DIR / "di_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(OUT_DIR / "di_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prior", "DI_MMD", "DI_KS", "inter_MMD", "intra_MMD", "inter_KS", "intra_KS"])
        for r in all_results:
            w.writerow([
                r["prior"], f"{r['di_mmd']:.4f}", f"{r['di_ks']:.4f}",
                f"{r['mean_inter_mmd']:.6f}", f"{r['mean_intra_mmd']:.6f}",
                f"{r['mean_inter_ks']:.4f}", f"{r['mean_intra_ks']:.4f}",
            ])

    # Summary table
    print("\n" + "=" * 70)
    print("DOMAIN INVARIANCE RESULTS")
    print("=" * 70)
    print(f"{'Prior':<20} {'DI(MMD)':>10} {'DI(KS)':>10} {'Assessment':<20}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["di_mmd"]):
        name = r["prior"]
        di = r["di_mmd"]
        di_ks = r["di_ks"]
        if di < 1.3:
            assess = "STRONG invariant"
        elif di < 2.0:
            assess = "weak invariant"
        else:
            assess = "domain-specific"
        print(f"{name:<20} {di:>10.4f} {di_ks:>10.4f} {assess:<20}")
    print("=" * 70)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", type=str, default=None,
                        help="Comma-separated prior names (default: all)")
    parser.add_argument("--max-images", type=int, default=500,
                        help="Max images per dataset")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run(args)
