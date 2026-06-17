"""Pillar 2.1: gate output distribution across synthetic vs real fog.

Forwards the DADG gate (seed=42) over 1000 sampled images from each of
5 benchmarks and dumps per-image (w_feat, w_attn, w_loc). Also computes
pairwise MMD (RBF) + per-branch KS between domains.

Output:
  gate/experiments/boundary/gate_distribution/
    - per_dataset.csv (image_path, dataset, w_f, w_a, w_l)
    - pairwise_mmd.csv (dataset_a, dataset_b, mmd)
    - per_branch_ks.csv (dataset_a, dataset_b, branch, ks_stat, p_value)
    - simplex_scatter.png
    - marginal_hist.png

Run:
  python3 gate/analysis/gate_distribution.py --seed 42 --max-images 1000
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gate.analysis.common import (  # noqa: E402
    DATASETS, load_dadg_gate, resolve_image_list, make_loader, iter_gate_outputs,
)

OUT_DIR = Path("/root/kd_visibility/gate/experiments/boundary/gate_distribution")


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Unbiased RBF MMD² between two sample sets."""
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    nx, ny = x.size(0), y.size(0)
    xy = torch.cat([x, y], dim=0)
    d2 = torch.cdist(xy, xy).pow(2)
    if sigma is None:
        sigma = float(d2[d2 > 0].median().sqrt())
    K = torch.exp(-d2 / (2 * sigma ** 2))
    Kxx = K[:nx, :nx].fill_diagonal_(0).sum() / (nx * (nx - 1))
    Kyy = K[nx:, nx:].fill_diagonal_(0).sum() / (ny * (ny - 1))
    Kxy = K[:nx, nx:].mean()
    return float(Kxx + Kyy - 2 * Kxy)


def simplex_project_2d(w: np.ndarray) -> np.ndarray:
    """Map (N, 3) barycentric → (N, 2) planar for plotting."""
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    return w @ corners


def run(seed: int, max_images: int, batch: int, workers: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate = load_dadg_gate(seed=seed, device=device)

    rows: list[tuple] = []
    per_dataset: dict[str, np.ndarray] = {}
    for tag, yaml_path in DATASETS.items():
        print(f"[{tag}] resolving images...", flush=True)
        paths = resolve_image_list(yaml_path, split="val")
        if len(paths) > max_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), size=max_images, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f"[{tag}] {len(paths)} images", flush=True)
        loader = make_loader(paths, batch=batch, workers=workers)
        all_w = []
        for w, ps in iter_gate_outputs(gate, loader, device=device):
            wn = w.numpy()
            for pth, wi in zip(ps, wn):
                rows.append((pth, tag, float(wi[0]), float(wi[1]), float(wi[2])))
            all_w.append(wn)
        per_dataset[tag] = np.concatenate(all_w, axis=0)

    with open(OUT_DIR / "per_dataset.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "dataset", "w_feat", "w_attn", "w_loc"])
        w.writerows(rows)

    tags = list(per_dataset)
    with open(OUT_DIR / "pairwise_mmd.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a", "b", "mmd2"])
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                m = rbf_mmd2(per_dataset[a], per_dataset[b])
                w.writerow([a, b, f"{m:.6f}"])
                print(f"MMD²({a}, {b}) = {m:.5f}")

    from scipy.stats import ks_2samp
    with open(OUT_DIR / "per_branch_ks.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a", "b", "branch", "ks", "p"])
        branches = ["w_feat", "w_attn", "w_loc"]
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                for k, bname in enumerate(branches):
                    stat, p = ks_2samp(per_dataset[a][:, k], per_dataset[b][:, k])
                    w.writerow([a, b, bname, f"{stat:.4f}", f"{p:.3e}"])

    # Simplex scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"beta_0.005": "#1f77b4", "beta_0.01": "#2ca02c", "beta_0.02": "#9467bd",
              "rtts": "#d62728", "foggy_driving": "#ff7f0e"}
    for tag in tags:
        xy = simplex_project_2d(per_dataset[tag])
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.3, label=tag, c=colors[tag])
    # Triangle edges
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
        ax.plot(*corners[[a, b]].T, "k-", lw=0.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(markerscale=3, fontsize=9, loc="upper right")
    ax.set_title(f"DADG gate output on simplex (seed={seed})")
    fig.tight_layout(); fig.savefig(OUT_DIR / "simplex_scatter.png", dpi=150)
    plt.close(fig)

    # Marginal histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, ax in enumerate(axes):
        for tag in tags:
            ax.hist(per_dataset[tag][:, k], bins=40, alpha=0.4, label=tag,
                    density=True, color=colors[tag])
        ax.set_title(["w_feat", "w_attn", "w_loc"][k])
        ax.set_xlim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT_DIR / "marginal_hist.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    run(**vars(ap.parse_args()))
