"""Reproduce the test-retest (non-)reproducibility result and the within-run
ladder from the released per-seed CSVs --- no gate checkpoints or image data
needed. Run:  python repro/cdu_testretest.py

Inputs (results/cdu_recompute/):
  e032_shared_bw.csv    : within-run ladder, 4 conditioning signals x 5 seeds (run A)
  e048f0_shared_bw.csv  : RGB, same 5 seeds, an independent identical-recipe run (run B)

Outputs to stdout:
  - test-retest agreement (Pearson r, Spearman rho, ICC(A,1), Bland-Altman) between
    runs A and B for every discrepancy variant  -> all ICC ~ 0
  - the within-run ladder summary (RGB highest under every variant; truncation 0)
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cdu_recompute"
RECIPES = ["cdu_old_percomp", "cdu_dist_shared", "cdu_sq_shared", "cdu_energy", "cdu_ks"]


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("variant") and r.get("cdu_dist_shared"):
            rows.append(r)
    return rows


def by_seed(rows, variant):
    return {int(r["seed"]): {k: float(r[k]) for k in RECIPES}
            for r in rows if r["variant"] == variant}


def icc_a1(X):
    """Two-way random, absolute-agreement, single-rater ICC(A,1)."""
    n, k = X.shape
    g = X.mean()
    SSR = k * ((X.mean(1) - g) ** 2).sum()
    SSC = n * ((X.mean(0) - g) ** 2).sum()
    SSE = ((X - g) ** 2).sum() - SSR - SSC
    MSR, MSC, MSE = SSR / (n - 1), SSC / (k - 1), SSE / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))


def main():
    e032 = load(DATA / "e032_shared_bw.csv")
    e048 = load(DATA / "e048f0_shared_bw.csv")
    rgbA, rgbB = by_seed(e032, "rgb"), by_seed(e048, "rgb")
    seeds = sorted(set(rgbA) & set(rgbB))

    print("=" * 68)
    print("TEST-RETEST: two independent identical-recipe RGB runs, per-seed CDU")
    print(f"shared seeds: {seeds}")
    print("=" * 68)
    for rec in RECIPES:
        a = np.array([rgbA[s][rec] for s in seeds])
        b = np.array([rgbB[s][rec] for s in seeds])
        diff = a - b
        loa = (diff.mean() - 1.96 * diff.std(ddof=1), diff.mean() + 1.96 * diff.std(ddof=1))
        print(f"\n{rec}:")
        print(f"  run A {np.round(a, 3)}  mean {a.mean():.3f}")
        print(f"  run B {np.round(b, 3)}  mean {b.mean():.3f}")
        print(f"  Pearson r={stats.pearsonr(a, b)[0]:+.3f}  Spearman rho={stats.spearmanr(a, b)[0]:+.3f}"
              f"  ICC(A,1)={icc_a1(np.column_stack([a, b])):+.3f}")
        print(f"  Bland-Altman bias(A-B)={diff.mean():+.3f}  LoA=[{loa[0]:+.3f},{loa[1]:+.3f}]")

    print("\n" + "=" * 68)
    print("WITHIN-RUN LADDER (run A, mean +/- s.d. over 5 seeds)")
    print("=" * 68)
    for rec in RECIPES:
        means = {}
        print(f"\n{rec}:")
        for v in ["rgb", "tsp_grad_mag", "raw_dark_channel", "raw_transmission"]:
            d = by_seed(e032, v)
            vals = np.array([d[s][rec] for s in sorted(d)])
            means[v] = vals.mean()
            print(f"  {v:20s} {vals.mean():.3f} +/- {vals.std(ddof=1):.3f}")
        order = sorted(means, key=means.get, reverse=True)
        print(f"  ordering: {' > '.join(order)}   RGB highest? {order[0] == 'rgb'}")

    trunc = [float(r["trunc_frac"]) for r in e032 + e048]
    print(f"\ntruncation: max over {len(trunc)} ladder+retest seed-runs = {max(trunc)}")


if __name__ == "__main__":
    main()
