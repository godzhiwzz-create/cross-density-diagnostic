"""Reproduce the intervention result (the index is movable) from the released
per-seed CSVs --- no gate checkpoints or image data needed. Run:
  python repro/cdu_intervention.py

Inputs (results/cdu_recompute/):
  e048f0_10seed.csv     : RGB reg_w=0     gate-output index, 10 seeds (corrected estimator)
  e048f1000_10seed.csv  : RGB reg_w=1000  gate-output index, 10 seeds

A density-sensitive regularizer (reg_w=1000) that targets the index's
cross-density denominator by construction moves the index substantially under
every discrepancy; mAP does not change (see the paper / preregistration).
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cdu_recompute"
RECIPES = ["cdu_dist_shared", "cdu_old_percomp", "cdu_sq_shared", "cdu_energy", "cdu_ks"]


def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        if r.get("variant") == "rgb" and r.get("cdu_dist_shared"):
            d[int(r["seed"])] = {k: float(r[k]) for k in RECIPES}
    return d


def main():
    f0 = load(DATA / "e048f0_10seed.csv")      # reg_w = 0
    f1 = load(DATA / "e048f1000_10seed.csv")   # reg_w = 1000
    seeds = sorted(set(f0) & set(f1))
    print(f"paired seeds (n={len(seeds)}): {seeds}\n")
    for rec in RECIPES:
        a = np.array([f0[s][rec] for s in seeds])   # reg0
        b = np.array([f1[s][rec] for s in seeds])   # reg1000
        pct = 100 * (b.mean() - a.mean()) / a.mean()
        w = stats.wilcoxon(a, b)
        print(f"{rec}: reg0 mean={a.mean():.3f}  reg1000 mean={b.mean():.3f}  "
              f"change={pct:+.1f}%  down {(b < a).sum()}/{len(seeds)}  Wilcoxon p={w.pvalue:.4f}")


if __name__ == "__main__":
    main()
