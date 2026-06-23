"""Reproduce the task-metric (guardrail) half of the intervention from released data.
Run: python repro/map_guardrail.py   (reads results/cdu_recompute/intervention_map50_perseed.csv)
The paired RTTS mAP50 difference (reg_w=1000 minus reg_w=0) shows no detectable change.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
d = [float(r["rtts_map50_delta_reg1000_minus_reg0"])
     for r in csv.DictReader(open(ROOT / "results/cdu_recompute/intervention_map50_perseed.csv"))]
d = np.array(d)
ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
print(f"n={len(d)}  mean delta={d.mean():+.4f}  95% CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  "
      f"paired Wilcoxon p={stats.wilcoxon(d).pvalue:.3f}")
