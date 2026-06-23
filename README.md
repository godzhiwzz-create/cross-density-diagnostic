# Cross-Density Diagnostic: a cross-density index over distillation gates is not reproducible across training runs

Code, seed-level outputs, and the preregistration for a **cautionary negative
result**: a cross-density normalized index (CDU) over the outputs of a
conditional knowledge-distillation gate, trained on synthetic fog, **is not a
reliable measurement.** Within a single training run the index looks clean and
kernel-robust; across two independent training runs with an *identical* recipe
and random seed it is essentially uncorrelated, and a regularizer can move it
with no change in real-fog detection accuracy.

This repository accompanies the note *"A Cross-Density Index over Conditional
Distillation Gates Is Not Reproducible Across Training Runs: A Preregistered
Cautionary Study"* (arXiv: `TODO-add-id`). It is a **measurement** release: no
detector, no training framework. Every headline number can be recomputed from
the released per-seed CSVs in seconds, without any gate checkpoints or image
data.

## The result in three numbers

Run these (only `numpy`/`scipy` needed; reads `results/cdu_recompute/*.csv`):

```bash
python repro/cdu_testretest.py     # within-run ladder + test-retest agreement
python repro/cdu_intervention.py   # the regularizer moves the index
```

1. **Within one run, well behaved.** RGB conditioning scores highest on the
   four-signal ladder, and the ordering is the same under squared-MMD, distance,
   energy-distance, and KS discrepancies; the rectification `max(0,·)` never
   fires (truncation 0 over all 20 ladder seed-runs).
2. **Across runs, not reproducible.** Two independent runs with byte-identical
   training config and seed (differing only by GPU nondeterminism) give per-seed
   CDU that is uncorrelated — `ICC(A,1)` between `-0.01` and `+0.02` for *every*
   discrepancy variant (Pearson `r ≈ 0`), with a ~2× run-to-run mean gap.
3. **Movable without a task footprint.** A density-sensitive regularizer that
   targets the index's cross-density denominator drops CDU by `-62%`
   (paired Wilcoxon `p=0.0039`, 9/10 seeds; `-52%` to `-79%` across variants)
   while real-fog mAP shows no detectable change (`p=0.375`).

Takeaway: absolute values of such gate-output indices are training-run noise,
not measurements; cross-density normalization does not buy cross-run
comparability. Only **within-run, paired** contrasts are interpretable.

## A corrected estimator

The natural implementation
(`src/gate/analysis/di_signal_ladder_probe.py`, the "old recipe") chooses a
median-heuristic kernel bandwidth *per comparison*, so the numerator and the
denominator terms of the ratio live in different RKHS scales. The corrected
estimator (`src/gate/analysis/cdu_recompute.py`) uses **one pooled bandwidth**
frozen across all five sample sets and the **unbiased** rectified MMD², and
reports squared-MMD / distance / energy-distance / KS side by side. The failure
in (2) is the same under the bandwidth-free energy and KS variants, so it is not
an artifact of any single kernel. (`cdu_recompute.py` produced the
`results/cdu_recompute/*.csv` files from gate checkpoints; it requires those
checkpoints and the Cityscapes-Foggy / RTTS data, which are external — see
`DATA.md`. The released CSVs make the headline claims reproducible without them.)

## Layout

- `results/cdu_recompute/` — per-seed outputs behind the negative result:
  `e032_shared_bw.csv` (run A ladder), `e048f0_shared_bw.csv` (run B, RGB
  test-retest partner), `e048f0_10seed.csv` / `e048f1000_10seed.csv`
  (intervention reg_w=0 vs reg_w=1000, 10 seeds).
- `repro/cdu_testretest.py`, `repro/cdu_intervention.py` — recompute every
  headline number from those CSVs.
- `src/gate/analysis/cdu_recompute.py` — the corrected estimator (production script).
- `src/gate/` — gate definitions (`models/dadg.py`), conditioning-signal
  extraction (`analysis/physics_priors.py`), and the original probe.
- `preregistration/E049_PREREGISTERED_PROTOCOL.md` — the frozen ten-seed
  protocol (primary metric RTTS mAP50, recorded SHA-256) for the intervention's
  task-metric test.
- `results/e032_di_signal_ladder/`, `results/e034_condition_control_probe/`,
  `repro/` (older scripts) — the broader exploratory measurement campaign that
  preceded the test-retest; kept for provenance.

## Scope and honesty

This release makes **no** causal or mechanistic claim, and does not claim CDU
measures any task-relevant property. It documents that one carefully defined,
intuitively appealing index fails a basic reproducibility check, that the
failure is robust to the estimator and to the choice of discrepancy, and that a
single run does not reveal it. Evidence is bounded: one gate architecture, one
teacher–student pair, one synthetic renderer, 5–10 seeds; the source of
nondeterminism is not decomposed, and whether the coarse RGB-highest *ordering*
reproduces across runs is left open.

## Citation

```
@misc{wang2026crossdensity,
  title  = {A Cross-Density Index over Conditional Distillation Gates Is Not
            Reproducible Across Training Runs: A Preregistered Cautionary Study},
  author = {Wang, Zhuangzhi},
  year   = {2026},
  note   = {arXiv: TODO-add-id}
}
```

License: see `LICENSE`.
