# Cross-Density Diagnostic: measuring gate-conditioning shortcuts in synthetic-fog detection

Reproducibility code and seed-level outputs for the diagnostic protocol that
measures whether a trained dynamic knowledge-distillation gate routes by
synthetic-domain identity (a rendering-fingerprint *shortcut*) rather than by
physical degradation. The instrument is a fixed gate whose only swapped
variable is the conditioning signal; cross-density units (CDU) normalise the
read-out, and a real-fog mAP guardrail bounds task relevance.

This is a **measurement** release: it lets you recompute the reported
quantities from gate outputs and verify every table number directly. It is
**not** a detector or a training framework.

> **Status.** This repository accompanies a paper under submission. It is kept private during review and will be **made public upon submission**.

## Layout
- `src/gate/analysis/` — condition-signal extraction (`physics_priors.py`:
  dark channel, transmission, TSP, rank, RGB-derived controls), the
  gate-output probe (`di_signal_ladder_probe.py`: MMD / KS / CDU ratio with
  absolute numerator and denominator), input-side DI (`di_evaluation.py`),
  the domain-classifier AUC (`gate_domain_classifier_probe.py`), dose-response
  (`e036_*`), and the dehaze counterexample prep (`dcp_dehaze_dir.py`).
- `src/gate/models/` — the gate definitions (`dadg.py`, `patch_mlp_gate.py`).
- `src/core/` — degradation / data utilities.
- `repro/` — content-covariate matching (`task3_covariate_probe.py`) and the
  DI bootstrap (`task5_di_bootstrap.py`).
- `results/` — seed-level output summaries (JSON/CSV) behind every table.

## Paper table -> output file
| Paper | File(s) in `results/` |
|---|---|
| Table 1 (nine-condition ladder), Table 2 (input DI) | `e032_di_signal_ladder/e032_gate_probe_summary.{json,csv}`, `e032_di_signal_ladder/e032_signal_ladder_summary.json` (`di_mmd`/`di_ks`) |
| Table 3 (multi-covariate content matching) | `task3_covariate_full.json` (per-covariate SMD, n_pairs, per-condition num/den/ratio, 5 seeds) |
| DI bootstrap (ordering robustness, sec. 5) | `di_bootstrap.json` |
| Domain-classifier AUC | `e034_condition_control_probe/gate_domain_classifier_summary.{json,csv}` |
| Match-tolerance sensitivity (5% -> 2%) | `e032_tol002/`, `e032_tol003/` |
| Dose-response calibration | `e036_di_dose_response/` |
| Planted-manipulation study | `e037_spike_recovery_dose2/`, `e037_spike_recovery_dose4/` |
| Non-convolutional (patch-MLP) instrument read | `task4pm_probe/` |

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Data and weights are not redistributed; see `DATA.md`.

## License
MIT (this repository). The training pipeline derives from Ultralytics
(AGPL-3.0) and is intentionally excluded; see `DATA.md`.

## Citation
> Measuring Gate-Conditioning Shortcuts in Synthetic-Fog Detection: A
> Cross-Density Diagnostic Protocol. (bibtex to be added on acceptance.)
