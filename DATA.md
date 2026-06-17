# Data and model weights (not redistributed)

This repository ships the **measurement code** and the **seed-level output
summaries** (`results/`). The underlying images and trained weights are **not**
included for licensing/size reasons; obtain them from the original sources.

## Datasets
- **Foggy Cityscapes / Cityscapes** — https://www.cityscapes-dataset.com/
  (register; redistribution of images is not permitted by the dataset license).
- **RTTS (RESIDE)** — https://sites.google.com/view/reside-dehaze-datasets
  (real-world hazy detection set used as the real-fog evaluation).

Synthetic fog is rendered on clear Cityscapes images with the atmospheric
scattering model at densities beta in {0.005, 0.01, 0.02}; see
`src/core/degradation.py` and `src/gate/analysis/physics_priors.py`.

## Weights
- Teacher (YOLOv8l) and the gate checkpoints are not redistributed. The gate
  is a small module (`src/gate/models/dadg.py`, `patch_mlp_gate.py`) trainable
  from the teacher; the training pipeline derives from Ultralytics (AGPL-3.0)
  and is therefore kept out of this MIT repository (available on request).

## Verifying reported numbers without any of the above
Every number in the paper's diagnostic tables is in `results/` as JSON/CSV;
see `README.md` for the paper-table -> file map.
