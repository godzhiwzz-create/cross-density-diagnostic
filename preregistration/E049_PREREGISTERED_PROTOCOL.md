# E049 — Preregistered Protocol: E048 Density-Sensitivity Regularizer, Significance Expansion

Frozen 2026-06-21 **before any new-seed result is computed** (sha256 recorded at launch in `e049_run.log` + DECISION_LOG). Addresses reviewer demand (GPT #1) for a CDU↔real-fog-performance coupling that reaches significance, by powering the existing 5-seed E048 result (RTTS 5/5, sign-flip p=0.0625 — the 5-seed floor).

> **2026-06-22 口径修正(出有效结果前,§0(C) 已报备)**:初稿误写 `batch=96`(= `run_tsp_dadg.py` 脚本默认值)。核对既有 E048 `tsp_dadg_rgb_e048f1000/seed42/args.yaml` 单一真相 → 成功的 5 seed 实为 **`batch=16, workers=8`**。为与既有 seed 同口径(否则新 seed 不可比、配对比较作废),改为 `batch=16, workers=8`。此修正发生在**任何有效新-seed 结果产出之前**(此前所有 launch 因 batch=96 近-OOM 全 hang、零产出),非 p-hacking。

## Hypothesis (directional, registered)
Under the fixed E048 protocol, the density-sensitivity regularizer (`reg_w=1000`) improves RTTS real-fog mAP50 vs `reg_w=0` in the predicted (positive) direction; across 10 seeds a paired test reaches p<0.05.

## Design (frozen — §0(C) no口径 change)
- Two arms, identical protocol except the regularizer weight:
  - Baseline `reg_w=0`  → dir `tsp_dadg_rgb_e048f0`
  - Intervention `reg_w=1000` → dir `tsp_dadg_rgb_e048f1000`
- Existing seeds (already run, 5): **42, 123, 456, 789, 2024**.
- NEW seeds (registered here, 5): **7, 17, 71, 137, 271** → 10 seeds total.
- Fixed config (= E048, unchanged): `variant=rgb`, teacher=YOLOv8l (cnn), data=`dataset_foggy_all_dlhost.yaml`, student=`yolov8n.pt`, `epochs=20`, **`batch=16`, `workers=8`**, `imgsz=640`, `gate-arch=conv`, `device=0`, `cache=false`, `amp=true`. Trainer = `dadg_trainer.py` (E048 patch in place). 全部对齐 `seed42/args.yaml`。
- Train (per arm): `run_tsp_dadg.py --variant rgb --density-reg-w {0|1000} --project-suffix _e048f{0|1000} --seeds 7 17 71 137 271 --epochs 20 --batch 16 --workers 8 --model <完整 yolov8n.pt 全路径>`
- Eval (per arm): `run_tsp_dadg_eval.py --variant rgb --run-suffix _e048f{0|1000} --seeds 7 17 71 137 271` → RTTS + Foggy Driving `external_metrics.json` (`results_dict["metrics/mAP50(B)"]`).
- Per-seed delta = RTTS mAP50(reg1000, seed) − RTTS mAP50(reg0, same seed).

## Primary endpoint & test (frozen)
- **Primary**: paired **Wilcoxon signed-rank** on the **10** per-seed RTTS-mAP50 deltas, two-sided.
- **Success**: p<0.05 **AND** median delta > 0.
- **Secondary (reported regardless)**: exact two-sided sign-flip; paired t-test; Foggy Driving and real-fog-avg (RTTS+FD)/2 deltas; CDU per arm (gate probe).

## Failure handling (no abstract drawer — preregistration)
If primary p≥0.05: report as **"directional improvement, not significant at 10 seeds"** and keep the proof-of-concept framing. **Result is reported either way.** Seeds / primary test / success criterion are **NOT** changed after seeing results.

## Explicitly NOT done (no p-hacking)
No adding/dropping seeds by result; no switching the primary test post-hoc; no tuning `reg_w` or any config. The 5 new seeds and the Wilcoxon primary are fixed here.
