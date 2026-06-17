"""Gate-output domain-classifier and bootstrap probe.

This reviewer-strengthening probe complements the MMD/KS gate-output shortcut
ratio. It asks whether the same trained gate weights are directly
domain-classifiable as real fog versus synthetic fog, and reports bootstrap CIs
for the MMD/KS real-vs-synthetic over synthetic-cross-density ratios.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from di_signal_ladder_probe import (  # noqa: E402
    DATASETS,
    EXP_ROOT,
    SYNTH_TAGS,
    gate_outputs,
    load_gate,
    mean_ks,
    mmd_rbf,
    resolve_images,
)


def sample_paths(tag: str, max_images: int, seed: int) -> list[str]:
    paths = resolve_images(DATASETS[tag])
    if max_images > 0 and len(paths) > max_images:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(paths), size=max_images, replace=False).tolist())
        paths = [paths[i] for i in idx]
    return [str(p) for p in paths]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comp = pos[:, None] - neg[None, :]
    return float((comp > 0).mean() + 0.5 * (comp == 0).mean())


def ci(values: list[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float | None, float | None]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return None, None
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def split_balanced(
    class0: np.ndarray,
    class1: np.ndarray,
    *,
    seed: int,
    train_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(len(class0), len(class1))
    if n < 10:
        raise ValueError("Need at least 10 samples per class for classifier probe.")
    idx0 = rng.choice(len(class0), size=n, replace=False)
    idx1 = rng.choice(len(class1), size=n, replace=False)
    n_train = max(4, min(n - 2, int(n * train_frac)))
    tr0, te0 = idx0[:n_train], idx0[n_train:]
    tr1, te1 = idx1[:n_train], idx1[n_train:]
    x_train = np.concatenate([class0[tr0], class1[tr1]], axis=0)
    y_train = np.concatenate([np.zeros(len(tr0)), np.ones(len(tr1))]).astype(int)
    x_test = np.concatenate([class0[te0], class1[te1]], axis=0)
    y_test = np.concatenate([np.zeros(len(te0)), np.ones(len(te1))]).astype(int)
    order = rng.permutation(len(y_train))
    return x_train[order], y_train[order], x_test, y_test


def logistic_probe(
    class0: np.ndarray,
    class1: np.ndarray,
    *,
    seed: int,
    train_frac: float,
    steps: int,
    lr: float,
    l2: float,
    bootstrap: int,
    permutations: int,
) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = split_balanced(
        class0, class1, seed=seed, train_frac=train_frac
    )
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_train = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)
    w = np.zeros(x_train.shape[1], dtype=np.float64)
    for _ in range(steps):
        p = sigmoid(x_train @ w)
        grad = x_train.T @ (p - y_train) / len(y_train)
        grad[:-1] += l2 * w[:-1]
        w -= lr * grad
    scores = sigmoid(x_test @ w)
    pred = (scores >= 0.5).astype(int)
    acc = float((pred == y_test).mean())
    auc = auc_binary(y_test, scores)

    rng = np.random.default_rng(seed + 100003)
    acc_boot: list[float] = []
    auc_boot: list[float] = []
    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    for _ in range(bootstrap):
        bi = np.concatenate([
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
        ])
        acc_boot.append(float(((scores[bi] >= 0.5).astype(int) == y_test[bi]).mean()))
        auc_boot.append(auc_binary(y_test[bi], scores[bi]))

    perm_aucs: list[float] = []
    for _ in range(permutations):
        perm_aucs.append(auc_binary(rng.permutation(y_test), scores))
    p_value = None
    if perm_aucs:
        obs = abs(auc - 0.5)
        p_value = float((sum(abs(x - 0.5) >= obs for x in perm_aucs) + 1) / (len(perm_aucs) + 1))

    acc_lo, acc_hi = ci(acc_boot)
    auc_lo, auc_hi = ci(auc_boot)
    return {
        "acc": acc,
        "auc": auc,
        "acc_ci_low": acc_lo,
        "acc_ci_high": acc_hi,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "auc_perm_p": p_value,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def ratio_metrics(real: np.ndarray, synth_by_tag: dict[str, np.ndarray]) -> dict[str, float | None]:
    mmd_rs = [mmd_rbf(real, synth_by_tag[tag]) for tag in SYNTH_TAGS]
    ks_rs = [mean_ks(real, synth_by_tag[tag]) for tag in SYNTH_TAGS]
    mmd_ss: list[float] = []
    ks_ss: list[float] = []
    for i, tag_i in enumerate(SYNTH_TAGS):
        for tag_j in SYNTH_TAGS[i + 1:]:
            mmd_ss.append(mmd_rbf(synth_by_tag[tag_i], synth_by_tag[tag_j]))
            ks_ss.append(mean_ks(synth_by_tag[tag_i], synth_by_tag[tag_j]))
    mmd_num = float(np.mean(mmd_rs))
    mmd_den = float(np.mean(mmd_ss))
    ks_num = float(np.mean(ks_rs))
    ks_den = float(np.mean(ks_ss))
    return {
        "mmd_real_synth": mmd_num,
        "mmd_synth_synth": mmd_den,
        "mmd_ratio": mmd_num / mmd_den if mmd_den > 1e-12 else None,
        "ks_real_synth": ks_num,
        "ks_synth_synth": ks_den,
        "ks_ratio": ks_num / ks_den if ks_den > 1e-12 else None,
    }


def bootstrap_ratio_metrics(
    real: np.ndarray,
    synth_by_tag: dict[str, np.ndarray],
    *,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 200003)
    mmd_vals: list[float] = []
    ks_vals: list[float] = []
    for _ in range(bootstrap):
        real_b = real[rng.integers(0, len(real), size=len(real))]
        synth_b = {
            tag: vals[rng.integers(0, len(vals), size=len(vals))]
            for tag, vals in synth_by_tag.items()
        }
        metrics = ratio_metrics(real_b, synth_b)
        if metrics["mmd_ratio"] is not None:
            mmd_vals.append(float(metrics["mmd_ratio"]))
        if metrics["ks_ratio"] is not None:
            ks_vals.append(float(metrics["ks_ratio"]))
    mmd_lo, mmd_hi = ci(mmd_vals)
    ks_lo, ks_hi = ci(ks_vals)
    return {
        "mmd_ratio_ci_low": mmd_lo,
        "mmd_ratio_ci_high": mmd_hi,
        "ks_ratio_ci_low": ks_lo,
        "ks_ratio_ci_high": ks_hi,
    }


def permutation_metric_p(
    real: np.ndarray,
    synth_by_tag: dict[str, np.ndarray],
    *,
    metric_name: str,
    observed: float,
    seed: int,
    permutations: int,
) -> float | None:
    if permutations <= 0 or not np.isfinite(observed):
        return None
    rng = np.random.default_rng(seed + 300003 + (0 if metric_name == "mmd" else 17))
    perm_vals: list[float] = []
    metric = mmd_rbf if metric_name == "mmd" else mean_ks
    for _ in range(permutations):
        vals = []
        for tag in SYNTH_TAGS:
            synth = synth_by_tag[tag]
            pooled = np.concatenate([real, synth], axis=0)
            order = rng.permutation(len(pooled))
            fake_real = pooled[order[: len(real)]]
            fake_synth = pooled[order[len(real) : len(real) + len(synth)]]
            vals.append(metric(fake_real, fake_synth))
        perm_vals.append(float(np.mean(vals)))
    return float((sum(v >= observed for v in perm_vals) + 1) / (len(perm_vals) + 1))


def summarize_values(values: list[float]) -> tuple[int, float | None, float | None]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return 0, None, None
    return int(len(vals)), float(vals.mean()), float(vals.std())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["rgb", "raw_dark_channel", "raw_transmission", "tsp_grad_mag", "tsp_rank"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--bootstrap", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=EXP_ROOT / "summaries/e034_condition_control_probe")
    args = parser.parse_args()

    import torch

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    paths_by_tag = {
        "rtts": sample_paths("rtts", args.max_images, 42),
        **{tag: sample_paths(tag, args.max_images, 42) for tag in SYNTH_TAGS},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = EXP_ROOT / f"tsp_dadg_{variant}{args.run_suffix}" / f"seed{seed}"
            try:
                gate, cfg = load_gate(run_dir, device=device)
            except FileNotFoundError as exc:
                raw_rows.append({"variant": variant, "seed": seed, "missing": str(exc)})
                print(f"[missing] {exc}", flush=True)
                continue

            real = gate_outputs(
                gate, cfg, paths_by_tag["rtts"],
                batch=args.batch, workers=args.workers, imgsz=args.imgsz, device=device
            )
            synth_by_tag = {
                tag: gate_outputs(
                    gate, cfg, paths_by_tag[tag],
                    batch=args.batch, workers=args.workers, imgsz=args.imgsz, device=device
                )
                for tag in SYNTH_TAGS
            }
            synth_all = np.concatenate([synth_by_tag[tag] for tag in SYNTH_TAGS], axis=0)

            metrics = ratio_metrics(real, synth_by_tag)
            boot = bootstrap_ratio_metrics(real, synth_by_tag, seed=seed, bootstrap=args.bootstrap)
            rs_probe = logistic_probe(
                synth_all, real,
                seed=seed, train_frac=args.train_frac, steps=args.steps,
                lr=args.lr, l2=args.l2, bootstrap=args.bootstrap, permutations=args.permutations,
            )

            synth_pair_metrics: list[dict[str, Any]] = []
            for i, tag_i in enumerate(SYNTH_TAGS):
                for tag_j in SYNTH_TAGS[i + 1:]:
                    synth_pair_metrics.append(logistic_probe(
                        synth_by_tag[tag_i], synth_by_tag[tag_j],
                        seed=seed + i * 31,
                        train_frac=args.train_frac, steps=args.steps,
                        lr=args.lr, l2=args.l2, bootstrap=args.bootstrap, permutations=args.permutations,
                    ))
            synth_acc = float(np.mean([m["acc"] for m in synth_pair_metrics]))
            synth_auc = float(np.mean([m["auc"] for m in synth_pair_metrics]))
            auc_excess_ratio = None
            if synth_auc > 0.5001:
                auc_excess_ratio = float(max(rs_probe["auc"] - 0.5, 0.0) / max(synth_auc - 0.5, 1e-6))

            row = {
                "variant": variant,
                "seed": seed,
                **metrics,
                **boot,
                "mmd_real_synth_perm_p": permutation_metric_p(
                    real, synth_by_tag, metric_name="mmd",
                    observed=float(metrics["mmd_real_synth"]), seed=seed, permutations=args.permutations,
                ),
                "ks_real_synth_perm_p": permutation_metric_p(
                    real, synth_by_tag, metric_name="ks",
                    observed=float(metrics["ks_real_synth"]), seed=seed, permutations=args.permutations,
                ),
                "real_synth_domain_acc": rs_probe["acc"],
                "real_synth_domain_auc": rs_probe["auc"],
                "real_synth_domain_acc_ci_low": rs_probe["acc_ci_low"],
                "real_synth_domain_acc_ci_high": rs_probe["acc_ci_high"],
                "real_synth_domain_auc_ci_low": rs_probe["auc_ci_low"],
                "real_synth_domain_auc_ci_high": rs_probe["auc_ci_high"],
                "real_synth_domain_auc_perm_p": rs_probe["auc_perm_p"],
                "synth_pair_domain_acc_mean": synth_acc,
                "synth_pair_domain_auc_mean": synth_auc,
                "domain_auc_excess_ratio": auc_excess_ratio,
                "n_real": len(real),
                "n_synth_total": len(synth_all),
            }
            raw_rows.append(row)
            print(f"[ok] {variant} seed{seed}: AUC={rs_probe['auc']:.4f} MMD={metrics['mmd_ratio']:.4f}", flush=True)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        if "missing" not in row:
            by_variant.setdefault(str(row["variant"]), []).append(row)
    metric_cols = [
        "mmd_ratio", "ks_ratio",
        "mmd_real_synth", "mmd_synth_synth",
        "ks_real_synth", "ks_synth_synth",
        "real_synth_domain_acc", "real_synth_domain_auc",
        "synth_pair_domain_acc_mean", "synth_pair_domain_auc_mean",
        "domain_auc_excess_ratio",
    ]
    for variant, rows in by_variant.items():
        out: dict[str, Any] = {"variant": variant, "n_seeds": len(rows)}
        for col in metric_cols:
            n, mean, std = summarize_values([float(r[col]) for r in rows if r.get(col) is not None])
            out[f"{col}_n"] = n
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = std
        summary_rows.append(out)

    raw_path = args.out_dir / "gate_domain_classifier_raw.csv"
    fieldnames = sorted({key for row in raw_rows for key in row})
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_path = args.out_dir / "gate_domain_classifier_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "experiment_id": "E034",
        "rows": summary_rows,
        "notes": {
            "class_1_for_real_synth": "RTTS real fog",
            "class_0_for_real_synth": "pooled synthetic beta splits",
            "distance_probe": "same gate-output MMD/KS numerator-denominator structure as E032, with bootstrap CI",
        },
    }
    (args.out_dir / "gate_domain_classifier_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary_rows, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
