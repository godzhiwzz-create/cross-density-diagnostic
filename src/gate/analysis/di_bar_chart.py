"""Produce DI ratio bar chart for P2 paper §3.3.

Reads ./gate/experiments_dlhost/di_evaluation/di_results.json
and produces a grouped bar chart (MMD + KS) sorted by DI(MMD) ascending,
with reference lines at DI=1.3 (strong invariance) and DI=2.0 (weak).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"


PRETTY = {
    "tsp_grad_mag": "TSP-grad-mag",
    "tsp_rank": "TSP-rank",
    "tsp_grad_dir": "TSP-grad-dir",
    "fdgp": "FDGP",
    "raw_transmission": "raw t(x)",
    "raw_dark_channel": "dark channel",
    "saturation": "saturation",
    "cap_extended": "CAP-ext",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default="./gate/experiments_dlhost/di_evaluation/di_results.json",
    )
    ap.add_argument(
        "--out-dir",
        default="./gate/experiments_dlhost/di_evaluation",
    )
    ap.add_argument(
        "--rgb-baseline-mmd",
        type=float,
        default=2.38,
        help="Paper-1 RGB baseline DI(MMD); drawn as a dashed horizontal reference.",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    # Sort ascending by DI(MMD) — smaller = more domain-invariant
    data_sorted = sorted(data, key=lambda x: x["di_mmd"])

    names = [PRETTY.get(d["prior"], d["prior"]) for d in data_sorted]
    mmd = np.array([d["di_mmd"] for d in data_sorted], dtype=float)
    ks = np.array(
        [d["di_ks"] if d["di_ks"] is not None and not math.isnan(d["di_ks"]) else np.nan
         for d in data_sorted], dtype=float,
    )

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    # Color code by invariance strength using MMD
    def color_for(v):
        if v < 1.3:
            return "#2a9d8f"  # strong (green)
        if v < 2.0:
            return "#e9c46a"  # weak (yellow)
        return "#e76f51"  # domain-specific (red)

    bar_mmd = ax.bar(
        x - width / 2, mmd, width, label="DI(MMD)",
        color=[color_for(v) for v in mmd], edgecolor="black", linewidth=0.4,
    )
    ks_safe = np.where(np.isnan(ks), 0.0, ks)
    bar_ks = ax.bar(
        x + width / 2, ks_safe, width, label="DI(KS)",
        color="#264653", alpha=0.85, edgecolor="black", linewidth=0.4,
    )

    # Mark NaN KS as hatched zero bars
    for i, v in enumerate(ks):
        if np.isnan(v):
            ax.bar(x[i] + width / 2, 0.05, width, color="none", edgecolor="gray",
                   hatch="////", linewidth=0.4)
            ax.text(x[i] + width / 2, 0.08, "N/A", ha="center", va="bottom",
                    fontsize=7, color="gray")

    # Reference lines
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.axhline(1.3, color="#2a9d8f", linewidth=0.8, linestyle="--", alpha=0.9,
               label="DI=1.3 (strong inv.)")
    ax.axhline(2.0, color="#e76f51", linewidth=0.8, linestyle="--", alpha=0.9,
               label="DI=2.0 (domain-spec.)")
    ax.axhline(args.rgb_baseline_mmd, color="gray", linewidth=0.8, linestyle="-.",
               alpha=0.7, label=f"RGB baseline (P1) DI={args.rgb_baseline_mmd}")

    # Numerical labels above bars
    for i, v in enumerate(mmd):
        ax.text(x[i] - width / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.5)
    for i, v in enumerate(ks):
        if not np.isnan(v):
            ax.text(x[i] + width / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#264653")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Domain Invariance ratio (lower = more invariant)")
    ax.set_title("Domain invariance of physical priors\n(Foggy ↔ RTTS ↔ Foggy Driving, 1000 imgs/domain)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    # Cap y for readability; cap_extended DI≈16.5 outlier
    ymax = 3.2
    ax.set_ylim(0, ymax)
    for i, v in enumerate(mmd):
        if v > ymax:
            ax.annotate(
                f"{v:.1f}",
                xy=(x[i] - width / 2, ymax - 0.05),
                xytext=(x[i] - width / 2, ymax - 0.3),
                ha="center", va="top", fontsize=8, color="#b33",
                arrowprops=dict(arrowstyle="->", color="#b33", lw=0.8),
            )

    plt.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "di_bar.png"
    pdf = out / "di_bar.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"Saved: {png}\n        {pdf}")


if __name__ == "__main__":
    main()
