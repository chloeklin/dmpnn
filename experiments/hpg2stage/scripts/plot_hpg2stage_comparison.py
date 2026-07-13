#!/usr/bin/env python3
"""Plot HPG-2Stage comparison: one figure per metric, subplots per target.

Usage:
    python scripts/python/plot_hpg2stage_comparison.py

Output:
    results/hpg2stage_rmse.png
    results/hpg2stage_r2.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Paths ────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
TARGET_SHORT = {"EA vs SHE (eV)": "EA", "IP vs SHE (eV)": "IP"}

# ── Model registry (same as build_hpg2stage_comparison_table.py) ────
MODELS = {
    "HPG_frac\n(+poly_type)": {
        "dir": "HPG",
        "pattern": "ea_ip__hpg_frac_polytype__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
    "wDMPNN": {
        "dir": "wDMPNN",
        "pattern": "ea_ip__a_held_out_results.csv",
        "per_target": False,
    },
    "HPG-2Stage\nFrac": {
        "dir": "HPG2Stage",
        "fallback_dir": "DMPNN",
        "pattern": "ea_ip__copoly_mix_meta__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
    "HPG-2Stage\nInteract-fixed": {
        "dir": "HPG2Stage",
        "fallback_dir": "DMPNN",
        "pattern": "ea_ip__copoly_mix_pair_meta__fusion_sum_fusion__poly_type__a_held_out__target_{target}_results.csv",
        "fallback": "ea_ip__copoly_mix_pair_meta__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
    "HPG-2Stage\nInteract-learned": {
        "dir": "HPG2Stage",
        "fallback_dir": "DMPNN",
        "pattern": "ea_ip__copoly_mix_pair_meta__fusion_scalar_residual_fusion__poly_type__a_held_out__target_{target}_results.csv",
        "per_target": True,
    },
}

COLORS = {
    "HPG_frac\n(+poly_type)": "#9e9e9e",
    "wDMPNN": "#78909c",
    "HPG-2Stage\nFrac": "#42a5f5",
    "HPG-2Stage\nInteract-fixed": "#66bb6a",
    "HPG-2Stage\nInteract-learned": "#ef5350",
}


def load_target_df(spec: dict, target: str) -> pd.DataFrame | None:
    dirs_to_try = [RESULTS_DIR / spec["dir"]]
    if "fallback_dir" in spec:
        dirs_to_try.append(RESULTS_DIR / spec["fallback_dir"])

    if spec["per_target"]:
        for base in dirs_to_try:
            path = base / spec["pattern"].format(target=target)
            if path.exists():
                break
            if "fallback" in spec:
                path = base / spec["fallback"].format(target=target)
                if path.exists():
                    break
        else:
            return None
        if not path.exists():
            return None
        df = pd.read_csv(path)
    else:
        path = None
        for base in dirs_to_try:
            candidate = base / spec["pattern"]
            if candidate.exists():
                path = candidate
                break
        if path is None:
            return None
        df = pd.read_csv(path)
        df = df[df["target"] == target]
    return df if not df.empty else None


def collect_data() -> dict:
    """Return {model_name: {target_short: {metric: (mean, std)}}}."""
    data = {}
    for name, spec in MODELS.items():
        data[name] = {}
        for target in TARGETS:
            short = TARGET_SHORT[target]
            df = load_target_df(spec, target)
            if df is None:
                data[name][short] = None
                continue
            data[name][short] = {
                "rmse": (df["test/rmse"].mean(), df["test/rmse"].std()),
                "r2": (df["test/r2"].mean(), df["test/r2"].std()),
            }
    return data


def plot_metric(data: dict, metric: str, ylabel: str, out_path: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Exclude HPG_frac from bars (off-scale); annotate instead
    bar_models = [n for n in MODELS if n != "HPG_frac\n(+poly_type)"]
    targets = ["EA", "IP"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, tgt in zip(axes, targets):
        means, stds, colors = [], [], []
        for name in bar_models:
            entry = data[name].get(tgt)
            if entry is None:
                means.append(0)
                stds.append(0)
                colors.append("#cccccc")
            else:
                means.append(entry[metric][0])
                stds.append(entry[metric][1])
                colors.append(COLORS[name])

        x = np.arange(len(bar_models))
        ax.bar(
            x, means, yerr=stds, width=0.55,
            color=colors, edgecolor="white", linewidth=0.8,
            capsize=5, error_kw={"linewidth": 1.3, "capthick": 1.3},
        )

        yhi = max(m + s for m, s in zip(means, stds)) * 1.25
        if metric == "r2":
            ylo_floor = min(m - s for m, s in zip(means, stds)) * 0.97
            ax.set_ylim(ylo_floor, min(yhi, 1.05))
        else:
            ax.set_ylim(0, yhi)

        for i, (m, s) in enumerate(zip(means, stds)):
            if data[bar_models[i]].get(tgt) is not None:
                ax.text(
                    i, m + s + yhi * 0.015, f"{m:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="medium",
                )

        # HPG_frac annotation
        hpg_entry = data["HPG_frac\n(+poly_type)"].get(tgt)
        if hpg_entry is not None:
            hm, hs = hpg_entry[metric]
            ax.text(
                0.98, 0.95,
                f"HPG_frac: {hm:.3f} $\\pm$ {hs:.3f}\n(off-scale)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="#757575", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bdbdbd", alpha=0.9),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bar_models, fontsize=9.5)
        ax.set_title(f"{tgt} vs SHE (eV)", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel if ax == axes[0] else "", fontsize=11)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle(
        f"HPG-2Stage Comparison — {ylabel}  (EA/IP, monomer-disjoint split)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> None:
    data = collect_data()
    plot_metric(data, "rmse", "RMSE (eV)", RESULTS_DIR / "hpg2stage_rmse.png")
    plot_metric(data, "r2", "R²", RESULTS_DIR / "hpg2stage_r2.png")


if __name__ == "__main__":
    main()
