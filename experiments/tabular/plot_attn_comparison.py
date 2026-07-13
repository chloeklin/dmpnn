"""Publication-quality plots comparing copolymer fusion strategies.

Compares mixture (baseline), attention, and fraction_aware_attention
across DMPNN, GIN, GAT models on EA/IP targets under a_held_out split.

Generates:
  1. ΔRMSE vs mixture  (negative = better)
  2. ΔR² vs mixture    (positive = better)
  3. Absolute RMSE
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "ea_ip_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["DMPNN", "GIN", "GAT"]
TARGETS = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}
# Internal mode name → display label
MODE_MAP = {
    "mix_meta": "mixture",
    "attention_meta": "attention",
    "frac_attn_meta": "frac_attn",
}
BASELINE = "mixture"
SPLIT = "a_held_out"

# Consistent palette across all plots
MODEL_PALETTE = {"DMPNN": "#1b9e77", "GIN": "#d95f02", "GAT": "#7570b3"}

# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _build_filename(model: str, mode: str, target: str) -> Path:
    """Construct the expected result CSV path."""
    return (
        RESULTS_DIR
        / model
        / f"ea_ip__copoly_{mode}__poly_type__{SPLIT}__target_{target}_results.csv"
    )


def load_results(
    results_dir: Path = RESULTS_DIR,
    models: list[str] = MODELS,
    mode_map: dict[str, str] = MODE_MAP,
    targets: dict[str, str] = TARGETS,
) -> pd.DataFrame:
    """Load all result CSVs into a tidy DataFrame.

    Returns
    -------
    DataFrame with columns: model, target, method, fold, rmse, r2, mae
    """
    rows = []
    for model in models:
        for mode_internal, method_label in mode_map.items():
            for target_full, target_short in targets.items():
                fpath = _build_filename(model, mode_internal, target_full)
                if not fpath.exists():
                    print(f"  SKIP (missing): {fpath.name}")
                    continue
                df = pd.read_csv(fpath)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "model": model,
                            "target": target_short,
                            "method": method_label,
                            "fold": int(r["split"]),
                            "rmse": float(r["test/rmse"]),
                            "r2": float(r["test/r2"]),
                            "mae": float(r["test/mae"]),
                        }
                    )
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows  ({df['model'].nunique()} models, "
          f"{df['target'].nunique()} targets, {df['method'].nunique()} methods)")
    return df


# ──────────────────────────────────────────────────────────────────────
# Delta computation
# ──────────────────────────────────────────────────────────────────────

def compute_deltas(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    metric: str = "rmse",
) -> pd.DataFrame:
    """Compute per-fold delta relative to baseline method.

    Returns a copy with an extra column ``delta_{metric}``.
    """
    baseline_df = df[df["method"] == baseline].set_index(["model", "target", "fold"])[
        [metric]
    ].rename(columns={metric: f"{metric}_baseline"})

    merged = df.merge(baseline_df, on=["model", "target", "fold"], how="inner")
    merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged[merged["method"] != baseline].copy()


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────

def _setup_style():
    """Set global matplotlib / seaborn style for publication plots."""
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.5,
        }
    )


def _sync_ylims(axes: list[plt.Axes]):
    """Force all axes to share the same y-limits."""
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    margin = (ymax - ymin) * 0.05
    for ax in axes:
        ax.set_ylim(ymin - margin, ymax + margin)


def plot_delta(
    df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "",
    ylabel: str = "",
    output_name: str = "delta_rmse.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    method_order: Optional[list[str]] = None,
):
    """Box + strip plot of Δmetric vs baseline, one subplot per target.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: model, target, method, fold, delta_{metric}.
    """
    _setup_style()
    if method_order is None:
        method_order = [m for m in ["attention", "frac_attn"] if m in df["method"].unique()]

    col_name = f"delta_{metric}"
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt)
            continue

        sns.boxplot(
            data=sub,
            x="method",
            y=col_name,
            hue="model",
            order=method_order,
            hue_order=[m for m in MODELS if m in sub["model"].unique()],
            palette=palette,
            width=0.6,
            linewidth=0.8,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="method",
            y=col_name,
            hue="model",
            order=method_order,
            hue_order=[m for m in MODELS if m in sub["model"].unique()],
            palette=palette,
            dodge=True,
            size=5,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
            ax=ax,
            legend=False,
        )
        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        # Keep only first axis legend
        if ax != axes[-1]:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    _sync_ylims(axes)

    # Collect handles across all subplots to show every model in legend
    all_handles, all_labels = {}, {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in all_labels:
                all_handles[li] = hi
                all_labels[li] = li
    # Remove per-axis legends; place single unified legend
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    ordered = [m for m in MODELS if m in all_labels]
    axes[-1].legend(
        [all_handles[m] for m in ordered],
        ordered,
        title="Model",
        loc="best",
        framealpha=0.9,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_absolute(
    df: pd.DataFrame,
    metric: str = "rmse",
    ylabel: str = "RMSE (eV)",
    title: str = "Absolute RMSE",
    output_name: str = "absolute_rmse.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    method_order: Optional[list[str]] = None,
):
    """Absolute metric values — box + strip, one subplot per target."""
    _setup_style()
    if method_order is None:
        method_order = [m for m in ["mixture", "attention", "frac_attn"]
                        if m in df["method"].unique()]

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt)
            continue

        sns.boxplot(
            data=sub,
            x="method",
            y=metric,
            hue="model",
            order=method_order,
            hue_order=[m for m in MODELS if m in sub["model"].unique()],
            palette=palette,
            width=0.6,
            linewidth=0.8,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="method",
            y=metric,
            hue="model",
            order=method_order,
            hue_order=[m for m in MODELS if m in sub["model"].unique()],
            palette=palette,
            dodge=True,
            size=5,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
            ax=ax,
            legend=False,
        )
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        if ax != axes[-1]:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    _sync_ylims(axes)

    # Collect handles across all subplots
    all_handles, all_labels = {}, {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in all_labels:
                all_handles[li] = hi
                all_labels[li] = li
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    ordered = [m for m in MODELS if m in all_labels]
    axes[-1].legend(
        [all_handles[m] for m in ordered],
        ordered,
        title="Model",
        loc="best",
        framealpha=0.9,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    df = load_results()
    if df.empty:
        print("No data found — check results directory.")
        return

    # Print summary table
    summary = (
        df.groupby(["target", "method", "model"])
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             r2_mean=("r2", "mean"), r2_std=("r2", "std"), n=("fold", "count"))
        .round(4)
    )
    print("\n=== Summary ===")
    print(summary.to_string())
    print()

    # Task 1: ΔRMSE
    delta_rmse = compute_deltas(df, baseline=BASELINE, metric="rmse")
    plot_delta(
        delta_rmse,
        metric="rmse",
        title="ΔRMSE vs mixture  (negative = better)",
        ylabel="ΔRMSE (eV)",
        output_name="delta_rmse_vs_mixture.png",
    )

    # Task 2: ΔR²
    delta_r2 = compute_deltas(df, baseline=BASELINE, metric="r2")
    plot_delta(
        delta_r2,
        metric="r2",
        title="ΔR² vs mixture  (positive = better)",
        ylabel="ΔR²",
        output_name="delta_r2_vs_mixture.png",
    )

    # Task 3: Absolute RMSE
    plot_absolute(
        df,
        metric="rmse",
        ylabel="RMSE (eV)",
        title="Absolute RMSE by method and model",
        output_name="absolute_rmse.png",
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
