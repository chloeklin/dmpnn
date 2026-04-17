"""Publication-quality plots comparing mixture-based pairwise copolymer strategies.

Compares mixture (baseline), mixture_pairwise, and mixture_pairwise_attention
across DMPNN, GIN, GAT models on EA/IP targets under a_held_out split.

Generates:
  1. Absolute RMSE
  2. ΔRMSE vs mixture  (negative = better)
  3. ΔR² vs mixture    (positive = better)
  4. Scatter: RMSE_method vs RMSE_mixture  (diagnostic)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
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
    "mix_pair_meta": "mix_pair",
    "mix_pair_attn_meta": "mix_pair_attn",
}
BASELINE = "mixture"
METHOD_ORDER = ["mixture", "mix_pair", "mix_pair_attn"]
DELTA_METHOD_ORDER = ["mix_pair", "mix_pair_attn"]
SPLIT = "a_held_out"

MODEL_PALETTE = {"DMPNN": "#1b9e77", "GIN": "#d95f02", "GAT": "#7570b3"}


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _build_filename(
    model: str,
    mode: str,
    target: str,
    split: str = SPLIT,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    return (
        results_dir
        / model
        / f"ea_ip__copoly_{mode}__poly_type__{split}__target_{target}_results.csv"
    )


def load_results(
    results_dir: Path = RESULTS_DIR,
    models: list[str] = MODELS,
    mode_map: dict[str, str] = MODE_MAP,
    targets: dict[str, str] = TARGETS,
    split: str = SPLIT,
) -> pd.DataFrame:
    """Load all result CSVs into a tidy DataFrame.

    Returns DataFrame with columns: model, target, method, fold, rmse, r2, mae
    """
    rows = []
    for model in models:
        for mode_internal, method_label in mode_map.items():
            for target_full, target_short in targets.items():
                fpath = _build_filename(model, mode_internal, target_full, split, results_dir)
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

    Returns a copy of non-baseline rows with an extra ``delta_{metric}`` column.
    """
    baseline_df = (
        df[df["method"] == baseline]
        .set_index(["model", "target", "fold"])[[metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )
    merged = df.merge(baseline_df, on=["model", "target", "fold"], how="inner")
    merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged[merged["method"] != baseline].copy()


# ──────────────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────────────

def setup_style():
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


def _unified_legend(axes: list[plt.Axes], model_order: list[str] = MODELS):
    """Collect handles from all axes and place a single unified legend."""
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    ordered = [m for m in model_order if m in seen]
    axes[-1].legend(
        [seen[m] for m in ordered],
        ordered,
        title="Model",
        loc="best",
        framealpha=0.9,
    )


# ──────────────────────────────────────────────────────────────────────
# Task 1: Absolute metric
# ──────────────────────────────────────────────────────────────────────

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
    """Box + strip plot of absolute metric, one subplot per target."""
    setup_style()
    method_order = method_order or [m for m in METHOD_ORDER if m in df["method"].unique()]

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt)
            continue
        hue_order = [m for m in MODELS if m in sub["model"].unique()]

        sns.boxplot(
            data=sub, x="method", y=metric, hue="model",
            order=method_order, hue_order=hue_order, palette=palette,
            width=0.6, linewidth=0.8, fliersize=0, ax=ax,
        )
        sns.stripplot(
            data=sub, x="method", y=metric, hue="model",
            order=method_order, hue_order=hue_order, palette=palette,
            dodge=True, size=5, alpha=0.7, edgecolor="black", linewidth=0.4,
            ax=ax, legend=False,
        )
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax == axes[0] else "")

    _sync_ylims(axes)
    _unified_legend(axes)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Tasks 2 & 3: Delta plots
# ──────────────────────────────────────────────────────────────────────

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
    """Box + strip plot of Δmetric vs baseline, one subplot per target."""
    setup_style()
    method_order = method_order or [m for m in DELTA_METHOD_ORDER if m in df["method"].unique()]
    col_name = f"delta_{metric}"

    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt)
            continue
        hue_order = [m for m in MODELS if m in sub["model"].unique()]

        sns.boxplot(
            data=sub, x="method", y=col_name, hue="model",
            order=method_order, hue_order=hue_order, palette=palette,
            width=0.6, linewidth=0.8, fliersize=0, ax=ax,
        )
        sns.stripplot(
            data=sub, x="method", y=col_name, hue="model",
            order=method_order, hue_order=hue_order, palette=palette,
            dodge=True, size=5, alpha=0.7, edgecolor="black", linewidth=0.4,
            ax=ax, legend=False,
        )
        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax == axes[0] else "")

    _sync_ylims(axes)
    _unified_legend(axes)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 4: Scatter vs baseline
# ──────────────────────────────────────────────────────────────────────

def plot_scatter_vs_baseline(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    metric: str = "rmse",
    xlabel: str = "RMSE_mixture (eV)",
    ylabel: str = "RMSE_method (eV)",
    title: str = "Scatter: method vs mixture",
    output_name: str = "scatter_rmse_vs_mixture.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    method_order: Optional[list[str]] = None,
):
    """Per-fold scatter of method RMSE vs baseline RMSE, one subplot per target."""
    setup_style()
    method_order = method_order or [m for m in DELTA_METHOD_ORDER if m in df["method"].unique()]

    baseline_df = (
        df[df["method"] == baseline]
        .set_index(["model", "target", "fold"])[[metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )
    non_base = df[df["method"] != baseline].copy()
    merged = non_base.merge(baseline_df, on=["model", "target", "fold"], how="inner")

    n_methods = len(method_order)
    fig, axes = plt.subplots(
        n_methods, len(targets),
        figsize=(5 * len(targets), 4.5 * n_methods),
        squeeze=False,
    )

    for col_idx, tgt in enumerate(targets):
        for row_idx, method in enumerate(method_order):
            ax = axes[row_idx, col_idx]
            sub = merged[(merged["target"] == tgt) & (merged["method"] == method)]
            if sub.empty:
                ax.set_title(f"{tgt} — {method}")
                continue

            for mdl in MODELS:
                m_sub = sub[sub["model"] == mdl]
                if m_sub.empty:
                    continue
                ax.scatter(
                    m_sub[f"{metric}_baseline"], m_sub[metric],
                    color=palette[mdl], label=mdl,
                    s=60, alpha=0.8, edgecolors="black", linewidths=0.4,
                )

            # Diagonal y=x
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            margin = (hi - lo) * 0.03
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    color="grey", linewidth=1.0, linestyle="--", zorder=0)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_aspect("equal")

            ax.set_title(f"{tgt} — {method}", fontweight="bold")
            ax.set_xlabel(xlabel if row_idx == n_methods - 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")

    # Single legend from last axes
    _unified_legend([axes[0, -1]])
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

    # Summary table
    summary = (
        df.groupby(["target", "method", "model"])
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             r2_mean=("r2", "mean"), r2_std=("r2", "std"), n=("fold", "count"))
        .round(4)
    )
    print("\n=== Summary ===")
    print(summary.to_string())
    print()

    # Task 1: Absolute RMSE
    plot_absolute(
        df, metric="rmse", ylabel="RMSE (eV)",
        title="Absolute RMSE by method and model",
        output_name="mix_pair_absolute_rmse.png",
    )

    # Task 2: ΔRMSE
    delta_rmse = compute_deltas(df, baseline=BASELINE, metric="rmse")
    plot_delta(
        delta_rmse, metric="rmse",
        title="ΔRMSE vs mixture  (negative = better)",
        ylabel="ΔRMSE (eV)",
        output_name="mix_pair_delta_rmse.png",
    )

    # Task 3: ΔR²
    delta_r2 = compute_deltas(df, baseline=BASELINE, metric="r2")
    plot_delta(
        delta_r2, metric="r2",
        title="ΔR² vs mixture  (positive = better)",
        ylabel="ΔR²",
        output_name="mix_pair_delta_r2.png",
    )

    # Task 4: Scatter
    plot_scatter_vs_baseline(
        df, baseline=BASELINE, metric="rmse",
        xlabel="RMSE_mixture (eV)", ylabel="RMSE_method (eV)",
        title="Scatter: method RMSE vs mixture RMSE",
        output_name="mix_pair_scatter_rmse.png",
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
