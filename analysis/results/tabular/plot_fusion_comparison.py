"""Publication-quality plots comparing fusion strategies for mix_pair_attn.

Compares sum_fusion (baseline), concat_fusion, gated_fusion, and
scalar_residual_fusion across DMPNN, GIN, GAT models on EA/IP targets
under a_held_out split.

Generates:
  1. Absolute RMSE by fusion strategy
  2. ΔRMSE vs sum_fusion   (negative = better)
  3. ΔR² vs sum_fusion     (positive = better)
  4. Scatter: RMSE_fusion vs RMSE_sum_fusion  (diagnostic)
  5. Win-rate: fraction of folds improved over sum_fusion
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
OUTPUT_DIR = Path(__file__).resolve().parent / "fusion_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["DMPNN", "GIN", "GAT"]
TARGETS = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}

# Fusion type → display label
FUSION_MAP = {
    "sum_fusion": "sum",
    "concat_fusion": "concat",
    "gated_fusion": "gated",
    "scalar_residual_fusion": "scalar_res",
}
BASELINE = "sum"
FUSION_ORDER = ["sum", "concat", "gated", "scalar_res"]
DELTA_FUSION_ORDER = ["concat", "gated", "scalar_res"]
SPLIT = "a_held_out"

MODEL_PALETTE = {"DMPNN": "#1b9e77", "GIN": "#d95f02", "GAT": "#7570b3"}


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _build_filename(
    model: str,
    fusion_type: str,
    target: str,
    split: str = SPLIT,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Build the result CSV path for a given (model, fusion_type, target).

    sum_fusion files have no ``__fusion_`` segment in the filename (backward
    compatible with the original mix_pair_attn runs).
    """
    if fusion_type == "sum_fusion":
        fusion_seg = ""
    else:
        fusion_seg = f"__fusion_{fusion_type}"
    return (
        results_dir
        / model
        / f"ea_ip__copoly_mix_pair_attn_meta{fusion_seg}__poly_type__{split}__target_{target}_results.csv"
    )


def load_results(
    results_dir: Path = RESULTS_DIR,
    models: list[str] = MODELS,
    fusion_map: dict[str, str] = FUSION_MAP,
    targets: dict[str, str] = TARGETS,
    split: str = SPLIT,
) -> pd.DataFrame:
    """Load all result CSVs into a tidy DataFrame.

    Returns DataFrame with columns:
        model, target, fusion_type, fold, rmse, r2, mae
    """
    rows: list[dict] = []
    for model in models:
        for ft_internal, ft_label in fusion_map.items():
            for target_full, target_short in targets.items():
                fpath = _build_filename(model, ft_internal, target_full, split, results_dir)
                if not fpath.exists():
                    print(f"  SKIP (missing): {fpath.name}")
                    continue
                df = pd.read_csv(fpath)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "model": model,
                            "target": target_short,
                            "fusion_type": ft_label,
                            "fold": int(r["split"]),
                            "rmse": float(r["test/rmse"]),
                            "r2": float(r["test/r2"]),
                            "mae": float(r["test/mae"]),
                        }
                    )
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows  ({df['model'].nunique()} models, "
          f"{df['target'].nunique()} targets, {df['fusion_type'].nunique()} fusion types)")
    return df


# ──────────────────────────────────────────────────────────────────────
# Delta computation
# ──────────────────────────────────────────────────────────────────────

def compute_deltas(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    metric: str = "rmse",
) -> pd.DataFrame:
    """Compute per-fold delta relative to baseline fusion strategy.

    Returns a copy of non-baseline rows with an extra ``delta_{metric}`` column.
    """
    baseline_df = (
        df[df["fusion_type"] == baseline]
        .set_index(["model", "target", "fold"])[[metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )
    merged = df.merge(baseline_df, on=["model", "target", "fold"], how="inner")
    merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged[merged["fusion_type"] != baseline].copy()


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
    seen: dict = {}
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
# Task 1: Absolute RMSE
# ──────────────────────────────────────────────────────────────────────

def plot_absolute(
    df: pd.DataFrame,
    metric: str = "rmse",
    ylabel: str = "RMSE (eV)",
    title: str = "RMSE by fusion strategy",
    output_name: str = "fusion_absolute_rmse.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    fusion_order: Optional[list[str]] = None,
):
    """Box + strip plot of absolute metric, one subplot per target."""
    setup_style()
    fusion_order = fusion_order or [f for f in FUSION_ORDER if f in df["fusion_type"].unique()]

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
            data=sub, x="fusion_type", y=metric, hue="model",
            order=fusion_order, hue_order=hue_order, palette=palette,
            width=0.6, linewidth=0.8, fliersize=0, ax=ax,
        )
        sns.stripplot(
            data=sub, x="fusion_type", y=metric, hue="model",
            order=fusion_order, hue_order=hue_order, palette=palette,
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
    output_name: str = "fusion_delta_rmse.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    fusion_order: Optional[list[str]] = None,
):
    """Box + strip plot of Δmetric vs baseline, one subplot per target."""
    setup_style()
    fusion_order = fusion_order or [f for f in DELTA_FUSION_ORDER if f in df["fusion_type"].unique()]
    col_name = f"delta_{metric}"

    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt)
            continue
        hue_order = [m for m in MODELS if m in sub["model"].unique()]

        sns.boxplot(
            data=sub, x="fusion_type", y=col_name, hue="model",
            order=fusion_order, hue_order=hue_order, palette=palette,
            width=0.6, linewidth=0.8, fliersize=0, ax=ax,
        )
        sns.stripplot(
            data=sub, x="fusion_type", y=col_name, hue="model",
            order=fusion_order, hue_order=hue_order, palette=palette,
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
    xlabel: str = "RMSE_sum (eV)",
    ylabel: str = "RMSE_fusion (eV)",
    title: str = "Scatter: fusion RMSE vs sum_fusion RMSE",
    output_name: str = "fusion_scatter_rmse.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    fusion_order: Optional[list[str]] = None,
):
    """Per-fold scatter of fusion RMSE vs baseline RMSE, rows=fusion, cols=target."""
    setup_style()
    fusion_order = fusion_order or [f for f in DELTA_FUSION_ORDER if f in df["fusion_type"].unique()]

    baseline_df = (
        df[df["fusion_type"] == baseline]
        .set_index(["model", "target", "fold"])[[metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )
    non_base = df[df["fusion_type"] != baseline].copy()
    merged = non_base.merge(baseline_df, on=["model", "target", "fold"], how="inner")

    n_fusions = len(fusion_order)
    fig, axes = plt.subplots(
        n_fusions, len(targets),
        figsize=(5 * len(targets), 4.5 * n_fusions),
        squeeze=False,
    )

    for col_idx, tgt in enumerate(targets):
        for row_idx, ft in enumerate(fusion_order):
            ax = axes[row_idx, col_idx]
            sub = merged[(merged["target"] == tgt) & (merged["fusion_type"] == ft)]
            if sub.empty:
                ax.set_title(f"{tgt} — {ft}")
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

            ax.set_title(f"{tgt} — {ft}", fontweight="bold")
            ax.set_xlabel(xlabel if row_idx == n_fusions - 1 else "")
            ax.set_ylabel(ylabel if col_idx == 0 else "")

    _unified_legend([axes[0, -1]])
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / output_name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────
# Task 5: Win-rate plot
# ──────────────────────────────────────────────────────────────────────

def plot_win_rate(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    metric: str = "rmse",
    title: str = "Win rate vs sum_fusion  (% folds with lower RMSE)",
    output_name: str = "fusion_win_rate.png",
    output_dir: Path = OUTPUT_DIR,
    palette: dict = MODEL_PALETTE,
    targets: list[str] = ["EA", "IP"],
    fusion_order: Optional[list[str]] = None,
):
    """Bar plot showing the fraction of folds where each fusion beats the baseline."""
    setup_style()
    fusion_order = fusion_order or [f for f in DELTA_FUSION_ORDER if f in df["fusion_type"].unique()]

    # Compute per-fold delta
    baseline_df = (
        df[df["fusion_type"] == baseline]
        .set_index(["model", "target", "fold"])[[metric]]
        .rename(columns={metric: f"{metric}_baseline"})
    )
    non_base = df[df["fusion_type"] != baseline].copy()
    merged = non_base.merge(baseline_df, on=["model", "target", "fold"], how="inner")
    # For RMSE, improvement = fusion < baseline → delta < 0
    merged["improved"] = merged[metric] < merged[f"{metric}_baseline"]

    # Aggregate: win % per (fusion_type, model, target)
    win_df = (
        merged
        .groupby(["fusion_type", "model", "target"])["improved"]
        .agg(["sum", "count"])
        .reset_index()
    )
    win_df["win_pct"] = 100.0 * win_df["sum"] / win_df["count"]

    fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub = win_df[win_df["target"] == tgt]
        if sub.empty:
            ax.set_title(tgt)
            continue
        hue_order = [m for m in MODELS if m in sub["model"].unique()]

        sns.barplot(
            data=sub, x="fusion_type", y="win_pct", hue="model",
            order=fusion_order, hue_order=hue_order, palette=palette,
            edgecolor="black", linewidth=0.6, ax=ax,
        )
        ax.axhline(50, color="grey", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_ylim(0, 105)
        ax.set_title(tgt, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Win rate (%)" if ax == axes[0] else "")

    _unified_legend(axes)
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
        df.groupby(["target", "fusion_type", "model"])
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             r2_mean=("r2", "mean"), r2_std=("r2", "std"), n=("fold", "count"))
        .round(4)
    )
    summary_path = OUTPUT_DIR / "fusion_summary.csv"
    summary.to_csv(summary_path)
    print(f"\n=== Summary ===\n{summary.to_string()}\n")
    print(f"Summary saved: {summary_path}")

    # Task 1: Absolute RMSE
    plot_absolute(
        df, metric="rmse", ylabel="RMSE (eV)",
        title="RMSE by fusion strategy",
        output_name="fusion_absolute_rmse.png",
    )

    # Task 2: ΔRMSE vs sum_fusion
    delta_rmse = compute_deltas(df, baseline=BASELINE, metric="rmse")
    plot_delta(
        delta_rmse, metric="rmse",
        title="ΔRMSE vs sum_fusion  (negative = better)",
        ylabel="ΔRMSE (eV)",
        output_name="fusion_delta_rmse.png",
    )

    # Task 3: ΔR² vs sum_fusion
    delta_r2 = compute_deltas(df, baseline=BASELINE, metric="r2")
    plot_delta(
        delta_r2, metric="r2",
        title="ΔR² vs sum_fusion  (positive = better)",
        ylabel="ΔR²",
        output_name="fusion_delta_r2.png",
    )

    # Task 4: Scatter vs sum_fusion
    plot_scatter_vs_baseline(
        df, baseline=BASELINE, metric="rmse",
        xlabel="RMSE_sum (eV)", ylabel="RMSE_fusion (eV)",
        title="Scatter: fusion RMSE vs sum_fusion RMSE",
        output_name="fusion_scatter_rmse.png",
    )

    # Task 5: Win rate
    plot_win_rate(
        df, baseline=BASELINE, metric="rmse",
        title="Win rate vs sum_fusion  (% folds with lower RMSE)",
        output_name="fusion_win_rate.png",
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
