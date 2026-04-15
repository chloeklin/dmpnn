"""Phase 1D analysis: HPG_frac vs HPG_frac_edgeTyped.

Usage
-----
    python plot_phase1d.py --results_csv path/to/results.csv [--out_dir ./figures]

The CSV must contain columns:
    model, target, method, fold, rmse, r2

All plotting functions are reusable for later phases (nodeInit, archAware, etc.)
by passing different ``methods`` and ``baseline`` arguments.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
#  Global style constants
# ---------------------------------------------------------------------------

# Method display order (extend here for later phases)
METHOD_ORDER_ALL = ["HPG_frac", "HPG_frac_edgeTyped"]
METHOD_ORDER_DELTA = ["HPG_frac_edgeTyped"]

# Model palette — fixed so colours are consistent across all figures
MODEL_PALETTE = {
    "DMPNN": "#4C72B0",
    "GIN":   "#DD8452",
    "GAT":   "#55A868",
}

TARGET_ORDER = ["EA", "IP"]

FONT_SIZE   = 13
TITLE_SIZE  = 14
TICK_SIZE   = 11
FIG_DPI     = 150
JITTER_SEED = 42

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "axes.titlesize":  TITLE_SIZE,
        "axes.labelsize":  FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi":      FIG_DPI,
    })


def _model_palette(df: pd.DataFrame) -> dict:
    """Return palette restricted to models present in df."""
    models = df["model"].unique()
    return {m: MODEL_PALETTE.get(m, "#999999") for m in models}


def _strip_jitter(ax, data: pd.Series, x_pos: float, color: str, rng: np.random.Generator,
                  jitter: float = 0.06, size: float = 4.5, alpha: float = 0.75) -> None:
    """Overlay individual fold points with jitter on a box."""
    xs = rng.uniform(x_pos - jitter, x_pos + jitter, size=len(data))
    ax.scatter(xs, data.values, color=color, s=size**2, alpha=alpha,
               zorder=3, linewidths=0.4, edgecolors="white")


def _share_ylims(axes_flat) -> None:
    """Share y-axis limits across a flat list of Axes."""
    ymin = min(ax.get_ylim()[0] for ax in axes_flat)
    ymax = max(ax.get_ylim()[1] for ax in axes_flat)
    for ax in axes_flat:
        ax.set_ylim(ymin, ymax)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _present_methods(df: pd.DataFrame, order: list[str]) -> list[str]:
    """Filter ordered method list to only methods present in df."""
    present = set(df["method"].unique())
    return [m for m in order if m in present]


def _mpl_grouped_boxplot(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    x_order: list[str],
    hue_order: list[str],
    palette: dict,
    rng: np.random.Generator,
    box_width: float = 0.18,
    jitter: float = 0.04,
    point_size: float = 4.5,
    hline: float | None = None,
) -> list:
    """Draw a grouped boxplot with jitter using matplotlib directly.

    Avoids the seaborn 0.13.2 bug with hue in sns.boxplot.  Returns
    a list of legend handles (one per hue value).
    """
    n_hue   = len(hue_order)
    spacing = box_width * 1.35
    offsets = np.linspace(-(n_hue - 1) / 2 * spacing,
                           (n_hue - 1) / 2 * spacing, n_hue)

    legend_handles = []
    for hue_idx, hue_val in enumerate(hue_order):
        color = palette.get(hue_val, "#999999")
        positions = []
        data_groups = []
        for x_idx, x_val in enumerate(x_order):
            group = df[(df[x_col] == x_val) & (df[hue_col] == hue_val)][y_col].dropna()
            if group.empty:
                continue
            pos = x_idx + offsets[hue_idx]
            positions.append(pos)
            data_groups.append(group.values)

        if not positions:
            continue

        bp = ax.boxplot(
            data_groups,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor=color, alpha=0.70, linewidth=1.2),
            medianprops=dict(color="white", linewidth=2.0),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker=""),
        )

        # Jitter points
        for pos, vals in zip(positions, data_groups):
            xs = rng.uniform(pos - jitter, pos + jitter, size=len(vals))
            ax.scatter(xs, vals, color=color, s=point_size**2, alpha=0.80,
                       zorder=4, linewidths=0.3, edgecolors="white")

        handle = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.80, linewidth=0)
        legend_handles.append((handle, hue_val))

    # Axis ticks
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=15, ha="right", fontsize=TICK_SIZE)

    if hline is not None:
        ax.axhline(hline, color="black", linewidth=1.2, linestyle="--", zorder=2)

    return legend_handles


# ---------------------------------------------------------------------------
#  Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(df: pd.DataFrame, baseline: str = "HPG_frac") -> pd.DataFrame:
    """Return a DataFrame with delta_rmse and delta_r2 vs *baseline*.

    Merges on (model, target, fold).  Rows where the baseline is missing
    for a given fold are dropped with a warning.
    """
    base = df[df["method"] == baseline][["model", "target", "fold", "rmse", "r2"]].copy()
    base = base.rename(columns={"rmse": "rmse_base", "r2": "r2_base"})

    other = df[df["method"] != baseline].copy()
    merged = other.merge(base, on=["model", "target", "fold"], how="inner")

    n_miss = len(other) - len(merged)
    if n_miss > 0:
        print(f"  [compute_deltas] Warning: {n_miss} rows dropped (no {baseline} match)")

    merged["delta_rmse"] = merged["rmse"] - merged["rmse_base"]
    merged["delta_r2"]   = merged["r2"]   - merged["r2_base"]
    return merged


# ---------------------------------------------------------------------------
#  Task 1: Absolute RMSE
# ---------------------------------------------------------------------------

def plot_absolute_rmse(
    df: pd.DataFrame,
    method_order: list[str] | None = None,
    out_path: Path | None = None,
    title: str = "Phase 1D1: RMSE by HPG edge-typing variant",
) -> plt.Figure:
    """Boxplot of absolute RMSE, columns = target, hue = model."""
    _set_style()
    methods = _present_methods(df, method_order or METHOD_ORDER_ALL)
    targets = [t for t in TARGET_ORDER if t in df["target"].unique()]
    palette = _model_palette(df)
    models  = [m for m in list(MODEL_PALETTE) + list(palette) if m in df["model"].unique()]
    models  = list(dict.fromkeys(models))  # deduplicate preserving order
    rng     = np.random.default_rng(JITTER_SEED)

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(5.5 * n_targets, 5), sharey=False)
    if n_targets == 1:
        axes = [axes]

    for col_idx, target in enumerate(targets):
        ax  = axes[col_idx]
        sub = df[df["target"] == target]

        handles = _mpl_grouped_boxplot(
            ax, sub, x_col="method", y_col="rmse", hue_col="model",
            x_order=methods, hue_order=models, palette=palette, rng=rng,
        )

        ax.set_title(f"Target: {target}", fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("RMSE (eV)" if col_idx == 0 else "")
        if col_idx == 0 and handles:
            patches, labels = zip(*handles)
            ax.legend(patches, labels, title="Model", loc="upper right", fontsize=TICK_SIZE)

    _share_ylims(axes)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
#  Task 2: ΔRMSE vs baseline
# ---------------------------------------------------------------------------

def plot_delta_rmse(
    df_delta: pd.DataFrame,
    method_order: list[str] | None = None,
    out_path: Path | None = None,
    title: str = "Phase 1D2: ΔRMSE vs HPG_frac",
) -> plt.Figure:
    """Boxplot of ΔRMSE = rmse_method − rmse_baseline, columns = target."""
    _set_style()
    methods = _present_methods(df_delta, method_order or METHOD_ORDER_DELTA)
    targets = [t for t in TARGET_ORDER if t in df_delta["target"].unique()]
    palette = _model_palette(df_delta)
    models  = [m for m in list(MODEL_PALETTE) + list(palette) if m in df_delta["model"].unique()]
    models  = list(dict.fromkeys(models))
    rng     = np.random.default_rng(JITTER_SEED)

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 5), sharey=False)
    if n_targets == 1:
        axes = [axes]

    for col_idx, target in enumerate(targets):
        ax  = axes[col_idx]
        sub = df_delta[(df_delta["target"] == target) & (df_delta["method"].isin(methods))]

        handles = _mpl_grouped_boxplot(
            ax, sub, x_col="method", y_col="delta_rmse", hue_col="model",
            x_order=methods, hue_order=models, palette=palette, rng=rng,
            hline=0.0,
        )

        ax.set_title(f"Target: {target}", fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("ΔRMSE (eV)\n(negative = better)" if col_idx == 0 else "")
        if col_idx == 0 and handles:
            patches, labels = zip(*handles)
            ax.legend(patches, labels, title="Model", loc="upper right", fontsize=TICK_SIZE)

    _share_ylims(axes)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
#  Task 3: ΔR² vs baseline
# ---------------------------------------------------------------------------

def plot_delta_r2(
    df_delta: pd.DataFrame,
    method_order: list[str] | None = None,
    out_path: Path | None = None,
    title: str = "Phase 1D3: ΔR² vs HPG_frac",
) -> plt.Figure:
    """Boxplot of ΔR² = r2_method − r2_baseline, columns = target."""
    _set_style()
    methods = _present_methods(df_delta, method_order or METHOD_ORDER_DELTA)
    targets = [t for t in TARGET_ORDER if t in df_delta["target"].unique()]
    palette = _model_palette(df_delta)
    models  = [m for m in list(MODEL_PALETTE) + list(palette) if m in df_delta["model"].unique()]
    models  = list(dict.fromkeys(models))
    rng     = np.random.default_rng(JITTER_SEED)

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 5), sharey=False)
    if n_targets == 1:
        axes = [axes]

    for col_idx, target in enumerate(targets):
        ax  = axes[col_idx]
        sub = df_delta[(df_delta["target"] == target) & (df_delta["method"].isin(methods))]

        handles = _mpl_grouped_boxplot(
            ax, sub, x_col="method", y_col="delta_r2", hue_col="model",
            x_order=methods, hue_order=models, palette=palette, rng=rng,
            hline=0.0,
        )

        ax.set_title(f"Target: {target}", fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("ΔR²\n(positive = better)" if col_idx == 0 else "")
        if col_idx == 0 and handles:
            patches, labels = zip(*handles)
            ax.legend(patches, labels, title="Model", loc="upper right", fontsize=TICK_SIZE)

    _share_ylims(axes)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
#  Task 4: Scatter — method vs baseline RMSE
# ---------------------------------------------------------------------------

def plot_scatter_vs_baseline(
    df: pd.DataFrame,
    compare_methods: list[str] | None = None,
    baseline: str = "HPG_frac",
    out_path: Path | None = None,
    title: str = "HPG_frac_edgeTyped vs HPG_frac",
) -> plt.Figure:
    """Scatter rmse_baseline (x) vs rmse_method (y), columns = target."""
    _set_style()
    compare  = compare_methods or ["HPG_frac_edgeTyped"]
    targets  = [t for t in TARGET_ORDER if t in df["target"].unique()]
    palette  = _model_palette(df)
    models   = list(palette.keys())

    base = df[df["method"] == baseline][["model", "target", "fold", "rmse"]].rename(
        columns={"rmse": "rmse_base"}
    )

    n_methods = len(compare)
    n_targets = len(targets)
    fig, axes = plt.subplots(
        n_methods, n_targets,
        figsize=(4.5 * n_targets, 4.5 * n_methods),
        squeeze=False,
    )

    for row_idx, method in enumerate(compare):
        sub_m = df[df["method"] == method].merge(base, on=["model", "target", "fold"], how="inner")

        for col_idx, target in enumerate(targets):
            ax  = axes[row_idx][col_idx]
            sub = sub_m[sub_m["target"] == target]

            for model in models:
                pts = sub[sub["model"] == model]
                if pts.empty:
                    continue
                ax.scatter(pts["rmse_base"], pts["rmse"], color=palette[model],
                           label=model, s=55, alpha=0.82, zorder=3,
                           edgecolors="white", linewidths=0.5)

            # y = x diagonal
            all_vals = pd.concat([sub["rmse_base"], sub["rmse"]])
            lo, hi   = all_vals.min(), all_vals.max()
            margin   = (hi - lo) * 0.05
            diag     = [lo - margin, hi + margin]
            ax.plot(diag, diag, "k--", linewidth=1.0, zorder=2, label="y = x")

            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_aspect("equal", adjustable="box")

            ax.set_title(f"{method} | Target: {target}", fontsize=FONT_SIZE)
            ax.set_xlabel(f"RMSE — {baseline} (eV)", fontsize=FONT_SIZE)
            ax.set_ylabel(f"RMSE — {method} (eV)", fontsize=FONT_SIZE)

            if row_idx == 0 and col_idx == 0:
                ax.legend(title="Model", fontsize=TICK_SIZE)

    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
#  Task 5: Win-rate
# ---------------------------------------------------------------------------

def plot_win_rate(
    df_delta: pd.DataFrame,
    method_order: list[str] | None = None,
    baseline: str = "HPG_frac",
    out_path: Path | None = None,
    title: str = "Fold win rate over HPG_frac",
) -> plt.Figure:
    """Bar chart of % folds where RMSE < baseline, columns = target."""
    _set_style()
    methods  = _present_methods(df_delta, method_order or METHOD_ORDER_DELTA)
    targets  = [t for t in TARGET_ORDER if t in df_delta["target"].unique()]
    palette  = _model_palette(df_delta)
    models   = list(palette.keys())

    # Compute win rate
    records = []
    for method in methods:
        sub_m = df_delta[df_delta["method"] == method]
        for target in targets:
            sub_t = sub_m[sub_m["target"] == target]
            for model in models:
                sub_mo = sub_t[sub_t["model"] == model]
                if sub_mo.empty:
                    continue
                n_total = len(sub_mo)
                n_wins  = (sub_mo["delta_rmse"] < 0).sum()
                records.append({
                    "method": method, "target": target, "model": model,
                    "win_rate": 100 * n_wins / n_total,
                    "n_wins": n_wins, "n_total": n_total,
                })
    wr_df = pd.DataFrame(records)

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.0 * n_targets, 4.5), sharey=True)
    if n_targets == 1:
        axes = [axes]

    for col_idx, target in enumerate(targets):
        ax  = axes[col_idx]
        sub = wr_df[(wr_df["target"] == target) & (wr_df["method"].isin(methods))]

        sns.barplot(
            data=sub, x="method", y="win_rate", hue="model",
            order=methods, hue_order=models, palette=palette,
            ax=ax, errorbar=None, width=0.55, alpha=0.85,
        )

        # Annotate counts on bars
        for p in ax.patches:
            h = p.get_height()
            if np.isnan(h) or h == 0:
                continue
            ax.annotate(f"{h:.0f}%", (p.get_x() + p.get_width() / 2, h + 1.5),
                        ha="center", va="bottom", fontsize=9)

        ax.axhline(50, color="grey", linestyle="--", linewidth=1.0, zorder=1, label="50%")
        ax.set_ylim(0, 110)
        ax.set_title(f"Target: {target}", fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("% folds improved (RMSE)" if col_idx == 0 else "")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha="right")
        if col_idx > 0:
            ax.get_legend().remove()
        else:
            ax.legend(title="Model", loc="upper right")

    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
#  Task 6: Variance / stability
# ---------------------------------------------------------------------------

def plot_variance(
    df: pd.DataFrame,
    method_order: list[str] | None = None,
    out_path: Path | None = None,
    title: str = "RMSE variance by HPG variant",
) -> plt.Figure:
    """Bar chart of std(RMSE) across folds, columns = target."""
    _set_style()
    methods  = _present_methods(df, method_order or METHOD_ORDER_ALL)
    targets  = [t for t in TARGET_ORDER if t in df["target"].unique()]
    palette  = _model_palette(df)
    models   = list(palette.keys())

    # Compute std per (model, target, method)
    var_df = (
        df[df["method"].isin(methods)]
        .groupby(["model", "target", "method"])["rmse"]
        .std()
        .reset_index()
        .rename(columns={"rmse": "std_rmse"})
    )

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(5.0 * n_targets, 4.5), sharey=True)
    if n_targets == 1:
        axes = [axes]

    for col_idx, target in enumerate(targets):
        ax  = axes[col_idx]
        sub = var_df[var_df["target"] == target]

        sns.barplot(
            data=sub, x="method", y="std_rmse", hue="model",
            order=methods, hue_order=models, palette=palette,
            ax=ax, errorbar=None, width=0.55, alpha=0.85,
        )

        ax.set_title(f"Target: {target}", fontsize=FONT_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("Std(RMSE) across folds (eV)" if col_idx == 0 else "")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha="right")
        if col_idx > 0:
            ax.get_legend().remove()
        else:
            ax.legend(title="Model", loc="upper right")

    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
#  Task 7: Summary tables
# ---------------------------------------------------------------------------

def print_summary_tables(df: pd.DataFrame, df_delta: pd.DataFrame,
                         baseline: str = "HPG_frac") -> None:
    """Print compact summary tables to stdout."""
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)

    # --- Table 1: per (model, target, method) ---
    summary = (
        df.groupby(["model", "target", "method"])
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse",  "std"),
            mean_r2=("r2",     "mean"),
            std_r2=("r2",      "std"),
        )
        .reset_index()
        .sort_values(["target", "method", "model"])
    )

    print("\n" + "=" * 80)
    print("  SUMMARY TABLE 1 — Mean/Std per (model, target, method)")
    print("=" * 80)
    print(summary.to_string(index=False))

    # --- Table 2: delta summary ---
    delta_summary = (
        df_delta.groupby(["model", "target", "method"])
        .agg(
            mean_delta_rmse=("delta_rmse", "mean"),
            std_delta_rmse=("delta_rmse",  "std"),
            mean_delta_r2=("delta_r2",     "mean"),
            std_delta_r2=("delta_r2",      "std"),
        )
        .reset_index()
    )

    # win rate
    wr = (
        df_delta.groupby(["model", "target", "method"])
        .apply(lambda g: (g["delta_rmse"] < 0).mean() * 100)
        .reset_index(name="win_rate_%")
    )
    delta_summary = delta_summary.merge(wr, on=["model", "target", "method"])
    delta_summary = delta_summary.sort_values(["target", "method", "model"])

    print("\n" + "=" * 80)
    print(f"  DELTA SUMMARY — vs {baseline}")
    print("=" * 80)
    print(delta_summary.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
#  Task 8: Decision support
# ---------------------------------------------------------------------------

def print_decision_support(df: pd.DataFrame, df_delta: pd.DataFrame,
                            compare_method: str = "HPG_frac_edgeTyped",
                            baseline: str = "HPG_frac") -> None:
    """Print a text analysis and next-step recommendation."""

    sub = df_delta[df_delta["method"] == compare_method]

    if sub.empty:
        print(f"\n[Decision support] No data found for method={compare_method!r}")
        return

    overall_mean_delta = sub["delta_rmse"].mean()
    overall_std_delta  = sub["delta_rmse"].std()
    overall_win_rate   = (sub["delta_rmse"] < 0).mean() * 100

    # Per-target
    target_stats = sub.groupby("target")["delta_rmse"].agg(["mean", "std"]).round(4)
    target_wr    = sub.groupby("target").apply(
        lambda g: (g["delta_rmse"] < 0).mean() * 100
    ).round(1)

    # Per-model
    model_stats = sub.groupby("model")["delta_rmse"].agg(["mean", "std"]).round(4)

    # Variance comparison
    var_base  = df[df["method"] == baseline].groupby(["model", "target"])["rmse"].std()
    var_comp  = df[df["method"] == compare_method].groupby(["model", "target"])["rmse"].std()
    var_delta = (var_comp - var_base).mean()  # negative = more stable

    # Determine improvement category
    improved_mean     = overall_mean_delta < -1e-4
    win_rate_majority = overall_win_rate > 50
    variance_reduced  = var_delta < 0

    print("\n" + "=" * 80)
    print("  DECISION SUPPORT — Phase 1D")
    print("=" * 80)

    print(f"\n[1] Does {compare_method} show consistent improvement over {baseline}?")
    print(f"    Overall mean ΔRMSE : {overall_mean_delta:+.4f} eV  "
          f"(std={overall_std_delta:.4f})")
    print(f"    Win rate (RMSE↓)   : {overall_win_rate:.1f}% of folds")
    if improved_mean and win_rate_majority:
        verdict = "YES — consistent improvement in both mean and majority of folds."
    elif win_rate_majority:
        verdict = "PARTIAL — majority of folds improve but mean gain is small."
    elif improved_mean:
        verdict = "WEAK — mean improves but not majority of folds (high variance)."
    else:
        verdict = "NO — edge typing does not consistently improve over the baseline."
    print(f"    Verdict: {verdict}")

    print(f"\n[2] Are gains stronger for EA or IP?")
    for target in target_stats.index:
        wr = target_wr.get(target, float("nan"))
        row = target_stats.loc[target]
        print(f"    {target}: mean ΔRMSE={row['mean']:+.4f}, std={row['std']:.4f}, "
              f"win rate={wr:.1f}%")

    ea_mean = target_stats.loc["EA", "mean"] if "EA" in target_stats.index else 0.0
    ip_mean = target_stats.loc["IP", "mean"] if "IP" in target_stats.index else 0.0
    stronger = "EA" if ea_mean < ip_mean else "IP"
    print(f"    → Gains are stronger for {stronger} (more negative ΔRMSE).")

    print(f"\n[3] Are gains concentrated in certain backbones?")
    for model, row in model_stats.iterrows():
        wr = (sub[sub["model"] == model]["delta_rmse"] < 0).mean() * 100
        print(f"    {model}: mean ΔRMSE={row['mean']:+.4f}, "
              f"std={row['std']:.4f}, win rate={wr:.1f}%")

    print(f"\n[4] Does edge typing help mean, variance, or both?")
    print(f"    Mean improvement  : {'YES' if improved_mean else 'NO'} "
          f"(mean ΔRMSE={overall_mean_delta:+.4f})")
    print(f"    Variance reduction: {'YES' if variance_reduced else 'NO'} "
          f"(mean Δstd(RMSE)={var_delta:+.4f})")

    print("\n" + "-" * 80)
    print("  RECOMMENDATION")
    print("-" * 80)
    if improved_mean and win_rate_majority:
        rec = textwrap.dedent(f"""\
            Edge typing gives clear and consistent gains.
            → Recommended next step: better fragment / polymer-node initialization
              (e.g. HPG_frac_nodeInit: replace ones(d_v) fragment nodes with
              RDKit/ECFP embeddings of each repeat unit).
        """)
    elif variance_reduced and not improved_mean:
        rec = textwrap.dedent(f"""\
            Edge typing reduces variance but does not consistently improve the mean.
            → Recommended next step: richer architecture encoding
              (e.g. global context injection, inter-fragment attention, or
              polymer-sequence-aware positional encodings).
        """)
    else:
        rec = textwrap.dedent(f"""\
            Edge typing gives neither consistent mean gain nor variance reduction.
            → Recommended next step: move to more faithful polymer graph construction
              or sequence encoding (e.g. SMILES-level positional awareness, block
              co-polymer topology embeddings).
        """)
    print(rec)
    print("=" * 80)


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------

def run_all(
    df: pd.DataFrame,
    out_dir: Path,
    baseline: str = "HPG_frac",
    compare_methods: list[str] | None = None,
) -> None:
    """Run all 8 tasks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    compare = compare_methods or ["HPG_frac_edgeTyped"]
    method_order_all   = [baseline] + compare
    method_order_delta = compare

    print(f"\n[Phase 1D] Methods in data: {sorted(df['method'].unique())}")
    print(f"[Phase 1D] Models in data : {sorted(df['model'].unique())}")
    print(f"[Phase 1D] Targets in data: {sorted(df['target'].unique())}")
    print(f"[Phase 1D] Folds in data  : {sorted(df['fold'].unique())}")

    # Compute deltas
    df_delta = compute_deltas(df, baseline=baseline)

    # Task 1
    print("\n[Task 1] Absolute RMSE...")
    plot_absolute_rmse(
        df, method_order=method_order_all,
        out_path=out_dir / "phase1d1_absolute_rmse.png",
    )

    # Task 2
    print("[Task 2] ΔRMSE vs baseline...")
    plot_delta_rmse(
        df_delta, method_order=method_order_delta,
        out_path=out_dir / "phase1d1_delta_rmse.png",
    )

    # Task 3
    print("[Task 3] ΔR² vs baseline...")
    plot_delta_r2(
        df_delta, method_order=method_order_delta,
        out_path=out_dir / "phase1d1_delta_r2.png",
    )

    # Task 4
    print("[Task 4] Scatter vs baseline...")
    plot_scatter_vs_baseline(
        df, compare_methods=compare, baseline=baseline,
        out_path=out_dir / "phase1d1_scatter_rmse.png",
    )

    # Task 5
    print("[Task 5] Win rate...")
    plot_win_rate(
        df_delta, method_order=method_order_delta, baseline=baseline,
        out_path=out_dir / "phase1d1_win_rate.png",
    )

    # Task 6
    print("[Task 6] Variance / stability...")
    plot_variance(
        df, method_order=method_order_all,
        out_path=out_dir / "phase1d1_variance.png",
    )

    # Task 7
    print("[Task 7] Summary tables...")
    print_summary_tables(df, df_delta, baseline=baseline)

    # Task 8
    print("[Task 8] Decision support...")
    for method in compare:
        print_decision_support(df, df_delta, compare_method=method, baseline=baseline)


# ---------------------------------------------------------------------------
#  HPG results loader
# ---------------------------------------------------------------------------

# Maps method keyword in filename → canonical method name
_FILENAME_METHOD_MAP = {
    "hpg_frac_edgetyped": "HPG_frac_edgeTyped",
    "hpg_frac_polytype":  "HPG_frac_polytype",
    "hpg_frac":           "HPG_frac",
    "hpg_baseline":       "HPG_baseline",
}

# Maps long target name → short label used in plots
_TARGET_LABEL_MAP = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}


def _parse_method_from_filename(stem: str) -> str | None:
    """Extract canonical method name from a result CSV filename stem."""
    s = stem.lower()
    # Match longest key first so 'hpg_frac_edgetyped' beats 'hpg_frac'
    for key in sorted(_FILENAME_METHOD_MAP, key=len, reverse=True):
        if key in s:
            return _FILENAME_METHOD_MAP[key]
    return None


def load_hpg_results(
    results_dir: str | Path,
    dataset: str = "ea_ip",
    model_name: str = "HPG",
) -> pd.DataFrame:
    """Auto-discover and load HPG result CSVs from *results_dir*.

    Expects files named like:
        ``{dataset}__{method}__{split_type}__target_{target}_results.csv``

    Returns a DataFrame with columns:
        model, target, method, fold, rmse, r2
    """
    results_dir = Path(results_dir)
    records: list[dict] = []

    for csv_path in sorted(results_dir.glob(f"{dataset}__*_results.csv")):
        stem   = csv_path.stem                          # strip .csv
        method = _parse_method_from_filename(stem)
        if method is None:
            continue                                     # skip unrecognised files

        df_raw = pd.read_csv(csv_path)

        # Normalise column names
        col_map = {}
        for col in df_raw.columns:
            lc = col.lower()
            if "rmse" in lc:
                col_map[col] = "rmse"
            elif "r2" in lc or "r²" in lc:
                col_map[col] = "r2"
            elif lc in ("split", "fold"):
                col_map[col] = "fold"
            elif lc == "target":
                col_map[col] = "target"
        df_raw = df_raw.rename(columns=col_map)

        # Infer target from filename if column missing
        if "target" not in df_raw.columns:
            for long, short in _TARGET_LABEL_MAP.items():
                if long.lower().replace(" ", "_") in stem.lower().replace(" ", "_"):
                    df_raw["target"] = long
                    break

        for _, row in df_raw.iterrows():
            raw_target = str(row.get("target", "unknown"))
            short_target = _TARGET_LABEL_MAP.get(raw_target, raw_target)
            records.append({
                "model":  model_name,
                "target": short_target,
                "method": method,
                "fold":   int(row["fold"]),
                "rmse":   float(row["rmse"]),
                "r2":     float(row["r2"]),
            })

    if not records:
        raise FileNotFoundError(
            f"No recognisable HPG result CSVs found in {results_dir} "
            f"for dataset={dataset!r}"
        )

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} rows from {results_dir}")
    print(f"  Methods : {sorted(df['method'].unique())}")
    print(f"  Targets : {sorted(df['target'].unique())}")
    print(f"  Folds   : {sorted(df['fold'].unique())}")
    return df


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1D HPG edge-typing analysis")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--results_csv", type=str, default=None,
        help="Pre-merged CSV with columns: model, target, method, fold, rmse, r2",
    )
    src.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing raw HPG result CSVs (auto-discovered)",
    )

    parser.add_argument(
        "--dataset", type=str, default="ea_ip",
        help="Dataset prefix used when discovering CSVs (default: ea_ip)",
    )
    parser.add_argument(
        "--model_name", type=str, default="HPG",
        help="Model label to use in plots when loading from --results_dir (default: HPG)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./figures_phase1d",
        help="Directory to save figures (default: ./figures_phase1d)",
    )
    parser.add_argument(
        "--baseline", type=str, default="HPG_frac",
        help="Reference method name (default: HPG_frac)",
    )
    parser.add_argument(
        "--compare", nargs="+", default=["HPG_frac_edgeTyped"],
        help="Methods to compare vs baseline (default: HPG_frac_edgeTyped)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.results_csv:
        df = pd.read_csv(args.results_csv)
        required_cols = {"model", "target", "method", "fold", "rmse", "r2"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
    else:
        df = load_hpg_results(
            results_dir=args.results_dir,
            dataset=args.dataset,
            model_name=args.model_name,
        )

    run_all(
        df=df,
        out_dir=Path(args.out_dir),
        baseline=args.baseline,
        compare_methods=args.compare,
    )
