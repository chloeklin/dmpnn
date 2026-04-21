"""HPG ablation analysis — bar plots, scatter plots, win rates, delta metrics.

Loads per-fold CSV results for multiple HPG variants, computes summary
statistics, and generates publication-quality matplotlib figures.

Data format
-----------
CSV files in a directory.  Two naming conventions are supported:

  (A) Per-target  (new): <dataset>__<variant>__[...]__target_<TARGET>_results.csv
  (B) Multi-target (old): <dataset>__<variant>__[...]__a_held_out_results.csv

All CSVs must have columns: test/mae, test/rmse, test/r2, split, target

The ``target`` column (e.g. "EA vs SHE (eV)") is shortened automatically
to a display label (e.g. "EA") using TARGET_SHORT_MAP.

Variant auto-detection
----------------------
File stems are matched against ``VARIANT_MAP`` (ordered dict of regex → label).
The first matching pattern wins.  Override via --variant_map_json if needed.

Usage
-----
    python plot_ablation_results.py \\
        --results_dir results/HPG \\
        --out_dir     figures_phase1e

    # or point at a single merged CSV (columns: model, target, fold, rmse, r2)
    python plot_ablation_results.py \\
        --results_csv path/to/merged.csv \\
        --out_dir     figures_phase1e
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Configuration constants
# ---------------------------------------------------------------------------

# Logical display order (extend here for new phases)
MODEL_ORDER = [
    "HPG_baseline",
    "HPG_frac",
    "HPG_frac_polytype",
    "HPG_frac_edgeTyped",
    "HPG_frac_archAware",
    "HPG_relMsg",
    "HPG_fragGraph",
    "HPG_frac_archGraph",   # Phase 2B rerun
    "HPG_attnPool",          # Phase 3A
    "HPG_pairInteract",      # Phase 3B
    "HPG_pairInteractAttn",  # Phase 3C
    "HPG_pairInteractGate",  # Phase 4
]

# Regex patterns matched against the file stem (first match wins).
# Keys are regex strings; values are canonical model names.
VARIANT_MAP: dict[str, str] = {
    r"hpg_frac_archAware":      "HPG_frac_archAware",
    r"hpg_frac_edgeTyped":      "HPG_frac_edgeTyped",
    r"hpg_frac_polytype":       "HPG_frac_polytype",
    r"hpg_relMsg":              "HPG_relMsg",
    r"hpg_fragGraph":           "HPG_fragGraph",
    r"hpg_archGraph":           "HPG_frac_archGraph",   # Phase 2B rerun
    r"hpg_pairInteractAttn":    "HPG_pairInteractAttn",  # Phase 3C — must stay before pairInteract
    r"hpg_pairInteractGate":    "HPG_pairInteractGate",  # Phase 4  — must stay before pairInteract
    r"hpg_pairInteract(?=[^AG]|$)": "HPG_pairInteract",  # Phase 3B — excludes Attn & Gate
    r"hpg_attnPool":            "HPG_attnPool",          # Phase 3A
    r"hpg_frac(?=__)":          "HPG_frac",
    r"HPG_baseline":            "HPG_baseline",
    # fallback patterns for unlabelled baseline files
    r"ea_ip__a_held_out":       "HPG_baseline",
}

# Map raw target strings from the 'target' column to short display labels
TARGET_SHORT_MAP: dict[str, str] = {
    "EA vs SHE (eV)": "EA",
    "IP vs SHE (eV)": "IP",
}

FIGURE_DPI   = 150
FONT_SIZE    = 12
TITLE_SIZE   = 13
TICK_SIZE    = 10


# ---------------------------------------------------------------------------
#  load_results
# ---------------------------------------------------------------------------

def _variant_from_stem(stem: str) -> str | None:
    """Return the canonical model name for a file stem, or None if unknown."""
    for pattern, label in VARIANT_MAP.items():
        if re.search(pattern, stem):
            return label
    return None


def load_results(
    results_dir: str | Path | None = None,
    results_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load per-fold results from a directory of CSVs or a single merged CSV.

    Parameters
    ----------
    results_dir : path to a directory containing result CSVs.
    results_csv : path to a single pre-merged CSV with columns
                  [model, target, fold, rmse, r2].

    Returns
    -------
    pd.DataFrame with columns: model, target, fold, mae, rmse, r2
    """
    if results_csv is not None:
        df = pd.read_csv(results_csv)
        _check_merged_columns(df)
        return df

    if results_dir is None:
        raise ValueError("Provide either results_dir or results_csv.")

    results_dir = Path(results_dir)
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(results_dir.glob("*.csv")):
        variant = _variant_from_stem(csv_path.stem)
        if variant is None:
            continue  # skip unrecognised files silently

        df_raw = pd.read_csv(csv_path)
        if df_raw.empty:
            continue

        # Normalise column names
        col_map = {
            "test/rmse": "rmse",
            "test/r2":   "r2",
            "test/mae":  "mae",
            "split":     "fold",
        }
        df_raw = df_raw.rename(columns=col_map)

        # Skip files that don't contain regression metrics (e.g. classification results)
        required_cols = {"rmse", "r2", "fold"}
        if not required_cols.issubset(df_raw.columns):
            continue

        # Short target label
        if "target" in df_raw.columns:
            df_raw["target"] = df_raw["target"].map(
                lambda t: TARGET_SHORT_MAP.get(t, t)
            )
        else:
            # Try to infer target from filename
            for raw, short in TARGET_SHORT_MAP.items():
                if short in csv_path.stem or raw in csv_path.stem:
                    df_raw["target"] = short
                    break

        df_raw["model"] = variant
        if "mae" not in df_raw.columns:
            df_raw["mae"] = np.nan
        frames.append(df_raw[["model", "target", "fold", "mae", "rmse", "r2"]])

    if not frames:
        raise FileNotFoundError(
            f"No recognised result CSVs found in {results_dir}.\n"
            "Check that VARIANT_MAP patterns match your file names."
        )

    combined = pd.concat(frames, ignore_index=True)
    # Remove duplicate (model, target, fold) rows — keep last
    combined = combined.drop_duplicates(
        subset=["model", "target", "fold"], keep="last"
    )
    return combined


def _check_merged_columns(df: pd.DataFrame) -> None:
    required = {"model", "target", "fold", "rmse", "r2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Merged CSV is missing columns: {missing}")


# ---------------------------------------------------------------------------
#  summarize_results
# ---------------------------------------------------------------------------

def summarize_results(
    df: pd.DataFrame,
    targets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute mean ± std across folds per model and target.

    Returns
    -------
    pd.DataFrame with columns:
        Model | RMSE_<T> | RMSE_<T>_std | R2_<T> | R2_<T>_std  ...
    """
    if targets is None:
        targets = sorted(df["target"].unique())

    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    # append any model not in MODEL_ORDER to preserve completeness
    models += [m for m in df["model"].unique() if m not in models]

    rows = []
    for model in models:
        row: dict = {"Model": model}
        for target in targets:
            sub = df[(df["model"] == model) & (df["target"] == target)]
            if sub.empty:
                row[f"RMSE_{target}"] = np.nan
                row[f"RMSE_{target}_std"] = np.nan
                row[f"R2_{target}"] = np.nan
                row[f"R2_{target}_std"] = np.nan
            else:
                row[f"RMSE_{target}"]     = sub["rmse"].mean()
                row[f"RMSE_{target}_std"] = sub["rmse"].std(ddof=1)
                row[f"R2_{target}"]       = sub["r2"].mean()
                row[f"R2_{target}_std"]   = sub["r2"].std(ddof=1)
        rows.append(row)

    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
#  plot_bar
# ---------------------------------------------------------------------------

def plot_bar(
    df: pd.DataFrame,
    metric: str,
    targets: Sequence[str] | None = None,
    out_dir: str | Path = ".",
    lower_is_better: bool | None = None,
) -> None:
    """Grouped bar plot of mean ± std for a metric across models and targets.

    One figure per metric; targets are side-by-side bar groups.
    Colors are assigned by matplotlib's default color cycle.

    Parameters
    ----------
    metric : "rmse" or "r2"
    targets : list of target labels to include (default: all)
    out_dir : directory to save figures
    lower_is_better : used only for the title annotation. Auto-detected if None.
    """
    if targets is None:
        targets = sorted(df["target"].unique())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    models += [m for m in df["model"].unique() if m not in models]
    n_models  = len(models)
    n_targets = len(targets)

    x = np.arange(n_models)
    width = 0.8 / n_targets  # bar width, all targets fit within unit spacing
    offsets = (np.arange(n_targets) - (n_targets - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.4), 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for t_idx, target in enumerate(targets):
        means, stds = [], []
        for model in models:
            sub = df[(df["model"] == model) & (df["target"] == target)][metric]
            means.append(sub.mean() if not sub.empty else np.nan)
            stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)

        color = colors[t_idx % len(colors)]
        ax.bar(
            x + offsets[t_idx], means, width,
            yerr=stds, capsize=4,
            label=target, color=color, alpha=0.85,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("HPG_", "") for m in models],
        rotation=30, ha="right", fontsize=TICK_SIZE,
    )
    metric_label = metric.upper()
    if lower_is_better is None:
        lower_is_better = metric.lower() in ("rmse", "mae")
    direction = "↓ better" if lower_is_better else "↑ better"
    ax.set_ylabel(f"{metric_label}  ({direction})", fontsize=FONT_SIZE)
    ax.set_title(f"{metric_label} by model variant", fontsize=TITLE_SIZE)
    ax.legend(title="Target", fontsize=TICK_SIZE)
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    fname = out_dir / f"bar_{metric}.png"
    fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ---------------------------------------------------------------------------
#  plot_scatter
# ---------------------------------------------------------------------------

def plot_scatter(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    target: str,
    out_dir: str | Path = ".",
    lower_is_better: bool | None = None,
) -> None:
    """Fold-wise scatter: model_a (x) vs model_b (y) with diagonal reference.

    Parameters
    ----------
    model_a : baseline on x-axis
    model_b : compared model on y-axis
    metric  : "rmse" or "r2"
    target  : "EA" or "IP"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_df = df[(df["model"] == model_a) & (df["target"] == target)][["fold", metric]].set_index("fold")
    b_df = df[(df["model"] == model_b) & (df["target"] == target)][["fold", metric]].set_index("fold")

    common_folds = a_df.index.intersection(b_df.index)
    if len(common_folds) == 0:
        print(f"  [skip] No common folds for {model_a} vs {model_b}, {target}, {metric}")
        return

    a_vals = a_df.loc[common_folds, metric].values
    b_vals = b_df.loc[common_folds, metric].values

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(a_vals, b_vals, s=60, zorder=3)

    # Per-fold annotations
    for fold, ax_v, bx_v in zip(common_folds, a_vals, b_vals):
        ax.annotate(str(fold), (ax_v, bx_v),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=8, color="dimgray")

    # Diagonal y = x
    all_vals = np.concatenate([a_vals, b_vals])
    lo, hi = all_vals.min(), all_vals.max()
    pad = (hi - lo) * 0.1 or 0.05
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "--", color="gray", linewidth=1.0, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    if lower_is_better is None:
        lower_is_better = metric.lower() in ("rmse", "mae")
    direction_note = "below diagonal = better" if lower_is_better else "above diagonal = better"

    a_short = model_a.replace("HPG_", "")
    b_short = model_b.replace("HPG_", "")
    ax.set_xlabel(f"{a_short}  ({metric.upper()})", fontsize=FONT_SIZE)
    ax.set_ylabel(f"{b_short}  ({metric.upper()})", fontsize=FONT_SIZE)
    ax.set_title(
        f"{target} — {a_short} vs {b_short}\n"
        f"{metric.upper()} per fold  ({direction_note})",
        fontsize=TITLE_SIZE - 1,
    )
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    safe_a = model_a.replace("/", "_")
    safe_b = model_b.replace("/", "_")
    fname = out_dir / f"scatter_{metric}_{target}_{safe_a}_vs_{safe_b}.png"
    fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ---------------------------------------------------------------------------
#  compute_win_rate
# ---------------------------------------------------------------------------

def compute_win_rate(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] = ("rmse", "r2"),
    targets: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute fold-level win rate of model_a over model_b.

    Win definition: lower for rmse/mae, higher for r2.

    Returns
    -------
    dict keyed by "<metric>_<target>" → win rate in [0, 1].
    """
    if targets is None:
        targets = sorted(df["target"].unique())

    lower_better = {"rmse", "mae"}
    results: dict[str, float] = {}

    for target in targets:
        a_df = (
            df[(df["model"] == model_a) & (df["target"] == target)]
            [["fold", *[m for m in metrics if m in df.columns]]]
            .set_index("fold")
        )
        b_df = (
            df[(df["model"] == model_b) & (df["target"] == target)]
            [["fold", *[m for m in metrics if m in df.columns]]]
            .set_index("fold")
        )
        common = a_df.index.intersection(b_df.index)
        if len(common) == 0:
            continue

        for metric in metrics:
            if metric not in a_df.columns or metric not in b_df.columns:
                continue
            a_vals = a_df.loc[common, metric].values
            b_vals = b_df.loc[common, metric].values
            if metric in lower_better:
                wins = (a_vals < b_vals).sum()
            else:
                wins = (a_vals > b_vals).sum()
            results[f"{metric}_{target}"] = wins / len(common)

    return results


def print_win_rates(
    win_rates: dict[str, float],
    model_a: str,
    model_b: str,
) -> None:
    """Pretty-print win rate dict."""
    print(f"\nWin rate: {model_a} vs {model_b}  (fraction of folds won by {model_a})")
    for key, rate in sorted(win_rates.items()):
        pct = f"{rate * 100:.0f}%"
        parts = key.split("_", 1)
        metric, target = parts[0], parts[1] if len(parts) > 1 else ""
        print(f"  {metric.upper():6s}  {target:4s}  →  {pct}")


# ---------------------------------------------------------------------------
#  plot_delta
# ---------------------------------------------------------------------------

def plot_delta(
    df: pd.DataFrame,
    baseline: str,
    metric: str,
    targets: Sequence[str] | None = None,
    out_dir: str | Path = ".",
) -> None:
    """Bar plot of mean delta (model − baseline) across folds per model.

    For RMSE: negative delta = improvement.
    For R²  : positive delta = improvement.

    Parameters
    ----------
    baseline : model name to use as reference (e.g. "HPG_frac")
    metric   : "rmse" or "r2"
    """
    if targets is None:
        targets = sorted(df["target"].unique())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_models = [
        m for m in MODEL_ORDER
        if m in df["model"].unique() and m != baseline
    ]
    comparison_models += [
        m for m in df["model"].unique()
        if m not in MODEL_ORDER and m != baseline
    ]
    if not comparison_models:
        print(f"  [skip] No comparison models for delta plot (baseline={baseline})")
        return

    n_models  = len(comparison_models)
    n_targets = len(targets)
    x = np.arange(n_models)
    width = 0.8 / n_targets
    offsets = (np.arange(n_targets) - (n_targets - 1) / 2) * width
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(5, n_models * 1.6), 4.5))

    for t_idx, target in enumerate(targets):
        base_df = (
            df[(df["model"] == baseline) & (df["target"] == target)]
            [["fold", metric]].set_index("fold")
        )
        delta_means, delta_stds = [], []
        for model in comparison_models:
            mod_df = (
                df[(df["model"] == model) & (df["target"] == target)]
                [["fold", metric]].set_index("fold")
            )
            common = base_df.index.intersection(mod_df.index)
            if len(common) == 0:
                delta_means.append(np.nan)
                delta_stds.append(0.0)
                continue
            deltas = mod_df.loc[common, metric].values - base_df.loc[common, metric].values
            delta_means.append(deltas.mean())
            delta_stds.append(deltas.std(ddof=1) if len(deltas) > 1 else 0.0)

        color = colors[t_idx % len(colors)]
        ax.bar(
            x + offsets[t_idx], delta_means, width,
            yerr=delta_stds, capsize=4,
            label=target, color=color, alpha=0.85,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("HPG_", "") for m in comparison_models],
        rotation=30, ha="right", fontsize=TICK_SIZE,
    )
    base_short = baseline.replace("HPG_", "")
    metric_label = f"Δ{metric.upper()}"
    direction = "(negative = improvement)" if metric.lower() in ("rmse", "mae") else "(positive = improvement)"
    ax.set_ylabel(f"{metric_label}  vs {base_short}  {direction}", fontsize=FONT_SIZE)
    ax.set_title(
        f"{metric_label}: each model − {base_short}",
        fontsize=TITLE_SIZE,
    )
    ax.legend(title="Target", fontsize=TICK_SIZE)
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    fname = out_dir / f"delta_{metric}_vs_{baseline.replace('HPG_', '')}.png"
    fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ---------------------------------------------------------------------------
#  print_text_summary
# ---------------------------------------------------------------------------

def print_text_summary(
    df: pd.DataFrame,
    baseline: str = "HPG_frac",
    targets: Sequence[str] | None = None,
) -> None:
    """Print best model per metric and biggest improvement over baseline."""
    if targets is None:
        targets = sorted(df["target"].unique())

    summary = summarize_results(df, targets)
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(summary.to_string(float_format="{:.4f}".format))
    print()

    for target in targets:
        rmse_col, r2_col = f"RMSE_{target}", f"R2_{target}"
        if rmse_col not in summary.columns:
            continue
        best_rmse = summary[rmse_col].idxmin()
        best_r2   = summary[r2_col].idxmax()
        print(f"  [{target}] Best RMSE: {best_rmse}  ({summary.loc[best_rmse, rmse_col]:.4f})")
        print(f"  [{target}] Best R²  : {best_r2}  ({summary.loc[best_r2, r2_col]:.4f})")

        if baseline in summary.index:
            base_rmse = summary.loc[baseline, rmse_col]
            base_r2   = summary.loc[baseline, r2_col]
            others = summary.index[summary.index != baseline]
            deltas = (summary.loc[others, rmse_col].dropna() - base_rmse)
            if not deltas.empty:
                biggest_gain_rmse = deltas.idxmin()
                print(
                    f"  [{target}] Biggest RMSE improvement over {baseline}: "
                    f"{biggest_gain_rmse}  (Δ = {deltas[biggest_gain_rmse]:+.4f})"
                )
            deltas_r2 = (summary.loc[others, r2_col].dropna() - base_r2)
            if not deltas_r2.empty:
                biggest_gain_r2 = deltas_r2.idxmax()
                print(
                    f"  [{target}] Biggest R² improvement over {baseline}: "
                    f"{biggest_gain_r2}  (Δ = {deltas_r2[biggest_gain_r2]:+.4f})"
                )
        print()


# ---------------------------------------------------------------------------
#  run_all  (convenience wrapper)
# ---------------------------------------------------------------------------

def run_all(
    results_dir: str | Path | None = None,
    results_csv: str | Path | None = None,
    out_dir: str | Path = "figures_ablation",
    baseline: str = "HPG_frac",
    comparisons: list[tuple[str, str]] | None = None,
    metrics: Sequence[str] = ("rmse", "r2"),
    targets: Sequence[str] | None = None,
    save_summary_csv: bool = True,
) -> pd.DataFrame:
    """Load data, generate all plots, print summary, return tidy DataFrame.

    Parameters
    ----------
    comparisons : list of (model_a, model_b) pairs for scatter + win-rate plots.
                  Defaults to all pairings of each non-baseline model vs baseline.
    """
    # 1. Load
    df = load_results(results_dir=results_dir, results_csv=results_csv)
    print(f"\nLoaded {len(df)} rows from "
          f"{df['model'].nunique()} models, "
          f"{df['target'].nunique()} targets, "
          f"{df['fold'].nunique()} folds.")
    print(f"  Models  : {sorted(df['model'].unique())}")
    print(f"  Targets : {sorted(df['target'].unique())}")

    if targets is None:
        targets = sorted(df["target"].unique())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Summary
    summary = summarize_results(df, targets)
    print_text_summary(df, baseline=baseline, targets=targets)

    if save_summary_csv:
        csv_path = out_dir / "summary_table.csv"
        summary.to_csv(csv_path)
        print(f"  Saved → {csv_path}")

    # 3. Bar plots
    for metric in metrics:
        plot_bar(df, metric=metric, targets=targets, out_dir=out_dir)

    # 4. Delta plots
    for metric in metrics:
        plot_delta(df, baseline=baseline, metric=metric,
                   targets=targets, out_dir=out_dir)

    # 5. Scatter plots + win rates
    if comparisons is None:
        non_baseline = [m for m in MODEL_ORDER
                        if m in df["model"].unique() and m != baseline]
        non_baseline += [m for m in df["model"].unique()
                         if m not in MODEL_ORDER and m != baseline]
        comparisons = [(baseline, b) for b in non_baseline]

    for model_a, model_b in comparisons:
        if model_a not in df["model"].unique() or model_b not in df["model"].unique():
            print(f"  [skip] {model_a} or {model_b} not in data")
            continue
        for target in targets:
            for metric in metrics:
                plot_scatter(df, model_a, model_b,
                             metric=metric, target=target, out_dir=out_dir)
        wr = compute_win_rate(df, model_a, model_b,
                              metrics=list(metrics), targets=list(targets))
        print_win_rates(wr, model_a, model_b)

    print(f"\nAll figures saved to {out_dir.resolve()}")
    return df


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            HPG ablation plotting pipeline.

            Examples
            --------
            # Auto-discover CSVs in a directory:
            python plot_ablation_results.py --results_dir results/HPG

            # Use a pre-merged CSV:
            python plot_ablation_results.py --results_csv merged.csv
        """),
    )
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory containing result CSVs (auto-detected).")
    p.add_argument("--results_csv", type=str, default=None,
                   help="Pre-merged CSV with columns [model, target, fold, rmse, r2].")
    p.add_argument("--out_dir", type=str, default="figures_ablation",
                   help="Output directory for figures (default: figures_ablation).")
    p.add_argument("--baseline", type=str, default="HPG_frac",
                   help="Baseline model for delta / win-rate comparisons.")
    p.add_argument("--metrics", nargs="+", default=["rmse", "r2"],
                   choices=["rmse", "r2", "mae"],
                   help="Metrics to plot (default: rmse r2).")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Target labels to include (default: all).")
    p.add_argument("--no_summary_csv", action="store_true",
                   help="Skip saving summary_table.csv.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.results_dir is None and args.results_csv is None:
        # Default: look in results/HPG relative to the repo root
        default_dir = Path(__file__).resolve().parents[2] / "results" / "HPG"
        if default_dir.is_dir():
            args.results_dir = str(default_dir)
        else:
            _build_parser().error(
                "Provide --results_dir or --results_csv. "
                f"Default directory {default_dir} does not exist."
            )

    run_all(
        results_dir=args.results_dir,
        results_csv=args.results_csv,
        out_dir=args.out_dir,
        baseline=args.baseline,
        metrics=args.metrics,
        targets=args.targets,
        save_summary_csv=not args.no_summary_csv,
    )
