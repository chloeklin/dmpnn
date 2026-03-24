"""Effect of copolymer representation strategy on ea_ip — +PT models, monomer split.

Compares mean vs mixture vs interaction strategies for DMPNN, GIN, GAT
(all with +PT) on the a_held_out (monomer) split.  Task 4 adds wDMPNN comparison.

Tasks
-----
1. Box plots:  RMSE and R² by strategy, hue = model.
2. Delta plots: per-fold improvement (strategy − mean).
3. Paired scatter: mean vs mixture / mean vs interaction, per model.
4. Best strategy vs wDMPNN paired scatter.

Outputs → analysis/ea_ip_report/
"""

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
ROOT_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
OUT_DIR     = SCRIPT_DIR / "ea_ip_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS = ["DMPNN", "GIN", "GAT"]
TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
STRATEGIES = ["mean", "mixture", "interaction"]
STRATEGY_ORDER = ["mean", "mixture", "interaction"]

MODEL_COLORS = {
    "DMPNN": "#4C72B0",
    "GIN":   "#DD8452",
    "GAT":   "#55A868",
}
STRATEGY_COLORS = {
    "mean":        "#888888",
    "mixture":     "#4C72B0",
    "interaction": "#C44E52",
}

# ── File mapping ──────────────────────────────────────────────────────────────
# Strategy name → naming convention suffix (all +PT, monomer)
# mean       → copoly_mean_meta__poly_type__a_held_out
# mixture    → copoly_mix_meta__poly_type__a_held_out
# interaction→ copoly_interact_meta__poly_type__a_held_out

def _make_file_list(model: str, copoly_suffix: str) -> list[str]:
    """Build per-target file paths for a model and copoly suffix."""
    base = f"ea_ip__copoly_{copoly_suffix}__poly_type__a_held_out"
    return [
        f"{model}/{base}__target_EA vs SHE (eV)_results.csv",
        f"{model}/{base}__target_IP vs SHE (eV)_results.csv",
    ]

STRATEGY_FILES = {}
for _model in MODELS:
    STRATEGY_FILES[_model] = {
        "mean":        _make_file_list(_model, "mean_meta"),
        "mixture":     _make_file_list(_model, "mix_meta"),
        "interaction": _make_file_list(_model, "interact_meta"),
    }

WDMPNN_FILES = [
    "wDMPNN/ea_ip__a_held_out__target_EA vs SHE (eV)_results.csv",
    "wDMPNN/ea_ip__a_held_out__target_IP vs SHE (eV)_results.csv",
]

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":   9,
    "figure.dpi":       100,
    "font.family":      "sans-serif",
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_path_for(per_target_path: Path):
    m = re.match(r'(.+?)__target_.+_results\.csv$', per_target_path.name)
    if m:
        return per_target_path.parent / f"{m.group(1)}_results.csv"
    return None


def _target_from_path(per_target_path: Path):
    m = re.match(r'.+?__target_(.+)_results\.csv$', per_target_path.name)
    return m.group(1) if m else None


def _load_files(file_list: list[str], **extra_cols) -> pd.DataFrame:
    """Load a list of result CSV paths, with aggregate fallback."""
    frames = []
    for rel in file_list:
        p = RESULTS_DIR / rel
        if p.exists():
            df = pd.read_csv(p)
        else:
            agg = _aggregate_path_for(p)
            target = _target_from_path(p)
            if agg is not None and agg.exists():
                df = pd.read_csv(agg)
                if "target" in df.columns and target:
                    df = df[df["target"] == target].copy()
            else:
                warnings.warn(f"[data] Missing: {p}")
                continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for k, v in extra_cols.items():
        df[k] = v
    if "split" in df.columns:
        df = df.rename(columns={"split": "fold"})
    return df


def load_strategies() -> pd.DataFrame:
    """Load all strategy × model data into one tidy DataFrame."""
    parts = []
    for model in MODELS:
        for strategy, flist in STRATEGY_FILES[model].items():
            df = _load_files(flist, model=model, strategy=strategy)
            if not df.empty:
                parts.append(df)
    if not parts:
        raise RuntimeError("No strategy data found")
    df = pd.concat(parts, ignore_index=True)
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], categories=STRATEGY_ORDER, ordered=True)
    return df


def load_wdmpnn() -> pd.DataFrame:
    """Load wDMPNN monomer results."""
    df = _load_files(WDMPNN_FILES, model="wDMPNN")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 – Box plots
# ─────────────────────────────────────────────────────────────────────────────

def task1_boxplots(df: pd.DataFrame):
    """Box plots of RMSE and R² by strategy, hue = model."""
    print("\n[Task 1] Box plots")
    metrics = {
        "test/rmse": ("RMSE (eV)", "box_rmse_monomer_strategy.png"),
        "test/r2":   ("R²",        "box_r2_monomer_strategy.png"),
    }
    targets_present = sorted(df["target"].unique())

    for metric, (ylabel, fname) in metrics.items():
        n_cols = len(targets_present)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
        fig.suptitle(f"{ylabel}  —  +PT models, monomer split",
                     fontsize=13, fontweight="bold")

        for col, target in enumerate(targets_present):
            ax = axes[0][col]
            sub = df[df["target"] == target].copy()

            sns.boxplot(
                data=sub, x="strategy", y=metric, hue="model",
                palette=MODEL_COLORS, width=0.6, linewidth=1.2,
                fliersize=0, ax=ax,
            )
            sns.stripplot(
                data=sub, x="strategy", y=metric, hue="model",
                palette=MODEL_COLORS, dodge=True,
                jitter=0.08, size=6, alpha=0.8,
                linewidth=0.5, edgecolor="white", ax=ax,
            )
            handles, labels = ax.get_legend_handles_labels()
            n_unique = len(MODEL_COLORS)
            ax.legend(handles[:n_unique], labels[:n_unique],
                      title="Model", loc="best", frameon=True)

            ax.set_title(target, fontsize=11)
            ax.set_xlabel("Strategy")
            ax.set_ylabel(ylabel)
            sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 – Delta vs mean
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deltas_vs_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold delta = strategy − mean for mixture and interaction."""
    mean_df = df[df["strategy"] == "mean"].set_index(["model", "target", "fold"])
    parts = []
    for strat in ["mixture", "interaction"]:
        strat_df = df[df["strategy"] == strat].set_index(["model", "target", "fold"])
        joined = mean_df[["test/rmse", "test/r2"]].join(
            strat_df[["test/rmse", "test/r2"]],
            lsuffix="_mean", rsuffix="_strat", how="inner",
        )
        joined["delta_rmse"] = joined["test/rmse_strat"] - joined["test/rmse_mean"]
        joined["delta_r2"]   = joined["test/r2_strat"]   - joined["test/r2_mean"]
        joined["strategy"] = strat
        parts.append(joined.reset_index())
    return pd.concat(parts, ignore_index=True)


def task2_delta(df: pd.DataFrame):
    """Delta plots: per-fold (strategy − mean)."""
    print("\n[Task 2] Delta vs mean plots")
    deltas = _compute_deltas_vs_mean(df)
    if deltas.empty:
        print("  ⚠  No paired data")
        return

    targets_present = sorted(deltas["target"].unique())
    metric_info = {
        "delta_rmse": ("ΔRMSE (strategy − mean)", "delta_rmse_strategy_vs_mean.png",
                       "negative = strategy better than mean"),
        "delta_r2":   ("ΔR² (strategy − mean)",   "delta_r2_strategy_vs_mean.png",
                       "positive = strategy better than mean"),
    }

    for metric, (ylabel, fname, note) in metric_info.items():
        n_cols = len(targets_present)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
        fig.suptitle(f"{ylabel}  —  +PT models, monomer split\n({note})",
                     fontsize=12, fontweight="bold")

        for col, target in enumerate(targets_present):
            ax = axes[0][col]
            sub = deltas[deltas["target"] == target].copy()
            sub["strategy"] = pd.Categorical(
                sub["strategy"], categories=["mixture", "interaction"], ordered=True
            )

            sns.stripplot(
                data=sub, x="strategy", y=metric, hue="model",
                palette=MODEL_COLORS, dodge=True,
                size=8, alpha=0.8, jitter=0.08,
                linewidth=0.5, edgecolor="white", ax=ax,
            )
            ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)

            # Mean diamond per model × strategy
            for i, strat in enumerate(["mixture", "interaction"]):
                for j, model in enumerate(MODELS):
                    vals = sub[(sub["strategy"] == strat) & (sub["model"] == model)][metric]
                    if not vals.empty:
                        n_groups = len(MODELS)
                        offset = (j - (n_groups - 1) / 2) * 0.27
                        ax.plot(i + offset, vals.mean(), marker="D",
                                color=MODEL_COLORS[model], markersize=9,
                                markeredgecolor="black", markeredgewidth=0.8,
                                zorder=5)

            handles, labels = ax.get_legend_handles_labels()
            n_unique = len(MODEL_COLORS)
            ax.legend(handles[:n_unique], labels[:n_unique],
                      title="Model", loc="best", frameon=True)
            ax.set_title(target, fontsize=11)
            ax.set_xlabel("Strategy")
            ax.set_ylabel(ylabel)
            sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 – Paired scatter (mean vs other strategies)
# ─────────────────────────────────────────────────────────────────────────────

def task3_paired(df: pd.DataFrame):
    """Paired scatter: mean (x) vs strategy (y), per model."""
    print("\n[Task 3] Paired scatter plots")
    deltas = _compute_deltas_vs_mean(df)
    if deltas.empty:
        print("  ⚠  No paired data")
        return

    targets_present = sorted(deltas["target"].unique())
    metrics = {
        "test/rmse": ("RMSE (eV)", "rmse", "below diagonal = strategy better"),
        "test/r2":   ("R²",        "r2",   "above diagonal = strategy better"),
    }

    for model in MODELS:
        for strat in ["mixture", "interaction"]:
            for metric, (mlabel, mshort, interp) in metrics.items():
                sub = deltas[(deltas["model"] == model) & (deltas["strategy"] == strat)]
                if sub.empty:
                    continue

                n_cols = len(targets_present)
                fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                                         squeeze=False)
                fig.suptitle(
                    f"{model} — {mlabel}: mean vs {strat}\n({interp})",
                    fontsize=12, fontweight="bold",
                )

                for col, target in enumerate(targets_present):
                    ax = axes[0][col]
                    t_sub = sub[sub["target"] == target]
                    if t_sub.empty:
                        ax.set_visible(False)
                        continue

                    x = t_sub[f"{metric}_mean"].values
                    y = t_sub[f"{metric}_strat"].values

                    lo = min(x.min(), y.min())
                    hi = max(x.max(), y.max())
                    pad = (hi - lo) * 0.08
                    lim = [lo - pad, hi + pad]
                    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y = x")
                    ax.scatter(x, y, s=60, alpha=0.85,
                               color=STRATEGY_COLORS[strat],
                               edgecolors="white", linewidth=0.5, zorder=3)

                    for i, fold in enumerate(t_sub["fold"].values):
                        ax.annotate(str(int(fold)), (x[i], y[i]),
                                    fontsize=7, textcoords="offset points",
                                    xytext=(5, 5))

                    ax.set_xlim(lim); ax.set_ylim(lim)
                    ax.set_aspect("equal")
                    ax.set_xlabel(f"mean  {mlabel}")
                    ax.set_ylabel(f"{strat}  {mlabel}")
                    ax.set_title(target, fontsize=11)
                    ax.legend(fontsize=8)
                    sns.despine(ax=ax)

                fig.tight_layout()
                fname = f"paired_{model.lower()}_{mshort}_mean_vs_{strat}.png"
                fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 – Best strategy vs wDMPNN
# ─────────────────────────────────────────────────────────────────────────────

def task4_vs_wdmpnn(df: pd.DataFrame, wdmpnn_df: pd.DataFrame):
    """Paired scatter: best strategy per model vs wDMPNN."""
    print("\n[Task 4] Best strategy vs wDMPNN")
    if wdmpnn_df.empty:
        print("  ⚠  wDMPNN data not available")
        return

    targets_present = sorted(df["target"].unique())
    metrics = {
        "test/rmse": ("RMSE (eV)", "rmse", "below diagonal = model better than wDMPNN"),
        "test/r2":   ("R²",        "r2",   "above diagonal = model better than wDMPNN"),
    }

    # For each model, pick the strategy with best mean RMSE across both targets
    best_strategy = {}
    for model in MODELS:
        m_df = df[df["model"] == model]
        mean_rmse = m_df.groupby("strategy", observed=True)["test/rmse"].mean()
        best = mean_rmse.idxmin()
        best_strategy[model] = best

    print(f"  Best strategies: {best_strategy}")

    wdmpnn_indexed = wdmpnn_df.set_index(["target", "fold"])

    for metric, (mlabel, mshort, interp) in metrics.items():
        n_cols = len(targets_present)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        fig.suptitle(f"{mlabel}: best +PT strategy vs wDMPNN\n({interp})",
                     fontsize=12, fontweight="bold")

        for col, target in enumerate(targets_present):
            ax = axes[0][col]

            all_vals = []
            for model in MODELS:
                strat = best_strategy[model]
                m_df = df[(df["model"] == model) & (df["strategy"] == strat)
                          & (df["target"] == target)]
                w_df = wdmpnn_df[wdmpnn_df["target"] == target]

                # Align by fold
                merged = m_df.merge(w_df, on="fold", suffixes=("_model", "_wdmpnn"))
                if merged.empty:
                    continue

                x_w = merged[f"{metric}_wdmpnn"].values
                y_m = merged[f"{metric}_model"].values
                all_vals.extend(x_w.tolist() + y_m.tolist())

                ax.scatter(x_w, y_m, s=60, alpha=0.85,
                           color=MODEL_COLORS[model],
                           edgecolors="white", linewidth=0.5, zorder=3,
                           label=f"{model} ({strat})")

                for i, fold in enumerate(merged["fold"].values):
                    ax.annotate(str(int(fold)), (x_w[i], y_m[i]),
                                fontsize=7, textcoords="offset points",
                                xytext=(5, 5))

            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                pad = (hi - lo) * 0.08
                lim = [lo - pad, hi + pad]
                ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_aspect("equal")

            ax.set_xlabel(f"wDMPNN  {mlabel}")
            ax.set_ylabel(f"Best +PT strategy  {mlabel}")
            ax.set_title(target, fontsize=11)
            ax.legend(fontsize=8, loc="best")
            sns.despine(ax=ax)

        fig.tight_layout()
        fname = f"paired_best_strategy_vs_wdmpnn_{mshort}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame, wdmpnn_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Summary: copolymer strategy comparison (+PT, monomer split)")
    print("=" * 70)

    deltas = _compute_deltas_vs_mean(df)

    for target in sorted(df["target"].unique()):
        print(f"\n  Target: {target}")

        # Strategy comparison
        print(f"    {'Model':<8} {'mean RMSE':>10} {'mix RMSE':>10} {'interact RMSE':>14}")
        for model in MODELS:
            vals = {}
            for strat in STRATEGIES:
                s = df[(df["model"] == model) & (df["strategy"] == strat)
                       & (df["target"] == target)]["test/rmse"]
                vals[strat] = f"{s.mean():.4f}" if not s.empty else "N/A"
            print(f"    {model:<8} {vals['mean']:>10} {vals['mixture']:>10} {vals['interaction']:>14}")

        # Deltas
        print(f"\n    Deltas vs mean (ΔRMSE / ΔR², negative RMSE = better):")
        for strat in ["mixture", "interaction"]:
            t = deltas[(deltas["target"] == target) & (deltas["strategy"] == strat)]
            for model in MODELS:
                m = t[t["model"] == model]
                if m.empty:
                    continue
                dr = m["delta_rmse"]
                d2 = m["delta_r2"]
                r_better = (dr < 0).sum()
                n = len(m)
                print(f"      {model} ({strat}):  ΔRMSE={dr.mean():+.4f}±{dr.std():.4f}"
                      f" ({r_better}/{n} folds↓)"
                      f"  ΔR²={d2.mean():+.4f}±{d2.std():.4f}")

        # wDMPNN comparison
        if not wdmpnn_df.empty:
            w = wdmpnn_df[wdmpnn_df["target"] == target]["test/rmse"]
            if not w.empty:
                print(f"\n    wDMPNN baseline: RMSE = {w.mean():.4f} ± {w.std():.4f}")
                for model in MODELS:
                    for strat in STRATEGIES:
                        s = df[(df["model"] == model) & (df["strategy"] == strat)
                               & (df["target"] == target)]["test/rmse"]
                        if not s.empty:
                            diff = s.mean() - w.mean()
                            tag = "worse" if diff > 0 else "better"
                            print(f"      {model} ({strat}): {s.mean():.4f}"
                                  f" ({diff:+.4f} vs wDMPNN, {tag})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ea_ip strategy comparison  (+PT models, monomer split)")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)

    df = load_strategies()
    wdmpnn_df = load_wdmpnn()

    print(f"\nLoaded {len(df)} strategy rows  "
          f"({df['model'].nunique()} models × {df['strategy'].nunique()} strategies × "
          f"{df['target'].nunique()} targets)")
    if not wdmpnn_df.empty:
        print(f"Loaded {len(wdmpnn_df)} wDMPNN rows")

    task1_boxplots(df)
    task2_delta(df)
    task3_paired(df)
    task4_vs_wdmpnn(df, wdmpnn_df)
    summarize(df, wdmpnn_df)

    print(f"\nDone. All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
