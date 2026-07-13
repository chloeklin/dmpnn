"""Effect of polymer-type (PT) features on ea_ip — mean strategy, monomer split.

Compares no-PT (copoly_mean) vs +PT (copoly_mean_meta + poly_type) for
DMPNN, GIN, and GAT on the a_held_out (monomer) split.

Tasks
-----
1. Box plots:  RMSE and R² by model, hue = PT condition.
2. Delta plots: per-fold improvement (+PT − no-PT).
3. Paired scatter: no-PT vs +PT per model.

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
PT_LABELS = {"no-PT": "no-PT", "+PT": "+PT"}
PT_COLORS = {"no-PT": "#888888", "+PT": "#4C72B0"}

# ── File mapping ──────────────────────────────────────────────────────────────
# no-PT:  copoly_mean, a_held_out
# +PT:    copoly_mean_meta + poly_type, a_held_out
#
# Files may be aggregate (both targets in one CSV with 'target' column)
# or per-target (one target per CSV).  The loader handles both.

NO_PT_FILES = {
    "DMPNN": ["DMPNN/ea_ip__copoly_mean__a_held_out_results.csv"],
    "GIN":   ["GIN/ea_ip__copoly_mean__a_held_out_results.csv"],
    "GAT": [
        "GAT/ea_ip__copoly_mean__a_held_out__target_EA vs SHE (eV)_results.csv",
        "GAT/ea_ip__copoly_mean__a_held_out__target_IP vs SHE (eV)_results.csv",
    ],
}

PT_FILES = {
    "DMPNN": [
        "DMPNN/ea_ip__copoly_mean_meta__poly_type__a_held_out__target_EA vs SHE (eV)_results.csv",
        "DMPNN/ea_ip__copoly_mean_meta__poly_type__a_held_out__target_IP vs SHE (eV)_results.csv",
    ],
    "GIN": [
        "GIN/ea_ip__copoly_mean_meta__poly_type__a_held_out__target_EA vs SHE (eV)_results.csv",
        "GIN/ea_ip__copoly_mean_meta__poly_type__a_held_out__target_IP vs SHE (eV)_results.csv",
    ],
    "GAT": [
        "GAT/ea_ip__copoly_mean_meta__poly_type__a_held_out__target_EA vs SHE (eV)_results.csv",
        "GAT/ea_ip__copoly_mean_meta__poly_type__a_held_out__target_IP vs SHE (eV)_results.csv",
    ],
}

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
    """Given .../base__target_X_results.csv → .../base_results.csv."""
    m = re.match(r'(.+?)__target_.+_results\.csv$', per_target_path.name)
    if m:
        return per_target_path.parent / f"{m.group(1)}_results.csv"
    return None


def _target_from_path(per_target_path: Path):
    m = re.match(r'.+?__target_(.+)_results\.csv$', per_target_path.name)
    return m.group(1) if m else None


def _load_files(file_list: list[str], model: str, pt_cond: str) -> pd.DataFrame:
    """Load a list of result CSV paths into a single DataFrame."""
    frames = []
    for rel in file_list:
        p = RESULTS_DIR / rel
        if p.exists():
            df = pd.read_csv(p)
        else:
            # Fallback: try aggregate file filtered by target
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
    df["model"] = model
    df["pt"] = pt_cond
    # Rename 'split' → 'fold' for clarity
    if "split" in df.columns:
        df = df.rename(columns={"split": "fold"})
    return df


def load_all() -> pd.DataFrame:
    """Load no-PT and +PT results for all models into one tidy DataFrame."""
    parts = []
    for model in MODELS:
        for pt_cond, file_map in [("no-PT", NO_PT_FILES), ("+PT", PT_FILES)]:
            df = _load_files(file_map.get(model, []), model, pt_cond)
            if not df.empty:
                parts.append(df)
    if not parts:
        raise RuntimeError("No data found")
    df = pd.concat(parts, ignore_index=True)
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    df["pt"] = pd.Categorical(df["pt"], categories=["no-PT", "+PT"], ordered=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 – Box plots
# ─────────────────────────────────────────────────────────────────────────────

def task1_boxplots(df: pd.DataFrame):
    """Box plots of RMSE and R² by model, hue = PT condition."""
    print("\n[Task 1] Box plots")
    metrics = {"test/rmse": ("RMSE (eV)", "box_rmse_monomer_pt_vs_nopt.png"),
               "test/r2":   ("R²",        "box_r2_monomer_pt_vs_nopt.png")}

    targets_present = sorted(df["target"].unique())

    for metric, (ylabel, fname) in metrics.items():
        n_cols = len(targets_present)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        fig.suptitle(f"{ylabel}  —  mean strategy, monomer split",
                     fontsize=13, fontweight="bold")

        for col, target in enumerate(targets_present):
            ax = axes[0][col]
            sub = df[df["target"] == target].copy()

            sns.boxplot(
                data=sub, x="model", y=metric, hue="pt",
                palette=PT_COLORS, width=0.5, linewidth=1.2,
                fliersize=0, ax=ax,
            )
            sns.stripplot(
                data=sub, x="model", y=metric, hue="pt",
                palette=PT_COLORS, dodge=True,
                jitter=0.08, size=6, alpha=0.8,
                linewidth=0.5, edgecolor="white", ax=ax,
            )
            # Remove duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            n_unique = len(PT_COLORS)
            ax.legend(handles[:n_unique], labels[:n_unique],
                      title="", loc="best", frameon=True)

            ax.set_title(target, fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 – Delta plots
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold delta = +PT − no-PT for each (model, target, fold)."""
    nopt = df[df["pt"] == "no-PT"].set_index(["model", "target", "fold"])
    pt   = df[df["pt"] == "+PT"].set_index(["model", "target", "fold"])
    # Inner join on shared folds
    joined = nopt[["test/rmse", "test/r2"]].join(
        pt[["test/rmse", "test/r2"]], lsuffix="_nopt", rsuffix="_pt", how="inner"
    )
    joined["delta_rmse"] = joined["test/rmse_pt"] - joined["test/rmse_nopt"]
    joined["delta_r2"]   = joined["test/r2_pt"]   - joined["test/r2_nopt"]
    return joined.reset_index()


def task2_delta(df: pd.DataFrame):
    """Delta plots: per-fold improvement (+PT − no-PT)."""
    print("\n[Task 2] Delta plots")
    deltas = _compute_deltas(df)
    if deltas.empty:
        print("  ⚠  Could not compute deltas (missing paired folds)")
        return

    targets_present = sorted(deltas["target"].unique())
    metric_info = {
        "delta_rmse": ("ΔRMSE (+PT − no-PT)", "delta_rmse_monomer_pt.png",
                       "negative = +PT better"),
        "delta_r2":   ("ΔR² (+PT − no-PT)",   "delta_r2_monomer_pt.png",
                       "positive = +PT better"),
    }

    for metric, (ylabel, fname, note) in metric_info.items():
        n_cols = len(targets_present)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        fig.suptitle(f"{ylabel}  —  mean strategy, monomer split\n({note})",
                     fontsize=12, fontweight="bold")

        for col, target in enumerate(targets_present):
            ax = axes[0][col]
            sub = deltas[deltas["target"] == target].copy()

            sns.stripplot(
                data=sub, x="model", y=metric,
                color="#4C72B0", size=8, alpha=0.8,
                jitter=0.1, linewidth=0.5, edgecolor="white",
                ax=ax,
            )
            # Horizontal reference at 0
            ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)

            # Mean marker
            for i, model in enumerate(MODELS):
                vals = sub[sub["model"] == model][metric]
                if not vals.empty:
                    ax.plot(i, vals.mean(), marker="D", color="#C44E52",
                            markersize=8, zorder=5)

            ax.set_title(target, fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 – Paired scatter plots
# ─────────────────────────────────────────────────────────────────────────────

def task3_paired(df: pd.DataFrame):
    """Paired scatter: no-PT (x) vs +PT (y), one plot per model × metric."""
    print("\n[Task 3] Paired scatter plots")
    deltas = _compute_deltas(df)
    if deltas.empty:
        print("  ⚠  Could not compute pairs")
        return

    targets_present = sorted(deltas["target"].unique())
    metrics = {
        "test/rmse": ("RMSE (eV)", "rmse", "below diagonal = +PT better"),
        "test/r2":   ("R²",        "r2",   "above diagonal = +PT better"),
    }

    for model in MODELS:
        for metric, (mlabel, mshort, interp) in metrics.items():
            sub = deltas[deltas["model"] == model]
            if sub.empty:
                continue

            n_cols = len(targets_present)
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                                     squeeze=False)
            fig.suptitle(f"{model} — {mlabel}: no-PT vs +PT\n({interp})",
                         fontsize=12, fontweight="bold")

            for col, target in enumerate(targets_present):
                ax = axes[0][col]
                t_sub = sub[sub["target"] == target]
                if t_sub.empty:
                    ax.set_visible(False)
                    continue

                x = t_sub[f"{metric}_nopt"].values
                y = t_sub[f"{metric}_pt"].values

                # Diagonal
                lo = min(x.min(), y.min())
                hi = max(x.max(), y.max())
                pad = (hi - lo) * 0.08
                lim = [lo - pad, hi + pad]
                ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y = x")

                ax.scatter(x, y, s=60, alpha=0.85, color="#4C72B0",
                           edgecolors="white", linewidth=0.5, zorder=3)

                # Annotate fold indices
                for i, fold in enumerate(t_sub["fold"].values):
                    ax.annotate(str(int(fold)),
                                (x[i], y[i]), fontsize=7,
                                textcoords="offset points", xytext=(5, 5))

                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.set_aspect("equal")
                ax.set_xlabel(f"no-PT  {mlabel}")
                ax.set_ylabel(f"+PT  {mlabel}")
                ax.set_title(target, fontsize=11)
                ax.legend(fontsize=8)
                sns.despine(ax=ax)

            fig.tight_layout()
            fname = f"paired_{model.lower()}_{mshort}_pt_vs_nopt.png"
            fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame):
    """Print a concise text summary of the PT effect."""
    print("\n" + "=" * 60)
    print("Summary: effect of +PT (mean strategy, monomer split)")
    print("=" * 60)
    deltas = _compute_deltas(df)
    if deltas.empty:
        print("  No paired data to summarize.")
        return

    for target in sorted(deltas["target"].unique()):
        print(f"\n  Target: {target}")
        t = deltas[deltas["target"] == target]
        for model in MODELS:
            m = t[t["model"] == model]
            if m.empty:
                print(f"    {model}: no data")
                continue
            dr = m["delta_rmse"]
            d2 = m["delta_r2"]
            rmse_better = (dr < 0).sum()
            r2_better   = (d2 > 0).sum()
            n = len(m)
            print(f"    {model}:  ΔRMSE = {dr.mean():+.4f} ± {dr.std():.4f}"
                  f"  ({rmse_better}/{n} folds improved)"
                  f"  |  ΔR² = {d2.mean():+.4f} ± {d2.std():.4f}"
                  f"  ({r2_better}/{n} folds improved)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ea_ip PT-effect analysis  (mean strategy, monomer split)")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    df = load_all()
    print(f"\nLoaded {len(df)} rows  "
          f"({df['model'].nunique()} models × {df['pt'].nunique()} conditions × "
          f"{df['target'].nunique()} targets)")

    task1_boxplots(df)
    task2_delta(df)
    task3_paired(df)
    summarize(df)

    print(f"\nDone. All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
