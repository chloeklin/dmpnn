"""Comparison plots for ea_ip copolymer models (EA/IP vs SHE).

Models Compared
---------------
- DMPNN+polytype (mix_meta for random, interact_meta for monomer)
- GIN+polytype (interact_meta + poly_type)
- GAT+polytype (interact_meta + poly_type)
- wDMPNN (monomer split only)

Tasks
-----
1. Parity plots  – requires per-sample predictions (predictions/).
                   Auto-generates when .npz files are present.
2. Box plots of RMSE / MAE / R² across CV folds.
3. Paired scatter plots: wDMPNN vs each baseline model, per fold.

Outputs (analysis/ea_ip_report/)
---------------------------------
Task 1 (when .npz data available):
    parity_random.png, parity_monomer.png

Task 2:
    box_rmse_random.png,   box_rmse_monomer.png
    box_mae_random.png,    box_mae_monomer.png
    box_r2_random.png,     box_r2_monomer.png

Task 3 (per metric × split):
    paired_rmse_monomer.png (random skipped: wDMPNN unavailable)
    paired_mae_monomer.png
    paired_r2_monomer.png

Configuration
-------------
Edit MODEL_FILES below to adjust which result CSV is used for each model.
"""

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
ROOT_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
PRED_DIR    = ROOT_DIR / "predictions"
OUT_DIR     = SCRIPT_DIR / "ea_ip_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
TARGET_LABELS = {
    "EA vs SHE (eV)": "EA vs SHE (eV)",
    "IP vs SHE (eV)": "IP vs SHE (eV)",
}

METRICS = ["test/rmse", "test/mae", "test/r2"]
METRIC_LABELS = {
    "test/rmse": "RMSE (eV)",
    "test/mae":  "MAE (eV)",
    "test/r2":   "R²",
}
METRIC_SHORT = {
    "test/rmse": "rmse",
    "test/mae":  "mae",
    "test/r2":   "r2",
}

SPLIT_LABELS = {
    "random":    "Random Split",
    "a_held_out": "Monomer (A-held-out) Split",
}
SPLIT_SHORT = {
    "random":    "random",
    "a_held_out": "monomer",
}

# ── Model colours ──────────────────────────────────────────────────────────────
MODEL_ORDER = ["DMPNN+polytype", "GIN+polytype", "GAT+polytype", "wDMPNN"]
MODEL_COLORS = {
    "DMPNN+polytype": "#4C72B0",
    "GIN+polytype":   "#DD8452",
    "GAT+polytype":   "#55A868",
    "wDMPNN":         "#C44E52",
}

# ── Prediction directory mapping ───────────────────────────────────────────────
# Map display names to actual prediction subdirectory names
PRED_DIR_MAP = {
    "DMPNN+polytype": "DMPNN",
    "GIN+polytype":   "GIN",
    "GAT+polytype":   "GAT",
    "wDMPNN":         "wDMPNN",
}

# ── Copolymer mode filtering ───────────────────────────────────────────────────
# Map (model, split_type) to expected copolymer_mode to match results CSVs
PRED_MODE_FILTER = {
    ("DMPNN+polytype", "random"):    "mean",
    ("DMPNN+polytype", "a_held_out"): "mean",
    ("GIN+polytype", "random"):       "mean",
    ("GIN+polytype", "a_held_out"):   "mean",
    ("GAT+polytype", "random"):       "mean",
    ("GAT+polytype", "a_held_out"):   "mean",
    ("wDMPNN", "random"):             "mix",
    ("wDMPNN", "a_held_out"):         "mix",
}

# ── Model → result-file mapping ────────────────────────────────────────────────
# List any number of CSV paths (relative to RESULTS_DIR).
# Files are concatenated; files with multiple targets are handled automatically.
MODEL_FILES = {
    "DMPNN+polytype": {
        "random": [
            "DMPNN/ea_ip__copoly_mean__target_EA vs SHE (eV)_results.csv",
            "DMPNN/ea_ip__copoly_mean__target_IP vs SHE (eV)_results.csv",
        ],
        "a_held_out": [
            "DMPNN/ea_ip__copoly_mean__a_held_out__target_EA vs SHE (eV)_results.csv",
            "DMPNN/ea_ip__copoly_mean__a_held_out__target_IP vs SHE (eV)_results.csv",
        ],
    },
    "GIN+polytype": {
        "random": [
            "GIN/ea_ip__copoly_mean__target_EA vs SHE (eV)_results.csv",
            "GIN/ea_ip__copoly_mean__target_IP vs SHE (eV)_results.csv",
        ],
        "a_held_out": [
            "GIN/ea_ip__copoly_mean__a_held_out__target_EA vs SHE (eV)_results.csv",
            "GIN/ea_ip__copoly_mean__a_held_out__target_IP vs SHE (eV)_results.csv",
        ],
    },
    "GAT+polytype": {
        "random": [
            "GAT/ea_ip__copoly_mean__target_EA vs SHE (eV)_results.csv",
            "GAT/ea_ip__copoly_mean__target_IP vs SHE (eV)_results.csv",
        ],
        "a_held_out": [
            "GAT/ea_ip__copoly_mean__a_held_out__target_EA vs SHE (eV)_results.csv",
            "GAT/ea_ip__copoly_mean__a_held_out__target_IP vs SHE (eV)_results.csv",
        ],
    },
    "wDMPNN": {
        "random": [],          # not available
        "a_held_out": [
            "wDMPNN/ea_ip__a_held_out__target_EA vs SHE (eV)_results.csv",
            "wDMPNN/ea_ip__a_held_out__target_IP vs SHE (eV)_results.csv",
        ],
    },
}

# ── Matplotlib style ───────────────────────────────────────────────────────────
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
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_path_for(per_target_path: Path) -> Path:
    """Given .../base__target_X_results.csv, return .../base_results.csv (or None)."""
    m = re.match(r'(.+?)__target_.+_results\.csv$', per_target_path.name)
    if m:
        return per_target_path.parent / f"{m.group(1)}_results.csv"
    return None


def _target_from_per_target_path(per_target_path: Path) -> str:
    """Extract target name from a per-target filename, or None."""
    m = re.match(r'.+?__target_(.+)_results\.csv$', per_target_path.name)
    return m.group(1) if m else None


def _load_single(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # For tabular results, filter by model column if it exists
    if "model" in df.columns and model_name in ["Linear", "RF", "XGB"]:
        df = df[df["model"] == model_name].copy()
    
    # Normalize column names for tabular CSVs to match graph model format
    if "mae" in df.columns and "test/mae" not in df.columns:
        # Convert mse to rmse
        if "mse" in df.columns:
            df["test/rmse"] = np.sqrt(df["mse"])
        df["test/mae"] = df["mae"]
        df["test/r2"] = df["r2"]
    
    df["model"] = model_name
    return df


def load_model_data(model_name: str, split_type: str):
    """Return concatenated DataFrame for one model+split, or None if unavailable."""
    file_list = MODEL_FILES.get(model_name, {}).get(split_type, [])
    if not file_list:
        return None
    frames = []
    for rel in file_list:
        p = RESULTS_DIR / rel
        if p.exists():
            frames.append(_load_single(p, model_name))
        else:
            # Per-target file may have been consolidated into an aggregate CSV.
            # Try loading the aggregate and filtering to the expected target.
            agg = _aggregate_path_for(p)
            target = _target_from_per_target_path(p)
            if agg is not None and agg.exists():
                df = pd.read_csv(agg)
                if "target" in df.columns and target:
                    df = df[df["target"] == target].copy()
                df["model"] = model_name
                frames.append(df)
            else:
                warnings.warn(f"[data] Missing result file: {p}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all(split_type: str) -> pd.DataFrame:
    """Load all models for a split into one tidy DataFrame."""
    parts = [load_model_data(m, split_type) for m in MODEL_ORDER]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        raise RuntimeError(f"No data found for split_type='{split_type}'")
    df = pd.concat(parts, ignore_index=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 – Parity plots
# ─────────────────────────────────────────────────────────────────────────────

def _has_predictions(model_name: str) -> bool:
    actual_dir = PRED_DIR_MAP.get(model_name, model_name)
    pred_subdir = PRED_DIR / actual_dir
    if not pred_subdir.exists():
        return False
    return any(pred_subdir.glob("ea_ip*.npz"))


def task1_parity():
    """Parity plots: create separate figures for each CV split (0-4)."""
    print("\n[Task 1] Parity plots (per-split)")

    available = [m for m in MODEL_ORDER if _has_predictions(m)]
    if not available:
        print(
            "  ⚠  predictions/ subdirectories are empty.\n"
            "     Re-run training with --save_predictions to generate .npz files,\n"
            "     then re-run this script — parity plots will be created automatically."
        )
        return

    # Create separate parity plot for each CV split
    for split_type, split_label in SPLIT_LABELS.items():
        split_short = SPLIT_SHORT[split_type]
        
        # Loop over each CV split (0-4)
        for split_idx in range(5):
            models_here = [m for m in available]
            if not models_here:
                continue

            n_models  = len(models_here)
            n_targets = len(TARGETS)
            fig, axes = plt.subplots(
                n_models, n_targets,
                figsize=(5 * n_targets, 4.5 * n_models),
                squeeze=False,
            )
            fig.suptitle(
                f"Parity plots — {split_label} — Split {split_idx}", 
                fontsize=14, fontweight="bold"
            )

            any_data = False  # Track if this split has any data

            for row, model in enumerate(models_here):
                actual_dir = PRED_DIR_MAP.get(model, model)
                pred_subdir = PRED_DIR / actual_dir
                
                for col, target in enumerate(TARGETS):
                    ax = axes[row][col]
                    
                    # Find the specific prediction file for this split
                    all_npz = sorted(pred_subdir.glob("ea_ip*.npz"))
                    split_tag = "a_held_out" if split_type == "a_held_out" else ""
                    expected_mode = PRED_MODE_FILTER.get((model, split_type), None)
                    
                    # Find file matching: target, split_type, split_idx, and mode
                    target_file = None
                    for f in all_npz:
                        if target not in f.name:
                            continue
                        if f"split{split_idx}" not in f.name:
                            continue
                        
                        # Check split_type
                        if model == "wDMPNN":
                            # wDMPNN: old naming, only has a_held_out
                            if split_type != "a_held_out":
                                continue
                        else:
                            # Other models: check for split_tag in filename
                            if split_tag:
                                if split_tag not in f.name:
                                    continue
                            else:
                                if "a_held_out" in f.name:
                                    continue
                        
                        # Check copolymer_mode
                        if expected_mode is not None:
                            data = np.load(f, allow_pickle=True)
                            meta = data.get("metadata", np.array({})).item() if "metadata" in data else {}
                            actual_mode = meta.get("copolymer_mode", None)
                            if actual_mode != expected_mode:
                                continue
                        
                        target_file = f
                        break
                    
                    if target_file is None:
                        ax.set_visible(False)
                        continue
                    
                    # Load predictions for this specific split
                    data = np.load(target_file, allow_pickle=True)
                    y_true = data["y_true"]
                    y_pred = data["y_pred"]
                    
                    # Flatten if needed
                    if y_true.ndim > 1:
                        y_true = y_true.flatten()
                    if y_pred.ndim > 1:
                        y_pred = y_pred.flatten()
                    
                    any_data = True
                    
                    # Plot this split's data
                    lo = min(y_true.min(), y_pred.min())
                    hi = max(y_true.max(), y_pred.max())
                    pad = (hi - lo) * 0.05
                    lim = [lo - pad, hi + pad]

                    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y = x")
                    ax.scatter(y_true, y_pred, s=15, alpha=0.4,
                               color=MODEL_COLORS.get(model, "#888"),
                               edgecolors="none")
                    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    ax.set_xlim(lim); ax.set_ylim(lim)
                    ax.set_xlabel(f"True {target}", fontsize=10)
                    ax.set_ylabel(f"Predicted {target}", fontsize=10)
                    ax.set_title(
                        f"{model}\nRMSE={rmse:.3f}  R²={r2:.4f}",
                        fontsize=10, fontweight="bold",
                    )
                    sns.despine(ax=ax)

            # Save figure for this split (only if it has data)
            if any_data:
                fig.tight_layout()
                fname = OUT_DIR / f"parity_{split_short}_split{split_idx}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {fname.name}")
            else:
                plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 – Box plots
# ─────────────────────────────────────────────────────────────────────────────

def task2_boxplots(split_type: str):
    split_label = SPLIT_LABELS[split_type]
    split_short = SPLIT_SHORT[split_type]

    try:
        df = load_all(split_type)
    except RuntimeError as e:
        print(f"  ⚠  {e}")
        return

    models_present = [m for m in MODEL_ORDER if m in df["model"].values]
    targets_present = sorted(df["target"].unique())
    multi_target = len(targets_present) > 1

    for metric in METRICS:
        metric_label = METRIC_LABELS[metric]
        metric_short = METRIC_SHORT[metric]

        # ── figure: one axis per target ────────────────────────────────────
        n_cols = len(targets_present)
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(max(5, len(models_present) * 1.6) * n_cols, 5),
            sharey=False,
            squeeze=False,
        )
        fig.suptitle(
            f"{metric_label} across folds — {split_label}",
            fontsize=13, fontweight="bold",
        )

        for col, target in enumerate(targets_present):
            ax = axes[0][col]
            sub = df[df["target"] == target].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            models_in_sub = [m for m in models_present if m in sub["model"].values]
            # Reset Categorical so only present models appear; avoids palette key errors
            sub = sub.copy()
            sub["model"] = pd.Categorical(sub["model"], categories=models_in_sub, ordered=True)
            full_palette = MODEL_COLORS  # dict covers all keys

            sns.boxplot(
                data=sub, x="model", y=metric,
                hue="model", legend=False,
                order=models_in_sub,
                palette=full_palette,
                width=0.45, linewidth=1.2, fliersize=0,
                ax=ax,
            )
            sns.stripplot(
                data=sub, x="model", y=metric,
                hue="model", legend=False,
                order=models_in_sub,
                palette=full_palette,
                jitter=0.12, size=6, alpha=0.75,
                linewidth=0.5, edgecolor="white",
                ax=ax,
            )

            ax.set_title(TARGET_LABELS.get(target, target), fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(metric_label if col == 0 else "")
            ax.tick_params(axis="x", rotation=20)

            # Annotate mean ± std above each box
            for i, model in enumerate(models_in_sub):
                vals = sub.loc[sub["model"] == model, metric].dropna()
                if vals.empty:
                    continue
                y_top = vals.max()
                ax.text(
                    i, y_top * (1.01 if metric != "test/r2" else 1.001),
                    f"μ={vals.mean():.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="dimgrey",
                )

            sns.despine(ax=ax)

        fig.tight_layout()
        fname = OUT_DIR / f"box_{metric_short}_{split_short}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 – Paired scatter plots (wDMPNN vs baselines)
# ─────────────────────────────────────────────────────────────────────────────

def task3_paired(split_type: str):
    split_label = SPLIT_LABELS[split_type]
    split_short = SPLIT_SHORT[split_type]

    try:
        df = load_all(split_type)
    except RuntimeError as e:
        print(f"  ⚠  {e}")
        return

    if "wDMPNN" not in df["model"].values:
        print(
            f"  ⚠  wDMPNN has no results for split='{split_type}' — "
            f"skipping paired plots."
        )
        return

    wdf = df[df["model"] == "wDMPNN"].copy()
    baselines = [m for m in ["DMPNN+polytype", "GIN+polytype", "GAT+polytype"]
                 if m in df["model"].values]
    targets = sorted(df["target"].unique())

    for metric in METRICS:
        metric_label = METRIC_LABELS[metric]
        metric_short = METRIC_SHORT[metric]

        n_rows = len(baselines)
        n_cols = len(targets)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 4.2 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"wDMPNN vs baselines — {metric_label}\n({split_label})",
            fontsize=13, fontweight="bold",
        )

        for row, baseline in enumerate(baselines):
            bdf = df[df["model"] == baseline].copy()

            for col, target in enumerate(targets):
                ax = axes[row][col]

                b_sub = bdf[bdf["target"] == target].sort_values("split")
                w_sub = wdf[wdf["target"] == target].sort_values("split")

                # Align on shared split indices
                common_splits = sorted(
                    set(b_sub["split"].values) & set(w_sub["split"].values)
                )
                if not common_splits:
                    ax.set_visible(False)
                    continue

                b_vals = b_sub.set_index("split").loc[common_splits, metric].values
                w_vals = w_sub.set_index("split").loc[common_splits, metric].values

                lo = min(b_vals.min(), w_vals.min())
                hi = max(b_vals.max(), w_vals.max())
                pad = max((hi - lo) * 0.12, 1e-4)
                lim = [lo - pad, hi + pad]

                ax.plot(lim, lim, "k--", lw=1, alpha=0.5)

                # Shade region: below diagonal = wDMPNN better (lower RMSE/MAE)
                # For R² it's above diagonal
                better_below = metric in ("test/rmse", "test/mae")
                if better_below:
                    ax.fill_between(lim, lim, [lim[0], lim[0]],
                                    color=MODEL_COLORS["wDMPNN"], alpha=0.05)
                    ax.text(lim[0] + pad * 0.3, lim[1] - pad * 1.5,
                            "wDMPNN\nbetter", fontsize=7.5, color=MODEL_COLORS["wDMPNN"],
                            alpha=0.7, ha="left")
                else:
                    ax.fill_between(lim, lim, [lim[1], lim[1]],
                                    color=MODEL_COLORS["wDMPNN"], alpha=0.05)
                    ax.text(lim[0] + pad * 0.3, lim[1] - pad * 0.5,
                            "wDMPNN\nbetter", fontsize=7.5, color=MODEL_COLORS["wDMPNN"],
                            alpha=0.7, ha="left")

                ax.scatter(
                    b_vals, w_vals,
                    color=MODEL_COLORS["wDMPNN"],
                    s=55, alpha=0.85,
                    edgecolors="white", linewidths=0.5, zorder=3,
                )
                for fold_i, (bv, wv) in zip(common_splits, zip(b_vals, w_vals)):
                    ax.annotate(
                        str(fold_i), (bv, wv),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=8, color="dimgrey",
                    )

                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xlabel(f"{baseline}", fontsize=10)
                ax.set_ylabel("wDMPNN" if col == 0 else "", fontsize=10)
                ax.set_title(
                    TARGET_LABELS.get(target, target)
                    if row == 0 else "",
                    fontsize=10,
                )
                # Row label
                if col == 0:
                    ax.set_ylabel(
                        f"wDMPNN\n{metric_label}",
                        fontsize=10,
                    )
                if col == n_cols - 1:
                    ax2 = ax.twinx()
                    ax2.set_ylabel(f"vs {baseline}", fontsize=9, rotation=270,
                                   labelpad=14, color="dimgrey")
                    ax2.set_yticks([])

                sns.despine(ax=ax)

        fig.tight_layout()
        fname = OUT_DIR / f"paired_{metric_short}_{split_short}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 60
    print(sep)
    print("ea_ip model comparison plots")
    print(f"Output directory: {OUT_DIR}")
    print(sep)

    # ── Task 1 ──
    task1_parity()

    # ── Task 2 ──
    print("\n[Task 2] Box plots of metrics across folds")
    for split_type in ["random", "a_held_out"]:
        print(f"  Split: {split_type}")
        task2_boxplots(split_type)

    # ── Task 3 ──
    print("\n[Task 3] Paired scatter plots (wDMPNN vs baselines)")
    for split_type in ["random", "a_held_out"]:
        print(f"  Split: {split_type}")
        task3_paired(split_type)

    print(f"\nDone. All figures saved to {OUT_DIR}")
