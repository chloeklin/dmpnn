#!/usr/bin/env python3
"""
Truth-curve (predicted vs actual) plots for the ea_ip dataset.

One figure per target (EA vs SHE (eV), IP vs SHE (eV)).
Each figure has 7 subplots — one per model:
  Row 1: DMPNN | GAT | GIN | IdentityBaseline
  Row 2: RF    | XGB | Linear | (empty)

All 5 CV splits are pooled into each subplot.
Scatter points are coloured by poly_type from ea_ip.csv.

Prediction files required:
  predictions/DMPNN/ea_ip__{target}__copoly_mix__split{i}.npz
  predictions/GAT/...
  predictions/GIN/...
  predictions/IdentityBaseline/ea_ip__{target}__copoly_mix__split{i}.npz
  predictions/Tabular/ea_ip__{target}__{feat_sfx}__{MODEL}__split{i}.npz

To generate missing files re-run the relevant training script with
--save_predictions, e.g.:
  python3 train_identity_baseline.py --dataset_name ea_ip ... --save_predictions
  python3 train_tabular.py           --dataset_name ea_ip ... --save_predictions
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_CSV      = PROJECT_DIR / "data" / "ea_ip.csv"
PRED_DIR      = PROJECT_DIR / "predictions"
OUTPUT_DIR    = PROJECT_DIR / "plots" / "ea_ip_truth_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
N_SPLITS = 5

# Set by CLI args at runtime
SPLIT_TYPE = "random"  # default; overridden by --split_type

# ── model configs ──────────────────────────────────────────────────────────
# Each entry: (display_label, pred_dir_name, file_pattern_fn)
#   file_pattern_fn(target, split_idx) -> Path relative to PRED_DIR

def _graph_pattern(model_dir, target, mode="copoly_mix"):
    """Return list of npz paths for all splits of a graph/identity model.

    For 'random' split: tries new naming first, falls back to old unsuffixed
    naming for backward compatibility.
    For other splits (e.g. 'a_held_out'): only looks for split-typed files;
    never falls back to unsuffixed files (which are random-split data).
    """
    paths = []
    for i in range(N_SPLITS):
        new_p = PRED_DIR / model_dir / f"ea_ip__{target}__{mode}__{SPLIT_TYPE}__split{i}.npz"
        if new_p.exists():
            paths.append(new_p)
        elif SPLIT_TYPE == "random":
            # backward compat: old files have no split suffix
            old_p = PRED_DIR / model_dir / f"ea_ip__{target}__{mode}__split{i}.npz"
            paths.append(old_p)
        else:
            # non-random split: return expected path; load_predictions skips missing
            paths.append(new_p)
    return paths


def _tabular_pattern(model_name, target):
    """Return the best-available npz file list for a tabular model.

    Tries feature sets in descending preference.
    For 'random' split: also falls back to old unsuffixed naming.
    For other splits: only uses split-typed filenames.
    """
    feat_candidates = ["__rdkit__ab", "__ab", "__rdkit", ""]
    for feat in feat_candidates:
        # new naming (with split_type)
        paths = []
        ok = True
        for i in range(N_SPLITS):
            p = PRED_DIR / "Tabular" / f"ea_ip__{target}{feat}__{SPLIT_TYPE}__{model_name}__split{i}.npz"
            if not p.exists():
                ok = False
                break
            paths.append(p)
        if ok:
            return paths

        # old naming (no split_type) — only valid as fallback for random split
        if SPLIT_TYPE == "random":
            paths = []
            ok = True
            for i in range(N_SPLITS):
                p = PRED_DIR / "Tabular" / f"ea_ip__{target}{feat}__{model_name}__split{i}.npz"
                if not p.exists():
                    ok = False
                    break
                paths.append(p)
            if ok:
                return paths

    # Partial-match fallback (any split present) — same split-type restriction
    for feat in feat_candidates:
        sfx_candidates = [f"__{SPLIT_TYPE}"]
        if SPLIT_TYPE == "random":
            sfx_candidates.append("")
        for sfx in sfx_candidates:
            paths = [
                PRED_DIR / "Tabular" / f"ea_ip__{target}{feat}{sfx}__{model_name}__split{i}.npz"
                for i in range(N_SPLITS)
            ]
            if any(p.exists() for p in paths):
                return paths
    return []


MODELS = [
    # (label, row, col, npz_getter)
    ("DMPNN",            0, 0, lambda t: _graph_pattern("DMPNN", t)),
    ("wDMPNN",           0, 1, lambda t: _graph_pattern("wDMPNN", t)),
    ("GAT",              0, 2, lambda t: _graph_pattern("GAT", t)),
    ("GIN",              0, 3, lambda t: _graph_pattern("GIN", t)),
    ("IdentityBaseline", 1, 0, lambda t: _graph_pattern("IdentityBaseline", t)),
    ("RF",               1, 1, lambda t: _tabular_pattern("RF", t)),
    ("XGB",              1, 2, lambda t: _tabular_pattern("XGB", t)),
    ("Linear",           1, 3, lambda t: _tabular_pattern("Linear", t)),
]

# ── colour palette for poly_type ──────────────────────────────────────────
# Colours assigned dynamically from this palette
POLY_PALETTE = [
    "#2171b5", "#cb181d", "#238b45", "#d94801",
    "#6a51a3", "#636363", "#8c6d31", "#3182bd",
]

# ── helpers ───────────────────────────────────────────────────────────────

def load_ea_ip():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"ea_ip.csv not found at {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded ea_ip.csv: {len(df)} rows, poly_types: {sorted(df['poly_type'].unique())}")
    return df


def load_predictions(npz_paths: list, df: pd.DataFrame, target_col: str):
    """Load and pool predictions across splits; return (y_true, y_pred, poly_type)."""
    all_true, all_pred, all_pt = [], [], []
    target_vals = df[target_col].values.astype(float)

    for p in npz_paths:
        if not p.exists():
            continue
        data = np.load(p, allow_pickle=True)
        yt = data["y_true"].flatten().astype(float)
        yp = data["y_pred"].flatten().astype(float)

        if "test_indices" in data:
            indices = data["test_indices"].astype(int)
            # Indices are into the filtered (valid) subset; fallback to nearest if OOB
            try:
                pts = df.iloc[indices]["poly_type"].values
            except IndexError:
                pts = np.array([df.iloc[int(np.argmin(np.abs(target_vals - v)))]["poly_type"]
                                for v in yt])
        else:
            # Nearest-neighbour matching by y_true value
            pts = np.array([df.iloc[int(np.argmin(np.abs(target_vals - v)))]["poly_type"]
                            for v in yt])

        all_true.append(yt)
        all_pred.append(yp)
        all_pt.append(pts)

    if not all_true:
        return None, None, None
    return (np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_pt))


def metrics_str(y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return f"R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"


# ── main plot ─────────────────────────────────────────────────────────────

def plot_target(df: pd.DataFrame, target: str, poly_colors: dict):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"Truth curves — {target}  [{SPLIT_TYPE}]", fontsize=14, fontweight="bold", y=1.01)

    # All 8 subplots used (4 graph + 4 tabular/identity)

    for label, row, col, getter in MODELS:
        ax = axes[row, col]
        npz_paths = getter(target)

        y_true, y_pred, poly_types = load_predictions(npz_paths, df, target)

        if y_true is None:
            ax.text(0.5, 0.5, "No predictions\navailable",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("True"); ax.set_ylabel("Predicted")
            continue

        # Plot diagonal
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=0.8, alpha=0.5)

        # Scatter coloured by poly_type
        for pt, color in poly_colors.items():
            mask = poly_types == pt
            if not mask.any():
                continue
            ax.scatter(y_true[mask], y_pred[mask], c=color, s=18,
                       alpha=0.65, linewidths=0, label=pt)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=8)
        ax.text(0.03, 0.97, metrics_str(y_true, y_pred),
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=pt)
        for pt, color in poly_colors.items()
    ]
    fig.legend(handles=legend_handles, title="poly_type",
               loc="lower right", bbox_to_anchor=(0.98, 0.02),
               ncol=1, fontsize=9, title_fontsize=9, frameon=True)

    fig.tight_layout()
    safe_name = target.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    out = OUTPUT_DIR / f"ea_ip_truth_curves_{safe_name}__{SPLIT_TYPE}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    global SPLIT_TYPE
    parser = argparse.ArgumentParser(description="Plot ea_ip truth curves coloured by poly_type")
    parser.add_argument("--split_type", default="random",
                        choices=["random", "a_held_out"],
                        help="Which split to plot (default: random)")
    args = parser.parse_args()
    SPLIT_TYPE = args.split_type
    print(f"Split type: {SPLIT_TYPE}")

    df = load_ea_ip()

    # Build poly_type → colour mapping
    poly_types_all = sorted(df["poly_type"].dropna().unique())
    poly_colors = {pt: POLY_PALETTE[i % len(POLY_PALETTE)]
                   for i, pt in enumerate(poly_types_all)}
    print(f"poly_type colours: {poly_colors}")

    for target in TARGETS:
        if target not in df.columns:
            print(f"[SKIP] target '{target}' not found in ea_ip.csv columns")
            continue
        print(f"\nPlotting: {target}")
        plot_target(df, target, poly_colors)

    print(f"\nAll plots → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
