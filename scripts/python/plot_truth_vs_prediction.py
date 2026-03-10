#!/usr/bin/env python3
"""
Plot Truth vs Prediction Scatter Plots for Model Comparison

Generates truth-vs-prediction scatter plots from saved .npz prediction files
produced by train_graph.py --save_predictions.

Filename format: {dataset}__{target}[__desc][__rdkit][__batch_norm][__copoly_{mode}][__size{N}]__split{i}.npz

All variants are auto-discovered; use filter flags to narrow down what is shown.

Usage:
    # Auto-discover all variants for GIN on pae_tg Tg (split 0, no size suffix)
    python plot_truth_vs_prediction.py \\
        --dataset pae_tg --target Tg --models GIN

    # Compare all three models, only copolymer mix and interact modes, with descriptors
    python plot_truth_vs_prediction.py \\
        --dataset pae_tg --target Tg \\
        --models DMPNN GIN GAT \\
        --copoly_mode mix interact \\
        --desc yes

    # Compare all copoly modes for DMPNN (aggregated across all splits)
    python plot_truth_vs_prediction.py \\
        --dataset pae_tg --target Tg \\
        --models DMPNN \\
        --aggregate_splits

    # Homopolymer files only (no copoly suffix)
    python plot_truth_vs_prediction.py \\
        --dataset block --target phase_label \\
        --models GIN --copoly_mode none

    # Size-filtered with residuals
    python plot_truth_vs_prediction.py \\
        --dataset opv_camb3lyp --target gap \\
        --models DMPNN AttentiveFP \\
        --size 12000 --residuals

    # Specific splits
    python plot_truth_vs_prediction.py \\
        --dataset pae_tg --target Tg \\
        --models DMPNN --splits 0 2 4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys
from collections import defaultdict

# Import project color scheme
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_colors import MODEL_COLORS, TOL_VIBRANT, STANDARD_GREY, LIGHT_GREY

# --------------------------------------------------------------------------- #
# Plotting configuration
# --------------------------------------------------------------------------- #
plt.style.use('seaborn-v0_8')
FIGSIZE_SINGLE = (5.5, 5)
FIGSIZE_RESIDUAL = (5.5, 3.5)
DPI = 200
SCATTER_ALPHA = 0.35
SCATTER_SIZE = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate truth-vs-prediction scatter plots from saved predictions."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pae_tg)")
    parser.add_argument("--target", required=True, help="Target property name (e.g. Tg)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to compare (e.g. DMPNN GIN GAT)")

    # --- Variant filters (all optional; if omitted, all matching variants are loaded) ---
    parser.add_argument("--copoly_mode", nargs="+", default=None,
                        help="Filter to specific copolymer mode(s) (e.g. mix interact mix_frac). "
                             "Use 'none' for homopolymer/no-copoly files. If omitted, all modes are loaded.")
    parser.add_argument("--size", type=str, default=None,
                        help="Training size filter (e.g. 1000). If omitted, only files WITHOUT a size suffix are loaded.")
    parser.add_argument("--desc", choices=["yes", "no"], default=None,
                        help="Filter by descriptor flag: 'yes' = only __desc files, 'no' = only non-desc.")
    parser.add_argument("--rdkit", choices=["yes", "no"], default=None,
                        help="Filter by rdkit flag: 'yes' = only __rdkit files, 'no' = only non-rdkit.")
    parser.add_argument("--batch_norm", choices=["yes", "no"], default=None,
                        help="Filter by batch norm flag: 'yes' = only __batch_norm files, 'no' = only non-BN.")

    parser.add_argument("--splits", nargs="+", type=int, default=[0],
                        help="Which split indices to plot (default: 0)")
    parser.add_argument("--aggregate_splits", action="store_true",
                        help="Aggregate predictions across all 5 splits (overrides --splits)")
    parser.add_argument("--residuals", action="store_true",
                        help="Also generate residual plots (y_pred - y_true vs y_true)")
    parser.add_argument("--predictions_dir", type=str, default=None,
                        help="Path to predictions directory (default: <project>/predictions)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: <project>/plots/truth_vs_pred)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

# Filename format produced by train_graph.py:
#   {dataset}__{target}[__desc][__rdkit][__batch_norm][__copoly_{mode}][__size{N}]__split{i}.npz

def find_prediction_files(predictions_dir, model_name, dataset, target, splits=None,
                          size=None, copoly_mode=None, desc=None, rdkit=None, batch_norm=None):
    """
    Auto-discover all prediction file variants for a given model/dataset/target.

    Returns dict: {variant_label: {split_idx: path}}

    Filters (all optional):
        size        : if set, only files with __size{size}; if None, only files WITHOUT a size suffix
        copoly_mode : list of mode strings to include (e.g. ["mix", "interact"]);
                      use ["none"] to match homopolymer files with no copoly suffix;
                      if None, all modes are loaded
        desc        : "yes" = only __desc files, "no" = only non-desc, None = both
        rdkit       : "yes" = only __rdkit files, "no" = only non-rdkit, None = both
        batch_norm  : "yes" = only __batch_norm files, "no" = only non-BN, None = both
    """
    model_dir = predictions_dir / model_name
    if not model_dir.exists():
        return {}

    # Match everything between {dataset}__{target} and __split{i}
    prefix_re = re.compile(
        r"^" + re.escape(f"{dataset}__{target}") + r"(.*)__split(\d+)$"
    )

    variants = defaultdict(dict)  # label -> {split_idx: path}

    for f in sorted(model_dir.glob("*.npz")):
        name = f.stem
        m = prefix_re.match(name)
        if not m:
            continue

        suffix_str = m.group(1)   # e.g. "__desc__rdkit__copoly_mix__size1000" or ""
        split_idx  = int(m.group(2))

        # Parse flags from suffix_str
        has_desc       = "__desc"       in suffix_str
        has_rdkit      = "__rdkit"      in suffix_str
        has_batch_norm = "__batch_norm" in suffix_str

        size_m    = re.search(r"__size(\w+)",      suffix_str)
        copoly_m  = re.search(r"__copoly_([a-z_]+)", suffix_str)
        file_size   = size_m.group(1)   if size_m   else None
        file_copoly = copoly_m.group(1) if copoly_m else None

        # --- Apply filters ---
        if size is not None:
            if file_size != str(size):
                continue
        else:
            if file_size is not None:
                continue

        if desc == "yes" and not has_desc:
            continue
        if desc == "no"  and has_desc:
            continue

        if rdkit == "yes" and not has_rdkit:
            continue
        if rdkit == "no"  and has_rdkit:
            continue

        if batch_norm == "yes" and not has_batch_norm:
            continue
        if batch_norm == "no"  and has_batch_norm:
            continue

        if copoly_mode is not None:
            if file_copoly is None:
                if "none" not in copoly_mode:
                    continue
            else:
                if file_copoly not in copoly_mode:
                    continue

        if splits is not None and split_idx not in splits:
            continue

        # Build human-readable variant label from suffix flags
        parts = [model_name]
        if has_desc:
            parts.append("desc")
        if has_rdkit:
            parts.append("rdkit")
        if has_batch_norm:
            parts.append("bn")
        if file_copoly:
            parts.append(file_copoly)
        if file_size:
            parts.append(f"size={file_size}")

        label = " | ".join(parts)
        variants[label][split_idx] = f

    return dict(variants)


def load_predictions(file_dict):
    """
    Load y_true and y_pred from a dict {split_idx: path}.

    Returns (y_true, y_pred) arrays.  When multiple splits are present,
    predictions are concatenated.

    Handles 2D y_pred from classification tasks:
      - shape (n, 1)  → squeeze to 1D (regression with extra dim)
      - shape (n, k)  → argmax over classes (multi-class classification)
    """
    y_true_all, y_pred_all = [], []
    for split_idx in sorted(file_dict.keys()):
        data = np.load(file_dict[split_idx], allow_pickle=True)
        yt = data["y_true"].flatten()
        yp = data["y_pred"]
        if yp.ndim == 2:
            if yp.shape[1] == 1:
                yp = yp.squeeze(1)          # regression with trailing dim
            else:
                yp = np.argmax(yp, axis=1)  # classification: predicted class
        else:
            yp = yp.flatten()
        y_true_all.append(yt)
        y_pred_all.append(yp)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_true, y_pred):
    """Return dict of MAE, RMSE, R2."""
    residuals = y_pred - y_true
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def get_color(label):
    """Get color for a variant label, falling back to Paul Tol palette."""
    # Try exact model name match first
    for model_name, color in MODEL_COLORS.items():
        if label.startswith(model_name):
            return color
    # Fallback cycle through Tol vibrant
    fallback = list(TOL_VIBRANT.values())
    return fallback[hash(label) % len(fallback)]


def plot_truth_vs_pred(all_data, dataset, target, size, output_dir,
                       aggregate_splits=False, splits=None):
    """
    Generate truth-vs-prediction scatter plots.

    all_data: dict {variant_label: (y_true, y_pred)}
    """
    n_variants = len(all_data)
    if n_variants == 0:
        print("No data to plot.")
        return

    # Determine shared axis limits across all variants
    global_min = min(min(yt.min(), yp.min()) for yt, yp in all_data.values())
    global_max = max(max(yt.max(), yp.max()) for yt, yp in all_data.values())
    margin = (global_max - global_min) * 0.05
    lim = (global_min - margin, global_max + margin)

    fig, axes = plt.subplots(1, n_variants, figsize=(FIGSIZE_SINGLE[0] * n_variants, FIGSIZE_SINGLE[1]),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (label, (y_true, y_pred)) in enumerate(all_data.items()):
        ax = axes[idx]
        color = get_color(label)
        metrics = compute_metrics(y_true, y_pred)

        # Diagonal reference
        ax.plot(lim, lim, ls="--", lw=1.2, color=STANDARD_GREY, zorder=1)

        # Scatter
        ax.scatter(y_true, y_pred, s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                   color=color, edgecolors="none", zorder=2)

        # Axis limits and labels
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")

        # Metrics annotation
        textstr = (f"MAE = {metrics['MAE']:.4f}\n"
                   f"RMSE = {metrics['RMSE']:.4f}\n"
                   f"R² = {metrics['R²']:.4f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=LIGHT_GREY))

    # Suptitle
    split_desc = "aggregated splits" if aggregate_splits else f"split(s) {splits}"
    size_desc = f"size={size}" if size else "full"
    fig.suptitle(f"{dataset} — {target}  ({size_desc}, {split_desc})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    models_tag = "_".join(sorted(all_data.keys())).replace(" ", "").replace("+", "plus")
    size_tag = f"_size{size}" if size else ""
    split_tag = "_agg" if aggregate_splits else f"_split{'_'.join(map(str, splits))}"
    fname = f"{dataset}__{target}{size_tag}{split_tag}__truth_vs_pred.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def plot_residuals(all_data, dataset, target, size, output_dir,
                   aggregate_splits=False, splits=None):
    """
    Generate residual plots (y_pred - y_true) vs y_true.

    all_data: dict {variant_label: (y_true, y_pred)}
    """
    n_variants = len(all_data)
    if n_variants == 0:
        return

    # Shared x-axis limits
    global_xmin = min(yt.min() for yt, _ in all_data.values())
    global_xmax = max(yt.max() for yt, _ in all_data.values())
    xmargin = (global_xmax - global_xmin) * 0.05
    xlim = (global_xmin - xmargin, global_xmax + xmargin)

    # Shared y-axis limits (residuals)
    all_residuals = [yp - yt for yt, yp in all_data.values()]
    global_rmin = min(r.min() for r in all_residuals)
    global_rmax = max(r.max() for r in all_residuals)
    rmargin = (global_rmax - global_rmin) * 0.05
    ylim = (global_rmin - rmargin, global_rmax + rmargin)

    fig, axes = plt.subplots(1, n_variants,
                             figsize=(FIGSIZE_RESIDUAL[0] * n_variants, FIGSIZE_RESIDUAL[1]),
                             squeeze=False)
    axes = axes.flatten()

    for idx, (label, (y_true, y_pred)) in enumerate(all_data.items()):
        ax = axes[idx]
        color = get_color(label)
        residuals = y_pred - y_true

        # Zero reference
        ax.axhline(0, ls="--", lw=1.2, color=STANDARD_GREY, zorder=1)

        # Scatter
        ax.scatter(y_true, residuals, s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                   color=color, edgecolors="none", zorder=2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("True", fontsize=11)
        ax.set_ylabel("Predicted − True", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")

        # Mean and std annotation
        textstr = (f"Mean res. = {np.mean(residuals):.4f}\n"
                   f"Std res.  = {np.std(residuals):.4f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=LIGHT_GREY))

    split_desc = "aggregated splits" if aggregate_splits else f"split(s) {splits}"
    size_desc = f"size={size}" if size else "full"
    fig.suptitle(f"{dataset} — {target} residuals  ({size_desc}, {split_desc})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    size_tag = f"_size{size}" if size else ""
    split_tag = "_agg" if aggregate_splits else f"_split{'_'.join(map(str, splits))}"
    fname = f"{dataset}__{target}{size_tag}{split_tag}__residuals.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    # Resolve directories
    project_root = Path(__file__).resolve().parent.parent.parent
    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else project_root / "predictions"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "plots" / "truth_vs_pred"

    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        sys.exit(1)

    # Determine splits
    if args.aggregate_splits:
        splits = list(range(5))
    else:
        splits = args.splits

    print(f"Dataset:     {args.dataset}")
    print(f"Target:      {args.target}")
    print(f"Models:      {args.models}")
    print(f"Size:        {args.size if args.size else 'full (no size suffix)'}")
    print(f"Copoly mode: {args.copoly_mode if args.copoly_mode else 'all'}")
    print(f"Desc:        {args.desc if args.desc else 'all'}")
    print(f"RDKit:       {args.rdkit if args.rdkit else 'all'}")
    print(f"Batch norm:  {args.batch_norm if args.batch_norm else 'all'}")
    print(f"Splits:      {'all (aggregated)' if args.aggregate_splits else splits}")
    print(f"Residuals:   {args.residuals}")
    print(f"Pred dir:    {predictions_dir}")
    print(f"Output dir:  {output_dir}")
    print()

    # Collect predictions for all model variants
    all_data = {}  # {label: (y_true, y_pred)}

    for model in args.models:
        variants = find_prediction_files(
            predictions_dir, model, args.dataset, args.target,
            splits=splits, size=args.size, copoly_mode=args.copoly_mode,
            desc=args.desc, rdkit=args.rdkit, batch_norm=args.batch_norm
        )
        if not variants:
            print(f"  WARNING: No prediction files found for {model} "
                  f"(dataset={args.dataset}, target={args.target}, size={args.size})")
            continue

        for label, file_dict in variants.items():
            n_splits = len(file_dict)
            y_true, y_pred = load_predictions(file_dict)
            all_data[label] = (y_true, y_pred)
            print(f"  Loaded {label}: {len(y_true)} samples from {n_splits} split(s)")

    if not all_data:
        print("\nERROR: No predictions found for any model. Check file paths and filters.")
        sys.exit(1)

    print()

    # Truth vs prediction plots
    plot_truth_vs_pred(all_data, args.dataset, args.target, args.size, output_dir,
                       aggregate_splits=args.aggregate_splits, splits=splits)

    # Residual plots
    if args.residuals:
        plot_residuals(all_data, args.dataset, args.target, args.size, output_dir,
                       aggregate_splits=args.aggregate_splits, splits=splits)

    print("\nDone.")


if __name__ == "__main__":
    main()
