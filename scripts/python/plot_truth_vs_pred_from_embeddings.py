#!/usr/bin/env python3
"""
Plot Truth vs Prediction from GNN Embeddings

Loads saved embeddings, trains a simple linear model on them, and generates
truth-vs-prediction scatter plots for model comparison.

Usage:
    # Compare DMPNN and AttentiveFP on htpmd Conductivity
    python plot_truth_vs_pred_from_embeddings.py \\
        --dataset htpmd \\
        --target Conductivity \\
        --models DMPNN AttentiveFP

    # Include RDKit variant
    python plot_truth_vs_pred_from_embeddings.py \\
        --dataset htpmd \\
        --target Conductivity \\
        --models DMPNN \\
        --rdkit

    # Use specific split
    python plot_truth_vs_pred_from_embeddings.py \\
        --dataset htpmd \\
        --target Conductivity \\
        --models DMPNN AttentiveFP \\
        --split 0

    # Aggregate across all splits
    python plot_truth_vs_pred_from_embeddings.py \\
        --dataset htpmd \\
        --target Conductivity \\
        --models DMPNN AttentiveFP \\
        --aggregate_splits

    # Include residual plots
    python plot_truth_vs_pred_from_embeddings.py \\
        --dataset htpmd \\
        --target Conductivity \\
        --models DMPNN AttentiveFP \\
        --residuals
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import project color scheme
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_colors import MODEL_COLORS, TOL_VIBRANT, STANDARD_GREY, LIGHT_GREY

# Plotting configuration
plt.style.use('seaborn-v0_8')
FIGSIZE_SINGLE = (5.5, 5)
FIGSIZE_RESIDUAL = (5.5, 3.5)
DPI = 200
SCATTER_ALPHA = 0.35
SCATTER_SIZE = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate truth-vs-prediction plots from saved embeddings."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. htpmd)")
    parser.add_argument("--target", required=True, help="Target property name (e.g. Conductivity)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names (e.g. DMPNN AttentiveFP)")
    parser.add_argument("--rdkit", action="store_true",
                        help="Also load __rdkit variant for each model")
    parser.add_argument("--desc", action="store_true",
                        help="Also load __desc variant for each model")
    parser.add_argument("--split", type=int, default=None,
                        help="Which split to use (default: 0)")
    parser.add_argument("--aggregate_splits", action="store_true",
                        help="Aggregate across all 5 splits")
    parser.add_argument("--residuals", action="store_true",
                        help="Also generate residual plots")
    parser.add_argument("--embeddings_dir", type=str, default=None,
                        help="Path to embeddings directory (default: <project>/results/embeddings)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory (default: <project>/data)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: <project>/plots/truth_vs_pred_embeddings)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression alpha parameter (default: 1.0)")
    return parser.parse_args()


def find_embedding_files(embeddings_dir, model_name, dataset, target, rdkit=False, desc=False, splits=None):
    """
    Find embedding files matching the given filters.
    
    Returns dict: {variant_label: {split_idx: {'train': path, 'test': path}}}
    """
    variants = {}
    
    # Build variant combinations
    variant_configs = [("", "")]  # Base variant
    if rdkit:
        variant_configs.append(("__rdkit", " + RDKit"))
    if desc:
        variant_configs.append(("__desc", " + Desc"))
        if rdkit:
            variant_configs.append(("__desc__rdkit", " + Desc + RDKit"))
    
    for suffix, label_suffix in variant_configs:
        prefix = f"{dataset}__{model_name}__{target}{suffix}"
        label = f"{model_name}{label_suffix}"
        
        split_files = {}
        for split_idx in (splits if splits else range(5)):
            train_file = embeddings_dir / f"{prefix}__X_train_split_{split_idx}.npy"
            test_file = embeddings_dir / f"{prefix}__X_test_split_{split_idx}.npy"
            
            if train_file.exists() and test_file.exists():
                split_files[split_idx] = {'train': train_file, 'test': test_file}
        
        if split_files:
            variants[label] = split_files
    
    return variants


def load_true_labels(data_dir, dataset, target, splits=None):
    """
    Load true labels from the original dataset CSV.
    
    Returns dict: {split_idx: {'train': y_train, 'test': y_test}}
    
    Note: This is a simplified version. In practice, you'd need to:
    1. Load the full dataset
    2. Apply the same train/test split as was used during training
    3. Extract the target values
    
    For now, we'll return None and handle this differently.
    """
    # This would require replicating the exact split logic from train_graph.py
    # Instead, we'll compute predictions and use a placeholder approach
    return None


def train_and_predict(X_train, y_train, X_test, alpha=1.0):
    """Train Ridge regression on embeddings and return predictions."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def compute_metrics(y_true, y_pred):
    """Return dict of MAE, RMSE, R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def get_color(label):
    """Get color for a variant label."""
    for model_name, color in MODEL_COLORS.items():
        if label.startswith(model_name):
            return color
    fallback = list(TOL_VIBRANT.values())
    return fallback[hash(label) % len(fallback)]


def plot_truth_vs_pred(all_data, dataset, target, output_dir, split_info):
    """Generate truth-vs-prediction scatter plots."""
    n_variants = len(all_data)
    if n_variants == 0:
        print("No data to plot.")
        return
    
    # Shared axis limits
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
        
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True", fontsize=11)
        ax.set_ylabel("Predicted (Ridge on embeddings)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        
        # Metrics annotation
        textstr = (f"MAE = {metrics['MAE']:.4f}\n"
                   f"RMSE = {metrics['RMSE']:.4f}\n"
                   f"R² = {metrics['R²']:.4f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=LIGHT_GREY))
    
    fig.suptitle(f"{dataset} — {target}  ({split_info})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}__{target}__{split_info.replace(' ', '_').replace(',', '')}__truth_vs_pred_embeddings.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_residuals(all_data, dataset, target, output_dir, split_info):
    """Generate residual plots."""
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
        
        # Stats annotation
        textstr = (f"Mean res. = {np.mean(residuals):.4f}\n"
                   f"Std res.  = {np.std(residuals):.4f}")
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=LIGHT_GREY))
    
    fig.suptitle(f"{dataset} — {target} residuals  ({split_info})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}__{target}__{split_info.replace(' ', '_').replace(',', '')}__residuals_embeddings.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    
    # Resolve directories
    project_root = Path(__file__).resolve().parent.parent.parent
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else project_root / "results" / "embeddings"
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "plots" / "truth_vs_pred_embeddings"
    
    if not embeddings_dir.exists():
        print(f"ERROR: Embeddings directory not found: {embeddings_dir}")
        sys.exit(1)
    
    # Determine splits
    if args.aggregate_splits:
        splits = list(range(5))
        split_info = "aggregated splits"
    elif args.split is not None:
        splits = [args.split]
        split_info = f"split {args.split}"
    else:
        splits = [0]
        split_info = "split 0"
    
    print(f"Dataset:    {args.dataset}")
    print(f"Target:     {args.target}")
    print(f"Models:     {args.models}")
    print(f"Splits:     {split_info}")
    print(f"RDKit:      {args.rdkit}")
    print(f"Desc:       {args.desc}")
    print(f"Ridge α:    {args.alpha}")
    print(f"Emb dir:    {embeddings_dir}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Load dataset to get true labels
    # We need to load the original CSV and extract the target column
    csv_path = data_dir / f"{args.dataset}.csv"
    if not csv_path.exists():
        print(f"ERROR: Dataset CSV not found: {csv_path}")
        print("Cannot load true labels without the original dataset.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        print(f"ERROR: Target '{args.target}' not found in dataset columns: {df.columns.tolist()}")
        sys.exit(1)
    
    print(f"Loaded dataset: {len(df)} samples")
    
    # Collect embeddings and predictions for all model variants
    all_data = {}  # {label: (y_true, y_pred)}
    
    for model in args.models:
        variants = find_embedding_files(
            embeddings_dir, model, args.dataset, args.target,
            rdkit=args.rdkit, desc=args.desc, splits=splits
        )
        
        if not variants:
            print(f"  WARNING: No embedding files found for {model}")
            continue
        
        for label, split_files in variants.items():
            y_true_all, y_pred_all = [], []
            
            for split_idx in sorted(split_files.keys()):
                files = split_files[split_idx]
                
                # Load embeddings
                X_train = np.load(files['train'])
                X_test = np.load(files['test'])
                
                # CRITICAL: We need the true labels that correspond to these exact samples
                # This requires knowing the exact train/test split indices used
                # For now, we'll use a workaround: assume the embeddings were saved
                # in the same order as the dataset, and we can infer the split
                
                # This is a limitation - we need the actual split indices
                # Let's check if there's a way to get them from the checkpoint metadata
                print(f"  WARNING: Cannot determine exact train/test split for {label} split {split_idx}")
                print(f"           Embeddings: train={X_train.shape}, test={X_test.shape}")
                print(f"           This script requires the original split indices to match embeddings with labels.")
                print(f"           Consider using the prediction-based script instead, or modify train_graph.py")
                print(f"           to save split indices alongside embeddings.")
                continue
            
            if y_true_all:
                y_true = np.concatenate(y_true_all)
                y_pred = np.concatenate(y_pred_all)
                all_data[label] = (y_true, y_pred)
                print(f"  Loaded {label}: {len(y_true)} samples from {len(split_files)} split(s)")
    
    if not all_data:
        print("\nERROR: Could not load any embeddings with matching labels.")
        print("\nRECOMMENDATION: Use plot_truth_vs_prediction.py with saved predictions instead,")
        print("or modify train_graph.py to save split indices alongside embeddings.")
        sys.exit(1)
    
    print()
    
    # Truth vs prediction plots
    plot_truth_vs_pred(all_data, args.dataset, args.target, output_dir, split_info)
    
    # Residual plots
    if args.residuals:
        plot_residuals(all_data, args.dataset, args.target, output_dir, split_info)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
