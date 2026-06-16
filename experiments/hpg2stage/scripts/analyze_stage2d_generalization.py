"""
Stage 2D Generalization Analysis
=================================
Analyzes predictions from group-disjoint and pair-disjoint generalization
experiments to determine whether architecture effects learned by 2D1
transfer to unseen chemistry.

Computes:
  - Overall metrics: EA R², EA MAE, IP R², IP MAE
  - Architecture-deviation metrics: R²(ΔEA), MAE(ΔEA), R²(ΔIP), MAE(ΔIP)
  - Comparison with original a_held_out results
  - ΔR² degradation analysis

Generates:
  - Summary comparison table (CSV + printed)
  - Comparison bar plots
  - Interpretation text

Usage:
    python analyze_stage2d_generalization.py

Output:
    experiments/hpg2stage/output/generalization/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

# ─── Project paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

DATA_PATH = PROJECT_ROOT / "data" / "ea_ip.csv"
PRED_DIR_GEN = PROJECT_ROOT / "predictions" / "HPG2Stage_Gen"
PRED_DIR_ORIG = PROJECT_ROOT / "predictions" / "HPG2Stage"
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "generalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
TARGET_SHORT = {"EA vs SHE (eV)": "EA", "IP vs SHE (eV)": "IP"}
MODELS = ["frac", "2d0_arch", "2d1_arch"]
MODEL_DISPLAY = {"frac": "Frac", "2d0_arch": "2D0-arch", "2d1_arch": "2D1-arch"}
SPLIT_TYPES = ["group_disjoint", "pair_disjoint"]
SPLIT_DISPLAY = {"a_held_out": "Original (A-held-out)",
                 "group_disjoint": "Group-disjoint",
                 "pair_disjoint": "Pair-disjoint"}
N_FOLDS = 5
DATASET_NAME = "ea_ip"


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_dataset():
    """Load the ea_ip dataset with metadata."""
    df = pd.read_csv(DATA_PATH)
    return df


def load_generalization_predictions(split_type, variant, target, n_folds=N_FOLDS):
    """Load predictions from generalization experiments.
    
    Returns list of (y_true, y_pred, test_indices, fold_idx) tuples.
    """
    per_split = []
    for fold_idx in range(n_folds):
        fname = f"{DATASET_NAME}__{target}__stage2d_{variant}__{split_type}__fold{fold_idx}.npz"
        fpath = PRED_DIR_GEN / fname
        if not fpath.exists():
            continue
        npz = np.load(fpath, allow_pickle=True)
        y_true = npz['y_true'].flatten()
        y_pred = npz['y_pred'].flatten()
        test_indices = npz['test_indices'].flatten() if 'test_indices' in npz else None
        per_split.append((y_true, y_pred, test_indices, fold_idx))
    return per_split


def load_original_predictions(variant, target, n_folds=N_FOLDS):
    """Load predictions from original a_held_out experiments.
    
    These predictions are in normalized space and need inverse transform.
    """
    per_split = []
    for fold_idx in range(n_folds):
        fname = f"ea_ip__{target}__copoly_stage2d_{variant}__a_held_out__split{fold_idx}.npz"
        fpath = PRED_DIR_ORIG / fname
        if not fpath.exists():
            continue
        npz = np.load(fpath, allow_pickle=True)
        y_true = npz['y_true'].flatten()
        y_pred = npz['y_pred'].flatten()
        
        # Apply per-file inverse transform (original predictions are normalized)
        slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
        y_pred_corrected = y_pred * slope + intercept
        
        test_indices = npz['test_indices'].flatten() if 'test_indices' in npz else None
        per_split.append((y_true, y_pred_corrected, test_indices, fold_idx))
    return per_split


def check_needs_inverse_transform(y_true, y_pred):
    """Check if predictions are in normalized space and need inverse transform."""
    r2 = r2_score(y_true, y_pred)
    if r2 < -1:
        return True
    if abs(y_pred.mean() - y_true.mean()) > 1.0:
        return True
    return False


def load_predictions_auto(split_type, variant, target):
    """Load predictions with automatic detection of normalization.
    
    For original predictions: always apply inverse transform.
    For new generalization predictions: check if needed (should not be
    after the UnscaleTransform fix).
    """
    if split_type == "a_held_out":
        return load_original_predictions(variant, target)
    
    per_split = load_generalization_predictions(split_type, variant, target)
    
    # Check first file: if predictions look normalized, apply correction
    if per_split:
        y_true, y_pred, _, _ = per_split[0]
        if check_needs_inverse_transform(y_true, y_pred):
            corrected = []
            for yt, yp, ti, fi in per_split:
                slope, intercept, _, _, _ = stats.linregress(yp, yt)
                yp_corr = yp * slope + intercept
                corrected.append((yt, yp_corr, ti, fi))
            return corrected
    
    return per_split


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_overall_metrics(per_split):
    """Compute overall R² and MAE from per-split predictions."""
    if not per_split:
        return {'R2': np.nan, 'MAE': np.nan, 'n_samples': 0}
    
    y_true_all = np.concatenate([s[0] for s in per_split])
    y_pred_all = np.concatenate([s[1] for s in per_split])
    
    return {
        'R2': r2_score(y_true_all, y_pred_all),
        'MAE': mean_absolute_error(y_true_all, y_pred_all),
        'n_samples': len(y_true_all),
    }


def compute_archdev_metrics(per_split, df, target):
    """Compute architecture-deviation R² and MAE.
    
    Δy = y - group_mean(y) within matched (A, B, fracA) groups.
    Only groups with >1 member (multiple architectures) contribute.
    """
    if not per_split:
        return {'R2_dev': np.nan, 'MAE_dev': np.nan, 'n_groups': 0}
    
    # Concatenate all predictions
    y_true_all = np.concatenate([s[0] for s in per_split])
    y_pred_all = np.concatenate([s[1] for s in per_split])
    test_indices_all = np.concatenate([s[2] for s in per_split]) if per_split[0][2] is not None else None
    
    if test_indices_all is None:
        # Fall back to y_true matching (less reliable)
        return _compute_archdev_by_ytrue_matching(y_true_all, y_pred_all, df, target)
    
    # Use test_indices to look up metadata
    pred_df = pd.DataFrame({
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'dataset_idx': test_indices_all.astype(int),
    })
    
    # Merge with dataset metadata
    meta = df[['smiles_A', 'smiles_B', 'fracA']].copy()
    meta['dataset_idx'] = meta.index
    pred_df = pred_df.merge(meta, on='dataset_idx', how='left')
    
    # Build group key
    pred_df['group'] = (pred_df['smiles_A'].astype(str) + '|' +
                        pred_df['smiles_B'].astype(str) + '|' +
                        pred_df['fracA'].astype(str))
    
    # Compute deviations
    group_means_true = pred_df.groupby('group')['y_true'].transform('mean')
    group_means_pred = pred_df.groupby('group')['y_pred'].transform('mean')
    group_sizes = pred_df.groupby('group')['y_true'].transform('count')
    
    pred_df['delta_true'] = pred_df['y_true'] - group_means_true
    pred_df['delta_pred'] = pred_df['y_pred'] - group_means_pred
    
    # Only use groups with >1 member
    df_multi = pred_df[group_sizes > 1]
    
    if len(df_multi) < 10:
        return {'R2_dev': np.nan, 'MAE_dev': np.nan, 'n_groups': 0}
    
    dt = df_multi['delta_true'].values
    dp = df_multi['delta_pred'].values
    
    n_groups = pred_df.loc[group_sizes > 1, 'group'].nunique()
    
    return {
        'R2_dev': r2_score(dt, dp),
        'MAE_dev': mean_absolute_error(dt, dp),
        'n_groups': n_groups,
    }


def _compute_archdev_by_ytrue_matching(y_true_all, y_pred_all, df, target):
    """Fallback: match predictions to dataset rows by y_true value."""
    tshort = TARGET_SHORT[target]
    vals = df[target].values
    lookup = {}
    for idx, v in enumerate(vals):
        if np.isfinite(v):
            key = round(float(v), 6)
            lookup[key] = idx
    
    rows_matched = []
    for i in range(len(y_true_all)):
        key = round(float(y_true_all[i]), 6)
        if key in lookup:
            rows_matched.append((y_true_all[i], y_pred_all[i], lookup[key]))
    
    if len(rows_matched) < 10:
        return {'R2_dev': np.nan, 'MAE_dev': np.nan, 'n_groups': 0}
    
    pred_df = pd.DataFrame(rows_matched, columns=['y_true', 'y_pred', 'dataset_idx'])
    meta = df[['smiles_A', 'smiles_B', 'fracA']].copy()
    meta['dataset_idx'] = meta.index
    pred_df = pred_df.merge(meta, on='dataset_idx', how='left')
    
    pred_df['group'] = (pred_df['smiles_A'].astype(str) + '|' +
                        pred_df['smiles_B'].astype(str) + '|' +
                        pred_df['fracA'].astype(str))
    
    group_means_true = pred_df.groupby('group')['y_true'].transform('mean')
    group_means_pred = pred_df.groupby('group')['y_pred'].transform('mean')
    group_sizes = pred_df.groupby('group')['y_true'].transform('count')
    
    pred_df['delta_true'] = pred_df['y_true'] - group_means_true
    pred_df['delta_pred'] = pred_df['y_pred'] - group_means_pred
    
    df_multi = pred_df[group_sizes > 1]
    if len(df_multi) < 10:
        return {'R2_dev': np.nan, 'MAE_dev': np.nan, 'n_groups': 0}
    
    dt = df_multi['delta_true'].values
    dp = df_multi['delta_pred'].values
    n_groups = pred_df.loc[group_sizes > 1, 'group'].nunique()
    
    return {
        'R2_dev': r2_score(dt, dp),
        'MAE_dev': mean_absolute_error(dt, dp),
        'n_groups': n_groups,
    }


def compute_per_fold_metrics(per_split, df, target):
    """Compute metrics per fold for significance testing."""
    fold_metrics = []
    for y_true, y_pred, test_indices, fold_idx in per_split:
        overall = {
            'fold': fold_idx,
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
        }
        
        # Arch-dev for this fold
        if test_indices is not None:
            fold_split = [(y_true, y_pred, test_indices, fold_idx)]
            dev = compute_archdev_metrics(fold_split, df, target)
            overall['R2_dev'] = dev['R2_dev']
            overall['MAE_dev'] = dev['MAE_dev']
        else:
            overall['R2_dev'] = np.nan
            overall['MAE_dev'] = np.nan
        
        fold_metrics.append(overall)
    return fold_metrics


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_analysis():
    """Run the full generalization analysis."""
    print("=" * 70)
    print("STAGE 2D GENERALIZATION ANALYSIS")
    print("=" * 70)
    
    df = load_dataset()
    print(f"Dataset: {len(df)} rows")
    
    # Collect all metrics
    all_split_types = ["a_held_out"] + SPLIT_TYPES
    results = []
    
    for split_type in all_split_types:
        for variant in MODELS:
            for target in TARGETS:
                tshort = TARGET_SHORT[target]
                
                # Load predictions
                per_split = load_predictions_auto(split_type, variant, target)
                
                if not per_split:
                    print(f"  [MISSING] {split_type} / {variant} / {tshort}")
                    continue
                
                # Overall metrics
                overall = compute_overall_metrics(per_split)
                
                # Architecture-deviation metrics
                dev = compute_archdev_metrics(per_split, df, target)
                
                results.append({
                    'split_type': split_type,
                    'model': variant,
                    'target': tshort,
                    'R2': overall['R2'],
                    'MAE': overall['MAE'],
                    'R2_dev': dev['R2_dev'],
                    'MAE_dev': dev['MAE_dev'],
                    'n_samples': overall['n_samples'],
                    'n_groups': dev['n_groups'],
                })
    
    if not results:
        print("\n[ERROR] No predictions found. Run training first.")
        print(f"  Expected predictions in: {PRED_DIR_GEN}")
        print(f"  And original predictions in: {PRED_DIR_ORIG}")
        return
    
    df_results = pd.DataFrame(results)
    
    # ── Summary Table ─────────────────────────────────────────────────
    print_summary_table(df_results)
    
    # ── Degradation Analysis ──────────────────────────────────────────
    print_degradation_analysis(df_results)
    
    # ── Plots ─────────────────────────────────────────────────────────
    generate_plots(df_results)
    
    # ── Interpretation ────────────────────────────────────────────────
    print_interpretation(df_results)
    
    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = OUT_DIR / "generalization_results.csv"
    df_results.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {csv_path}")


def print_summary_table(df_results):
    """Print the comparison summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    # Pivot for display
    header = f"{'Split Type':<25} {'Model':<12} {'EA R²':<8} {'IP R²':<8} {'R²(ΔEA)':<9} {'R²(ΔIP)':<9}"
    print(header)
    print("-" * len(header))
    
    for split_type in ["a_held_out"] + SPLIT_TYPES:
        for model in MODELS:
            row_ea = df_results[(df_results['split_type'] == split_type) & 
                               (df_results['model'] == model) & 
                               (df_results['target'] == 'EA')]
            row_ip = df_results[(df_results['split_type'] == split_type) & 
                               (df_results['model'] == model) & 
                               (df_results['target'] == 'IP')]
            
            ea_r2 = row_ea['R2'].values[0] if len(row_ea) > 0 else np.nan
            ip_r2 = row_ip['R2'].values[0] if len(row_ip) > 0 else np.nan
            ea_dev = row_ea['R2_dev'].values[0] if len(row_ea) > 0 else np.nan
            ip_dev = row_ip['R2_dev'].values[0] if len(row_ip) > 0 else np.nan
            
            split_disp = SPLIT_DISPLAY.get(split_type, split_type)
            model_disp = MODEL_DISPLAY.get(model, model)
            
            print(f"{split_disp:<25} {model_disp:<12} "
                  f"{ea_r2:<8.4f} {ip_r2:<8.4f} "
                  f"{ea_dev:<9.4f} {ip_dev:<9.4f}")
        print()


def print_degradation_analysis(df_results):
    """Compute and print ΔR² = R²(new) - R²(original) for each metric."""
    print("\n" + "=" * 70)
    print("DEGRADATION ANALYSIS: ΔR² = R²(new split) - R²(original)")
    print("=" * 70)
    
    orig = df_results[df_results['split_type'] == 'a_held_out']
    
    if orig.empty:
        print("  [SKIP] No original a_held_out results found for comparison.")
        return
    
    header = f"{'Split Type':<25} {'Model':<12} {'ΔEA R²':<9} {'ΔIP R²':<9} {'ΔR²(ΔEA)':<10} {'ΔR²(ΔIP)':<10}"
    print(header)
    print("-" * len(header))
    
    for split_type in SPLIT_TYPES:
        for model in MODELS:
            deltas = {}
            for target in ['EA', 'IP']:
                orig_row = orig[(orig['model'] == model) & (orig['target'] == target)]
                new_row = df_results[(df_results['split_type'] == split_type) & 
                                    (df_results['model'] == model) & 
                                    (df_results['target'] == target)]
                
                if orig_row.empty or new_row.empty:
                    deltas[f'd_R2_{target}'] = np.nan
                    deltas[f'd_R2dev_{target}'] = np.nan
                else:
                    deltas[f'd_R2_{target}'] = new_row['R2'].values[0] - orig_row['R2'].values[0]
                    deltas[f'd_R2dev_{target}'] = new_row['R2_dev'].values[0] - orig_row['R2_dev'].values[0]
            
            split_disp = SPLIT_DISPLAY.get(split_type, split_type)
            model_disp = MODEL_DISPLAY.get(model, model)
            
            print(f"{split_disp:<25} {model_disp:<12} "
                  f"{deltas['d_R2_EA']:+<9.4f} {deltas['d_R2_IP']:+<9.4f} "
                  f"{deltas['d_R2dev_EA']:+<10.4f} {deltas['d_R2dev_IP']:+<10.4f}")
        print()


def generate_plots(df_results):
    """Generate comparison bar plots."""
    try:
        _plot_overall_r2(df_results)
        _plot_archdev_r2(df_results)
        _plot_relative_degradation(df_results)
        print(f"\nPlots saved to: {OUT_DIR}")
    except Exception as e:
        print(f"\n[WARNING] Plot generation failed: {e}")


def _plot_overall_r2(df_results):
    """Bar plot: Overall EA/IP R² across split types."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_splits = ["a_held_out"] + SPLIT_TYPES
    x = np.arange(len(MODELS))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for t_idx, target in enumerate(['EA', 'IP']):
        ax = axes[t_idx]
        for s_idx, split_type in enumerate(all_splits):
            vals = []
            for model in MODELS:
                row = df_results[(df_results['split_type'] == split_type) & 
                                (df_results['model'] == model) & 
                                (df_results['target'] == target)]
                vals.append(row['R2'].values[0] if len(row) > 0 else 0)
            
            ax.bar(x + s_idx * width, vals, width, 
                   label=SPLIT_DISPLAY.get(split_type, split_type),
                   color=colors[s_idx], alpha=0.85)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('R²')
        ax.set_title(f'{target} Overall R²')
        ax.set_xticks(x + width)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
        ax.legend(fontsize=8)
        ax.set_ylim(0.9, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "overall_r2_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_archdev_r2(df_results):
    """Bar plot: Architecture-deviation R² across split types."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_splits = ["a_held_out"] + SPLIT_TYPES
    x = np.arange(len(MODELS))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for t_idx, target in enumerate(['EA', 'IP']):
        ax = axes[t_idx]
        for s_idx, split_type in enumerate(all_splits):
            vals = []
            for model in MODELS:
                row = df_results[(df_results['split_type'] == split_type) & 
                                (df_results['model'] == model) & 
                                (df_results['target'] == target)]
                v = row['R2_dev'].values[0] if len(row) > 0 else 0
                vals.append(v if np.isfinite(v) else 0)
            
            ax.bar(x + s_idx * width, vals, width,
                   label=SPLIT_DISPLAY.get(split_type, split_type),
                   color=colors[s_idx], alpha=0.85)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('R²(Δy)')
        ax.set_title(f'Architecture-Deviation R² ({target})')
        ax.set_xticks(x + width)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "archdev_r2_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_relative_degradation(df_results):
    """Bar plot: ΔR² relative to original split."""
    orig = df_results[df_results['split_type'] == 'a_held_out']
    if orig.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [('R2', 'Overall R²'), ('R2_dev', 'Arch-Deviation R²')]
    targets = ['EA', 'IP']
    
    x = np.arange(len(MODELS))
    width = 0.35
    colors = ['#FF9800', '#4CAF50']
    
    for m_idx, (metric, metric_name) in enumerate(metrics):
        for t_idx, target in enumerate(targets):
            ax = axes[m_idx, t_idx]
            
            for s_idx, split_type in enumerate(SPLIT_TYPES):
                deltas = []
                for model in MODELS:
                    orig_row = orig[(orig['model'] == model) & (orig['target'] == target)]
                    new_row = df_results[(df_results['split_type'] == split_type) &
                                        (df_results['model'] == model) &
                                        (df_results['target'] == target)]
                    
                    if orig_row.empty or new_row.empty:
                        deltas.append(0)
                    else:
                        d = new_row[metric].values[0] - orig_row[metric].values[0]
                        deltas.append(d if np.isfinite(d) else 0)
                
                ax.bar(x + s_idx * width, deltas, width,
                       label=SPLIT_DISPLAY[split_type],
                       color=colors[s_idx], alpha=0.85)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(f'Δ{metric_name}')
            ax.set_title(f'{target}: Δ{metric_name} vs Original')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
            ax.legend(fontsize=8)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "degradation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def print_interpretation(df_results):
    """Print interpretation of results."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    orig = df_results[df_results['split_type'] == 'a_held_out']
    
    # Helper to get a metric value
    def get_val(split_type, model, target, metric):
        row = df_results[(df_results['split_type'] == split_type) &
                        (df_results['model'] == model) &
                        (df_results['target'] == target)]
        if row.empty:
            return np.nan
        return row[metric].values[0]
    
    # 1. Group-disjoint retention
    print("\n1. Does 2D1 retain architecture-deviation performance under")
    print("   group-disjoint evaluation?")
    print("   " + "-" * 50)
    
    gd_present = not df_results[df_results['split_type'] == 'group_disjoint'].empty
    if gd_present:
        for target in ['EA', 'IP']:
            orig_dev = get_val('a_held_out', '2d1_arch', target, 'R2_dev')
            gd_dev = get_val('group_disjoint', '2d1_arch', target, 'R2_dev')
            if np.isfinite(orig_dev) and np.isfinite(gd_dev):
                retention = gd_dev / orig_dev * 100 if orig_dev != 0 else np.nan
                print(f"   {target}: R²(Δ) original={orig_dev:.4f} → group-disjoint={gd_dev:.4f} "
                      f"(retention: {retention:.1f}%)")
        
        # Verdict
        ea_gd = get_val('group_disjoint', '2d1_arch', 'EA', 'R2_dev')
        ip_gd = get_val('group_disjoint', '2d1_arch', 'IP', 'R2_dev')
        if np.isfinite(ea_gd) and np.isfinite(ip_gd):
            mean_gd = (ea_gd + ip_gd) / 2
            if mean_gd > 0.5:
                print("   → YES: 2D1 retains substantial arch-dev performance on unseen compositions.")
            elif mean_gd > 0.2:
                print("   → PARTIAL: 2D1 retains some arch-dev performance, with notable degradation.")
            else:
                print("   → NO: 2D1 arch-dev performance does not generalize to unseen compositions.")
    else:
        print("   [NO DATA] Group-disjoint predictions not available.")
    
    # 2. Pair-disjoint retention
    print("\n2. Does 2D1 retain architecture-deviation performance under")
    print("   pair-disjoint evaluation?")
    print("   " + "-" * 50)
    
    pd_present = not df_results[df_results['split_type'] == 'pair_disjoint'].empty
    if pd_present:
        for target in ['EA', 'IP']:
            orig_dev = get_val('a_held_out', '2d1_arch', target, 'R2_dev')
            pd_dev = get_val('pair_disjoint', '2d1_arch', target, 'R2_dev')
            if np.isfinite(orig_dev) and np.isfinite(pd_dev):
                retention = pd_dev / orig_dev * 100 if orig_dev != 0 else np.nan
                print(f"   {target}: R²(Δ) original={orig_dev:.4f} → pair-disjoint={pd_dev:.4f} "
                      f"(retention: {retention:.1f}%)")
        
        ea_pd = get_val('pair_disjoint', '2d1_arch', 'EA', 'R2_dev')
        ip_pd = get_val('pair_disjoint', '2d1_arch', 'IP', 'R2_dev')
        if np.isfinite(ea_pd) and np.isfinite(ip_pd):
            mean_pd = (ea_pd + ip_pd) / 2
            if mean_pd > 0.5:
                print("   → YES: 2D1 transfers arch effects to completely unseen chemistry pairs.")
            elif mean_pd > 0.2:
                print("   → PARTIAL: Some transfer, but substantial loss on unseen pairs.")
            else:
                print("   → NO: 2D1 arch-dev does not transfer to unseen chemistry pairs.")
    else:
        print("   [NO DATA] Pair-disjoint predictions not available.")
    
    # 3. Relative degradation comparison
    print("\n3. Is degradation larger for architecture metrics than for")
    print("   overall EA/IP metrics?")
    print("   " + "-" * 50)
    
    for split_type in SPLIT_TYPES:
        if df_results[df_results['split_type'] == split_type].empty:
            continue
        
        overall_drops = []
        dev_drops = []
        for target in ['EA', 'IP']:
            orig_r2 = get_val('a_held_out', '2d1_arch', target, 'R2')
            new_r2 = get_val(split_type, '2d1_arch', target, 'R2')
            orig_dev = get_val('a_held_out', '2d1_arch', target, 'R2_dev')
            new_dev = get_val(split_type, '2d1_arch', target, 'R2_dev')
            
            if np.isfinite(orig_r2) and np.isfinite(new_r2):
                overall_drops.append(new_r2 - orig_r2)
            if np.isfinite(orig_dev) and np.isfinite(new_dev):
                dev_drops.append(new_dev - orig_dev)
        
        if overall_drops and dev_drops:
            mean_overall_drop = np.mean(overall_drops)
            mean_dev_drop = np.mean(dev_drops)
            split_disp = SPLIT_DISPLAY.get(split_type, split_type)
            print(f"   {split_disp}:")
            print(f"     Mean ΔR²(overall) = {mean_overall_drop:+.4f}")
            print(f"     Mean ΔR²(arch-dev) = {mean_dev_drop:+.4f}")
            if abs(mean_dev_drop) > abs(mean_overall_drop) * 1.5:
                print(f"     → YES: Arch-dev degrades more than overall ({abs(mean_dev_drop)/abs(mean_overall_drop):.1f}× larger)")
            else:
                print(f"     → NO: Similar or smaller degradation for arch-dev metrics")
    
    # 4. Evidence assessment
    print("\n4. Does evidence support:")
    print("   A) Transferable architecture learning, or")
    print("   B) Chemistry-specific architecture learning?")
    print("   " + "-" * 50)
    
    # Assess based on pair-disjoint (the stronger test)
    if pd_present:
        ea_pd = get_val('pair_disjoint', '2d1_arch', 'EA', 'R2_dev')
        ip_pd = get_val('pair_disjoint', '2d1_arch', 'IP', 'R2_dev')
        ea_frac_pd = get_val('pair_disjoint', 'frac', 'EA', 'R2_dev')
        ip_frac_pd = get_val('pair_disjoint', 'frac', 'IP', 'R2_dev')
        
        if all(np.isfinite(v) for v in [ea_pd, ip_pd]):
            mean_2d1_pd = (ea_pd + ip_pd) / 2
            mean_frac_pd = np.nanmean([ea_frac_pd, ip_frac_pd])
            
            improvement_over_frac = mean_2d1_pd - mean_frac_pd
            
            print(f"\n   Pair-disjoint arch-dev R² (strongest test):")
            print(f"     2D1-arch: mean R²(Δ) = {mean_2d1_pd:.4f}")
            print(f"     Frac:     mean R²(Δ) = {mean_frac_pd:.4f}")
            print(f"     Improvement: {improvement_over_frac:+.4f}")
            
            if mean_2d1_pd > 0.5 and improvement_over_frac > 0.3:
                print("\n   CONCLUSION: Strong evidence for (A) TRANSFERABLE architecture learning.")
                print("   2D1 captures genuine, generalizable architecture effects that transfer")
                print("   to completely unseen chemistry pairs.")
            elif mean_2d1_pd > 0.2 and improvement_over_frac > 0.1:
                print("\n   CONCLUSION: Moderate evidence for (A) TRANSFERABLE architecture learning.")
                print("   2D1 captures partially generalizable effects, with some chemistry specificity.")
            elif improvement_over_frac > 0.05:
                print("\n   CONCLUSION: Weak evidence for (A). Mostly (B) chemistry-specific.")
                print("   2D1's architecture signal largely depends on seen chemistry.")
            else:
                print("\n   CONCLUSION: Evidence supports (B) CHEMISTRY-SPECIFIC architecture learning.")
                print("   2D1's superior arch-dev performance does not generalize to unseen pairs.")
    elif gd_present:
        print("   [Only group-disjoint available — pair-disjoint is the definitive test]")
        ea_gd = get_val('group_disjoint', '2d1_arch', 'EA', 'R2_dev')
        ip_gd = get_val('group_disjoint', '2d1_arch', 'IP', 'R2_dev')
        if all(np.isfinite(v) for v in [ea_gd, ip_gd]):
            mean_gd = (ea_gd + ip_gd) / 2
            if mean_gd > 0.5:
                print(f"   Group-disjoint R²(Δ) = {mean_gd:.4f} suggests SOME transferability,")
                print("   but pair-disjoint is needed for a definitive conclusion.")
    else:
        print("   [NO DATA] Cannot assess — run generalization experiments first.")
    
    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_analysis()
