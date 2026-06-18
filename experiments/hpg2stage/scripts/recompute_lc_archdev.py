"""
Recompute architecture-deviation metrics for 100% learning-curve checkpoints
using the EXACT same metric function from analyze_pair_disjoint_transfer.py

This script isolates whether the R² discrepancy is due to:
- Metric computation differences, OR
- Actual model/training differences

Compares: 100% learning-curve vs final Stage 2D checkpoints
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

DATA_PATH = PROJECT_ROOT / "data" / "ea_ip.csv"
PRED_DIR_LC = PROJECT_ROOT / "predictions" / "HPG2Stage_LC"  # Learning curve predictions
PRED_DIR_GEN = PROJECT_ROOT / "predictions" / "HPG2Stage_Gen"  # Generalization predictions
OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
MODELS = ["2d0_arch", "2d1_arch"]
N_FOLDS = 5

# ═══════════════════════════════════════════════════════════════════════
# EXACT functions from analyze_pair_disjoint_transfer.py
# ═══════════════════════════════════════════════════════════════════════

def load_dataset():
    """Load dataset with group_key (EXACT same as pair_disjoint_transfer)."""
    df = pd.read_csv(DATA_PATH)
    df['group_key'] = (df['smiles_A'].astype(str) + '||' +
                       df['smiles_B'].astype(str) + '||' +
                       df['fracA'].astype(str) + '||' +
                       df['fracB'].astype(str))
    return df


def compute_archdev_metrics(y_true, y_pred, row_indices, df):
    """
    Compute architecture-deviation R² and MAE.
    EXACT copy from analyze_pair_disjoint_transfer.py (lines 167-194)
    """
    valid = row_indices >= 0
    if valid.sum() < 20:
        return np.nan, np.nan

    yt = y_true[valid]
    yp = y_pred[valid]
    groups = df.iloc[row_indices[valid]]['group_key'].values
    arch = df.iloc[row_indices[valid]]['poly_type'].values

    gdf = pd.DataFrame({'y_true': yt, 'y_pred': yp, 'group': groups, 'arch': arch})
    ga = gdf.groupby('group')['arch'].nunique()
    multi = ga[ga >= 2].index
    gdf_m = gdf[gdf['group'].isin(multi)]

    if len(gdf_m) < 20:
        return np.nan, np.nan

    gmt = gdf_m.groupby('group')['y_true'].transform('mean')
    gmp = gdf_m.groupby('group')['y_pred'].transform('mean')
    dt = gdf_m['y_true'] - gmt
    dp = gdf_m['y_pred'] - gmp

    if dt.std() < 1e-10:
        return np.nan, np.nan

    return r2_score(dt, dp), mean_absolute_error(dt, dp)


def match_to_dataset(y_true, df, target_long):
    """Match y_true values to dataset row indices (from pair_disjoint_transfer)."""
    vals = df[target_long].values
    lookup = {}
    for idx, v in enumerate(vals):
        if np.isfinite(v):
            lookup[round(float(v), 6)] = idx
    indices = np.array([lookup.get(round(float(v), 6), -1) for v in y_true])
    return indices


# ═══════════════════════════════════════════════════════════════════════
# LOAD PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════

def apply_inverse_transform(y_true, y_pred):
    """Apply linregress inverse transform (same as analyze_pair_disjoint_transfer.py line 142)."""
    slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
    return y_pred * slope + intercept


def needs_inverse_transform(y_true, y_pred):
    """Check if predictions are in normalized space (same heuristic as LC script)."""
    if r2_score(y_true, y_pred) < -1:
        return True
    if abs(y_pred.mean() - y_true.mean()) > 1.0:
        return True
    return False


def load_lc_100_preds(model, target_long):
    """
    Load 100% learning-curve predictions.
    Pattern: ea_ip__{target}__stage2d_{model}__fold{fold}__frac100__split{fold}.npz
    
    NOTE: LC predictions are stored in NORMALIZED space.
    We apply linregress inverse transform (same as analyze_pair_disjoint_transfer.py
    does for original A-held-out predictions).
    """
    per_fold = []
    for fold in range(N_FOLDS):
        fname = f"ea_ip__{target_long}__stage2d_{model}__fold{fold}__frac100__split{fold}.npz"
        fpath = PRED_DIR_LC / fname
        if not fpath.exists():
            # Try alternative pattern
            fname = f"ea_ip__{target_long}__copoly_stage2d_{model}__fold{fold}__frac100__split{fold}.npz"
            fpath = PRED_DIR_LC / fname
            if not fpath.exists():
                return None
        
        npz = np.load(fpath, allow_pickle=True)
        y_true = npz['y_true'].flatten().astype(float)
        y_pred = npz['y_pred'].flatten().astype(float)
        
        # Apply inverse transform if predictions are in normalized space
        if needs_inverse_transform(y_true, y_pred):
            y_pred = apply_inverse_transform(y_true, y_pred)
        
        per_fold.append({
            'y_true': y_true,
            'y_pred': y_pred,
            'test_indices': npz['test_indices'].flatten().astype(int) if 'test_indices' in npz else None,
        })
    return per_fold


def load_gen_preds(model, target_long):
    """
    Load generalization predictions (Stage 2D final).
    Tries multiple split types: a_held_out, group_disjoint, pair_disjoint
    Pattern: ea_ip__{target}__stage2d_{model}__{split_type}__fold{fold}.npz
    """
    split_types = ["a_held_out", "group_disjoint", "pair_disjoint"]
    
    for split_type in split_types:
        per_fold = []
        all_exist = True
        for fold in range(N_FOLDS):
            fname = f"ea_ip__{target_long}__stage2d_{model}__{split_type}__fold{fold}.npz"
            fpath = PRED_DIR_GEN / fname
            if not fpath.exists():
                all_exist = False
                break
        
        if all_exist:
            # Load all folds
            for fold in range(N_FOLDS):
                fname = f"ea_ip__{target_long}__stage2d_{model}__{split_type}__fold{fold}.npz"
                fpath = PRED_DIR_GEN / fname
                npz = np.load(fpath, allow_pickle=True)
                per_fold.append({
                    'y_true': npz['y_true'].flatten().astype(float),
                    'y_pred': npz['y_pred'].flatten().astype(float),
                    'test_indices': npz['test_indices'].flatten().astype(int) if 'test_indices' in npz else None,
                })
            return per_fold, split_type
    
    return None, None


# ═══════════════════════════════════════════════════════════════════════
# COMPUTE METRICS (per-fold, then average)
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics_per_fold(per_fold, df, target_long):
    """
    Compute arch-dev metrics per fold using EXACT pair_disjoint_transfer method.
    Returns list of (r2_dev, mae_dev) per fold.
    """
    fold_metrics = []
    for fold_data in per_fold:
        y_true = fold_data['y_true']
        y_pred = fold_data['y_pred']
        test_indices = fold_data['test_indices']
        
        # Get row indices (exact same logic as pair_disjoint_transfer)
        if test_indices is not None:
            row_idx = test_indices
        else:
            row_idx = match_to_dataset(y_true, df, target_long)
        
        # Compute arch-dev using EXACT same function
        r2_dev, mae_dev = compute_archdev_metrics(y_true, y_pred, row_idx, df)
        fold_metrics.append((r2_dev, mae_dev))
    
    return fold_metrics


# ═══════════════════════════════════════════════════════════════════════
# MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def run_comparison():
    """Compare 100% LC vs Stage 2D using identical metric computation."""
    print("=" * 70)
    print("RECOMPUTE: Architecture-Deviation Metrics")
    print("Method: EXACT same as analyze_pair_disjoint_transfer.py")
    print("=" * 70)
    
    df = load_dataset()
    print(f"Dataset: {len(df)} rows")
    print(f"Unique groups: {df['group_key'].nunique()}")
    print()
    
    results = []
    
    for target_short, target_long in TARGETS.items():
        print(f"\n--- Target: {target_short} ---")
        
        for model in MODELS:
            # Load 100% learning curve predictions
            lc_preds = load_lc_100_preds(model, target_long)
            
            # Load Stage 2D generalization predictions
            gen_preds, gen_split_type = load_gen_preds(model, target_long)
            
            if lc_preds is None:
                print(f"  {model}: LC(100%) - NO PREDICTIONS FOUND")
            else:
                lc_metrics = compute_metrics_per_fold(lc_preds, df, target_long)
                lc_r2_vals = [m[0] for m in lc_metrics if not np.isnan(m[0])]
                
                if lc_r2_vals:
                    lc_r2_mean = np.mean(lc_r2_vals)
                    lc_r2_std = np.std(lc_r2_vals)
                    print(f"  {model}: LC(100%)  R²(Δy) = {lc_r2_mean:.4f} ± {lc_r2_std:.4f} (n={len(lc_r2_vals)} folds)")
                else:
                    print(f"  {model}: LC(100%)  R²(Δy) = NaN (no valid groups)")
                
                results.append({
                    'target': target_short,
                    'model': model,
                    'checkpoint': 'LC_100',
                    'r2_dev_mean': np.mean(lc_r2_vals) if lc_r2_vals else np.nan,
                    'r2_dev_std': np.std(lc_r2_vals) if lc_r2_vals else np.nan,
                    'n_folds': len(lc_r2_vals),
                })
            
            if gen_preds is None:
                print(f"  {model}: Stage2D  - NO PREDICTIONS FOUND")
                gen_split_type = None
            else:
                gen_metrics = compute_metrics_per_fold(gen_preds, df, target_long)
                gen_r2_vals = [m[0] for m in gen_metrics if not np.isnan(m[0])]
                
                if gen_r2_vals:
                    gen_r2_mean = np.mean(gen_r2_vals)
                    gen_r2_std = np.std(gen_r2_vals)
                    print(f"  {model}: Stage2D  R²(Δy) = {gen_r2_mean:.4f} ± {gen_r2_std:.4f} (n={len(gen_r2_vals)} folds) [{gen_split_type}]")
                else:
                    print(f"  {model}: Stage2D  R²(Δy) = NaN (no valid groups)")
                
                results.append({
                    'target': target_short,
                    'model': model,
                    'checkpoint': 'Stage2D',
                    'split_type': gen_split_type,
                    'r2_dev_mean': np.mean(gen_r2_vals) if gen_r2_vals else np.nan,
                    'r2_dev_std': np.std(gen_r2_vals) if gen_r2_vals else np.nan,
                    'n_folds': len(gen_r2_vals),
                })
            
            # Compare if both available
            if lc_preds is not None and gen_preds is not None:
                lc_r2_vals = [m[0] for m in lc_metrics if not np.isnan(m[0])]
                gen_r2_vals = [m[0] for m in gen_metrics if not np.isnan(m[0])]
                
                if lc_r2_vals and gen_r2_vals:
                    gap = np.mean(gen_r2_vals) - np.mean(lc_r2_vals)
                    print(f"  {model}: GAP = {gap:+.4f} (Stage2D - LC100)")
    
    # Save results
    if not results:
        print("\n[WARNING] No results to save - no prediction files found!")
        return None
    
    df_results = pd.DataFrame(results)
    csv_path = OUT_DIR / "lc_vs_stage2d_archdev_comparison.csv"
    df_results.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {csv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for target in ['EA', 'IP']:
        print(f"\n{target}:")
        sub = df_results[df_results['target'] == target]
        
        lc_data = sub[sub['checkpoint'] == 'LC_100']
        gen_data = sub[sub['checkpoint'] == 'Stage2D']
        
        if not lc_data.empty and not gen_data.empty:
            lc_mean = lc_data['r2_dev_mean'].mean()
            gen_mean = gen_data['r2_dev_mean'].mean()
            gap = gen_mean - lc_mean
            
            print(f"  LC(100%) avg:  {lc_mean:.4f}")
            print(f"  Stage2D avg:   {gen_mean:.4f}")
            print(f"  Discrepancy:   {gap:+.4f}")
            
            if abs(gap) < 0.05:
                print(f"  -> CONCLUSION: Discrepancy RESOLVED (gap < 0.05)")
            else:
                print(f"  -> CONCLUSION: Discrepancy REMAINS (gap > 0.05)")
    
    return df_results


if __name__ == "__main__":
    run_comparison()
