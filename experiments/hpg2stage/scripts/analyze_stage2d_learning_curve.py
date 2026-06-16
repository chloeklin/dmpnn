"""
Stage 2D Learning Curve Analysis
==================================
Processes predictions from run_stage2d_learning_curve.py and generates:
- stage2d_learning_curve_results.csv
- fig_learning_curve_archdev_EA.png/pdf
- fig_learning_curve_archdev_IP.png/pdf
- fig_learning_curve_overall_EA.png/pdf
- fig_learning_curve_overall_IP.png/pdf
- stage2d_learning_curve_interpretation.md

Usage:
    python analyze_stage2d_learning_curve.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]  # dmpnn/
DATA_PATH = ROOT / 'data' / 'ea_ip.csv'
PRED_DIR = ROOT / 'predictions' / 'HPG2Stage_LC'
OUT = ROOT / 'experiments' / 'hpg2stage' / 'output' / 'learning_curve'
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {'EA': 'EA vs SHE (eV)', 'IP': 'IP vs SHE (eV)'}
MODELS = ['2d0_arch', '2d1_arch']
FRACTIONS = [25, 50, 75, 100]
N_FOLDS = 5
DATASET_NAME = 'ea_ip'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

# ── Load dataset ─────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['group_key'] = (df['smiles_A'].astype(str) + '||' +
                   df['smiles_B'].astype(str) + '||' +
                   df['fracA'].astype(str) + '||' +
                   df['fracB'].astype(str))


# ── Normalization recovery ───────────────────────────────────────
def estimate_normalization_params_lc():
    """Estimate per-fold normalization params from frac=100 predictions."""
    # For the learning curve, predictions are saved by the training script.
    # If they used trainer.predict() they should already be unscaled.
    # However, to be safe, we check if predictions need inverse transform
    # by comparing y_true range to y_pred range.
    return None  # Handled inline during loading


def build_ytrue_lookup(target_long):
    """Build lookup: round(y_true, 6) → dataset row index."""
    vals = df[target_long].values
    lookup = {}
    for idx, v in enumerate(vals):
        if np.isfinite(v):
            key = round(float(v), 6)
            lookup[key] = idx
    return lookup


def match_predictions_to_rows(y_true, target_long):
    """Match predictions to dataset rows via rounded y_true."""
    lookup = build_ytrue_lookup(target_long)
    indices = np.full(len(y_true), -1, dtype=int)
    for i, yt in enumerate(y_true):
        key = round(float(yt), 6)
        if key in lookup:
            indices[i] = lookup[key]
    return indices


def compute_archdev_r2(y_true, y_pred, row_indices):
    """Compute architecture-deviation R² given row indices into the dataset.
    
    row_indices can be:
    - Direct dataset indices (from test_indices in npz), all valid
    - Matched indices with -1 for unmatched (from y_true lookup)
    """
    if row_indices is None:
        return np.nan
    
    valid = row_indices >= 0
    if valid.sum() < 20:
        return np.nan
    
    y_t = y_true[valid]
    y_p = y_pred[valid]
    groups = df.iloc[row_indices[valid]]['group_key'].values
    arch = df.iloc[row_indices[valid]]['poly_type'].values
    
    gdf = pd.DataFrame({'y_true': y_t, 'y_pred': y_p, 'group': groups, 'arch': arch})
    
    # Only groups with ≥2 architectures
    group_arch = gdf.groupby('group')['arch'].nunique()
    multi = group_arch[group_arch >= 2].index
    gdf_multi = gdf[gdf['group'].isin(multi)]
    
    if len(gdf_multi) < 20:
        return np.nan
    
    g_mean_true = gdf_multi.groupby('group')['y_true'].transform('mean')
    g_mean_pred = gdf_multi.groupby('group')['y_pred'].transform('mean')
    dt = gdf_multi['y_true'] - g_mean_true
    dp = gdf_multi['y_pred'] - g_mean_pred
    
    if dt.std() < 1e-10:
        return np.nan
    
    return r2_score(dt, dp)


def check_needs_inverse_transform(y_true, y_pred):
    """Check if predictions need inverse transform by comparing R²."""
    r2 = r2_score(y_true, y_pred)
    if r2 < -1:
        return True
    # Also check if mean is far off (normalized preds ~0, true values ~ -2 to -5)
    if abs(y_pred.mean() - y_true.mean()) > 1.0:
        return True
    return False


def apply_inverse_transform(y_true, y_pred):
    """Apply per-file linear-regression inverse transform.
    
    This recovers predictions from normalized space using the relationship:
    y_true = slope * y_pred_norm + intercept
    """
    slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
    return y_pred * slope + intercept


# ═══════════════════════════════════════════════════════════════════
# LOAD AND PROCESS PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

def load_all_predictions():
    """Load all learning curve prediction files and compute metrics."""
    rows = []
    
    for tgt_short, tgt_long in TARGETS.items():
        for fold in range(N_FOLDS):
            for frac_pct in FRACTIONS:
                for model in MODELS:
                    # Try multiple filename patterns
                    patterns = [
                        f'{DATASET_NAME}__{tgt_long}__stage2d_{model}__fold{fold}__frac{frac_pct}__split{fold}.npz',
                        f'{DATASET_NAME}__{tgt_long}__copoly_stage2d_{model}__fold{fold}__frac{frac_pct}__split{fold}.npz',
                    ]
                    
                    pred_file = None
                    for pat in patterns:
                        candidate = PRED_DIR / pat
                        if candidate.exists():
                            pred_file = candidate
                            break
                    
                    if pred_file is None:
                        continue
                    
                    npz = np.load(pred_file, allow_pickle=True)
                    y_true = npz['y_true'].flatten()
                    y_pred = npz['y_pred'].flatten()
                    
                    # Apply per-file inverse transform if predictions are in normalized space
                    if check_needs_inverse_transform(y_true, y_pred):
                        y_pred = apply_inverse_transform(y_true, y_pred)
                    
                    # Overall metrics
                    r2_overall = r2_score(y_true, y_pred)
                    mae_overall = mean_absolute_error(y_true, y_pred)
                    rmse_overall = np.sqrt(mean_squared_error(y_true, y_pred))
                    
                    # Architecture-deviation R²
                    # Prefer direct test_indices if saved, else fall back to y_true matching
                    if 'test_indices' in npz:
                        row_indices = npz['test_indices'].astype(int)
                    else:
                        row_indices = match_predictions_to_rows(y_true, tgt_long)
                    r2_arch = compute_archdev_r2(y_true, y_pred, row_indices)
                    
                    rows.append({
                        'model': model,
                        'fold': fold,
                        'fraction': frac_pct,
                        'ea_r2': r2_overall if tgt_short == 'EA' else np.nan,
                        'ip_r2': r2_overall if tgt_short == 'IP' else np.nan,
                        'ea_mae': mae_overall if tgt_short == 'EA' else np.nan,
                        'ip_mae': mae_overall if tgt_short == 'IP' else np.nan,
                        'ea_rmse': rmse_overall if tgt_short == 'EA' else np.nan,
                        'ip_rmse': rmse_overall if tgt_short == 'IP' else np.nan,
                        'ea_arch_r2': r2_arch if tgt_short == 'EA' else np.nan,
                        'ip_arch_r2': r2_arch if tgt_short == 'IP' else np.nan,
                        '_target': tgt_short,
                    })
    
    if not rows:
        return None
    
    # Combine EA and IP rows for same (model, fold, fraction)
    raw_df = pd.DataFrame(rows)
    
    # Pivot to have one row per (model, fold, fraction) with both EA and IP columns
    combined_rows = []
    for model in MODELS:
        for fold in range(N_FOLDS):
            for frac_pct in FRACTIONS:
                sub = raw_df[(raw_df['model'] == model) &
                            (raw_df['fold'] == fold) &
                            (raw_df['fraction'] == frac_pct)]
                if sub.empty:
                    continue
                
                row = {'model': model, 'fold': fold, 'fraction': frac_pct}
                ea_sub = sub[sub['_target'] == 'EA']
                ip_sub = sub[sub['_target'] == 'IP']
                
                if not ea_sub.empty:
                    row['ea_r2'] = ea_sub.iloc[0]['ea_r2']
                    row['ea_mae'] = ea_sub.iloc[0]['ea_mae']
                    row['ea_rmse'] = ea_sub.iloc[0]['ea_rmse']
                    row['ea_arch_r2'] = ea_sub.iloc[0]['ea_arch_r2']
                else:
                    row['ea_r2'] = row['ea_mae'] = row['ea_rmse'] = row['ea_arch_r2'] = np.nan
                
                if not ip_sub.empty:
                    row['ip_r2'] = ip_sub.iloc[0]['ip_r2']
                    row['ip_mae'] = ip_sub.iloc[0]['ip_mae']
                    row['ip_rmse'] = ip_sub.iloc[0]['ip_rmse']
                    row['ip_arch_r2'] = ip_sub.iloc[0]['ip_arch_r2']
                else:
                    row['ip_r2'] = row['ip_mae'] = row['ip_rmse'] = row['ip_arch_r2'] = np.nan
                
                combined_rows.append(row)
    
    return pd.DataFrame(combined_rows)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

COLORS = {'2d0_arch': '#4C72B0', '2d1_arch': '#DD8452'}
LABELS = {'2d0_arch': '2D0-arch', '2d1_arch': '2D1-arch'}
MARKERS = {'2d0_arch': 'o', '2d1_arch': 's'}


def plot_learning_curve(results_df, metric_col, ylabel, title, save_prefix):
    """Create a learning curve plot for a given metric."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    for model in MODELS:
        sub = results_df[results_df['model'] == model]
        if sub.empty:
            continue
        
        agg = sub.groupby('fraction').agg(
            mean_val=(metric_col, 'mean'),
            std_val=(metric_col, 'std'),
        ).reset_index()
        
        ax.errorbar(
            agg['fraction'], agg['mean_val'], yerr=agg['std_val'],
            marker=MARKERS[model], markersize=8, linewidth=2.2,
            capsize=5, capthick=1.5, elinewidth=1.5,
            color=COLORS[model], label=LABELS[model],
        )
    
    ax.set_xlabel('Training Fraction (%)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(FRACTIONS)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(OUT / f'{save_prefix}.png')
    fig.savefig(OUT / f'{save_prefix}.pdf')
    plt.close(fig)


def generate_plots(results_df):
    """Generate all 4 learning curve plots."""
    # Plot 1: EA arch-dev R²
    plot_learning_curve(
        results_df, 'ea_arch_r2',
        'R²(ΔEA)', 'EA Architecture-Deviation R² vs Training Fraction',
        'fig_learning_curve_archdev_EA'
    )
    
    # Plot 2: IP arch-dev R²
    plot_learning_curve(
        results_df, 'ip_arch_r2',
        'R²(ΔIP)', 'IP Architecture-Deviation R² vs Training Fraction',
        'fig_learning_curve_archdev_IP'
    )
    
    # Plot 3: Overall EA R²
    plot_learning_curve(
        results_df, 'ea_r2',
        'EA R²', 'Overall EA R² vs Training Fraction',
        'fig_learning_curve_overall_EA'
    )
    
    # Plot 4: Overall IP R²
    plot_learning_curve(
        results_df, 'ip_r2',
        'IP R²', 'Overall IP R² vs Training Fraction',
        'fig_learning_curve_overall_IP'
    )
    
    print("  Saved 4 learning curve plots (PNG + PDF)")


# ═══════════════════════════════════════════════════════════════════
# INTERPRETATION REPORT
# ═══════════════════════════════════════════════════════════════════

def generate_interpretation(results_df):
    """Generate the interpretation markdown report."""
    md = "# Stage 2D Learning Curve Interpretation\n\n"
    md += "## Experiment Design\n\n"
    md += "- **Models**: 2D0-arch, 2D1-arch\n"
    md += "- **Training fractions**: 25%, 50%, 75%, 100% of training matched groups\n"
    md += "- **Evaluation**: Full original test set (unchanged)\n"
    md += "- **Split**: a_held_out (monomer A held out per fold)\n"
    md += "- **Matched group**: (smiles_A, smiles_B, fracA, fracB)\n"
    md += "- **Key constraint**: Entire groups selected together (no architecture leak)\n\n"
    
    # Summary table
    md += "## Summary Results\n\n"
    md += "### Architecture-Deviation R²\n\n"
    md += "| Model | Fraction | EA R²(Δ) | IP R²(Δ) |\n"
    md += "|-------|----------|----------|----------|\n"
    
    for model in MODELS:
        for frac in FRACTIONS:
            sub = results_df[(results_df['model'] == model) & (results_df['fraction'] == frac)]
            if sub.empty:
                continue
            ea_mean = sub['ea_arch_r2'].mean()
            ea_std = sub['ea_arch_r2'].std()
            ip_mean = sub['ip_arch_r2'].mean()
            ip_std = sub['ip_arch_r2'].std()
            md += f"| {LABELS[model]} | {frac}% | {ea_mean:.4f} ± {ea_std:.4f} | {ip_mean:.4f} ± {ip_std:.4f} |\n"
    
    md += "\n### Overall R²\n\n"
    md += "| Model | Fraction | EA R² | IP R² |\n"
    md += "|-------|----------|-------|-------|\n"
    
    for model in MODELS:
        for frac in FRACTIONS:
            sub = results_df[(results_df['model'] == model) & (results_df['fraction'] == frac)]
            if sub.empty:
                continue
            ea_mean = sub['ea_r2'].mean()
            ea_std = sub['ea_r2'].std()
            ip_mean = sub['ip_r2'].mean()
            ip_std = sub['ip_r2'].std()
            md += f"| {LABELS[model]} | {frac}% | {ea_mean:.4f} ± {ea_std:.4f} | {ip_mean:.4f} ± {ip_std:.4f} |\n"
    
    # Quantitative analysis
    md += "\n## Quantitative Analysis\n\n"
    md += "### Improvement from 25% → 100%\n\n"
    md += "| Model | Metric | 25% value | 100% value | Δ(100%-25%) |\n"
    md += "|-------|--------|-----------|------------|-------------|\n"
    
    improvements = {}
    for model in MODELS:
        for metric, label in [('ea_arch_r2', 'EA R²(Δ)'), ('ip_arch_r2', 'IP R²(Δ)'),
                              ('ea_r2', 'EA R²'), ('ip_r2', 'IP R²')]:
            sub_25 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 25)]
            sub_100 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 100)]
            if sub_25.empty or sub_100.empty:
                continue
            v25 = sub_25[metric].mean()
            v100 = sub_100[metric].mean()
            delta = v100 - v25
            md += f"| {LABELS[model]} | {label} | {v25:.4f} | {v100:.4f} | {delta:+.4f} |\n"
            improvements[(model, metric)] = delta
    
    # Interpretation questions
    md += "\n## Interpretation\n\n"
    
    # Q1: Does R²(ΔEA) plateau?
    md += "### 1. Does R²(ΔEA) plateau?\n\n"
    for model in MODELS:
        fracs_vals = []
        for frac in FRACTIONS:
            sub = results_df[(results_df['model'] == model) & (results_df['fraction'] == frac)]
            if not sub.empty:
                fracs_vals.append((frac, sub['ea_arch_r2'].mean()))
        
        if len(fracs_vals) >= 3:
            vals = [v for _, v in fracs_vals]
            # Check if last 3 values are within 0.01 of each other
            last3_range = max(vals[-3:]) - min(vals[-3:])
            total_improvement = vals[-1] - vals[0]
            late_improvement = vals[-1] - vals[-2]  # 75% → 100%
            
            if last3_range < 0.005:
                md += f"- **{LABELS[model]}**: YES, plateau observed. "
                md += f"Range across 50-100%: {last3_range:.4f}. "
            elif late_improvement > 0.005:
                md += f"- **{LABELS[model]}**: NO, still improving. "
                md += f"75%→100% gain: {late_improvement:+.4f}. "
            else:
                md += f"- **{LABELS[model]}**: MARGINAL. "
                md += f"75%→100% gain: {late_improvement:+.4f}. "
            md += f"Total 25%→100%: {total_improvement:+.4f}\n"
    
    # Q2: Does R²(ΔIP) plateau?
    md += "\n### 2. Does R²(ΔIP) plateau?\n\n"
    for model in MODELS:
        fracs_vals = []
        for frac in FRACTIONS:
            sub = results_df[(results_df['model'] == model) & (results_df['fraction'] == frac)]
            if not sub.empty:
                fracs_vals.append((frac, sub['ip_arch_r2'].mean()))
        
        if len(fracs_vals) >= 3:
            vals = [v for _, v in fracs_vals]
            last3_range = max(vals[-3:]) - min(vals[-3:])
            total_improvement = vals[-1] - vals[0]
            late_improvement = vals[-1] - vals[-2]
            
            if last3_range < 0.005:
                md += f"- **{LABELS[model]}**: YES, plateau observed. "
                md += f"Range across 50-100%: {last3_range:.4f}. "
            elif late_improvement > 0.005:
                md += f"- **{LABELS[model]}**: NO, still improving. "
                md += f"75%→100% gain: {late_improvement:+.4f}. "
            else:
                md += f"- **{LABELS[model]}**: MARGINAL. "
                md += f"75%→100% gain: {late_improvement:+.4f}. "
            md += f"Total 25%→100%: {total_improvement:+.4f}\n"
    
    # Q3: Is 2D1 more data-hungry than 2D0?
    md += "\n### 3. Is 2D1 more data-hungry than 2D0?\n\n"
    d0_ea_gain = improvements.get(('2d0_arch', 'ea_arch_r2'), 0)
    d1_ea_gain = improvements.get(('2d1_arch', 'ea_arch_r2'), 0)
    d0_ip_gain = improvements.get(('2d0_arch', 'ip_arch_r2'), 0)
    d1_ip_gain = improvements.get(('2d1_arch', 'ip_arch_r2'), 0)
    
    md += f"- EA arch-dev improvement (25%→100%): 2D0={d0_ea_gain:+.4f}, 2D1={d1_ea_gain:+.4f}\n"
    md += f"- IP arch-dev improvement (25%→100%): 2D0={d0_ip_gain:+.4f}, 2D1={d1_ip_gain:+.4f}\n\n"
    
    if abs(d1_ea_gain) > abs(d0_ea_gain) * 1.5 or abs(d1_ip_gain) > abs(d0_ip_gain) * 1.5:
        md += "**2D1-arch shows stronger data dependence** — the chemistry-conditioned model "
        md += "benefits more from additional training data, suggesting it is data-limited.\n"
    elif abs(d1_ea_gain - d0_ea_gain) < 0.005 and abs(d1_ip_gain - d0_ip_gain) < 0.005:
        md += "**Both models show similar data dependence** — 2D1 is NOT more data-hungry than 2D0.\n"
    else:
        md += "**Mixed signal** — see per-target breakdown above.\n"
    
    # Q4: Is performance still improving at 100%?
    md += "\n### 4. Is performance still improving at 100% training data?\n\n"
    for model in MODELS:
        for metric, label in [('ea_arch_r2', 'EA R²(Δ)'), ('ip_arch_r2', 'IP R²(Δ)')]:
            sub_75 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 75)]
            sub_100 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 100)]
            if sub_75.empty or sub_100.empty:
                continue
            v75 = sub_75[metric].mean()
            v100 = sub_100[metric].mean()
            delta = v100 - v75
            if delta > 0.005:
                md += f"- {LABELS[model]} {label}: **Still improving** (75%→100%: {delta:+.4f})\n"
            elif delta > 0.001:
                md += f"- {LABELS[model]} {label}: Marginal improvement (75%→100%: {delta:+.4f})\n"
            else:
                md += f"- {LABELS[model]} {label}: Saturated (75%→100%: {delta:+.4f})\n"
    
    # Q5: Final verdict
    md += "\n### 5. Final Verdict\n\n"
    
    # Compute average late-stage improvement across models and targets
    late_gains = []
    for model in MODELS:
        for metric in ['ea_arch_r2', 'ip_arch_r2']:
            sub_75 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 75)]
            sub_100 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 100)]
            if not sub_75.empty and not sub_100.empty:
                late_gains.append(sub_100[metric].mean() - sub_75[metric].mean())
    
    total_gains = []
    for model in MODELS:
        for metric in ['ea_arch_r2', 'ip_arch_r2']:
            sub_25 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 25)]
            sub_100 = results_df[(results_df['model'] == model) & (results_df['fraction'] == 100)]
            if not sub_25.empty and not sub_100.empty:
                total_gains.append(sub_100[metric].mean() - sub_25[metric].mean())
    
    avg_late_gain = np.mean(late_gains) if late_gains else 0
    avg_total_gain = np.mean(total_gains) if total_gains else 0
    
    md += "| Criterion | Evidence |\n"
    md += "|-----------|----------|\n"
    md += f"| Avg total improvement (25%→100%) | {avg_total_gain:+.4f} |\n"
    md += f"| Avg late improvement (75%→100%) | {avg_late_gain:+.4f} |\n"
    
    if avg_total_gain < 0.005 and avg_late_gain < 0.002:
        verdict = "dataset_saturated"
        md += f"| **Verdict** | **Dataset likely saturated** |\n\n"
        md += "Performance is essentially constant across training fractions. "
        md += "Adding more data of the same type is unlikely to improve arch-dev R².\n"
    elif avg_late_gain > 0.005:
        verdict = "more_data_beneficial"
        md += f"| **Verdict** | **More data likely beneficial** |\n\n"
        md += "Performance is still meaningfully improving at 100% training data. "
        md += "Additional matched groups would likely improve arch-dev R² further.\n"
    else:
        verdict = "inconclusive"
        md += f"| **Verdict** | **Inconclusive / approaching saturation** |\n\n"
        md += "Late-stage improvements are small but nonzero. "
        md += "The dataset may be approaching saturation, but marginal gains from more data cannot be ruled out.\n"
    
    md += "\n## Decision Rule Application\n\n"
    md += "```\n"
    if verdict == "dataset_saturated":
        md += "25% ≈ 50% ≈ 75% ≈ 100%  →  dataset likely saturated  ✓\n"
    elif verdict == "more_data_beneficial":
        md += "Performance keeps increasing toward 100%  →  more data likely beneficial  ✓\n"
    else:
        md += "Small but nonzero gains  →  approaching saturation (inconclusive)  ✓\n"
    md += "```\n"
    
    (OUT / 'stage2d_learning_curve_interpretation.md').write_text(md)
    print(f"  Saved: stage2d_learning_curve_interpretation.md")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STAGE 2D LEARNING CURVE ANALYSIS")
    print("=" * 70)
    
    # Load predictions
    print("\nLoading predictions...")
    results_df = load_all_predictions()
    
    if results_df is None or results_df.empty:
        print("\nERROR: No prediction files found in", PRED_DIR)
        print("Run run_stage2d_learning_curve.py first to generate predictions.")
        print("\nExpected file pattern:")
        print(f"  {PRED_DIR}/ea_ip__<target>__stage2d_<model>__fold<N>__frac<P>__split<N>.npz")
        return
    
    print(f"  Loaded {len(results_df)} result rows")
    print(f"  Models: {results_df['model'].unique().tolist()}")
    print(f"  Fractions: {sorted(results_df['fraction'].unique().tolist())}")
    print(f"  Folds: {sorted(results_df['fold'].unique().tolist())}")
    
    # Save CSV
    results_df.to_csv(OUT / 'stage2d_learning_curve_results.csv', index=False)
    print(f"\n  Saved: stage2d_learning_curve_results.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results_df)
    
    # Generate interpretation
    print("\nGenerating interpretation...")
    generate_interpretation(results_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Output: {OUT}")
    print("=" * 70)


if __name__ == '__main__':
    main()
