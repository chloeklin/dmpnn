"""
Stage 2D Learning Curve — Final Evaluation
============================================
Computes overall and architecture-deviation metrics from predictions
produced by run_stage2d_learning_curve_final.py.

Architecture-deviation metric (canonical definition):
    group = (smiles_A, smiles_B, fracA, fracB)
    Δy_true = y_true − mean(y_true within test group)
    Δy_pred = y_pred − mean(y_pred within test group)
    Only groups with ≥ 2 distinct architectures contribute.

Outputs:
    stage2d_learning_curve_final_metrics.csv
    stage2d_learning_curve_final_summary.md
    fig_final_learning_curve_{EA,IP}_overall.{png,pdf}
    fig_final_learning_curve_{EA,IP}_archdev.{png,pdf}

Usage:
    python evaluate_stage2d_learning_curve_final.py [--fractions 25,50,75,100]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

DATA_PATH = PROJECT_ROOT / 'data' / 'ea_ip.csv'
PRED_DIR = PROJECT_ROOT / 'predictions' / 'HPG2Stage_LC_Final'
PRED_DIR_ORIG = PROJECT_ROOT / 'predictions' / 'HPG2Stage'
OUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'learning_curve_final'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {'EA': 'EA vs SHE (eV)', 'IP': 'IP vs SHE (eV)'}
MODELS = ['2d0_arch', '2d1_arch']
MODEL_LABELS = {'2d0_arch': '2D0-arch', '2d1_arch': '2D1-arch'}
N_FOLDS = 5
SEED = 42

# Reference final Stage 2D R²(Δ) values for verification
REFERENCE_ARCHDEV = {
    '2d0_arch': {'EA': 0.84, 'IP': 0.91},
    '2d1_arch': {'EA': 0.86, 'IP': 0.91},
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


# ═══════════════════════════════════════════════════════════════════════
# CANONICAL ARCH-DEV METRIC
# ═══════════════════════════════════════════════════════════════════════

def compute_archdev_metrics(y_true, y_pred, test_orig_indices, df):
    """Compute architecture-deviation R² and MAE.

    Canonical definition from analyze_pair_disjoint_transfer.py:
        group = (smiles_A, smiles_B, fracA, fracB)
        Filter to groups with ≥ 2 unique poly_type values.
        Δy = y − group_mean(y), then R²(Δy_true, Δy_pred).
    """
    if len(y_true) < 20:
        return np.nan, np.nan

    groups = df.iloc[test_orig_indices]['group_key'].values
    arch = df.iloc[test_orig_indices]['poly_type'].values

    gdf = pd.DataFrame({
        'y_true': y_true, 'y_pred': y_pred,
        'group': groups, 'arch': arch,
    })

    # Filter to groups with ≥ 2 architectures
    ga = gdf.groupby('group')['arch'].nunique()
    multi = ga[ga >= 2].index
    gdf_m = gdf[gdf['group'].isin(multi)]

    if len(gdf_m) < 20:
        return np.nan, np.nan

    # Deviation from group mean
    gmt = gdf_m.groupby('group')['y_true'].transform('mean')
    gmp = gdf_m.groupby('group')['y_pred'].transform('mean')
    dt = gdf_m['y_true'].values - gmt.values
    dp = gdf_m['y_pred'].values - gmp.values

    if np.std(dt) < 1e-10:
        return np.nan, np.nan

    return r2_score(dt, dp), mean_absolute_error(dt, dp)


def apply_inverse_transform(y_true, y_pred):
    """Apply linregress inverse transform if predictions are normalized."""
    slope, intercept, _, _, _ = linregress(y_pred, y_true)
    if abs(slope) < 1e-10:
        return y_pred
    return y_pred * slope + intercept


def check_needs_inverse_transform(y_true, y_pred, target_name):
    """Check if predictions need inverse transform by comparing scales."""
    true_range = y_true.max() - y_true.min()
    pred_range = y_pred.max() - y_pred.min()
    if true_range < 1e-10:
        return False
    ratio = pred_range / true_range
    if ratio < 0.5 or ratio > 2.0:
        return True
    # Also check mean offset
    mean_diff = abs(y_true.mean() - y_pred.mean())
    if mean_diff > 0.5 * true_range:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# LOAD PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_lc_predictions(model, target_long, fold, frac_pct):
    """Load learning curve predictions."""
    fname = (f'ea_ip__{target_long}__stage2d_{model}__'
             f'a_held_out__fold{fold}__frac{frac_pct}.npz')
    fpath = PRED_DIR / fname
    if not fpath.exists():
        return None
    d = np.load(fpath, allow_pickle=True)
    return {
        'y_true': d['y_true'].flatten(),
        'y_pred': d['y_pred'].flatten(),
        'test_indices': d['test_indices'].flatten().astype(int),
    }


def load_original_predictions(model, target_long, fold):
    """Load original final Stage 2D predictions for comparison."""
    fname = (f'ea_ip__{target_long}__copoly_stage2d_{model}__'
             f'a_held_out__split{fold}.npz')
    fpath = PRED_DIR_ORIG / fname
    if not fpath.exists():
        return None
    d = np.load(fpath, allow_pickle=True)
    return {
        'y_true': d['y_true'].flatten(),
        'y_pred': d['y_pred'].flatten(),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Stage 2D Learning Curve (Final Pipeline)')
    parser.add_argument('--fractions', type=str, default='25,50,75,100',
                        help='Fractions to evaluate (default: 25,50,75,100)')
    args = parser.parse_args()

    fracs = [int(x) for x in args.fractions.split(',')]

    df = pd.read_csv(DATA_PATH)

    # ── Pair canonicalization + fraction normalization ─────────────────
    # Must match load_and_preprocess_data() / run_stage2d_learning_curve_final.py
    def _canon_pair(a, b, wa, wb):
        a = "" if pd.isna(a) else str(a)
        b = "" if pd.isna(b) else str(b)
        if b < a:
            return b, a, wb, wa
        return a, b, wa, wb

    raw_A = df['smiles_A'].astype(str).tolist()
    raw_B = df['smiles_B'].astype(str).tolist()
    raw_fA = df['fracA'].values.astype(float)
    raw_fB = df['fracB'].values.astype(float)

    canA, canB, fA_list, fB_list = [], [], [], []
    for a, b, wa, wb in zip(raw_A, raw_B, raw_fA, raw_fB):
        a2, b2, wa2, wb2 = _canon_pair(a, b, wa, wb)
        canA.append(a2)
        canB.append(b2)
        fA_list.append(wa2)
        fB_list.append(wb2)

    fracA_arr = np.array(fA_list, dtype=float)
    fracB_arr = np.array(fB_list, dtype=float)
    fsum = fracA_arr + fracB_arr
    fracA_arr = fracA_arr / fsum
    fracB_arr = 1.0 - fracA_arr

    df['smilesA'] = canA
    df['smilesB'] = canB
    df['fracA'] = fracA_arr
    df['fracB'] = fracB_arr

    _round6 = lambda x: round(float(x), 6)
    df['group_key'] = [
        f"{a}||{b}||{_round6(fa)}||{_round6(fb)}"
        for a, b, fa, fb in zip(canA, canB, fracA_arr, fracB_arr)
    ]

    rows = []

    for tshort, tlong in TARGETS.items():
        for model in MODELS:
            for frac_pct in fracs:
                fold_r2, fold_mae, fold_rmse = [], [], []
                fold_arch_r2, fold_arch_mae = [], []

                # Pooled arrays for cross-fold pooled metrics
                pool_yt, pool_yp, pool_dt, pool_dp = [], [], [], []

                for fold in range(N_FOLDS):
                    pred = load_lc_predictions(model, tlong, fold, frac_pct)
                    if pred is None:
                        continue

                    yt = pred['y_true']
                    yp = pred['y_pred']
                    te_idx = pred['test_indices']

                    # Check if inverse transform needed
                    if check_needs_inverse_transform(yt, yp, tlong):
                        yp = apply_inverse_transform(yt, yp)

                    # Overall metrics
                    r2 = r2_score(yt, yp)
                    mae = mean_absolute_error(yt, yp)
                    rmse = root_mean_squared_error(yt, yp)
                    fold_r2.append(r2)
                    fold_mae.append(mae)
                    fold_rmse.append(rmse)

                    # Arch-dev metrics
                    arch_r2, arch_mae = compute_archdev_metrics(
                        yt, yp, te_idx, df)
                    fold_arch_r2.append(arch_r2)
                    fold_arch_mae.append(arch_mae)

                    # Pool for cross-fold
                    pool_yt.append(yt)
                    pool_yp.append(yp)

                    # Pool arch-dev deviations
                    groups = df.iloc[te_idx]['group_key'].values
                    arch = df.iloc[te_idx]['poly_type'].values
                    gdf = pd.DataFrame({
                        'y_true': yt, 'y_pred': yp,
                        'group': groups, 'arch': arch,
                    })
                    ga = gdf.groupby('group')['arch'].nunique()
                    multi = ga[ga >= 2].index
                    gdf_m = gdf[gdf['group'].isin(multi)]
                    if len(gdf_m) >= 2:
                        gmt = gdf_m.groupby('group')['y_true'].transform('mean')
                        gmp = gdf_m.groupby('group')['y_pred'].transform('mean')
                        pool_dt.append(
                            (gdf_m['y_true'].values - gmt.values))
                        pool_dp.append(
                            (gdf_m['y_pred'].values - gmp.values))

                if not fold_r2:
                    continue

                # Per-fold mean ± std
                row = {
                    'model': model,
                    'target': tshort,
                    'fraction': frac_pct,
                    # Per-fold aggregated
                    'ea_r2' if tshort == 'EA' else 'ip_r2':
                        np.nanmean(fold_r2),
                    'ea_mae' if tshort == 'EA' else 'ip_mae':
                        np.nanmean(fold_mae),
                    'ea_rmse' if tshort == 'EA' else 'ip_rmse':
                        np.nanmean(fold_rmse),
                    'ea_arch_r2' if tshort == 'EA' else 'ip_arch_r2':
                        np.nanmean(fold_arch_r2),
                    'ea_arch_mae' if tshort == 'EA' else 'ip_arch_mae':
                        np.nanmean(fold_arch_mae),
                }

                # Also store per-fold for the CSV
                for fi, (r2, mae, rmse, ar2, amae) in enumerate(
                    zip(fold_r2, fold_mae, fold_rmse,
                        fold_arch_r2, fold_arch_mae)
                ):
                    rows.append({
                        'model': model,
                        'fold': fi,
                        'fraction': frac_pct,
                        'target': tshort,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'arch_r2': ar2,
                        'arch_mae': amae,
                    })

    if not rows:
        print("No predictions found. Ensure training has completed.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(rows)

    # Pivot to wide format with all columns requested
    ea_df = metrics_df[metrics_df['target'] == 'EA'].rename(columns={
        'r2': 'ea_r2', 'mae': 'ea_mae', 'rmse': 'ea_rmse',
        'arch_r2': 'ea_arch_r2', 'arch_mae': 'ea_arch_mae',
    })
    ip_df = metrics_df[metrics_df['target'] == 'IP'].rename(columns={
        'r2': 'ip_r2', 'mae': 'ip_mae', 'rmse': 'ip_rmse',
        'arch_r2': 'ip_arch_r2', 'arch_mae': 'ip_arch_mae',
    })
    merge_cols = ['model', 'fold', 'fraction']
    wide_df = ea_df[merge_cols + ['ea_r2', 'ea_mae', 'ea_rmse',
                                   'ea_arch_r2', 'ea_arch_mae']].merge(
        ip_df[merge_cols + ['ip_r2', 'ip_mae', 'ip_rmse',
                            'ip_arch_r2', 'ip_arch_mae']],
        on=merge_cols, how='outer',
    )
    wide_df = wide_df.sort_values(['model', 'fraction', 'fold']).reset_index(drop=True)

    csv_path = OUT_DIR / 'stage2d_learning_curve_final_metrics.csv'
    wide_df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("# Stage 2D Learning Curve — Final Pipeline Results\n")
    summary_lines.append("## Metrics: per-fold mean ± std\n")

    for tshort, tlong in TARGETS.items():
        summary_lines.append(f"\n### {tshort} ({tlong})\n")
        summary_lines.append(
            "| Model | Fraction | R² | MAE | RMSE | R²(Δ) | MAE(Δ) |")
        summary_lines.append(
            "|-------|----------|-----|-----|------|-------|--------|")

        sub = metrics_df[metrics_df['target'] == tshort]
        for model in MODELS:
            for frac_pct in fracs:
                s = sub[(sub['model'] == model) &
                        (sub['fraction'] == frac_pct)]
                if s.empty:
                    continue
                r2_m, r2_s = s['r2'].mean(), s['r2'].std()
                mae_m, mae_s = s['mae'].mean(), s['mae'].std()
                rmse_m, rmse_s = s['rmse'].mean(), s['rmse'].std()
                ar2_m, ar2_s = s['arch_r2'].mean(), s['arch_r2'].std()
                amae_m, amae_s = s['arch_mae'].mean(), s['arch_mae'].std()
                summary_lines.append(
                    f"| {MODEL_LABELS[model]} | {frac_pct}% "
                    f"| {r2_m:.4f}±{r2_s:.4f} "
                    f"| {mae_m:.4f}±{mae_s:.4f} "
                    f"| {rmse_m:.4f}±{rmse_s:.4f} "
                    f"| {ar2_m:.4f}±{ar2_s:.4f} "
                    f"| {amae_m:.4f}±{amae_s:.4f} |"
                )

    # Verification check
    summary_lines.append("\n## 100% Verification\n")
    all_pass = True
    for model in MODELS:
        for tshort in TARGETS:
            sub = metrics_df[(metrics_df['model'] == model) &
                             (metrics_df['target'] == tshort) &
                             (metrics_df['fraction'] == 100)]
            if sub.empty:
                summary_lines.append(
                    f"- **{MODEL_LABELS[model]} {tshort}**: NO DATA")
                all_pass = False
                continue
            ar2_mean = sub['arch_r2'].mean()
            ref = REFERENCE_ARCHDEV[model][tshort]
            delta = abs(ar2_mean - ref)
            status = "PASS" if delta < 0.05 else "FAIL"
            if status == "FAIL":
                all_pass = False
            summary_lines.append(
                f"- **{MODEL_LABELS[model]} {tshort}**: "
                f"R²(Δ) = {ar2_mean:.4f} (ref = {ref:.2f}, "
                f"Δ = {delta:.4f}) → **{status}**"
            )

    if all_pass:
        summary_lines.append(
            "\n**All 100% verification checks PASSED.** "
            "Safe to run 25/50/75% fractions."
        )
    else:
        summary_lines.append(
            "\n**WARNING: Some verification checks FAILED.** "
            "Debug before running other fractions."
        )

    summary_path = OUT_DIR / 'stage2d_learning_curve_final_summary.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f"Saved summary: {summary_path}")
    print('\n'.join(summary_lines))

    # ── Figures ───────────────────────────────────────────────────────
    if len(fracs) > 1:
        _plot_curves(metrics_df, fracs)


def _plot_curves(metrics_df, fracs):
    """Generate learning curve figures."""
    colors = {'2d0_arch': '#E24A33', '2d1_arch': '#348ABD'}
    markers = {'2d0_arch': 'o', '2d1_arch': 's'}

    for tshort in TARGETS:
        # ── Overall R² ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = metrics_df[metrics_df['target'] == tshort]
        for model in MODELS:
            ms = sub[sub['model'] == model]
            means = ms.groupby('fraction')['r2'].mean()
            stds = ms.groupby('fraction')['r2'].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        label=MODEL_LABELS[model], color=colors[model],
                        marker=markers[model], capsize=4, linewidth=2,
                        markersize=8)
        ax.set_xlabel('Training Fraction (%)')
        ax.set_ylabel('R²')
        ax.set_title(f'{tshort} — Overall R² vs Training Fraction')
        ax.set_xticks(fracs)
        ax.legend()
        ax.grid(alpha=0.3)
        for ext in ('png', 'pdf'):
            fig.savefig(OUT_DIR / f'fig_final_learning_curve_{tshort}_overall.{ext}')
        plt.close(fig)

        # ── Arch-dev R² ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        for model in MODELS:
            ms = sub[sub['model'] == model]
            means = ms.groupby('fraction')['arch_r2'].mean()
            stds = ms.groupby('fraction')['arch_r2'].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        label=MODEL_LABELS[model], color=colors[model],
                        marker=markers[model], capsize=4, linewidth=2,
                        markersize=8)
        ax.set_xlabel('Training Fraction (%)')
        ax.set_ylabel('R²(Δy)')
        ax.set_title(f'{tshort} — Architecture-Deviation R² vs Training Fraction')
        ax.set_xticks(fracs)
        ax.legend()
        ax.grid(alpha=0.3)
        for ext in ('png', 'pdf'):
            fig.savefig(OUT_DIR / f'fig_final_learning_curve_{tshort}_archdev.{ext}')
        plt.close(fig)

    print(f"Saved figures to {OUT_DIR}")


if __name__ == '__main__':
    main()
