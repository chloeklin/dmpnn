#!/usr/bin/env python3
"""
Fusion Ablation Consistency Analysis
======================================
Paired fold-wise analysis of Additive (2D1) vs FiLM / NLMix / FiLM+NLMix.
No retraining; no re-evaluation. All data from existing .npz prediction files.

Outputs (all to output/fusion_ablation/):
  fusion_per_fold_metrics.csv
  fusion_fold_deltas.csv
  fusion_fold_win_summary.md
  fusion_significance_tests.md / .csv
  fusion_foldwise_ea_r2.png
  fusion_foldwise_ip_r2.png
  fusion_foldwise_ea_archdev_r2.png
  fusion_foldwise_ip_archdev_r2.png
  film_vs_additive_difference_plots.png
  fusion_without_fold6.md
  fusion_ablation_consistency_report.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PRED_ABLATION = ROOT / 'predictions' / 'hpg2stage_ablation'
PRED_LOMAO    = ROOT / 'predictions' / 'HPG2Stage_LOMAO'
OUT_DIR       = ROOT / 'output' / 'fusion_ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'experiments' / 'hpg2stage' / 'scripts'))
import generate_stage2d_paper_outputs as g2d

N_FOLDS  = 9
TARGETS  = {'EA': 'EA vs SHE (eV)', 'IP': 'IP vs SHE (eV)'}
VARIANTS = {
    'Additive (2D1)': None,
    'FiLM':           '2d1_film',
    'NLMix':          '2d1_nlmix',
    'FiLM+NLMix':     '2d1_film_nlmix',
}
VARIANT_NAMES = list(VARIANTS.keys())
NON_BASE      = [v for v in VARIANT_NAMES if v != 'Additive (2D1)']

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
COLORS = {
    'Additive (2D1)': '#2ca02c',
    'FiLM':           '#d62728',
    'NLMix':          '#ff7f0e',
    'FiLM+NLMix':     '#9467bd',
}
MARKERS = {'Additive (2D1)': 'o', 'FiLM': 's', 'NLMix': '^', 'FiLM+NLMix': 'D'}

OUTLIER_FOLD = 6   # hard OOD fold


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_additive(target_key: str, fold: int):
    target_full = TARGETS[target_key]
    fname = f"ea_ip__{target_full}__copoly_stage2d_2d1_arch__a_held_out__split{fold}.npz"
    path  = PRED_LOMAO / fname
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    yt = data['y_true'].flatten()
    yp = data['y_pred'].flatten()
    # test_ids are 'idx_N' strings — extract integer suffix as df row index
    if 'test_ids' in data:
        try:
            indices = np.array([int(str(tid).split('_')[-1]) for tid in data['test_ids']], dtype=int)
        except (ValueError, IndexError):
            indices = np.full(len(yt), -1, dtype=int)
    else:
        indices = np.full(len(yt), -1, dtype=int)
    return {'y_true': yt, 'y_pred': yp, 'indices': indices}


def _load_ablation(suffix: str, target_key: str, fold: int):
    target_full = TARGETS[target_key]
    fname = f"ea_ip__{target_full}__stage2d_{suffix}__lomao__fold{fold}.npz"
    path  = PRED_ABLATION / fname
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    yt   = data['y_true'].flatten()
    yp   = data['y_pred'].flatten()
    indices = data['test_indices'].astype(int) if 'test_indices' in data else np.full(len(yt), -1, dtype=int)
    return {'y_true': yt, 'y_pred': yp, 'indices': indices}


def load_all(df):
    """Returns dict: variant → target → fold → {'y_true', 'y_pred', 'indices'}"""
    out = {v: {t: {} for t in TARGETS} for v in VARIANT_NAMES}
    for fold in range(N_FOLDS):
        for tk in TARGETS:
            r = _load_additive(tk, fold)
            if r:
                out['Additive (2D1)'][tk][fold] = r
            for vname, vsuffix in VARIANTS.items():
                if vsuffix is None:
                    continue
                r = _load_ablation(vsuffix, tk, fold)
                if r:
                    out[vname][tk][fold] = r
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(d, df):
    yt, yp, idx = d['y_true'], d['y_pred'], d['indices']
    r2   = float(r2_score(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    ar2, amae = g2d.compute_archdev_metrics(yt, yp, idx, df)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'ArchR2': ar2, 'ArchMAE': amae}


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Per-fold metrics CSV
# ═══════════════════════════════════════════════════════════════════════════════

def task1(data, df):
    print("Task 1: Per-fold metrics")
    rows = []
    # Store for later use: metric_table[variant][target][fold] = metrics_dict
    metric_table = {v: {t: {} for t in TARGETS} for v in VARIANT_NAMES}

    for vname in VARIANT_NAMES:
        for tk in TARGETS:
            for fold in range(N_FOLDS):
                d = data[vname][tk].get(fold)
                if d is None:
                    continue
                m = compute_metrics(d, df)
                metric_table[vname][tk][fold] = m
                rows.append({
                    'Model': vname, 'Target': tk, 'Fold': fold,
                    'R2': m['R2'], 'RMSE': m['RMSE'], 'MAE': m['MAE'],
                    'ArchDev_R2': m['ArchR2'], 'ArchDev_MAE': m['ArchMAE'],
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / 'fusion_per_fold_metrics.csv', index=False)
    print("  Saved: fusion_per_fold_metrics.csv")
    return metric_table


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Paired fold deltas
# ═══════════════════════════════════════════════════════════════════════════════

def task2(metric_table):
    print("Task 2: Paired fold deltas")
    rows = []
    METRICS = [
        ('EA', 'R2',      'ΔEA_R2'),
        ('EA', 'RMSE',    'ΔEA_RMSE'),
        ('EA', 'ArchR2',  'ΔEA_ArchDev_R2'),
        ('IP', 'R2',      'ΔIP_R2'),
        ('IP', 'RMSE',    'ΔIP_RMSE'),
        ('IP', 'ArchR2',  'ΔIP_ArchDev_R2'),
    ]
    for fold in range(N_FOLDS):
        base_ea = metric_table['Additive (2D1)']['EA'].get(fold, {})
        base_ip = metric_table['Additive (2D1)']['IP'].get(fold, {})
        for vname in NON_BASE:
            var_ea = metric_table[vname]['EA'].get(fold, {})
            var_ip = metric_table[vname]['IP'].get(fold, {})
            row = {'Variant': vname, 'Fold': fold}
            for target, metric, col in METRICS:
                base = base_ea if target == 'EA' else base_ip
                var  = var_ea  if target == 'EA' else var_ip
                bv   = base.get(metric, np.nan)
                vv   = var.get(metric, np.nan)
                # For RMSE lower is better: delta = variant - additive (negative = variant better)
                row[col] = vv - bv if not (np.isnan(bv) or np.isnan(vv)) else np.nan
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / 'fusion_fold_deltas.csv', index=False)
    print("  Saved: fusion_fold_deltas.csv")
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Fold win counts
# ═══════════════════════════════════════════════════════════════════════════════

def task3(metric_table, df_deltas):
    print("Task 3: Fold win summary")
    COMPARISONS = [
        ('EA', 'R2',     True,  'Overall EA R²'),
        ('IP', 'R2',     True,  'Overall IP R²'),
        ('EA', 'ArchR2', True,  'EA ArchDev R²'),
        ('IP', 'ArchR2', True,  'IP ArchDev R²'),
        ('EA', 'MAE',    False, 'EA MAE'),
        ('IP', 'MAE',    False, 'IP MAE'),
    ]

    lines = ["# Fusion Fold Win Summary\n\n",
             "Comparing each fusion variant against Additive (2D1) on a fold-by-fold basis.\n\n"]

    for tk, metric, higher_better, label in COMPARISONS:
        lines.append(f"## {label}\n\n")
        for vname in NON_BASE:
            base_vals = [metric_table['Additive (2D1)'][tk].get(f, {}).get(metric, np.nan)
                         for f in range(N_FOLDS)]
            var_vals  = [metric_table[vname][tk].get(f, {}).get(metric, np.nan)
                         for f in range(N_FOLDS)]

            lines.append(f"### {vname} vs Additive (2D1)\n\n")
            lines.append("| Fold | Additive | Variant | Winner |\n")
            lines.append("|---|---|---|---|\n")

            var_wins = add_wins = ties = 0
            for fold in range(N_FOLDS):
                bv, vv = base_vals[fold], var_vals[fold]
                if np.isnan(bv) or np.isnan(vv):
                    winner = 'N/A'
                elif higher_better:
                    if vv > bv + 1e-6:   winner = vname; var_wins += 1
                    elif bv > vv + 1e-6: winner = 'Additive (2D1)'; add_wins += 1
                    else:                winner = 'Tie'; ties += 1
                else:
                    if vv < bv - 1e-6:   winner = vname; var_wins += 1
                    elif bv < vv - 1e-6: winner = 'Additive (2D1)'; add_wins += 1
                    else:                winner = 'Tie'; ties += 1
                fold6_flag = ' ← **fold 6 (OOD)**' if fold == OUTLIER_FOLD else ''
                lines.append(f"| {fold} | {bv:.4f} | {vv:.4f} | {winner}{fold6_flag} |\n")

            lines.append(f"\n**{vname} wins: {var_wins}/9 folds**  "
                         f"| Additive wins: {add_wins}/9 folds  "
                         f"| Ties: {ties}\n\n")

    with open(OUT_DIR / 'fusion_fold_win_summary.md', 'w') as f:
        f.writelines(lines)
    print("  Saved: fusion_fold_win_summary.md")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d(x, y):
    """Paired Cohen's d: mean(diff) / std(diff)."""
    diff = np.array(x) - np.array(y)
    if diff.std(ddof=1) == 0:
        return np.nan
    return float(diff.mean() / diff.std(ddof=1))


def bootstrap_ci(diff, n_boot=5000, ci=0.95, rng_seed=42):
    rng  = np.random.default_rng(rng_seed)
    boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    lo   = np.percentile(boot, (1 - ci) / 2 * 100)
    hi   = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def task4(metric_table):
    print("Task 4: Statistical tests")

    TEST_METRICS = [
        ('EA', 'R2',     True,  'EA Overall R²'),
        ('IP', 'R2',     True,  'IP Overall R²'),
        ('EA', 'ArchR2', True,  'EA ArchDev R²'),
        ('IP', 'ArchR2', True,  'IP ArchDev R²'),
        ('EA', 'MAE',    False, 'EA MAE'),
        ('IP', 'MAE',    False, 'IP MAE'),
    ]

    stat_rows = []
    lines = ["# Fusion Ablation — Paired Statistical Tests\n\n",
             "Nine folds treated as paired observations (variant vs Additive 2D1).\n",
             "**Δ = variant − additive** (positive = variant better for R², "
             "negative = variant better for MAE/RMSE).\n\n"]

    for tk, metric, higher_better, label in TEST_METRICS:
        lines.append(f"## {label}\n\n")
        lines.append("| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | "
                     "Cohen's d | 95% CI (bootstrap) | Significant? |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")

        for vname in NON_BASE:
            base_vals = np.array([metric_table['Additive (2D1)'][tk].get(f, {}).get(metric, np.nan)
                                  for f in range(N_FOLDS)])
            var_vals  = np.array([metric_table[vname][tk].get(f, {}).get(metric, np.nan)
                                  for f in range(N_FOLDS)])

            valid = ~(np.isnan(base_vals) | np.isnan(var_vals))
            bv    = base_vals[valid]
            vv    = var_vals[valid]
            diff  = vv - bv

            mean_d   = float(diff.mean())
            med_d    = float(np.median(diff))
            cd       = cohens_d(vv, bv)
            ci_lo, ci_hi = bootstrap_ci(diff)

            t_stat, t_p   = stats.ttest_rel(vv, bv)
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_p = np.nan

            sig = ('Yes' if (not np.isnan(w_p) and w_p < 0.05)
                   else 'No' if not np.isnan(w_p)
                   else 'N/A')

            ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
            lines.append(f"| {vname} vs Additive | {mean_d:+.4f} | {med_d:+.4f} | "
                         f"{t_p:.4f} | {w_p:.4f} | {cd:+.4f} | {ci_str} | {sig} |\n")

            stat_rows.append({
                'Metric': label, 'Comparison': f"{vname} vs Additive",
                'Mean_delta': mean_d, 'Median_delta': med_d,
                't_pvalue': float(t_p), 'Wilcoxon_pvalue': float(w_p) if not np.isnan(w_p) else np.nan,
                'Cohens_d': cd, 'CI_95_lo': ci_lo, 'CI_95_hi': ci_hi,
                'Significant_Wilcoxon_p05': sig,
            })
        lines.append("\n")

    with open(OUT_DIR / 'fusion_significance_tests.md', 'w') as f:
        f.writelines(lines)
    pd.DataFrame(stat_rows).to_csv(OUT_DIR / 'fusion_significance_tests.csv', index=False)
    print("  Saved: fusion_significance_tests.md / .csv")
    return stat_rows


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Fold-wise line plots
# ═══════════════════════════════════════════════════════════════════════════════

def _foldwise_lineplot(metric_table, target_key, metric_key, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    folds = list(range(N_FOLDS))

    for vname in VARIANT_NAMES:
        vals = [metric_table[vname][target_key].get(f, {}).get(metric_key, np.nan)
                for f in folds]
        ax.plot(folds, vals, marker=MARKERS[vname], color=COLORS[vname],
                label=vname, linewidth=1.8, markersize=6, zorder=3)

    # Highlight fold 6
    ax.axvspan(OUTLIER_FOLD - 0.4, OUTLIER_FOLD + 0.4, alpha=0.15, color='grey',
               label=f'Fold {OUTLIER_FOLD} (OOD)', zorder=1)
    ax.axvline(OUTLIER_FOLD, color='grey', linewidth=1.0, linestyle='--', zorder=2)

    ax.set_xticks(folds)
    ax.set_xticklabels([f'F{f}' for f in folds])
    ax.set_xlabel('Fold (held-out monomer)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right', framealpha=0.85)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def task5(metric_table):
    print("Task 5: Fold-wise line plots")
    plots = [
        ('EA', 'R2',     'R²',                    'Overall EA R² per fold',           'fusion_foldwise_ea_r2.png'),
        ('IP', 'R2',     'R²',                    'Overall IP R² per fold',           'fusion_foldwise_ip_r2.png'),
        ('EA', 'ArchR2', 'Architecture-Dev R²',   'EA Architecture-Deviation R² per fold', 'fusion_foldwise_ea_archdev_r2.png'),
        ('IP', 'ArchR2', 'Architecture-Dev R²',   'IP Architecture-Deviation R² per fold', 'fusion_foldwise_ip_archdev_r2.png'),
    ]
    for tk, mk, ylabel, title, fname in plots:
        _foldwise_lineplot(metric_table, tk, mk, ylabel, title, OUT_DIR / fname)
        print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6 — Bland-Altman / difference plots (FiLM vs Additive)
# ═══════════════════════════════════════════════════════════════════════════════

def task6(metric_table):
    print("Task 6: Bland-Altman difference plots (FiLM vs Additive)")

    PANELS = [
        ('EA', 'R2',     'EA Overall R²'),
        ('IP', 'R2',     'IP Overall R²'),
        ('EA', 'ArchR2', 'EA ArchDev R²'),
        ('IP', 'ArchR2', 'IP ArchDev R²'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (tk, mk, label) in zip(axes, PANELS):
        add_vals  = np.array([metric_table['Additive (2D1)'][tk].get(f, {}).get(mk, np.nan)
                               for f in range(N_FOLDS)])
        film_vals = np.array([metric_table['FiLM'][tk].get(f, {}).get(mk, np.nan)
                               for f in range(N_FOLDS)])
        valid     = ~(np.isnan(add_vals) | np.isnan(film_vals))
        means     = (film_vals[valid] + add_vals[valid]) / 2
        diffs     = film_vals[valid] - add_vals[valid]
        folds_v   = np.array(range(N_FOLDS))[valid]

        mean_d = diffs.mean()
        std_d  = diffs.std(ddof=1)
        loa_hi = mean_d + 1.96 * std_d
        loa_lo = mean_d - 1.96 * std_d

        scatter_colors = ['#d62728' if f == OUTLIER_FOLD else '#1f77b4' for f in folds_v]
        ax.scatter(means, diffs, c=scatter_colors, zorder=4, s=60)

        # Annotate fold labels
        for i, (m, d, f) in enumerate(zip(means, diffs, folds_v)):
            ax.annotate(f'F{f}', (m, d), textcoords='offset points',
                        xytext=(5, 3), fontsize=7,
                        color='#d62728' if f == OUTLIER_FOLD else 'black')

        ax.axhline(0,       color='black',  linewidth=1.0, linestyle='-',  zorder=2)
        ax.axhline(mean_d,  color='blue',   linewidth=1.2, linestyle='--', zorder=3,
                   label=f'Mean Δ={mean_d:+.3f}')
        ax.axhline(loa_hi,  color='red',    linewidth=0.9, linestyle=':',  zorder=3,
                   label=f'+1.96SD={loa_hi:+.3f}')
        ax.axhline(loa_lo,  color='red',    linewidth=0.9, linestyle=':',  zorder=3,
                   label=f'−1.96SD={loa_lo:+.3f}')
        ax.fill_between(ax.get_xlim(), loa_lo, loa_hi, alpha=0.06, color='blue')

        ax.set_xlabel('Mean of FiLM and Additive')
        ax.set_ylabel('Difference (FiLM − Additive)')
        ax.set_title(label)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.25)

    fig.suptitle('Bland-Altman Difference Plots: FiLM vs Additive (2D1)\n'
                 'Red dot = fold 6 (OOD outlier)', fontweight='bold', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'film_vs_additive_difference_plots.png')
    plt.close(fig)
    print("  Saved: film_vs_additive_difference_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7 — Analysis excluding fold 6
# ═══════════════════════════════════════════════════════════════════════════════

def _compare_subset(metric_table, folds_subset, label):
    """Return dict of summary stats for a given set of folds."""
    TEST_METRICS = [
        ('EA', 'R2',     'EA Overall R²'),
        ('IP', 'R2',     'IP Overall R²'),
        ('EA', 'ArchR2', 'EA ArchDev R²'),
        ('IP', 'ArchR2', 'IP ArchDev R²'),
    ]
    results = {}
    for tk, mk, mlabel in TEST_METRICS:
        for vname in NON_BASE:
            bv = np.array([metric_table['Additive (2D1)'][tk].get(f, {}).get(mk, np.nan)
                           for f in folds_subset])
            vv = np.array([metric_table[vname][tk].get(f, {}).get(mk, np.nan)
                           for f in folds_subset])
            valid = ~(np.isnan(bv) | np.isnan(vv))
            if valid.sum() < 2:
                continue
            diff = vv[valid] - bv[valid]
            try:
                _, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_p = np.nan
            results[(mlabel, vname)] = {
                'mean_add': float(np.mean(bv[valid])),
                'mean_var': float(np.mean(vv[valid])),
                'median_add': float(np.median(bv[valid])),
                'median_var': float(np.median(vv[valid])),
                'mean_delta': float(diff.mean()),
                'median_delta': float(np.median(diff)),
                'wilcoxon_p': float(w_p) if not np.isnan(w_p) else np.nan,
                'cohens_d': cohens_d(vv[valid], bv[valid]),
                'n_folds': int(valid.sum()),
            }
    return results


def task7(metric_table):
    print("Task 7: Analysis excluding fold 6")

    folds_all   = list(range(N_FOLDS))
    folds_no6   = [f for f in folds_all if f != OUTLIER_FOLD]

    res_all = _compare_subset(metric_table, folds_all,  'all 9 folds')
    res_no6 = _compare_subset(metric_table, folds_no6,  'folds 0-5,7-8 (excl. fold 6)')

    lines = ["# Fusion Ablation — Analysis Excluding Fold 6\n\n",
             f"Fold {OUTLIER_FOLD} is a hard OOD fold (held-out monomer: "
             "OB(O)c1ccc(B(O)O)c2nsnc12).\n",
             "This document compares paired analyses with and without this fold.\n\n"]

    TEST_METRICS = [
        ('EA', 'R2',     'EA Overall R²'),
        ('IP', 'R2',     'IP Overall R²'),
        ('EA', 'ArchR2', 'EA ArchDev R²'),
        ('IP', 'ArchR2', 'IP ArchDev R²'),
    ]

    for tk, mk, mlabel in TEST_METRICS:
        lines.append(f"## {mlabel}\n\n")
        lines.append("| Comparison | Subset | Mean Δ | Median Δ | "
                     "Wilcoxon p | Cohen's d | Conclusion |\n")
        lines.append("|---|---|---|---|---|---|---|\n")

        for vname in NON_BASE:
            for subset_label, res in [('All 9 folds', res_all),
                                       ('Excl. fold 6 (n=8)', res_no6)]:
                r = res.get((mlabel, vname), {})
                if not r:
                    lines.append(f"| {vname} | {subset_label} | N/A | N/A | N/A | N/A | N/A |\n")
                    continue
                mean_d = r['mean_delta']
                med_d  = r['median_delta']
                wp     = r['wilcoxon_p']
                cd     = r['cohens_d']
                sig    = 'p<0.05' if not np.isnan(wp) and wp < 0.05 else 'n.s.'
                direction = ('variant better' if mean_d > 0
                             else 'additive better' if mean_d < 0
                             else 'equal')
                concl  = f"{direction}, {sig}"
                lines.append(f"| {vname} | {subset_label} | {mean_d:+.4f} | "
                              f"{med_d:+.4f} | {wp:.4f} | {cd:+.4f} | {concl} |\n")
        lines.append("\n")

    # Per-fold R² table (easy comparison)
    lines.append("## Per-fold EA Overall R² (all models)\n\n")
    lines.append("| Fold | OOD? | Additive | FiLM | NLMix | FiLM+NLMix |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for fold in folds_all:
        ood = '**YES**' if fold == OUTLIER_FOLD else 'No'
        vals = []
        for vname in VARIANT_NAMES:
            v = metric_table[vname]['EA'].get(fold, {}).get('R2', np.nan)
            vals.append(f"{v:.4f}" if not np.isnan(v) else 'N/A')
        lines.append(f"| {fold} | {ood} | {' | '.join(vals)} |\n")

    lines.append("\n## Summary: Do conclusions change after excluding fold 6?\n\n")
    # Build a simple summary comparing sig/direction
    changed = []
    for tk, mk, mlabel in TEST_METRICS:
        for vname in NON_BASE:
            r_all = res_all.get((mlabel, vname), {})
            r_no6 = res_no6.get((mlabel, vname), {})
            if not r_all or not r_no6:
                continue
            dir_all = 'variant' if r_all['mean_delta'] > 0 else 'additive'
            dir_no6 = 'variant' if r_no6['mean_delta'] > 0 else 'additive'
            sig_all = r_all['wilcoxon_p'] < 0.05 if not np.isnan(r_all['wilcoxon_p']) else False
            sig_no6 = r_no6['wilcoxon_p'] < 0.05 if not np.isnan(r_no6['wilcoxon_p']) else False
            if dir_all != dir_no6:
                changed.append(f"- **{mlabel} / {vname}**: direction reverses "
                                f"({dir_all} → {dir_no6})\n")
            elif sig_all != sig_no6:
                changed.append(f"- **{mlabel} / {vname}**: significance changes "
                                f"({'sig' if sig_all else 'n.s.'} → {'sig' if sig_no6 else 'n.s.'})\n")

    if changed:
        lines.append("The following comparisons change direction or significance:\n\n")
        lines.extend(changed)
    else:
        lines.append("No comparison changes direction after excluding fold 6. "
                     "Significance levels may shift but overall patterns are robust.\n")

    with open(OUT_DIR / 'fusion_without_fold6.md', 'w') as f:
        f.writelines(lines)
    print("  Saved: fusion_without_fold6.md")
    return res_all, res_no6


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8 — Final interpretation report
# ═══════════════════════════════════════════════════════════════════════════════

def task8(metric_table, stat_rows, res_all, res_no6):
    print("Task 8: Final interpretation report")

    lines = [
        "# Fusion Ablation — Consistency Report\n\n",
        "All conclusions are based solely on paired fold analyses and statistical tests.\n",
        "No model speculation or extrapolation beyond observed data.\n\n",
    ]

    # ── Helper: retrieve stat row ─────────────────────────────────────────────
    def get_stat(metric_label, comparison):
        for r in stat_rows:
            if r['Metric'] == metric_label and r['Comparison'] == comparison:
                return r
        return {}

    # ── Build per-question evidence ───────────────────────────────────────────

    # Q1: Does FiLM consistently outperform Additive for overall EA?
    lines.append("## Q1: Does FiLM consistently outperform Additive for overall EA R²?\n\n")
    tk, mk = 'EA', 'R2'
    ea_r2_add  = [metric_table['Additive (2D1)']['EA'].get(f,{}).get('R2', np.nan) for f in range(N_FOLDS)]
    ea_r2_film = [metric_table['FiLM']['EA'].get(f,{}).get('R2', np.nan) for f in range(N_FOLDS)]
    film_better = sum(1 for a, b in zip(ea_r2_add, ea_r2_film)
                      if not (np.isnan(a) or np.isnan(b)) and b > a)
    add_better  = sum(1 for a, b in zip(ea_r2_add, ea_r2_film)
                      if not (np.isnan(a) or np.isnan(b)) and a > b)
    s_all = get_stat('EA Overall R²', 'FiLM vs Additive')
    s_no6 = res_no6.get(('EA Overall R²', 'FiLM'), {})
    lines.append(f"- FiLM wins EA R² in {film_better}/9 folds; Additive wins {add_better}/9 folds.\n")
    lines.append(f"- All-fold: mean Δ={s_all.get('Mean_delta',np.nan):+.4f}, "
                 f"median Δ={s_all.get('Median_delta',np.nan):+.4f}, "
                 f"Wilcoxon p={s_all.get('Wilcoxon_pvalue',np.nan):.4f}, "
                 f"Cohen's d={s_all.get('Cohens_d',np.nan):+.4f}\n")
    if s_no6:
        lines.append(f"- Excl. fold 6: mean Δ={s_no6['mean_delta']:+.4f}, "
                     f"median Δ={s_no6['median_delta']:+.4f}, "
                     f"Wilcoxon p={s_no6['wilcoxon_p']:.4f}\n")
    lines.append(f"- Per-fold EA R² for reference: "
                 f"Additive medians={np.nanmedian(ea_r2_add):.4f}, "
                 f"FiLM median={np.nanmedian(ea_r2_film):.4f}\n\n")

    # Q2: Does Additive consistently outperform fusion for EA ArchDev R²?
    lines.append("## Q2: Does Additive consistently outperform fusion for EA ArchDev R²?\n\n")
    lines.append("| Variant | Additive wins | Variant wins | Wilcoxon p (all) | Wilcoxon p (excl. fold 6) |\n")
    lines.append("|---|---|---|---|---|\n")
    for vname in NON_BASE:
        bv = [metric_table['Additive (2D1)']['EA'].get(f,{}).get('ArchR2', np.nan) for f in range(N_FOLDS)]
        vv = [metric_table[vname]['EA'].get(f,{}).get('ArchR2', np.nan) for f in range(N_FOLDS)]
        aw = sum(1 for a, b in zip(bv, vv) if not (np.isnan(a) or np.isnan(b)) and a > b)
        vw = sum(1 for a, b in zip(bv, vv) if not (np.isnan(a) or np.isnan(b)) and b > a)
        s  = get_stat('EA ArchDev R²', f'{vname} vs Additive')
        r6 = res_no6.get(('EA ArchDev R²', vname), {})
        wp_all = s.get('Wilcoxon_pvalue', np.nan)
        wp_no6 = r6.get('wilcoxon_p', np.nan)
        lines.append(f"| {vname} | {aw}/9 | {vw}/9 | {wp_all:.4f} | {wp_no6:.4f} |\n")
    lines.append("\n")

    # Q3: Statistical significance
    lines.append("## Q3: Are the observed differences statistically significant?\n\n")
    lines.append("Summary of Wilcoxon signed-rank tests (all 9 folds):\n\n")
    lines.append("| Metric | Comparison | Mean Δ | Wilcoxon p | Significant (p<0.05)? |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in stat_rows:
        sig = 'Yes' if r['Significant_Wilcoxon_p05'] == 'Yes' else 'No'
        wp  = r['Wilcoxon_pvalue']
        lines.append(f"| {r['Metric']} | {r['Comparison']} | "
                     f"{r['Mean_delta']:+.4f} | {wp:.4f} | {sig} |\n")
    lines.append("\n")

    # Q4: Robustness after excluding fold 6
    lines.append("## Q4: Are conclusions robust after excluding fold 6?\n\n")
    changed_any = False
    for tk, mk, mlabel in [('EA', 'R2', 'EA Overall R²'), ('IP', 'R2', 'IP Overall R²'),
                             ('EA', 'ArchR2', 'EA ArchDev R²'), ('IP', 'ArchR2', 'IP ArchDev R²')]:
        for vname in NON_BASE:
            r_a = res_all.get((mlabel, vname), {})
            r_n = res_no6.get((mlabel, vname), {})
            if not r_a or not r_n:
                continue
            dir_a = r_a['mean_delta'] > 0
            dir_n = r_n['mean_delta'] > 0
            if dir_a != dir_n:
                lines.append(f"- **{mlabel} / {vname}**: direction reverses between "
                              f"all-folds ({r_a['mean_delta']:+.4f}) and "
                              f"excl. fold 6 ({r_n['mean_delta']:+.4f}).\n")
                changed_any = True
    if not changed_any:
        lines.append("No comparison changes direction after excluding fold 6. "
                     "The mean-delta sign is consistent in all cases.\n")
    lines.append("\n")

    # Q5: Evidence for genuine trade-off
    lines.append("## Q5: Is there evidence for a genuine trade-off between overall prediction "
                 "and architecture recovery?\n\n")
    lines.append("For each variant, compare direction of Δ for Overall R² vs ArchDev R²:\n\n")
    lines.append("| Variant | EA Overall Δ (mean) | EA ArchDev Δ (mean) | "
                 "IP Overall Δ (mean) | IP ArchDev Δ (mean) | Trade-off observed? |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for vname in NON_BASE:
        ea_ov  = res_all.get(('EA Overall R²',  vname), {}).get('mean_delta', np.nan)
        ea_ad  = res_all.get(('EA ArchDev R²',  vname), {}).get('mean_delta', np.nan)
        ip_ov  = res_all.get(('IP Overall R²',  vname), {}).get('mean_delta', np.nan)
        ip_ad  = res_all.get(('IP ArchDev R²',  vname), {}).get('mean_delta', np.nan)
        # Trade-off: overall improves but archdev degrades (or vice versa)
        ea_tradeoff = (not np.isnan(ea_ov) and not np.isnan(ea_ad) and
                       np.sign(ea_ov) != np.sign(ea_ad))
        ip_tradeoff = (not np.isnan(ip_ov) and not np.isnan(ip_ad) and
                       np.sign(ip_ov) != np.sign(ip_ad))
        to = 'Yes (EA)' if ea_tradeoff else ''
        to += ' Yes (IP)' if ip_tradeoff else ''
        if not to:
            to = 'No'
        def fmtv(v): return f"{v:+.4f}" if not np.isnan(v) else 'N/A'
        lines.append(f"| {vname} | {fmtv(ea_ov)} | {fmtv(ea_ad)} | "
                     f"{fmtv(ip_ov)} | {fmtv(ip_ad)} | {to} |\n")
    lines.append("\n")

    # ── Per-fold summary tables ────────────────────────────────────────────────
    lines.append("## Reference: Per-fold R² table\n\n")
    for tk in ('EA', 'IP'):
        lines.append(f"### {tk} Overall R²\n\n")
        lines.append("| Fold | OOD? | " + " | ".join(VARIANT_NAMES) + " |\n")
        lines.append("|---|---|" + "|".join(["---"] * len(VARIANT_NAMES)) + "|\n")
        for fold in range(N_FOLDS):
            ood = "**YES**" if fold == OUTLIER_FOLD else "No"
            vals = [metric_table[v][tk].get(fold,{}).get('R2', np.nan) for v in VARIANT_NAMES]
            row  = " | ".join(f"{v:.4f}" if not np.isnan(v) else "N/A" for v in vals)
            lines.append(f"| {fold} | {ood} | {row} |\n")
        lines.append("\n")

    with open(OUT_DIR / 'fusion_ablation_consistency_report.md', 'w') as f:
        f.writelines(lines)
    print("  Saved: fusion_ablation_consistency_report.md")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Fusion Ablation Consistency Analysis")
    print("=" * 70)

    df = g2d.load_dataset()
    g2d._ensure_value_map(df)

    data = load_all(df)

    # Verify completeness
    for vname in VARIANT_NAMES:
        for tk in TARGETS:
            n = len(data[vname][tk])
            status = "OK" if n == N_FOLDS else f"MISSING {N_FOLDS - n} folds"
            print(f"  {vname:20s} {tk}: {n}/{N_FOLDS} folds — {status}")

    metric_table = task1(data, df)
    df_deltas    = task2(metric_table)
    task3(metric_table, df_deltas)
    stat_rows    = task4(metric_table)
    task5(metric_table)
    task6(metric_table)
    res_all, res_no6 = task7(metric_table)
    task8(metric_table, stat_rows, res_all, res_no6)

    print("=" * 70)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
