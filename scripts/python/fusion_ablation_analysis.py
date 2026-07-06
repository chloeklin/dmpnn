#!/usr/bin/env python3
"""
Stage 2D1 Fusion Ablation Analysis
====================================
Gathers, summarises and visualises completed fusion-ablation experiments.
No retraining; no re-evaluation. All data from existing .npz prediction files.

Variants analysed (LOMAO, 9 folds each):
  Additive   — predictions/HPG2Stage_LOMAO/   *copoly_stage2d_2d1_arch*  (denorm required)
  FiLM       — predictions/hpg2stage_ablation/ *stage2d_2d1_film*        (real scale)
  NLMix      — predictions/hpg2stage_ablation/ *stage2d_2d1_nlmix*       (real scale)
  FiLM+NLMix — predictions/hpg2stage_ablation/ *stage2d_2d1_film_nlmix*  (real scale)

Outputs (all to output/fusion_ablation/):
  fusion_experiment_inventory.md
  fusion_ablation_summary.csv / .md
  fusion_model_ranking.md
  fusion_fold_metrics.csv
  fusion_prediction_inventory.md
  fusion_results_for_chatgpt.md
  fusion_overall_r2.png / .pdf
  fusion_archdev_r2.png / .pdf
  fusion_delta_vs_additive.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PRED_ABLATION  = ROOT / 'predictions' / 'hpg2stage_ablation'
PRED_LOMAO     = ROOT / 'predictions' / 'HPG2Stage_LOMAO'
CKPT_ABLATION  = ROOT / 'checkpoints'  / 'HPG2Stage_Ablation'
OUT_DIR        = ROOT / 'output' / 'fusion_ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'experiments' / 'hpg2stage' / 'scripts'))
import generate_stage2d_paper_outputs as g2d

N_FOLDS   = 9
TARGETS   = {'EA': 'EA vs SHE (eV)', 'IP': 'IP vs SHE (eV)'}
VARIANTS  = {
    'Additive (2D1)' : '2d1_arch',
    'FiLM'           : '2d1_film',
    'NLMix'          : '2d1_nlmix',
    'FiLM+NLMix'     : '2d1_film_nlmix',
}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})
COLORS = {
    'Additive (2D1)' : '#2ca02c',
    'FiLM'           : '#d62728',
    'NLMix'          : '#ff7f0e',
    'FiLM+NLMix'     : '#9467bd',
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_additive_fold(target_key: str, fold: int):
    """Load 2D1-arch (Additive) from HPG2Stage_LOMAO — predictions already in real scale."""
    target_full = TARGETS[target_key]
    fname = f"ea_ip__{target_full}__copoly_stage2d_2d1_arch__a_held_out__split{fold}.npz"
    path  = PRED_LOMAO / fname

    if not path.exists():
        return None

    data = np.load(path, allow_pickle=True)
    yt   = data['y_true'].flatten()
    yp   = data['y_pred'].flatten()

    # test_ids stores strings like 'idx_0', 'idx_1' — the integer suffix is the df row index
    if 'test_ids' in data:
        ids = data['test_ids']
        try:
            indices = np.array([int(str(tid).split('_')[-1]) for tid in ids], dtype=int)
        except (ValueError, IndexError):
            indices = np.full(len(yt), -1, dtype=int)
    else:
        indices = np.full(len(yt), -1, dtype=int)

    return {'y_true': yt, 'y_pred': yp, 'indices': indices,
            'fold': fold, 'pred_file': str(path)}


def _load_ablation_fold(variant_suffix: str, target_key: str, fold: int):
    """Load ablation variant from hpg2stage_ablation/ — already real scale."""
    target_full = TARGETS[target_key]
    fname = f"ea_ip__{target_full}__stage2d_{variant_suffix}__lomao__fold{fold}.npz"
    path  = PRED_ABLATION / fname

    if not path.exists():
        return None

    data    = np.load(path, allow_pickle=True)
    yt      = data['y_true'].flatten()
    yp      = data['y_pred'].flatten()
    indices = data['test_indices'] if 'test_indices' in data else np.full(len(yt), -1, dtype=int)

    return {'y_true': yt, 'y_pred': yp, 'indices': indices,
            'fold': fold, 'pred_file': str(path)}


def load_all_preds(df):
    """Return dict: variant_name → target_key → list of fold dicts."""
    preds = {v: {'EA': [], 'IP': []} for v in VARIANTS}

    for target_key in ('EA', 'IP'):
        for fold in range(N_FOLDS):
            # Additive baseline
            r = _load_additive_fold(target_key, fold)
            if r is not None:
                preds['Additive (2D1)'][target_key].append(r)

            # Ablation variants
            for vname, vsuffix in VARIANTS.items():
                if vname == 'Additive (2D1)':
                    continue
                r = _load_ablation_fold(vsuffix, target_key, fold)
                if r is not None:
                    preds[vname][target_key].append(r)

    return preds


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def fold_metrics(p, df):
    yt, yp = p['y_true'], p['y_pred']
    r2   = r2_score(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    ar2, amae = g2d.compute_archdev_metrics(yt, yp, p['indices'], df)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'ArchR2': ar2, 'ArchMAE': amae}


def aggregate(metric_list, key):
    vals = [m[key] for m in metric_list if not np.isnan(m.get(key, np.nan))]
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Inventory
# ═══════════════════════════════════════════════════════════════════════════════

def task1_inventory(preds):
    print("Task 1: Experiment inventory")
    lines = [
        "# Fusion Experiment Inventory\n",
        "| Variant | Suffix | Split | Folds EA | Folds IP | Checkpoint dir | Pred dir |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    pred_inv_lines = [
        "# Fusion Prediction Inventory\n\n",
        "| Variant | Target | Fold | Prediction file | Checkpoint dir |\n",
        "|---|---|---|---|---|\n",
    ]

    for vname, vsuffix in VARIANTS.items():
        n_ea = len(preds[vname]['EA'])
        n_ip = len(preds[vname]['IP'])

        if vname == 'Additive (2D1)':
            ckpt_dir = 'checkpoints/HPG2Stage_LOMAO/'
            pred_dir = 'predictions/HPG2Stage_LOMAO/'
        else:
            ckpt_dir = f'checkpoints/HPG2Stage_Ablation/'
            pred_dir = 'predictions/hpg2stage_ablation/'

        lines.append(f"| {vname} | {vsuffix} | lomao | {n_ea}/9 | {n_ip}/9 "
                     f"| {ckpt_dir} | {pred_dir} |\n")

        for target_key in ('EA', 'IP'):
            for p in preds[vname][target_key]:
                fold = p['fold']
                ckpt_name = (
                    f"ea_ip__{TARGETS[target_key]}__stage2d_{vsuffix}__lomao__fold{fold}"
                    if vname != 'Additive (2D1)' else
                    f"ea_ip__{TARGETS[target_key]}__copoly_stage2d_2d1_arch__a_held_out__split{fold}"
                )
                pred_inv_lines.append(
                    f"| {vname} | {target_key} | {fold} | {Path(p['pred_file']).name} "
                    f"| {ckpt_dir}{ckpt_name} |\n"
                )

    with open(OUT_DIR / 'fusion_experiment_inventory.md', 'w') as f:
        f.writelines(lines)
    print(f"  Saved: fusion_experiment_inventory.md")

    with open(OUT_DIR / 'fusion_prediction_inventory.md', 'w') as f:
        f.writelines(pred_inv_lines)
    print(f"  Saved: fusion_prediction_inventory.md")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Comparison table + fold metrics
# ═══════════════════════════════════════════════════════════════════════════════

def task2_summary(preds, df):
    print("Task 2: Comparison table")
    summary_rows = []
    fold_rows    = []

    for vname in VARIANTS:
        fold_metrics_ea = []
        fold_metrics_ip = []

        for p in preds[vname]['EA']:
            m = fold_metrics(p, df)
            fold_metrics_ea.append(m)
            fold_rows.append({
                'Model': vname, 'Fold': p['fold'], 'Target': 'EA',
                'R2': m['R2'], 'RMSE': m['RMSE'], 'MAE': m['MAE'],
                'ArchDev_R2': m['ArchR2'], 'ArchDev_MAE': m['ArchMAE'],
            })

        for p in preds[vname]['IP']:
            m = fold_metrics(p, df)
            fold_metrics_ip.append(m)
            fold_rows.append({
                'Model': vname, 'Fold': p['fold'], 'Target': 'IP',
                'R2': m['R2'], 'RMSE': m['RMSE'], 'MAE': m['MAE'],
                'ArchDev_R2': m['ArchR2'], 'ArchDev_MAE': m['ArchMAE'],
            })

        def fmt(mean, std, median):
            if np.isnan(mean):
                return 'N/A'
            return f"{mean:.4f}±{std:.4f} (med={median:.4f})"

        def fmt_median(median):
            return 'N/A' if np.isnan(median) else f"{median:.4f}"

        row = {'Model': vname}
        for key, mlist in [('EA', fold_metrics_ea), ('IP', fold_metrics_ip)]:
            for metric in ('R2', 'RMSE', 'MAE', 'ArchR2', 'ArchMAE'):
                col = f"{key}_ArchDev_R2" if metric == 'ArchR2' else \
                      f"{key}_ArchDev_MAE" if metric == 'ArchMAE' else \
                      f"{key}_{metric}"
                m, s, med = aggregate(mlist, metric)
                row[col]              = m    # numeric mean
                row[col + '_std']     = s
                row[col + '_median']  = med
                row[col + '_fmt']     = fmt(m, s, med)
                row[col + '_medfmt']  = fmt_median(med)
        summary_rows.append(row)

    df_fold = pd.DataFrame(fold_rows)
    df_fold.to_csv(OUT_DIR / 'fusion_fold_metrics.csv', index=False)
    print("  Saved: fusion_fold_metrics.csv")

    # Build readable summary CSV (means + medians)
    cols_csv = ['Model',
                'EA_R2','EA_R2_median','EA_RMSE','EA_RMSE_median','EA_MAE','EA_MAE_median',
                'EA_ArchDev_R2','EA_ArchDev_R2_median','EA_ArchDev_MAE','EA_ArchDev_MAE_median',
                'IP_R2','IP_R2_median','IP_RMSE','IP_RMSE_median','IP_MAE','IP_MAE_median',
                'IP_ArchDev_R2','IP_ArchDev_R2_median','IP_ArchDev_MAE','IP_ArchDev_MAE_median']
    df_sum = pd.DataFrame([{c: r.get(c, np.nan) for c in cols_csv} for r in summary_rows])
    df_sum.to_csv(OUT_DIR / 'fusion_ablation_summary.csv', index=False)
    print("  Saved: fusion_ablation_summary.csv")

    # Markdown table — show mean±std and median separately
    header_labels = ['Model',
                     'EA R² (mean±std)','EA R² median','EA MAE (mean±std)',
                     'EA ΔR² (mean±std)','EA ΔR² median',
                     'IP R² (mean±std)','IP R² median','IP MAE (mean±std)',
                     'IP ΔR² (mean±std)','IP ΔR² median']
    cols_md = ['Model',
               'EA_R2_fmt','EA_R2_medfmt','EA_MAE_fmt',
               'EA_ArchDev_R2_fmt','EA_ArchDev_R2_medfmt',
               'IP_R2_fmt','IP_R2_medfmt','IP_MAE_fmt',
               'IP_ArchDev_R2_fmt','IP_ArchDev_R2_medfmt']
    md_lines = [
        "# Fusion Ablation Summary (LOMAO, 9 folds)\n\n",
        "> Note: fold 6 (held-out monomer OB(O)c1ccc(B(O)O)c2nsnc12) causes catastrophic "
        "failure for all models (EA R²≈−12 to −17). Median is more robust than mean here.\n\n",
        "| " + " | ".join(header_labels) + " |\n",
        "|" + "|".join(["---"] * len(header_labels)) + "|\n",
    ]
    for r in summary_rows:
        md_lines.append("| " + " | ".join(str(r.get(c, 'N/A')) for c in cols_md) + " |\n")

    with open(OUT_DIR / 'fusion_ablation_summary.md', 'w') as f:
        f.writelines(md_lines)
    print("  Saved: fusion_ablation_summary.md")

    return summary_rows, df_fold


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Rankings
# ═══════════════════════════════════════════════════════════════════════════════

def task3_ranking(summary_rows):
    print("Task 3: Rankings")
    rankings = {
        'Overall EA R²'    : ('EA_R2',          True),
        'Overall IP R²'    : ('IP_R2',          True),
        'EA ArchDev R²'    : ('EA_ArchDev_R2',  True),
        'IP ArchDev R²'    : ('IP_ArchDev_R2',  True),
        'EA MAE'           : ('EA_MAE',         False),
        'IP MAE'           : ('IP_MAE',         False),
        'EA RMSE'          : ('EA_RMSE',        False),
        'IP RMSE'          : ('IP_RMSE',        False),
    }
    lines = ["# Fusion Model Rankings\n\n"]
    for title, (col, higher_better) in rankings.items():
        rows_sorted = sorted(
            [(r['Model'], r.get(col, np.nan)) for r in summary_rows],
            key=lambda x: (-x[1] if higher_better else x[1]) if not np.isnan(x[1]) else float('inf')
        )
        lines.append(f"## {title}\n\n")
        for rank, (model, val) in enumerate(rows_sorted, 1):
            marker = " ← **best**" if rank == 1 else ""
            val_str = f"{val:.4f}" if not np.isnan(val) else "N/A"
            lines.append(f"{rank}. {model}: {val_str}{marker}\n")
        lines.append("\n")

    with open(OUT_DIR / 'fusion_model_ranking.md', 'w') as f:
        f.writelines(lines)
    print("  Saved: fusion_model_ranking.md")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Figures
# ═══════════════════════════════════════════════════════════════════════════════

def _bar_panel(ax, variant_names, means, stds, metric_label, colors):
    x = np.arange(len(variant_names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.55,
                  color=[colors.get(v, '#333333') for v in variant_names],
                  edgecolor='white', alpha=0.88, error_kw={'linewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(metric_label)
    # Annotate values
    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (s or 0) + 0.002,
                    f"{m:.3f}", ha='center', va='bottom', fontsize=7)


def task4_figures(summary_rows):
    print("Task 4: Figures")
    vnames = [r['Model'] for r in summary_rows]

    def _get_median(col):
        medians = np.array([r.get(col + '_median', np.nan) for r in summary_rows])
        # IQR-based spread: compute per-model from fold_metrics is unavailable here;
        # use std/2 as an approximate spread bar (visually honest for median plots)
        stds = np.array([r.get(col + '_std', 0.0) for r in summary_rows])
        return medians, stds

    # Figure 1 — Overall R² (median, more robust than mean due to fold-6 outlier)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, tkey in zip(axes, ['EA', 'IP']):
        medians, stds = _get_median(f'{tkey}_R2')
        _bar_panel(ax, vnames, medians, stds, 'R² (median across folds)', COLORS)
        ax.set_title(f'{tkey} Overall R²')
    fig.suptitle('Overall Prediction Performance — median R² (LOMAO, 9 folds)\n'
                 'Note: fold 6 is a hard OOD outlier; median is more informative than mean',
                 fontweight='bold', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fusion_overall_r2.png')
    fig.savefig(OUT_DIR / 'fusion_overall_r2.pdf')
    plt.close(fig)
    print("  Saved: fusion_overall_r2.{png,pdf}")

    # Figure 2 — ArchDev R² (median)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, tkey in zip(axes, ['EA', 'IP']):
        medians, stds = _get_median(f'{tkey}_ArchDev_R2')
        _bar_panel(ax, vnames, medians, stds, 'Architecture-Deviation R² (median)', COLORS)
        ax.set_title(f'{tkey} Architecture Recovery R²')
    fig.suptitle('Architecture-Deviation Performance — median R² (LOMAO, 9 folds)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fusion_archdev_r2.png')
    fig.savefig(OUT_DIR / 'fusion_archdev_r2.pdf')
    plt.close(fig)
    print("  Saved: fusion_archdev_r2.{png,pdf}")

    # Figure 3 — Delta vs Additive
    base = {r['Model']: r for r in summary_rows}.get('Additive (2D1)', {})
    metrics_delta = [
        ('EA_R2',         'EA Overall R²'),
        ('IP_R2',         'IP Overall R²'),
        ('EA_ArchDev_R2', 'EA ArchDev R²'),
        ('IP_ArchDev_R2', 'IP ArchDev R²'),
    ]
    fig, axes = plt.subplots(1, len(metrics_delta), figsize=(14, 4))
    for ax, (col, label) in zip(axes, metrics_delta):
        base_val = base.get(col + '_median', np.nan)
        deltas = []
        labels = []
        for r in summary_rows:
            if r['Model'] == 'Additive (2D1)':
                continue
            delta = r.get(col + '_median', np.nan) - base_val
            deltas.append(delta)
            labels.append(r['Model'])
        x = np.arange(len(labels))
        bar_colors = [COLORS.get(l, '#333333') for l in labels]
        bars = ax.bar(x, deltas, color=bar_colors, edgecolor='white', alpha=0.88, width=0.55)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(f'Δ {label}')
        ax.set_title(label)
        for bar, d in zip(bars, deltas):
            if not np.isnan(d):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.001 if d >= 0 else -0.003),
                        f"{d:+.4f}", ha='center', va='bottom' if d >= 0 else 'top', fontsize=7)
    fig.suptitle('Improvement over Additive 2D1 (Δ median R² = Variant − Additive)', fontweight='bold', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fusion_delta_vs_additive.png')
    plt.close(fig)
    print("  Saved: fusion_delta_vs_additive.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7 — ChatGPT Summary
# ═══════════════════════════════════════════════════════════════════════════════

def task7_chatgpt(summary_rows):
    print("Task 7: ChatGPT summary")

    def best(col, higher_better=True):
        valid = [(r['Model'], r.get(col, np.nan)) for r in summary_rows if not np.isnan(r.get(col, np.nan))]
        if not valid:
            return 'N/A', np.nan
        return sorted(valid, key=lambda x: -x[1] if higher_better else x[1])[0]

    # Use median as primary ranking metric (fold 6 is a hard OOD outlier)
    best_ea_r2   = best('EA_R2_median',        True)[0]
    best_ip_model= best('IP_R2_median',        True)[0]
    best_ea_arch = best('EA_ArchDev_R2_median',True)[0]
    best_ip_arch = best('IP_ArchDev_R2_median',True)[0]

    base     = {r['Model']: r for r in summary_rows}.get('Additive (2D1)', {})
    non_base = [r for r in summary_rows if r['Model'] != 'Additive (2D1)']
    max_delta_ea = max((r.get('EA_R2_median', np.nan) - base.get('EA_R2_median', np.nan)
                        for r in non_base if not np.isnan(r.get('EA_R2_median', np.nan))),
                       default=np.nan)
    max_delta_ip = max((r.get('IP_R2_median', np.nan) - base.get('IP_R2_median', np.nan)
                        for r in non_base if not np.isnan(r.get('IP_R2_median', np.nan))),
                       default=np.nan)

    lines = [
        "# Fusion Ablation — Results for ChatGPT\n\n",
        "## Context\n\n",
        "Stage 2D1 architecture ablation. All experiments use LOMAO split (9 folds, leave-one-monomer-out).\n",
        "Baseline: Additive 2D1 (current model). Variants: FiLM, NLMix, FiLM+NLMix.\n\n",
        "**Important:** Fold 6 (held-out monomer OB(O)c1ccc(B(O)O)c2nsnc12) causes catastrophic\n",
        "failure for ALL models (EA R²≈−12 to −17). This single fold dominates the mean.\n",
        "Median R² is reported as the primary metric; mean is provided for completeness.\n\n",
        "## Overall Metrics — Median R² across 9 folds (primary)\n\n",
        "| Model | EA R² median | EA R² mean±std | IP R² median | IP R² mean±std |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in summary_rows:
        def fmtm(col):
            v = r.get(col + '_median', np.nan)
            return 'N/A' if np.isnan(v) else f"{v:.4f}"
        def fmtms(col):
            m, s = r.get(col, np.nan), r.get(col+'_std', np.nan)
            return 'N/A' if np.isnan(m) else f"{m:.4f}±{s:.4f}"
        lines.append(f"| {r['Model']} | {fmtm('EA_R2')} | {fmtms('EA_R2')} "
                     f"| {fmtm('IP_R2')} | {fmtms('IP_R2')} |\n")

    lines += [
        "\n## Architecture-Deviation Metrics (median R² across 9 folds)\n\n",
        "| Model | EA ΔR² median | EA ΔR² mean±std | IP ΔR² median | IP ΔR² mean±std |\n",
        "|---|---|---|---|---|\n",
    ]
    for r in summary_rows:
        def fmtm(col):
            v = r.get(col + '_median', np.nan)
            return 'N/A' if np.isnan(v) else f"{v:.4f}"
        def fmtms(col):
            m, s = r.get(col, np.nan), r.get(col+'_std', np.nan)
            return 'N/A' if np.isnan(m) else f"{m:.4f}±{s:.4f}"
        lines.append(f"| {r['Model']} | {fmtm('EA_ArchDev_R2')} | {fmtms('EA_ArchDev_R2')} "
                     f"| {fmtm('IP_ArchDev_R2')} | {fmtms('IP_ArchDev_R2')} |\n")

    lines += [
        "\n## Best-Performing Models\n\n",
        f"- Best overall EA model: **{best_ea_r2}**\n",
        f"- Best overall IP model: **{best_ip_model}**\n",
        f"- Best EA architecture recovery model: **{best_ea_arch}**\n",
        f"- Best IP architecture recovery model: **{best_ip_arch}**\n\n",
        "## Magnitude of Improvement over Additive 2D1\n\n",
        f"- Max ΔEA R² (best variant vs Additive): {max_delta_ea:+.4f}\n" if not np.isnan(max_delta_ea) else "- Max ΔEA R²: N/A\n",
        f"- Max ΔIP R² (best variant vs Additive): {max_delta_ip:+.4f}\n" if not np.isnan(max_delta_ip) else "- Max ΔIP R²: N/A\n",
    ]

    with open(OUT_DIR / 'fusion_results_for_chatgpt.md', 'w') as f:
        f.writelines(lines)
    print("  Saved: fusion_results_for_chatgpt.md")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Stage 2D1 Fusion Ablation Analysis")
    print("=" * 70)

    df = g2d.load_dataset()
    g2d._ensure_value_map(df)

    preds = load_all_preds(df)

    # Report discovery
    for vname in VARIANTS:
        n_ea = len(preds[vname]['EA'])
        n_ip = len(preds[vname]['IP'])
        print(f"  {vname:20s}: EA={n_ea}/9 folds, IP={n_ip}/9 folds")

    task1_inventory(preds)
    summary_rows, df_fold = task2_summary(preds, df)
    task3_ranking(summary_rows)
    task4_figures(summary_rows)
    task7_chatgpt(summary_rows)

    print("=" * 70)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
