#!/usr/bin/env python3
"""
Stage 2D Paper — Tables 1-3 + Figures
======================================
Generates paper outputs from canonical prediction files. No retraining.

Outputs (all saved to experiments/hpg2stage/output/paper/):

  Tables:
    table1_overall_performance.csv / .md
    table2_architecture_performance.csv / .md
    table3_generalization_comparison.csv / .md

  Figures:
    fig_overall_vs_arch_recovery_mean.pdf/.png      (mean R² across LOMO folds)
    fig_overall_vs_arch_recovery_median.pdf/.png    (median R² across LOMO folds)
    fig_overall_vs_arch_recovery_per_fold.pdf/.png  (per-fold diagnostic)
    fig_generalization_performance.pdf/.png

Predictions are loaded via evaluation/naming.py — already in physical units (eV).
Global df indices come from metadata/splits/*.json.

Usage:
    python experiments/hpg2stage/scripts/generate_stage2d_paper_outputs.py [--skip-figures]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

DATA_PATH = PROJECT_ROOT / 'data' / 'ea_ip.csv'
META_DIR  = PROJECT_ROOT / 'metadata' / 'splits'
PRED_ROOT = PROJECT_ROOT / 'predictions'
OUT_DIR   = PROJECT_ROOT / 'experiments' / 'hpg2stage' / 'output' / 'paper'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from evaluation.naming import make_prediction_filename
from evaluation.metrics import compute_archdev_r2

# ── Constants ────────────────────────────────────────────────────────
TARGETS = {'EA': 'EA vs SHE (eV)', 'IP': 'IP vs SHE (eV)'}
N_LOMO_FOLDS = 9
N_GEN_FOLDS  = 5

MODEL_DISPLAY = {
    'frac':       'Frac',
    'wdmpnn':     'wDMPNN',
    'globalarch': 'GlobalArch',
    'chemarch':   'ChemArch',
}
MODEL_ORDER = ['frac', 'wdmpnn', 'globalarch', 'chemarch']

SPLIT_SUBDIRS = {
    'monomer_heldout': 'ea_ip_lomo',
    'group_disjoint':  'ea_ip_group',
    'pair_disjoint':   'ea_ip_pair',
}
N_FOLDS_FOR_SPLIT = {
    'monomer_heldout': N_LOMO_FOLDS,
    'group_disjoint':  N_GEN_FOLDS,
    'pair_disjoint':   N_GEN_FOLDS,
}

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})
COLORS = {
    'wDMPNN':     '#7f7f7f',
    'Frac':       '#1f77b4',
    'GlobalArch': '#ff7f0e',
    'ChemArch':   '#2ca02c',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """Load ea_ip.csv with pair canonicalization (adds group_key, smilesA, smilesB)."""
    df = pd.read_csv(DATA_PATH)

    def _canon(a, b, wa, wb):
        a = "" if pd.isna(a) else str(a)
        b = "" if pd.isna(b) else str(b)
        if b < a:
            return b, a, wb, wa
        return a, b, wa, wb

    raw_A  = df['smiles_A'].astype(str).tolist()
    raw_B  = df['smiles_B'].astype(str).tolist()
    raw_fA = df['fracA'].values.astype(float)
    raw_fB = df['fracB'].values.astype(float)

    canA, canB, fA_list, fB_list = [], [], [], []
    for a, b, wa, wb in zip(raw_A, raw_B, raw_fA, raw_fB):
        a2, b2, wa2, wb2 = _canon(a, b, wa, wb)
        canA.append(a2); canB.append(b2)
        fA_list.append(wa2); fB_list.append(wb2)

    fA = np.array(fA_list, dtype=float)
    fB = np.array(fB_list, dtype=float)
    s  = fA + fB
    fA = fA / s
    fB = 1.0 - fA

    df['smilesA'] = canA;  df['smilesB'] = canB
    df['fracA']   = fA;    df['fracB']   = fB

    _r6 = lambda x: round(float(x), 6)
    df['group_key'] = [
        f"{a}||{b}||{_r6(fa)}||{_r6(fb)}"
        for a, b, fa, fb in zip(canA, canB, fA, fB)
    ]
    return df


def load_split_meta(split: str) -> list:
    path = META_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/generate_split_metadata.py"
        )
    import json
    return json.loads(path.read_text())["folds"]


def _get_global_idx(fold: int, meta_folds: list) -> np.ndarray:
    for rec in meta_folds:
        if rec["fold"] == fold:
            return np.asarray(rec["global_test_indices"], dtype=int)
    raise KeyError(f"Fold {fold} not in metadata")


# =============================================================================
# PREDICTION LOADER
# =============================================================================

def load_predictions(model: str, target_key: str, split: str,
                     df: pd.DataFrame, meta_folds: list):
    """
    Load all folds for one model/target/split from canonical prediction files.
    Returns list of dicts {y_true, y_pred, global_idx, fold}, or None if all missing.
    Predictions are already in physical units (eV) — no denormalization needed.
    """
    target_full = TARGETS[target_key]
    subdir      = PRED_ROOT / SPLIT_SUBDIRS[split]
    n_folds     = N_FOLDS_FOR_SPLIT[split]

    results, missing = [], []
    for fold in range(n_folds):
        fname = make_prediction_filename(target_full, model, split, fold)
        fpath = subdir / fname
        if not fpath.exists():
            missing.append(fold)
            continue
        p          = np.load(fpath, allow_pickle=True)
        y_true     = p['y_true'].flatten().astype(np.float64)
        y_pred     = p['y_pred'].flatten().astype(np.float64)
        global_idx = _get_global_idx(fold, meta_folds)
        results.append({'y_true': y_true, 'y_pred': y_pred,
                        'global_idx': global_idx, 'fold': fold})

    if not results:
        return None
    if missing:
        print(f"  [WARN] {model}/{target_key}/{split}: missing folds {missing}")
    return results


# =============================================================================
# METRICS AGGREGATION
# =============================================================================

def compute_fold_metrics(preds: list, df: pd.DataFrame) -> dict:
    """
    Return per-fold arrays for R2, MAE, ArchR2, ArchMAE.
    Keys: 'folds', 'r2', 'mae', 'arch_r2', 'arch_mae'  (parallel lists, NaN where unavailable).
    """
    folds, r2s, maes, ar2s, amaes = [], [], [], [], []
    for p in preds:
        yt, yp, idx = p['y_true'], p['y_pred'], p['global_idx']
        folds.append(p['fold'])
        r2s.append(r2_score(yt, yp))
        maes.append(mean_absolute_error(yt, yp))
        ad = compute_archdev_r2(df, yt, yp, idx)
        ar2s.append(ad['r2'])
        amaes.append(ad['mae'])
    return {
        'folds':    folds,
        'r2':       np.array(r2s),
        'mae':      np.array(maes),
        'arch_r2':  np.array(ar2s),
        'arch_mae': np.array(amaes),
    }


def compute_metrics(preds: list, df: pd.DataFrame) -> dict:
    """Aggregate overall and arch-deviation metrics across folds (mean ± SEM)."""
    fm = compute_fold_metrics(preds, df)
    r2s   = fm['r2'];       maes  = fm['mae']
    ar2s  = fm['arch_r2'];  amaes = fm['arch_mae']
    valid_ar  = ar2s[~np.isnan(ar2s)]
    valid_am  = amaes[~np.isnan(amaes)]
    n   = max(len(r2s), 1)
    na  = max(len(valid_ar), 1)
    return {
        'R2_mean':      float(np.mean(r2s)),
        'R2_median':    float(np.median(r2s)),
        'R2_std':       float(np.std(r2s) / n**0.5),
        'MAE_mean':     float(np.mean(maes)),
        'MAE_std':      float(np.std(maes) / n**0.5),
        'ArchR2_mean':  float(np.nanmean(ar2s))  if len(valid_ar) else np.nan,
        'ArchR2_median':float(np.nanmedian(ar2s)) if len(valid_ar) else np.nan,
        'ArchR2_std':   float(np.nanstd(ar2s) / na**0.5) if len(valid_ar) else np.nan,
        'ArchMAE_mean': float(np.nanmean(amaes)) if len(valid_am) else np.nan,
        'ArchMAE_std':  float(np.nanstd(amaes) / na**0.5) if len(valid_am) else np.nan,
        '_fold_r2':     r2s,
        '_fold_arch_r2':ar2s,
        '_folds':       fm['folds'],
    }


# =============================================================================
# HELPER
# =============================================================================

def save_fig(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f'{name}.pdf')
    fig.savefig(OUT_DIR / f'{name}.png')
    plt.close(fig)
    print(f"  Saved: {name}.pdf/.png")


# =============================================================================
# TABLES 1, 2, 3
# =============================================================================

def generate_tables(df: pd.DataFrame, lomo_meta: list,
                    group_meta: list, pair_meta: list) -> None:
    print("\n" + "=" * 60)
    print("Tables 1-3")
    print("=" * 60)

    meta = {
        'monomer_heldout': lomo_meta,
        'group_disjoint':  group_meta,
        'pair_disjoint':   pair_meta,
    }

    # Table 1: Overall performance (LOMO)
    rows = []
    for model in MODEL_ORDER:
        row = {'Model': MODEL_DISPLAY[model]}
        for tkey in ['EA', 'IP']:
            preds = load_predictions(model, tkey, 'monomer_heldout', df, lomo_meta)
            if preds:
                m = compute_metrics(preds, df)
                row[f'{tkey} R2']  = f"{m['R2_mean']:.4f} +/- {m['R2_std']:.4f}"
                row[f'{tkey} MAE'] = f"{m['MAE_mean']:.4f} +/- {m['MAE_std']:.4f}"
            else:
                row[f'{tkey} R2'] = row[f'{tkey} MAE'] = 'NA'
        rows.append(row)
    df_t1 = pd.DataFrame(rows)
    df_t1.to_csv(OUT_DIR / 'table1_overall_performance.csv', index=False)
    md = ["# Table 1: Overall EA/IP Performance (Leave-One-Monomer-Out)\n",
          df_t1.to_markdown(index=False), ""]
    (OUT_DIR / 'table1_overall_performance.md').write_text('\n'.join(md))
    print("  Written: table1_overall_performance.csv/.md")

    # Table 2: Architecture-deviation performance (LOMO)
    rows = []
    for model in MODEL_ORDER:
        row = {'Model': MODEL_DISPLAY[model]}
        for tkey in ['EA', 'IP']:
            preds = load_predictions(model, tkey, 'monomer_heldout', df, lomo_meta)
            if preds:
                m = compute_metrics(preds, df)
                if not np.isnan(m['ArchR2_mean']):
                    row[f'{tkey} R2(d)']  = f"{m['ArchR2_mean']:.4f} +/- {m['ArchR2_std']:.4f}"
                    row[f'{tkey} MAE(d)'] = f"{m['ArchMAE_mean']:.4f} +/- {m['ArchMAE_std']:.4f}"
                else:
                    row[f'{tkey} R2(d)'] = row[f'{tkey} MAE(d)'] = 'NA'
            else:
                row[f'{tkey} R2(d)'] = row[f'{tkey} MAE(d)'] = 'NA'
        rows.append(row)
    df_t2 = pd.DataFrame(rows)
    df_t2.to_csv(OUT_DIR / 'table2_architecture_performance.csv', index=False)
    md = ["# Table 2: Architecture-Deviation Performance (Leave-One-Monomer-Out)\n",
          df_t2.to_markdown(index=False), ""]
    (OUT_DIR / 'table2_architecture_performance.md').write_text('\n'.join(md))
    print("  Written: table2_architecture_performance.csv/.md")

    # Table 3: Generalization comparison
    split_configs = [
        ('group_disjoint',  'Group-disjoint'),
        ('pair_disjoint',   'Pair-disjoint'),
        ('monomer_heldout', 'LOMO'),
    ]
    rows = []
    for split_key, split_label in split_configs:
        for model in MODEL_ORDER:
            row = {'Model': MODEL_DISPLAY[model], 'Split': split_label}
            for tkey in ['EA', 'IP']:
                preds = load_predictions(model, tkey, split_key, df, meta[split_key])
                if preds:
                    m = compute_metrics(preds, df)
                    row[f'{tkey} R2']    = f"{m['R2_mean']:.4f}"
                    row[f'{tkey} R2(d)'] = f"{m['ArchR2_mean']:.4f}" if not np.isnan(m['ArchR2_mean']) else 'NA'
                else:
                    row[f'{tkey} R2'] = row[f'{tkey} R2(d)'] = 'NA'
            rows.append(row)
    df_t3 = pd.DataFrame(rows)
    df_t3.to_csv(OUT_DIR / 'table3_generalization_comparison.csv', index=False)
    md = ["# Table 3: Generalization Comparison\n",
          df_t3.to_markdown(index=False), ""]
    (OUT_DIR / 'table3_generalization_comparison.md').write_text('\n'.join(md))
    print("  Written: table3_generalization_comparison.csv/.md")


# =============================================================================
# FIGURE: Overall vs Architecture Recovery (shared helpers)
# =============================================================================

def _collect_lomo_stats(df: pd.DataFrame, lomo_meta: list, stat: str
                        ) -> tuple[dict, dict]:
    """
    Collect per-model summary stats for LOMO predictions.
    stat: 'mean' or 'median'.
    Returns (overall_vals, arch_vals) each keyed by tkey -> list[float per model].
    Also returns (overall_errs, arch_errs) for mean mode (SEM); zeros for median.
    """
    overall_vals, arch_vals   = {}, {}
    overall_errs, arch_errs   = {}, {}
    fold_data: dict = {}  # (model, tkey) -> metrics dict

    for tkey in ['EA', 'IP']:
        ov, ae, oe, are = [], [], [], []
        for model in MODEL_ORDER:
            preds = load_predictions(model, tkey, 'monomer_heldout', df, lomo_meta)
            if preds:
                m = compute_metrics(preds, df)
                fold_data[(model, tkey)] = m
                if stat == 'mean':
                    ov.append(m['R2_mean']);       oe.append(m['R2_std'])
                    ae.append(m['ArchR2_mean']);   are.append(m['ArchR2_std'])
                else:
                    ov.append(m['R2_median']);     oe.append(0.0)
                    ae.append(m['ArchR2_median']); are.append(0.0)
            else:
                ov.append(np.nan); oe.append(0.0)
                ae.append(np.nan); are.append(0.0)
        overall_vals[tkey] = ov;  overall_errs[tkey] = oe
        arch_vals[tkey]    = ae;  arch_errs[tkey]    = are

    return overall_vals, overall_errs, arch_vals, arch_errs, fold_data


def _draw_bar_panel(ax, vals, errs, colors, x, ylo, yhi, label_fmt='.3f'):
    """Draw one bar panel; hatch missing bars, label present ones."""
    c_errs = [min(e, yhi - v) if not np.isnan(v) else 0.0
              for v, e in zip(vals, errs)]
    bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in vals],
                  yerr=c_errs, capsize=4, color=colors,
                  alpha=0.85, edgecolor='white', error_kw={'elinewidth': 1.2})
    ax.set_ylim(ylo, yhi)
    for bar, mv, sv in zip(bars, vals, c_errs):
        if np.isnan(mv):
            bar.set_hatch('///')
            bar.set_facecolor('#dddddd')
            bar.set_edgecolor('#aaaaaa')
        else:
            label_y = min(mv + sv + (yhi - ylo) * 0.02, yhi - (yhi - ylo) * 0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    format(mv, label_fmt),
                    ha='center', va='bottom', fontsize=6, clip_on=True)
    return bars


def _figure_overall_vs_arch(df: pd.DataFrame, lomo_meta: list,
                            stat: str, fname: str) -> None:
    """
    2×2 bar figure: top = overall R², bottom = ΔR².
    stat: 'mean' or 'median'.
    """
    label = 'Mean' if stat == 'mean' else 'Median'
    print(f"\n  Overall vs arch-recovery ({label} per-fold R²)")

    overall_vals, overall_errs, arch_vals, arch_errs, _ = \
        _collect_lomo_stats(df, lomo_meta, stat)

    display_names = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    colors = [COLORS[MODEL_DISPLAY[m]] for m in MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), constrained_layout=False)
    fig.suptitle(
        f'LOMO: Overall Prediction vs Architecture-Recovery ({label} per-fold R\u00b2)',
        fontweight='bold'
    )

    for i, tkey in enumerate(['EA', 'IP']):
        ov = overall_vals[tkey];  oe = overall_errs[tkey]
        av = arch_vals[tkey];     ae = arch_errs[tkey]

        # Top row: overall R²
        ax = axes[0, i]
        valid = [v for v in ov if not np.isnan(v)]
        ylo = round(max(-1.0, (min(valid) - 0.15) if valid else 0.0) * 10) / 10
        yhi = min(1.0, round(((max(valid) + 0.10) if valid else 1.0) * 10 + 1) / 10)
        _draw_bar_panel(ax, ov, oe, colors, x, ylo, yhi)
        ax.set_xticks(x);  ax.set_xticklabels(display_names, rotation=15, ha='right')
        ax.set_ylabel('R\u00b2(y)')
        ax.set_title(f'{"A" if i == 0 else "B"}. Overall {tkey} Prediction')
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)

        # Bottom row: arch-deviation R²
        ax = axes[1, i]
        valid_a = [v for v in av if not np.isnan(v)]
        ylo_r = round(max(-1.0, (min(valid_a) - 0.15) if valid_a else -0.15) * 10) / 10
        yhi_r = min(1.0, round(((max(valid_a) + 0.10) if valid_a else 1.0) * 10 + 1) / 10)
        _draw_bar_panel(ax, av, ae, colors, x, ylo_r, yhi_r)
        ax.set_xticks(x);  ax.set_xticklabels(display_names, rotation=15, ha='right')
        ax.set_ylabel('R\u00b2(\u0394y)')
        ax.set_title(f'{"C" if i == 0 else "D"}. Architecture Recovery: \u0394{tkey}')
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)

    fig.subplots_adjust(bottom=0.12, top=0.88, hspace=0.48, wspace=0.30)
    save_fig(fig, fname)


def figure_overall_vs_arch_mean(df: pd.DataFrame, lomo_meta: list) -> None:
    _figure_overall_vs_arch(df, lomo_meta, 'mean',
                            'fig_overall_vs_arch_recovery_mean')


def figure_overall_vs_arch_median(df: pd.DataFrame, lomo_meta: list) -> None:
    _figure_overall_vs_arch(df, lomo_meta, 'median',
                            'fig_overall_vs_arch_recovery_median')


def figure_overall_vs_arch_per_fold(df: pd.DataFrame, lomo_meta: list) -> None:
    """2×2 per-fold diagnostic: one line per model, x=fold index."""
    print("\n  Overall vs arch-recovery (per-fold diagnostic)")

    LOMO_META_DICT = {r['fold']: r for r in lomo_meta}
    FOLD6_MONOMER  = LOMO_META_DICT.get(6, {}).get('held_out_monomer_A', '')
    FOLD5_MONOMER  = LOMO_META_DICT.get(5, {}).get('held_out_monomer_A', '')

    # Gather fold-wise data: {(model, tkey): {'folds': [...], 'r2': [...], 'arch_r2': [...]}}
    fd: dict = {}
    for model in MODEL_ORDER:
        for tkey in ['EA', 'IP']:
            preds = load_predictions(model, tkey, 'monomer_heldout', df, lomo_meta)
            if preds:
                fm = compute_fold_metrics(preds, df)
                fd[(model, tkey)] = fm
            else:
                fd[(model, tkey)] = None

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=False)
    fig.suptitle('LOMO: Per-Fold R\u00b2 Diagnostic (Monomer-Heldout)',
                 fontweight='bold')

    panel_cfg = [
        (0, 0, 'EA', 'r2',      'A. Overall EA R\u00b2',          'R\u00b2(y)'),
        (0, 1, 'IP', 'r2',      'B. Overall IP R\u00b2',          'R\u00b2(y)'),
        (1, 0, 'EA', 'arch_r2', 'C. Architecture-Deviation \u0394EA R\u00b2', 'R\u00b2(\u0394y)'),
        (1, 1, 'IP', 'arch_r2', 'D. Architecture-Deviation \u0394IP R\u00b2', 'R\u00b2(\u0394y)'),
    ]

    markers = {'Frac': 'o', 'wDMPNN': 's', 'GlobalArch': '^', 'ChemArch': 'D'}
    all_folds = list(range(N_LOMO_FOLDS))

    for row, col, tkey, metric_key, title, ylabel in panel_cfg:
        ax = axes[row, col]
        ymin_all, ymax_all = [], []

        for model in MODEL_ORDER:
            mname = MODEL_DISPLAY[model]
            color = COLORS[mname]
            data  = fd[(model, tkey)]
            if data is None:
                continue
            fold_idx = data['folds']
            vals     = data[metric_key]
            ymin_all.extend([v for v in vals if not np.isnan(v)])
            ymax_all.extend([v for v in vals if not np.isnan(v)])
            ax.plot(fold_idx, vals, marker=markers[mname], color=color,
                    linewidth=1.5, markersize=5, label=mname, alpha=0.9)

        # Highlight fold 6 for EA, fold 5 for IP
        if tkey == 'EA':
            hl_fold, hl_monomer = 6, FOLD6_MONOMER or 'thiadiazole'
        else:
            hl_fold, hl_monomer = 5, FOLD5_MONOMER or f'fold {5}'
        if hl_fold in all_folds:
            ax.axvspan(hl_fold - 0.4, hl_fold + 0.4, alpha=0.12, color='red', zorder=0)
            ax.text(hl_fold, ax.get_ylim()[1] if ymax_all else 1.0,
                    f'fold {hl_fold}\n({hl_monomer})', ha='center', va='top',
                    fontsize=6, color='darkred')

        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xticks(all_folds)
        ax.set_xlabel('Fold')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if ymin_all and ymax_all:
            span = max(ymax_all) - min(ymin_all)
            pad  = max(span * 0.12, 0.05)
            ax.set_ylim(min(ymin_all) - pad, min(1.05, max(ymax_all) + pad))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.10, top=0.91, hspace=0.48, wspace=0.28)
    save_fig(fig, 'fig_overall_vs_arch_recovery_per_fold')


# =============================================================================
# FIGURE: Generalization
# =============================================================================

def figure_generalization(df: pd.DataFrame, lomo_meta: list,
                          group_meta: list, pair_meta: list) -> None:
    """Grouped bar: arch-deviation R2 by split type for all 4 models."""
    print("\n  Generalization figure")

    splits       = ['group_disjoint', 'pair_disjoint', 'monomer_heldout']
    split_labels = ['Group-disjoint', 'Pair-disjoint', 'LOMO']
    meta = {
        'monomer_heldout': lomo_meta,
        'group_disjoint':  group_meta,
        'pair_disjoint':   pair_meta,
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ylo, yhi = -0.15, 1.05
    width = 0.2

    for i, tkey in enumerate(['EA', 'IP']):
        ax = axes[i]
        x  = np.arange(len(splits))

        for j, model in enumerate(MODEL_ORDER):
            vals, pending_mask = [], []
            for split in splits:
                preds = load_predictions(model, tkey, split, df, meta[split])
                if preds:
                    m = compute_metrics(preds, df)
                    vals.append(m['ArchR2_mean'])
                    pending_mask.append(False)
                else:
                    vals.append(np.nan)
                    pending_mask.append(True)

            offset   = (j - 1.5) * width
            bar_vals = [0.5 if p else (v if not np.isnan(v) else 0)
                        for v, p in zip(vals, pending_mask)]
            color    = COLORS[MODEL_DISPLAY[model]]
            bars     = ax.bar(x + offset, bar_vals, width,
                              label=MODEL_DISPLAY[model],
                              color=color, alpha=0.85, edgecolor='white')

            for bar, val, pending in zip(bars, vals, pending_mask):
                if pending:
                    bar.set_hatch('///')
                    bar.set_facecolor('#dddddd')
                    bar.set_edgecolor('#aaaaaa')
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.25,
                            'pend.', ha='center', va='center', fontsize=4,
                            color='#666666', style='italic', clip_on=True)
                elif not np.isnan(val) and val > 0.01:
                    label_y = min(val + 0.03, yhi - 0.04)
                    ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                            f'{val:.3f}', ha='center', va='bottom',
                            fontsize=5, clip_on=True)

        ax.set_xticks(x);  ax.set_xticklabels(split_labels, fontsize=8)
        ax.set_ylabel('R\u00b2(\u0394)' if i == 0 else '')
        ax.set_title(tkey)
        ax.set_ylim(ylo, yhi)
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False)
    fig.suptitle('Generalization: Architecture-Deviation R\u00b2 by Split Type',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    save_fig(fig, 'fig_generalization_performance')


def figure_generalization_overall_r2(df: pd.DataFrame, lomo_meta: list,
                                     group_meta: list, pair_meta: list) -> None:
    """Grouped bar: overall R2 by split type for all 4 models."""
    print("\n  Generalization overall-R\u00b2 figure")

    splits       = ['group_disjoint', 'pair_disjoint', 'monomer_heldout']
    split_labels = ['Group-disjoint', 'Pair-disjoint', 'LOMO']
    meta = {
        'monomer_heldout': lomo_meta,
        'group_disjoint':  group_meta,
        'pair_disjoint':   pair_meta,
    }

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ylo, yhi = -1.5, 1.20
    width = 0.2

    for i, tkey in enumerate(['EA', 'IP']):
        ax = axes[i]
        x  = np.arange(len(splits))

        for j, model in enumerate(MODEL_ORDER):
            vals, pending_mask = [], []
            for split in splits:
                preds = load_predictions(model, tkey, split, df, meta[split])
                if preds:
                    m = compute_metrics(preds, df)
                    vals.append(m['R2_mean'])
                    pending_mask.append(False)
                else:
                    vals.append(np.nan)
                    pending_mask.append(True)

            offset   = (j - 1.5) * width
            bar_vals = [0.5 if p else (v if not np.isnan(v) else 0)
                        for v, p in zip(vals, pending_mask)]
            color    = COLORS[MODEL_DISPLAY[model]]
            bars     = ax.bar(x + offset, bar_vals, width,
                              label=MODEL_DISPLAY[model],
                              color=color, alpha=0.85, edgecolor='white')

            for bar, val, pending in zip(bars, vals, pending_mask):
                if pending:
                    bar.set_hatch('///')
                    bar.set_facecolor('#dddddd')
                    bar.set_edgecolor('#aaaaaa')
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.25,
                            'pend.', ha='center', va='center', fontsize=4,
                            color='#666666', style='italic', clip_on=True)
                elif not np.isnan(val):
                    label_y = min(val + 0.03, yhi - 0.04)
                    ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                            f'{val:.3f}', ha='center', va='bottom',
                            fontsize=5, clip_on=True)

        ax.set_xticks(x);  ax.set_xticklabels(split_labels, fontsize=8)
        ax.set_ylabel('R\u00b2(y)' if i == 0 else '')
        ax.set_title(tkey)
        ax.set_ylim(ylo, yhi)
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False)
    fig.suptitle('Generalization: Overall R\u00b2 by Split Type',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    save_fig(fig, 'fig_generalization_overall_r2')


# =============================================================================
# CSV EXPORTS
# =============================================================================

def export_lomo_cv_csvs(df: pd.DataFrame, lomo_meta: list) -> None:
    """Export per-fold and summary CSVs for LOMO cross-validation results."""
    print("\n  Exporting LOMO CV CSVs")

    fd: dict = {}
    for model in MODEL_ORDER:
        for tkey in ['EA', 'IP']:
            preds = load_predictions(model, tkey, 'monomer_heldout', df, lomo_meta)
            fd[(model, tkey)] = compute_fold_metrics(preds, df) if preds else None

    all_folds = list(range(N_LOMO_FOLDS))

    # ── Per-fold CSV ──────────────────────────────────────────────────
    per_fold_rows = []
    for model in MODEL_ORDER:
        mname   = MODEL_DISPLAY[model]
        ea_data = fd[(model, 'EA')]
        ip_data = fd[(model, 'IP')]
        ea_r2   = dict(zip(ea_data['folds'], ea_data['r2']))       if ea_data else {}
        ea_ar2  = dict(zip(ea_data['folds'], ea_data['arch_r2']))  if ea_data else {}
        ip_r2   = dict(zip(ip_data['folds'], ip_data['r2']))       if ip_data else {}
        ip_ar2  = dict(zip(ip_data['folds'], ip_data['arch_r2'])) if ip_data else {}
        for fold in all_folds:
            per_fold_rows.append({
                'Model':     mname,
                'Fold':      fold,
                'R2_EA':     round(ea_r2[fold],  4) if fold in ea_r2  else None,
                'R2_IP':     round(ip_r2[fold],  4) if fold in ip_r2  else None,
                'ArchR2_EA': round(ea_ar2[fold], 4) if fold in ea_ar2 else None,
                'ArchR2_IP': round(ip_ar2[fold], 4) if fold in ip_ar2 else None,
            })
    df_pf = pd.DataFrame(per_fold_rows)
    df_pf.to_csv(OUT_DIR / 'lomao_cv_per_fold.csv', index=False)
    print("  Written: lomao_cv_per_fold.csv")

    # ── Summary CSV ───────────────────────────────────────────────────
    summary_rows = []
    for model in MODEL_ORDER:
        mname = MODEL_DISPLAY[model]
        row   = {'Model': mname}
        for tkey in ['EA', 'IP']:
            data = fd[(model, tkey)]
            if data is not None:
                r2s  = data['r2']
                ar2s = data['arch_r2']
                row[f'R2_{tkey}_mean']     = round(float(np.mean(r2s)),        4)
                row[f'R2_{tkey}_std']      = round(float(np.std(r2s)),         4)
                row[f'ArchR2_{tkey}_mean'] = round(float(np.nanmean(ar2s)),    4)
                row[f'ArchR2_{tkey}_std']  = round(float(np.nanstd(ar2s)),     4)
            else:
                row[f'R2_{tkey}_mean'] = row[f'R2_{tkey}_std'] = None
                row[f'ArchR2_{tkey}_mean'] = row[f'ArchR2_{tkey}_std'] = None
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows,
                          columns=['Model',
                                   'R2_EA_mean', 'R2_EA_std',
                                   'R2_IP_mean', 'R2_IP_std',
                                   'ArchR2_EA_mean', 'ArchR2_EA_std',
                                   'ArchR2_IP_mean', 'ArchR2_IP_std'])
    df_sum.to_csv(OUT_DIR / 'lomao_cv_summary.csv', index=False)
    print("  Written: lomao_cv_summary.csv")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_lomo_summary(df: pd.DataFrame, lomo_meta: list) -> None:
    """Print fold-wise R² values, mean±SD, median, min/max for all models × targets."""
    print("\n" + "=" * 72)
    print("LOMO FOLD-WISE SUMMARY")
    print("=" * 72)

    LOMO_META_DICT = {r['fold']: r for r in lomo_meta}

    for tkey in ['EA', 'IP']:
        print(f"\n{'─'*72}")
        print(f"  Target: {tkey}")
        print(f"{'─'*72}")
        print(f"  {'Model':<12} {'Fold':<5} {'R²':>8} {'ΔR²':>8}  held-out monomer (abbrev.)")
        print(f"  {'-'*68}")
        for model in MODEL_ORDER:
            mname = MODEL_DISPLAY[model]
            preds = load_predictions(model, tkey, 'monomer_heldout', df, lomo_meta)
            if preds is None:
                print(f"  {mname:<12}  [NO PREDICTIONS]")
                continue
            fm = compute_fold_metrics(preds, df)
            for fold, r2, ar2 in zip(fm['folds'], fm['r2'], fm['arch_r2']):
                mon = str(LOMO_META_DICT.get(fold, {}).get('held_out_monomer_A', '?'))[:35]
                ar2_str = f"{ar2:>8.4f}" if not np.isnan(ar2) else f"{'NA':>8}"
                print(f"  {mname:<12} {fold:<5} {r2:>8.4f} {ar2_str}  {mon}")
            r2s  = fm['r2']
            ar2s = fm['arch_r2']
            ar2s_valid = ar2s[~np.isnan(ar2s)]
            print(f"  {mname:<12}  "
                  f"R²  mean={np.mean(r2s):.4f} median={np.median(r2s):.4f} "
                  f"std={np.std(r2s):.4f} min={np.min(r2s):.4f} max={np.max(r2s):.4f}")
            if len(ar2s_valid):
                print(f"  {mname:<12}  "
                      f"ΔR² mean={np.nanmean(ar2s):.4f} median={np.nanmedian(ar2s):.4f} "
                      f"std={np.nanstd(ar2s):.4f} min={np.nanmin(ar2s):.4f} max={np.nanmax(ar2s):.4f}")
            print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--skip-figures', action='store_true',
                   help='Generate tables only, skip matplotlib figures')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Stage 2D Paper Output Generation")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}\n")

    print("Loading dataset...")
    df = load_dataset()
    print(f"  {len(df)} rows, {df['group_key'].nunique()} unique groups\n")

    print("Loading split metadata...")
    lomo_meta  = load_split_meta('monomer_heldout')
    group_meta = load_split_meta('group_disjoint')
    pair_meta  = load_split_meta('pair_disjoint')
    print(f"  monomer_heldout: {len(lomo_meta)} folds")
    print(f"  group_disjoint:  {len(group_meta)} folds")
    print(f"  pair_disjoint:   {len(pair_meta)} folds\n")

    generate_tables(df, lomo_meta, group_meta, pair_meta)
    export_lomo_cv_csvs(df, lomo_meta)
    print_lomo_summary(df, lomo_meta)

    if not args.skip_figures:
        print("\n" + "=" * 60)
        print("Figures")
        print("=" * 60)
        figure_overall_vs_arch_mean(df, lomo_meta)
        figure_overall_vs_arch_median(df, lomo_meta)
        figure_overall_vs_arch_per_fold(df, lomo_meta)
        figure_generalization(df, lomo_meta, group_meta, pair_meta)
        figure_generalization_overall_r2(df, lomo_meta, group_meta, pair_meta)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"Outputs: {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()

