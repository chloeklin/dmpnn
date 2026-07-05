#!/usr/bin/env python3
"""
LOMAO Results Inspector
=======================
Plots per-fold RMSE and R² for wDMPNN / Frac / 2D0 / 2D1,
plus parity (truth) plots from NPZ predictions.

Outputs saved to: experiments/hpg2stage/output/lomao_inspect/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'python'))

RESULTS_DIR  = PROJECT_ROOT / 'results'  / 'HPG2Stage_LOMAO'
PRED_DIR     = PROJECT_ROOT / 'predictions' / 'HPG2Stage_LOMAO'
OUT_DIR      = PROJECT_ROOT / 'experiments' / 'hpg2stage' / 'output' / 'lomao_inspect'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    'EA': 'EA vs SHE (eV)',
    'IP': 'IP vs SHE (eV)',
}
N_FOLDS = 9

# ── Model catalogue ───────────────────────────────────────────────────
MODELS = {
    'wDMPNN': {
        'result_prefix': 'ea_ip__copoly_mix__a_held_out__target',
        'pred_prefix':   None,   # uses bare ea_ip__{target}__split{fold}.npz
    },
    'Frac': {
        'result_prefix': 'ea_ip__copoly_stage2d_frac__a_held_out__target',
        'pred_prefix':   'ea_ip__{target}__copoly_stage2d_frac__a_held_out__split{fold}.npz',
    },
    '2D0': {
        'result_prefix': 'ea_ip__copoly_stage2d_2d0_arch__a_held_out__target',
        'pred_prefix':   'ea_ip__{target}__copoly_stage2d_2d0_arch__a_held_out__split{fold}.npz',
    },
    '2D1': {
        'result_prefix': 'ea_ip__copoly_stage2d_2d1_arch__a_held_out__target',
        'pred_prefix':   'ea_ip__{target}__copoly_stage2d_2d1_arch__a_held_out__split{fold}.npz',
    },
}

COLORS = {
    'wDMPNN': '#7f7f7f',
    'Frac':   '#1f77b4',
    '2D0':    '#ff7f0e',
    '2D1':    '#2ca02c',
}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_results(model_name: str, tkey: str) -> pd.DataFrame | None:
    prefix = MODELS[model_name]['result_prefix']
    target_full = TARGETS[tkey]
    fname = f"{prefix}_{target_full}_results.csv"
    path = RESULTS_DIR / fname
    if not path.exists():
        print(f"  [WARN] Results not found: {path.name}")
        return None
    df = pd.read_csv(path)
    df['model'] = model_name
    df['tkey'] = tkey
    return df


def load_predictions(model_name: str, tkey: str) -> list[dict] | None:
    """Load per-fold NPZ predictions from PRED_DIR."""
    target_full = TARGETS[tkey]
    other_tkey  = 'IP' if tkey == 'EA' else 'EA'
    other_full  = TARGETS[other_tkey]
    results = []

    for fold in range(N_FOLDS):
        if model_name == 'wDMPNN':
            fname       = f"ea_ip__{target_full}__split{fold}.npz"
            other_fname = f"ea_ip__{other_full}__split{fold}.npz"
        else:
            tmpl        = MODELS[model_name]['pred_prefix']
            fname       = tmpl.format(target=target_full,  fold=fold)
            other_fname = tmpl.format(target=other_full,   fold=fold)

        path = PRED_DIR / fname
        if not path.exists():
            print(f"  [WARN] Missing pred: {fname}")
            continue

        npz = np.load(path, allow_pickle=True)
        yt  = npz['y_true'].flatten()
        yp  = npz['y_pred'].flatten()

        # Denormalize stage2d predictions via regression on frac model
        if model_name != 'wDMPNN':
            frac_fname = f"ea_ip__{target_full}__copoly_stage2d_frac__a_held_out__split{fold}.npz"
            frac_path  = PRED_DIR / frac_fname
            if frac_path.exists():
                fnpz     = np.load(frac_path, allow_pickle=True)
                fyt      = fnpz['y_true'].flatten()
                fyp      = fnpz['y_pred'].flatten()
                slope, intercept, *_ = sp_stats.linregress(fyp, fyt)
                yp = yp * slope + intercept
                yt = yt * slope + intercept  # y_true is also in norm space; rescale

        results.append({'fold': fold, 'y_true': yt, 'y_pred': yp})

    return results if results else None


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 & 2 — Per-fold RMSE and R²
# ═══════════════════════════════════════════════════════════════════════

def plot_metric_bars():
    """Two figures (RMSE, R²), each with EA | IP panels, one bar per fold per model."""
    for metric, ylabel, ylim in [
        ('test/rmse', 'RMSE (eV)',  (0, None)),
        ('test/r2',   'R²',         (None, 1.0)),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        fig.suptitle(f'LOMO per-fold {ylabel}', fontweight='bold')

        for col, tkey in enumerate(['EA', 'IP']):
            ax = axes[col]
            folds = np.arange(N_FOLDS)
            n_models = len(MODELS)
            width = 0.18
            offsets = np.linspace(-(n_models - 1) / 2 * width,
                                   (n_models - 1) / 2 * width, n_models)

            for k, mname in enumerate(MODELS):
                df = load_results(mname, tkey)
                if df is None:
                    continue
                vals = df.sort_values('split')[metric].values
                bars = ax.bar(folds + offsets[k], vals, width,
                              label=mname, color=COLORS[mname], alpha=0.85,
                              edgecolor='white')
                # value labels
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        va_pos = bar.get_height() + (0.01 if metric == 'test/rmse' else 0.02)
                        ax.text(bar.get_x() + bar.get_width() / 2, va_pos,
                                f'{v:.2f}', ha='center', va='bottom', fontsize=5, rotation=90)

            ax.set_xticks(folds)
            ax.set_xticklabels([f'fold {i}' for i in folds], rotation=30, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(tkey)
            if metric == 'test/r2':
                ax.axhline(0, color='grey', lw=0.8, ls=':')
            if ylim[0] is not None:
                ax.set_ylim(bottom=ylim[0])
            if ylim[1] is not None:
                ax.set_ylim(top=ylim[1])

        axes[0].legend(loc='upper right', frameon=False)
        fig.tight_layout()

        safe = ylabel.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        fname = f'lomao_perfold_{safe}'
        fig.savefig(OUT_DIR / f'{fname}.pdf')
        fig.savefig(OUT_DIR / f'{fname}.png')
        plt.close(fig)
        print(f"  Saved: {fname}.pdf/.png")

    # Also plot mean ± std summary bars
    _plot_summary_bars()


def _plot_summary_bars():
    """Summary bar chart: mean ± std across folds + individual fold dots overlaid.

    Two panels per target (EA / IP): top = RMSE, bottom = R².
    For R² EA the y-axis is clipped to (-1.5, 1.0) so fold-6 outliers don't
    compress the readable region; outlier values are annotated with arrows.
    """
    rng = np.random.default_rng(42)  # for reproducible jitter

    # y-axis clip limits per (metric, tkey) — None means auto
    CLIP = {
        ('test/r2',   'EA'): (-1.5, 1.05),
        ('test/r2',   'IP'): (None, 1.05),
        ('test/rmse', 'EA'): (0, None),
        ('test/rmse', 'IP'): (0, None),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle('LOMO: mean ± std across folds  (dots = individual folds)',
                 fontweight='bold')

    for row, (metric, ylabel) in enumerate([('test/rmse', 'RMSE (eV)'), ('test/r2', 'R²')]):
        for col, tkey in enumerate(['EA', 'IP']):
            ax = axes[row, col]
            names_list = list(MODELS.keys())
            x = np.arange(len(names_list))
            width = 0.55

            all_vals_per_model = []
            means, stds = [], []
            for mname in names_list:
                df = load_results(mname, tkey)
                if df is None:
                    all_vals_per_model.append(np.array([np.nan]))
                    means.append(np.nan); stds.append(0)
                else:
                    vals = df.sort_values('split')[metric].values.astype(float)
                    all_vals_per_model.append(vals)
                    means.append(np.nanmean(vals))
                    stds.append(np.nanstd(vals))

            clip_lo, clip_hi = CLIP.get((metric, tkey), (None, None))

            # bars
            colors = [COLORS[n] for n in names_list]
            bars = ax.bar(x, means, width, color=colors, alpha=0.75,
                          edgecolor='white', zorder=2)

            # std caps (manual, so they respect clipping)
            for xi, (m, s) in enumerate(zip(means, stds)):
                if not np.isnan(m):
                    lo_cap = m - s
                    hi_cap = m + s
                    ax.plot([xi, xi], [lo_cap, hi_cap], color='black',
                            lw=1.5, zorder=3)
                    ax.plot([xi - 0.08, xi + 0.08], [lo_cap, lo_cap],
                            color='black', lw=1.5, zorder=3)
                    ax.plot([xi - 0.08, xi + 0.08], [hi_cap, hi_cap],
                            color='black', lw=1.5, zorder=3)

            # individual fold dots with jitter
            for xi, (mname, vals) in enumerate(zip(names_list, all_vals_per_model)):
                jitter = rng.uniform(-0.14, 0.14, size=len(vals))
                clipped = (clip_lo is not None and vals < clip_lo) | \
                          (clip_hi is not None and vals > clip_hi)
                visible = ~clipped if clip_lo is not None or clip_hi is not None \
                          else np.ones(len(vals), dtype=bool)

                ax.scatter(xi + jitter[visible], vals[visible],
                           s=30, color=COLORS[mname], edgecolor='black',
                           linewidth=0.4, zorder=4, alpha=0.9)

                # annotate clipped outliers with arrow + value
                for fi, (v, c) in enumerate(zip(vals, clipped)):
                    if c:
                        clip_edge = clip_lo if (clip_lo is not None and v < clip_lo) else clip_hi
                        direction = -1 if v < (clip_lo or -999) else 1
                        ax.annotate(f'f{fi}:{v:.2f}',
                                    xy=(xi, clip_edge),
                                    xytext=(xi + 0.3, clip_edge + direction * 0.12),
                                    fontsize=6, color=COLORS[mname],
                                    arrowprops=dict(arrowstyle='->', color=COLORS[mname],
                                                    lw=0.8),
                                    zorder=5)

            # mean value labels
            for xi, (m, s) in enumerate(zip(means, stds)):
                if not np.isnan(m):
                    label_y = m + s + (0.01 if metric == 'test/rmse' else 0.03)
                    if clip_hi is not None:
                        label_y = min(label_y, clip_hi - 0.05)
                    ax.text(xi, label_y, f'{m:.3f}', ha='center',
                            va='bottom', fontsize=8, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(names_list, rotation=15, ha='right')
            ax.set_ylabel(ylabel if col == 0 else '')
            ax.set_title(f'{tkey}')
            if metric == 'test/r2':
                ax.axhline(0, color='grey', lw=0.8, ls=':', zorder=1)
            if clip_lo is not None:
                ax.set_ylim(bottom=clip_lo)
            if clip_hi is not None:
                ax.set_ylim(top=clip_hi)
            ax.grid(axis='y', alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'lomao_summary_bars.pdf')
    fig.savefig(OUT_DIR / 'lomao_summary_bars.png')
    plt.close(fig)
    print("  Saved: lomao_summary_bars.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3 — Parity / truth plots
# ═══════════════════════════════════════════════════════════════════════

def plot_parity():
    """4×2 parity grid: rows = models, cols = EA | IP.
    Each panel pools all folds, coloured by fold index.
    """
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))
    fig.suptitle('LOMO Parity Plots (all folds pooled)', fontweight='bold', y=1.01)

    cmap = plt.get_cmap('tab10')

    for row, mname in enumerate(MODELS):
        for col, tkey in enumerate(['EA', 'IP']):
            ax = axes[row, col]
            preds = load_predictions(mname, tkey)

            if preds is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='grey')
                ax.set_title(f'{mname} — {tkey}')
                continue

            all_yt, all_yp = [], []
            for p in preds:
                fold_color = cmap(p['fold'] / N_FOLDS)
                ax.scatter(p['y_true'], p['y_pred'], s=8, alpha=0.5,
                           color=fold_color, rasterized=True,
                           label=f"fold {p['fold']}")
                all_yt.extend(p['y_true'])
                all_yp.extend(p['y_pred'])

            all_yt = np.array(all_yt)
            all_yp = np.array(all_yp)

            # y = x line
            lo = min(all_yt.min(), all_yp.min()) - 0.1
            hi = max(all_yt.max(), all_yp.max()) + 0.1
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y=x')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal', adjustable='box')

            r2  = r2_score(all_yt, all_yp)
            rmse = np.sqrt(np.mean((all_yt - all_yp) ** 2))
            ax.set_title(f'{mname} — {tkey}  |  R²={r2:.3f}  RMSE={rmse:.3f}')
            ax.set_xlabel('True (eV)')
            ax.set_ylabel('Predicted (eV)')

    # shared fold legend from last panel
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    fold_handles = [h for h, l in zip(handles, labels) if l.startswith('fold')]
    fold_labels  = [l for l in labels if l.startswith('fold')]
    fig.legend(fold_handles, fold_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=N_FOLDS, frameon=False,
               fontsize=8, title='Fold (colour)')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'lomao_parity.pdf', bbox_inches='tight')
    fig.savefig(OUT_DIR / 'lomao_parity.png', bbox_inches='tight')
    plt.close(fig)
    print("  Saved: lomao_parity.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("LOMAO Results Inspector")
    print("=" * 60)
    print(f"Results:     {RESULTS_DIR}")
    print(f"Predictions: {PRED_DIR}")
    print(f"Output:      {OUT_DIR}")
    print()

    # Quick sanity print
    print("Per-model fold summary:")
    for tkey in ['EA', 'IP']:
        print(f"\n  {tkey}:")
        print(f"  {'Model':<10}  {'Mean R²':>8}  {'Std R²':>7}  {'Mean RMSE':>9}  {'Std RMSE':>8}")
        for mname in MODELS:
            df = load_results(mname, tkey)
            if df is None:
                print(f"  {mname:<10}  {'N/A':>8}")
                continue
            r2s   = df['test/r2'].values
            rmses = df['test/rmse'].values
            print(f"  {mname:<10}  {np.nanmean(r2s):>8.4f}  {np.nanstd(r2s):>7.4f}"
                  f"  {np.nanmean(rmses):>9.4f}  {np.nanstd(rmses):>8.4f}")

    print()
    print("Generating metric bar charts...")
    plot_metric_bars()

    print("Generating parity plots...")
    plot_parity()

    print()
    print("=" * 60)
    print(f"Done. Outputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
