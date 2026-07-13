"""Shared plotting utilities for the diagnostics pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS, DPI, COLORS, MARKERS, MODEL_DISPLAY


def savefig(fig, path):
    """Save figure and close."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Variance geometry figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_variance_decomposition(vg_df: pd.DataFrame) -> None:
    """Stacked bar: between vs within variance fraction by split."""
    out_dir = STEP_DIRS['02_variance_geometry']

    for tkey in TARGETS:
        fig, ax = plt.subplots(figsize=(6, 4))
        sub = vg_df[vg_df['target'] == tkey]
        if len(sub) == 0:
            plt.close(fig)
            continue

        # Average across folds
        agg = sub.groupby('split')[['frac_between', 'frac_within']].mean()
        agg = agg.reindex(SPLITS)

        x = np.arange(len(SPLITS))
        ax.bar(x, agg['frac_between'], label='Between-group', color='#1f77b4')
        ax.bar(x, agg['frac_within'], bottom=agg['frac_between'],
               label='Within-group (arch)', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in SPLITS], fontsize=8)
        ax.set_ylabel('Fraction of SS_total')
        ax.set_title(f'{tkey}: Variance Decomposition')
        ax.legend(frameon=False)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        savefig(fig, out_dir / f'variance_decomposition_{tkey}.png')

    # Distribution of group means vs deltas
    # (requires raw data, skip if not provided)


def plot_error_decomposition(ed_df: pd.DataFrame) -> None:
    """Stacked bar: between vs within SSE per model, by split and target."""
    out_dir = STEP_DIRS['02_variance_geometry']

    for tkey in TARGETS:
        for split in SPLITS:
            sub = ed_df[(ed_df['target'] == tkey) & (ed_df['split'] == split)]
            if len(sub) == 0:
                continue

            # Average across folds
            agg = sub.groupby('model')[['SSE_between', 'SSE_within']].mean()
            agg = agg.reindex(MODELS)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Absolute
            ax = axes[0]
            x = np.arange(len(MODELS))
            labels = [MODEL_DISPLAY[m] for m in MODELS]
            colors_b = [COLORS[m] for m in MODELS]
            ax.bar(x, agg['SSE_between'], label='Between-group SSE',
                   color='#1f77b4', alpha=0.8)
            ax.bar(x, agg['SSE_within'], bottom=agg['SSE_between'],
                   label='Within-group SSE', color='#ff7f0e', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('SSE')
            ax.set_title(f'{tkey} / {split}: Absolute Error Decomposition')
            ax.legend(frameon=False, fontsize=8)

            # Normalised
            ax = axes[1]
            total = agg['SSE_between'] + agg['SSE_within']
            frac_b = agg['SSE_between'] / total
            frac_w = agg['SSE_within'] / total
            ax.bar(x, frac_b, label='Between-group', color='#1f77b4', alpha=0.8)
            ax.bar(x, frac_w, bottom=frac_b, label='Within-group',
                   color='#ff7f0e', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Fraction of total SSE')
            ax.set_title(f'{tkey} / {split}: Normalised')
            ax.set_ylim(0, 1.05)
            ax.legend(frameon=False, fontsize=8)

            fig.tight_layout()
            savefig(fig, out_dir / f'error_decomposition_{tkey}_{split}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Group-mean scatter plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_group_mean_scatter(gm_df: pd.DataFrame, gpred_df: pd.DataFrame) -> None:
    """Group-mean true vs predicted scatter plots."""
    out_dir = STEP_DIRS['03_group_mean_prediction']

    for tkey in TARGETS:
        for split in SPLITS:
            sub = gpred_df[(gpred_df['target'] == tkey) & (gpred_df['split'] == split)]
            if len(sub) == 0:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for i, model in enumerate(MODELS):
                ax = axes[i // 2, i % 2]
                msub = sub[sub['model'] == model]
                if len(msub) == 0:
                    ax.set_title(f'{MODEL_DISPLAY[model]} [no data]')
                    continue
                ax.scatter(msub['y_bar_true'], msub['y_bar_pred'],
                           alpha=0.3, s=8, color=COLORS[model])
                # y=x line
                lims = [min(msub['y_bar_true'].min(), msub['y_bar_pred'].min()),
                        max(msub['y_bar_true'].max(), msub['y_bar_pred'].max())]
                ax.plot(lims, lims, 'k--', lw=0.8)
                # Metrics
                metrics = gm_df[
                    (gm_df['model'] == model) &
                    (gm_df['split'] == split) &
                    (gm_df['target'] == tkey)
                ]
                if len(metrics) > 0:
                    r2 = metrics['gm_r2'].mean()
                    mae = metrics['gm_mae'].mean()
                    ax.text(0.05, 0.95, f'R²={r2:.4f}\nMAE={mae:.4f}',
                            transform=ax.transAxes, va='top', fontsize=8)
                ax.set_xlabel('True group mean (eV)')
                ax.set_ylabel('Predicted group mean (eV)')
                ax.set_title(MODEL_DISPLAY[model])

            fig.suptitle(f'{tkey} Group-Mean Prediction ({split})', fontweight='bold')
            fig.tight_layout()
            savefig(fig, out_dir / f'group_mean_scatter_{tkey}_{split}.png')

    # Per-fold line plots for monomer_heldout
    for tkey in TARGETS:
        lomo_gm = gm_df[(gm_df['target'] == tkey) & (gm_df['split'] == 'monomer_heldout')]
        if len(lomo_gm) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        for model in MODELS:
            msub = lomo_gm[lomo_gm['model'] == model].sort_values('fold')
            if len(msub) == 0:
                continue
            ax.plot(msub['fold'], msub['gm_r2'],
                    marker=MARKERS[model], color=COLORS[model],
                    label=MODEL_DISPLAY[model], linewidth=1.5, markersize=5)

        ax.axhline(0, color='grey', ls=':', lw=0.8)
        # Highlight fold 6 for EA, fold 5 for IP
        hl = 6 if tkey == 'EA' else 5
        ax.axvspan(hl - 0.4, hl + 0.4, alpha=0.12, color='red', zorder=0)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Group-mean R²')
        ax.set_title(f'{tkey}: Per-fold Group-Mean R² (Monomer-Heldout)')
        ax.legend(frameon=False)
        ax.set_xticks(range(N_FOLDS['monomer_heldout']))
        fig.tight_layout()
        savefig(fig, out_dir / f'group_mean_foldwise_{tkey}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Calibration scatter plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_calibration(cal_df: pd.DataFrame, df: pd.DataFrame, meta: dict) -> None:
    """Delta calibration scatter plots and summary bars."""
    from .data_loading import load_predictions_single
    from .grouping import build_fold_df, add_group_means, filter_matched_groups

    out_dir = STEP_DIRS['04_architecture_calibration']

    for tkey in TARGETS:
        for split in SPLITS:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            meta_folds = meta[split]

            for i, model in enumerate(MODELS):
                ax = axes[i // 2, i % 2]
                # Collect all fold data for this model/split/target
                all_dt, all_dp = [], []
                for fold in range(N_FOLDS[split]):
                    pred = load_predictions_single(model, tkey, split, fold, meta_folds)
                    if pred is None:
                        continue
                    fdf = build_fold_df(df, pred['y_true'], pred['y_pred'], pred['global_idx'])
                    matched = filter_matched_groups(fdf)
                    if len(matched) == 0:
                        continue
                    matched = add_group_means(matched)
                    all_dt.extend(matched['delta_true'].values)
                    all_dp.extend(matched['delta_pred'].values)

                if not all_dt:
                    ax.set_title(f'{MODEL_DISPLAY[model]} [no data]')
                    continue

                dt = np.array(all_dt)
                dp = np.array(all_dp)
                ax.scatter(dt, dp, alpha=0.15, s=4, color=COLORS[model])
                lims = [min(dt.min(), dp.min()), max(dt.max(), dp.max())]
                ax.plot(lims, lims, 'k--', lw=0.8)

                # Regression line
                from scipy import stats as sp_stats
                slope, intercept, _, _, _ = sp_stats.linregress(dt, dp)
                x_fit = np.linspace(lims[0], lims[1], 100)
                ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=1, alpha=0.7)

                disp = np.std(dp) / np.std(dt) if np.std(dt) > 1e-10 else np.nan
                from sklearn.metrics import r2_score
                r2 = r2_score(dt, dp) if np.std(dt) > 1e-10 else np.nan
                ax.text(0.05, 0.95,
                        f'R²={r2:.3f}\nslope={slope:.3f}\ndisp={disp:.3f}',
                        transform=ax.transAxes, va='top', fontsize=7)
                ax.set_xlabel('Δy_true (eV)')
                ax.set_ylabel('Δy_pred (eV)')
                ax.set_title(MODEL_DISPLAY[model])

            fig.suptitle(f'{tkey} Delta Calibration ({split})', fontweight='bold')
            fig.tight_layout()
            savefig(fig, out_dir / f'delta_calibration_{tkey}_{split}.png')

    # Summary bar plots
    for metric, ylabel, fname_prefix in [
        ('delta_slope', 'Calibration Slope', 'delta_slope_summary'),
        ('dispersion_ratio', 'Dispersion Ratio', 'delta_dispersion_summary'),
    ]:
        for tkey in TARGETS:
            fig, ax = plt.subplots(figsize=(7, 4))
            sub = cal_df[cal_df['target'] == tkey]
            if len(sub) == 0:
                plt.close(fig)
                continue

            x = np.arange(len(SPLITS))
            width = 0.18
            for j, model in enumerate(MODELS):
                msub = sub[sub['model'] == model]
                vals = [msub[msub['split'] == s][metric].mean() for s in SPLITS]
                offset = (j - 1.5) * width
                ax.bar(x + offset, vals, width, label=MODEL_DISPLAY[model],
                       color=COLORS[model], alpha=0.85)

            ax.axhline(1.0, color='grey', ls=':', lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace('_', '\n') for s in SPLITS], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{tkey}: {ylabel} by Split')
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            savefig(fig, out_dir / f'{fname_prefix}_{tkey}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Architecture ordering plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ordering(ord_df: pd.DataFrame) -> None:
    """Grouped bar: ordering metrics by model and split."""
    out_dir = STEP_DIRS['05_architecture_ordering']

    for tkey in TARGETS:
        sub = ord_df[ord_df['target'] == tkey]
        if len(sub) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        metrics = ['exact_rank_acc', 'pairwise_acc', 'median_kendall']
        titles = ['Full Ranking Accuracy', 'Pairwise Ordering Accuracy', 'Median Kendall τ']

        for ax, metric, title in zip(axes, metrics, titles):
            x = np.arange(len(SPLITS))
            width = 0.18
            for j, model in enumerate(MODELS):
                msub = sub[sub['model'] == model]
                vals = [msub[msub['split'] == s][metric].mean() for s in SPLITS]
                offset = (j - 1.5) * width
                ax.bar(x + offset, vals, width, label=MODEL_DISPLAY[model],
                       color=COLORS[model], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace('_', '\n') for s in SPLITS], fontsize=7)
            ax.set_ylabel(title)
            ax.set_title(title)

        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle(f'{tkey}: Architecture Ordering', fontweight='bold')
        fig.tight_layout()
        savefig(fig, out_dir / f'architecture_ordering_{tkey}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Effect magnitude plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_effect_magnitude(em_df: pd.DataFrame) -> None:
    """Binned delta error by effect size."""
    out_dir = STEP_DIRS['06_effect_magnitude']

    for tkey in TARGETS:
        fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 4))
        if len(SPLITS) == 1:
            axes = [axes]

        for ax, split in zip(axes, SPLITS):
            sub = em_df[(em_df['target'] == tkey) & (em_df['split'] == split)]
            if len(sub) == 0:
                continue

            bins = ['Q1_smallest', 'Q2', 'Q3', 'Q4_largest']
            x = np.arange(len(bins))
            width = 0.18
            for j, model in enumerate(MODELS):
                msub = sub[sub['model'] == model]
                vals = [msub[msub['bin'] == b]['delta_mae'].mean() for b in bins]
                offset = (j - 1.5) * width
                ax.bar(x + offset, vals, width, label=MODEL_DISPLAY[model],
                       color=COLORS[model], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(bins, fontsize=7, rotation=15)
            ax.set_ylabel('Delta MAE (eV)')
            ax.set_title(f'{split}')

        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle(f'{tkey}: Delta Error by |Δy| Quartile', fontweight='bold')
        fig.tight_layout()
        savefig(fig, out_dir / f'delta_error_by_effect_size_{tkey}.png')

    # wDMPNN vs ChemArch advantage plots
    adv_path = out_dir / 'wdmpnn_vs_chemarch_advantage.csv'
    if adv_path.exists():
        adv_df = pd.read_csv(adv_path)
        for tkey in TARGETS:
            for adv_col, fname in [
                ('overall_advantage', f'wdmpnn_vs_chemarch_overall_advantage_{tkey}'),
                ('delta_advantage', f'wdmpnn_vs_chemarch_delta_advantage_{tkey}'),
            ]:
                sub = adv_df[adv_df['target'] == tkey]
                if len(sub) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(7, 4))
                # Bin by abs_delta_true and compute mean advantage
                sub_s = sub.sort_values('abs_delta_true')
                n_bins = 20
                bins = pd.qcut(sub_s['abs_delta_true'], n_bins, duplicates='drop')
                binned = sub_s.groupby(bins)[adv_col].mean()

                ax.plot(range(len(binned)), binned.values, 'o-', color='#2ca02c', ms=4)
                ax.axhline(0, color='grey', ls=':', lw=0.8)
                ax.set_xlabel('|Δy_true| quantile bin')
                ax.set_ylabel(f'{adv_col} (positive = wDMPNN better)')
                ax.set_title(f'{tkey}: {adv_col}')
                fig.tight_layout()
                savefig(fig, out_dir / f'{fname}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Novelty plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_novelty(nov_df: pd.DataFrame) -> None:
    """Scatter: novelty vs performance."""
    out_dir = STEP_DIRS['07_monomer_novelty']

    if 'max_tanimoto' not in nov_df.columns:
        return

    for tkey in TARGETS:
        for perf_suffix, ylabel in [
            ('_r2', 'Overall R²'),
            ('_gm_r2', 'Group-mean R²'),
            ('_delta_r2', 'ΔR²'),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            for model in ['wdmpnn', 'chemarch']:
                col = f'{model}_{tkey}{perf_suffix}'
                if col not in nov_df.columns:
                    continue
                ax.scatter(nov_df['max_tanimoto'], nov_df[col],
                           marker=MARKERS[model], color=COLORS[model],
                           s=40, label=MODEL_DISPLAY[model], zorder=3)
                # Label by fold
                for _, row in nov_df.iterrows():
                    ax.annotate(f"{int(row['fold'])}", (row['max_tanimoto'], row[col]),
                                fontsize=6, ha='left', va='bottom')

            ax.set_xlabel('Max Tanimoto to Training')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{tkey}: Novelty vs {ylabel}')
            ax.legend(frameon=False)
            fig.tight_layout()

            fname_map = {'_r2': 'overall', '_gm_r2': 'group_mean', '_delta_r2': 'delta'}
            savefig(fig, out_dir / f'novelty_vs_{fname_map[perf_suffix]}_{tkey}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: Target shift plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_target_shift(ts_df: pd.DataFrame) -> None:
    """Per-fold target shift visualisation."""
    out_dir = STEP_DIRS['08_target_shift']

    for tkey in TARGETS:
        sub = ts_df[ts_df['target'] == tkey]
        if len(sub) == 0:
            continue

        # Mean shift and test std by fold
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.bar(sub['fold'], sub['mean_shift'], color='#1f77b4', alpha=0.8)
        ax.axhline(0, color='grey', ls=':', lw=0.8)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Mean Shift (test - train)')
        ax.set_title(f'{tkey}: Target Mean Shift by Fold')

        ax = axes[1]
        ax.bar(sub['fold'], sub['test_std'], color='#ff7f0e', alpha=0.8)
        ax.axhline(sub['train_std'].mean(), color='grey', ls='--', lw=0.8,
                   label='avg train std')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Test Std (eV)')
        ax.set_title(f'{tkey}: Test Target Std by Fold')
        ax.legend(frameon=False)

        fig.tight_layout()
        savefig(fig, out_dir / f'target_shift_{tkey}_by_fold.png')
