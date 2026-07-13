"""Step 7: Performance vs architecture-effect magnitude."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS
from .data_loading import load_predictions_single
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def run_effect_magnitude(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Step 7: Binned analysis by architecture-effect magnitude.
    Saves effect_magnitude_metrics.csv.
    """
    out_dir = STEP_DIRS['06_effect_magnitude']
    rows = []

    for split in SPLITS:
        meta_folds = meta[split]
        for tkey in TARGETS:
            for fold in range(N_FOLDS[split]):
                # Collect all models' predictions for this fold
                model_dfs = {}
                for model in MODELS:
                    pred = load_predictions_single(model, tkey, split, fold, meta_folds)
                    if pred is None:
                        continue
                    fdf = build_fold_df(
                        df, pred['y_true'], pred['y_pred'], pred['global_idx']
                    )
                    matched = filter_matched_groups(fdf)
                    if len(matched) == 0:
                        continue
                    matched = add_group_means(matched)
                    model_dfs[model] = matched

                if not model_dfs:
                    continue

                # Use true delta from any model (they share y_true)
                ref_model = list(model_dfs.keys())[0]
                ref_df = model_dfs[ref_model]
                abs_delta = ref_df['delta_true'].abs().values

                # Quantile bins
                try:
                    bin_edges = np.quantile(abs_delta, [0, 0.25, 0.5, 0.75, 1.0])
                    bin_edges[-1] += 1e-6  # include max
                    bin_labels = ['Q1_smallest', 'Q2', 'Q3', 'Q4_largest']
                    ref_df = ref_df.copy()
                    ref_df['abs_delta_bin'] = pd.cut(
                        ref_df['delta_true'].abs(), bins=bin_edges,
                        labels=bin_labels, include_lowest=True
                    )
                except Exception:
                    continue

                for model, mdf in model_dfs.items():
                    mdf = mdf.copy()
                    mdf['abs_delta_bin'] = ref_df['abs_delta_bin'].values

                    for bin_label in bin_labels:
                        bsub = mdf[mdf['abs_delta_bin'] == bin_label]
                        if len(bsub) < 5:
                            continue

                        dt = bsub['delta_true'].values
                        dp = bsub['delta_pred'].values
                        delta_err = dt - dp

                        rows.append({
                            'model': model,
                            'split': split,
                            'target': tkey,
                            'fold': fold,
                            'bin': bin_label,
                            'n_samples': len(bsub),
                            'delta_mae': float(np.abs(delta_err).mean()),
                            'delta_rmse': float(np.sqrt((delta_err ** 2).mean())),
                            'mean_signed_error': float(delta_err.mean()),
                            'frac_correct_sign': float(
                                np.mean(np.sign(dt) == np.sign(dp))
                            ),
                            'mean_abs_delta_true': float(np.abs(dt).mean()),
                        })

    em_df = pd.DataFrame(rows)
    if len(em_df) > 0:
        col_order = ['model', 'split', 'target', 'fold', 'bin'] + [
            c for c in em_df.columns
            if c not in ('model', 'split', 'target', 'fold', 'bin')
        ]
        em_df = em_df[col_order]
    em_df.to_csv(out_dir / 'effect_magnitude_metrics.csv', index=False)

    # Model-advantage analysis (wDMPNN vs ChemArch)
    _compute_model_advantage(df, meta, out_dir)

    print(f"  Step 7 (effect magnitude) complete: {out_dir}")
    return em_df


def _compute_model_advantage(df: pd.DataFrame, meta: dict[str, list],
                             out_dir) -> None:
    """Compute per-sample wDMPNN vs ChemArch advantage."""
    adv_rows = []

    for split in SPLITS:
        meta_folds = meta[split]
        for tkey in TARGETS:
            for fold in range(N_FOLDS[split]):
                pred_w = load_predictions_single('wdmpnn', tkey, split, fold, meta_folds)
                pred_c = load_predictions_single('chemarch', tkey, split, fold, meta_folds)
                if pred_w is None or pred_c is None:
                    continue

                fdf_w = build_fold_df(df, pred_w['y_true'], pred_w['y_pred'], pred_w['global_idx'])
                fdf_c = build_fold_df(df, pred_c['y_true'], pred_c['y_pred'], pred_c['global_idx'])

                matched_w = filter_matched_groups(fdf_w)
                matched_c = filter_matched_groups(fdf_c)
                if len(matched_w) == 0 or len(matched_c) == 0:
                    continue

                matched_w = add_group_means(matched_w).sort_values('global_idx').reset_index(drop=True)
                matched_c = add_group_means(matched_c).sort_values('global_idx').reset_index(drop=True)

                # Align on global_idx
                merged = matched_w[['global_idx', 'y_true', 'y_pred', 'delta_true', 'delta_pred']].merge(
                    matched_c[['global_idx', 'y_pred', 'delta_pred']],
                    on='global_idx', suffixes=('_w', '_c')
                )
                if len(merged) == 0:
                    continue

                # Overall advantage: |err_ChemArch| - |err_wDMPNN| (positive = wDMPNN better)
                err_w = np.abs(merged['y_true'] - merged['y_pred_w'])
                err_c = np.abs(merged['y_true'] - merged['y_pred_c'])
                overall_adv = (err_c - err_w).values

                # Delta advantage
                dt = merged['delta_true'].values
                d_err_w = np.abs(dt - merged['delta_pred_w'].values)
                d_err_c = np.abs(dt - merged['delta_pred_c'].values)
                delta_adv = d_err_c - d_err_w

                for i in range(len(merged)):
                    adv_rows.append({
                        'split': split,
                        'target': tkey,
                        'fold': fold,
                        'abs_delta_true': abs(dt[i]),
                        'overall_advantage': overall_adv[i],
                        'delta_advantage': float(delta_adv[i]),
                    })

    if adv_rows:
        adv_df = pd.DataFrame(adv_rows)
        adv_df.to_csv(out_dir / 'wdmpnn_vs_chemarch_advantage.csv', index=False)
