"""Step 10: Per-fold case studies for monomer-heldout."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats as sp_stats

from .config import (
    MODELS, TARGETS, N_FOLDS, STEP_DIRS, MODEL_DISPLAY, COLORS,
    FOLD_MONOMER_NAMES,
)
from .data_loading import load_predictions_single
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def _fold_case_summary(df: pd.DataFrame, meta_folds: list, fold: int,
                       tkey: str) -> dict:
    """Generate summary dict for one fold/target."""
    fold_meta = next((r for r in meta_folds if r['fold'] == fold), {})
    held_out = fold_meta.get('held_out_monomer_A', '')

    summary = {
        'fold': fold,
        'target': tkey,
        'held_out_smiles': held_out,
        'monomer_name': FOLD_MONOMER_NAMES.get(fold, ''),
    }

    for model in MODELS:
        pred = load_predictions_single(model, tkey, 'monomer_heldout', fold, meta_folds)
        if pred is None:
            for metric in ['r2', 'mae', 'gm_r2', 'gm_mae', 'delta_r2', 'delta_mae',
                           'cal_slope', 'disp_ratio', 'n_matched_groups']:
                summary[f'{model}_{metric}'] = np.nan
            continue

        yt, yp, gidx = pred['y_true'], pred['y_pred'], pred['global_idx']
        summary[f'{model}_r2'] = float(r2_score(yt, yp))
        summary[f'{model}_mae'] = float(mean_absolute_error(yt, yp))

        fdf = build_fold_df(df, yt, yp, gidx)
        matched = filter_matched_groups(fdf)
        if len(matched) < 10:
            for m in ['gm_r2', 'gm_mae', 'delta_r2', 'delta_mae',
                      'cal_slope', 'disp_ratio', 'n_matched_groups']:
                summary[f'{model}_{m}'] = np.nan
            continue

        matched = add_group_means(matched)
        summary[f'{model}_n_matched_groups'] = int(matched['group_key'].nunique())

        # Group-mean metrics
        gm = matched.groupby('group_key').agg(
            y_bar_t=('y_true', 'mean'), y_bar_p=('y_pred', 'mean')
        )
        if len(gm) > 2 and gm['y_bar_t'].std() > 1e-10:
            summary[f'{model}_gm_r2'] = float(r2_score(gm['y_bar_t'], gm['y_bar_p']))
            summary[f'{model}_gm_mae'] = float(mean_absolute_error(gm['y_bar_t'], gm['y_bar_p']))
        else:
            summary[f'{model}_gm_r2'] = np.nan
            summary[f'{model}_gm_mae'] = np.nan

        # Delta metrics
        dt = matched['delta_true'].values
        dp = matched['delta_pred'].values
        if np.std(dt) > 1e-10:
            summary[f'{model}_delta_r2'] = float(r2_score(dt, dp))
            summary[f'{model}_delta_mae'] = float(mean_absolute_error(dt, dp))
            slope, _, _, _, _ = sp_stats.linregress(dt, dp)
            summary[f'{model}_cal_slope'] = float(slope)
            summary[f'{model}_disp_ratio'] = float(np.std(dp) / np.std(dt))
        else:
            summary[f'{model}_delta_r2'] = np.nan
            summary[f'{model}_delta_mae'] = np.nan
            summary[f'{model}_cal_slope'] = np.nan
            summary[f'{model}_disp_ratio'] = np.nan

    return summary


def _find_extreme_groups(df: pd.DataFrame, meta_folds: list, fold: int,
                         tkey: str, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find groups where wDMPNN most outperforms ChemArch (overall)
    and where ChemArch most outperforms wDMPNN (delta).
    """
    pred_w = load_predictions_single('wdmpnn', tkey, 'monomer_heldout', fold, meta_folds)
    pred_c = load_predictions_single('chemarch', tkey, 'monomer_heldout', fold, meta_folds)

    if pred_w is None or pred_c is None:
        return pd.DataFrame(), pd.DataFrame()

    fdf_w = build_fold_df(df, pred_w['y_true'], pred_w['y_pred'], pred_w['global_idx'])
    fdf_c = build_fold_df(df, pred_c['y_true'], pred_c['y_pred'], pred_c['global_idx'])

    matched_w = filter_matched_groups(fdf_w)
    matched_c = filter_matched_groups(fdf_c)
    if len(matched_w) == 0 or len(matched_c) == 0:
        return pd.DataFrame(), pd.DataFrame()

    matched_w = add_group_means(matched_w)
    matched_c = add_group_means(matched_c)

    # Group-level advantage: wDMPNN overall advantage
    gw = matched_w.groupby('group_key').agg(
        y_bar_true=('y_true', 'mean'),
        y_bar_pred_w=('y_pred', 'mean'),
    ).reset_index()
    gc = matched_c.groupby('group_key').agg(
        y_bar_pred_c=('y_pred', 'mean'),
    ).reset_index()
    merged = gw.merge(gc, on='group_key')
    merged['err_w'] = (merged['y_bar_pred_w'] - merged['y_bar_true']).abs()
    merged['err_c'] = (merged['y_bar_pred_c'] - merged['y_bar_true']).abs()
    merged['w_advantage'] = merged['err_c'] - merged['err_w']

    # Top groups where wDMPNN wins overall
    wdmpnn_wins = merged.nlargest(n, 'w_advantage')

    # Delta advantage (ChemArch wins in architecture recovery)
    # Compute per-group delta MSE
    delta_w = matched_w.groupby('group_key').apply(
        lambda g: np.mean((g['delta_true'] - g['delta_pred']) ** 2)
    ).rename('delta_mse_w')
    delta_c = matched_c.groupby('group_key').apply(
        lambda g: np.mean((g['delta_true'] - g['delta_pred']) ** 2)
    ).rename('delta_mse_c')
    delta_merged = pd.DataFrame({'delta_mse_w': delta_w, 'delta_mse_c': delta_c}).dropna()
    delta_merged['c_advantage'] = delta_merged['delta_mse_w'] - delta_merged['delta_mse_c']
    chemarch_wins = delta_merged.nlargest(n, 'c_advantage')

    return wdmpnn_wins, chemarch_wins


def run_case_studies(df: pd.DataFrame, meta: dict[str, list]) -> None:
    """
    Step 10: Per-fold case studies.
    Saves fold-specific results under 09_per_fold_case_studies/.
    """
    out_dir = STEP_DIRS['09_per_fold_case_studies']
    split = 'monomer_heldout'
    meta_folds = meta[split]
    n_folds = N_FOLDS[split]

    all_summaries = []

    # Focus on fold 6 (EA) and find hardest IP fold
    key_folds = {6}  # EA hard fold

    # Find hardest IP fold
    ip_r2s = []
    for fold in range(n_folds):
        pred = load_predictions_single('wdmpnn', 'IP', split, fold, meta_folds)
        if pred is not None:
            ip_r2s.append((fold, float(r2_score(pred['y_true'], pred['y_pred']))))
    if ip_r2s:
        worst_ip = min(ip_r2s, key=lambda x: x[1])[0]
        key_folds.add(worst_ip)

    for fold in range(n_folds):
        fold_dir = out_dir / f'fold_{fold:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        for tkey in TARGETS:
            summary = _fold_case_summary(df, meta_folds, fold, tkey)
            all_summaries.append(summary)

        # Extreme groups (only for key folds to save computation)
        if fold in key_folds:
            for tkey in TARGETS:
                wdmpnn_wins, chemarch_wins = _find_extreme_groups(
                    df, meta_folds, fold, tkey, n=5
                )
                if len(wdmpnn_wins) > 0:
                    wdmpnn_wins.to_csv(
                        fold_dir / f'wdmpnn_wins_overall_{tkey}.csv', index=False
                    )
                if len(chemarch_wins) > 0:
                    chemarch_wins.to_csv(
                        fold_dir / f'chemarch_wins_delta_{tkey}.csv', index=False
                    )

    # Save combined summary
    sum_df = pd.DataFrame(all_summaries)
    sum_df.to_csv(out_dir / 'fold_case_summaries.csv', index=False)
    print(f"  Step 10 (case studies) complete: {out_dir}")
