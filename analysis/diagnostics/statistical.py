"""Step 11: Statistical comparisons between models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import r2_score

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS
from .data_loading import load_predictions_single
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def _paired_comparison(vals_a: np.ndarray, vals_b: np.ndarray,
                       name_a: str, name_b: str, metric_name: str) -> dict:
    """Perform paired Wilcoxon signed-rank test on fold-level values."""
    valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
    a = vals_a[valid]
    b = vals_b[valid]
    n = len(a)

    if n < 3:
        return {
            'metric': metric_name,
            'model_a': name_a,
            'model_b': name_b,
            'n_folds': n,
            'median_diff': np.nan,
            'mean_diff': np.nan,
            'wilcoxon_p': np.nan,
            'wins_a': 0,
            'wins_b': 0,
            'ties': 0,
        }

    diff = a - b
    median_diff = float(np.median(diff))
    mean_diff = float(np.mean(diff))

    try:
        stat, p_val = sp_stats.wilcoxon(diff, alternative='two-sided')
    except Exception:
        p_val = np.nan

    wins_a = int((diff > 0).sum())
    wins_b = int((diff < 0).sum())
    ties = int((diff == 0).sum())

    return {
        'metric': metric_name,
        'model_a': name_a,
        'model_b': name_b,
        'n_folds': n,
        'median_diff': median_diff,
        'mean_diff': mean_diff,
        'wilcoxon_p': float(p_val) if not np.isnan(p_val) else np.nan,
        'wins_a': wins_a,
        'wins_b': wins_b,
        'ties': ties,
    }


def run_statistical(df: pd.DataFrame, meta: dict[str, list],
                    cal_df: pd.DataFrame = None,
                    ord_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Step 11: Statistical comparisons.
    Saves statistical_comparisons.csv.
    """
    out_dir = STEP_DIRS['10_summary']
    rows = []

    # Focus on monomer_heldout paired comparisons (wDMPNN vs ChemArch)
    split = 'monomer_heldout'
    meta_folds = meta[split]
    n_folds = N_FOLDS[split]

    for tkey in TARGETS:
        # Collect fold-level metrics for each model
        model_r2 = {m: [] for m in MODELS}
        model_gm_r2 = {m: [] for m in MODELS}
        model_delta_r2 = {m: [] for m in MODELS}

        for fold in range(n_folds):
            for model in MODELS:
                pred = load_predictions_single(model, tkey, split, fold, meta_folds)
                if pred is None:
                    model_r2[model].append(np.nan)
                    model_gm_r2[model].append(np.nan)
                    model_delta_r2[model].append(np.nan)
                    continue

                yt, yp = pred['y_true'], pred['y_pred']
                gidx = pred['global_idx']

                # Overall R²
                model_r2[model].append(float(r2_score(yt, yp)))

                # Group-mean and delta R²
                fdf = build_fold_df(df, yt, yp, gidx)
                matched = filter_matched_groups(fdf)
                if len(matched) > 10:
                    matched = add_group_means(matched)
                    gm = matched.groupby('group_key').agg(
                        y_bar_t=('y_true', 'mean'),
                        y_bar_p=('y_pred', 'mean'),
                    )
                    if len(gm) > 2 and gm['y_bar_t'].std() > 1e-10:
                        model_gm_r2[model].append(float(r2_score(gm['y_bar_t'], gm['y_bar_p'])))
                    else:
                        model_gm_r2[model].append(np.nan)

                    dt = matched['delta_true'].values
                    dp = matched['delta_pred'].values
                    if np.std(dt) > 1e-10:
                        model_delta_r2[model].append(float(r2_score(dt, dp)))
                    else:
                        model_delta_r2[model].append(np.nan)
                else:
                    model_gm_r2[model].append(np.nan)
                    model_delta_r2[model].append(np.nan)

        # Compare wDMPNN vs ChemArch
        for metric_name, metric_dict in [
            (f'{tkey}_overall_R2', model_r2),
            (f'{tkey}_group_mean_R2', model_gm_r2),
            (f'{tkey}_delta_R2', model_delta_r2),
        ]:
            a = np.array(metric_dict['wdmpnn'])
            b = np.array(metric_dict['chemarch'])
            rows.append(_paired_comparison(a, b, 'wdmpnn', 'chemarch', metric_name))

        # Also wDMPNN vs GlobalArch
        for metric_name, metric_dict in [
            (f'{tkey}_overall_R2', model_r2),
            (f'{tkey}_delta_R2', model_delta_r2),
        ]:
            a = np.array(metric_dict['wdmpnn'])
            b = np.array(metric_dict['globalarch'])
            rows.append(_paired_comparison(a, b, 'wdmpnn', 'globalarch', metric_name))

    # Ordering accuracy comparison if available
    if ord_df is not None and len(ord_df) > 0:
        lomo_ord = ord_df[ord_df['split'] == 'monomer_heldout']
        for tkey in TARGETS:
            tsub = lomo_ord[lomo_ord['target'] == tkey]
            for metric_col in ['pairwise_acc', 'median_kendall']:
                a_vals, b_vals = [], []
                for fold in range(n_folds):
                    a_row = tsub[(tsub['model'] == 'wdmpnn') & (tsub['fold'] == fold)]
                    b_row = tsub[(tsub['model'] == 'chemarch') & (tsub['fold'] == fold)]
                    a_vals.append(a_row[metric_col].values[0] if len(a_row) else np.nan)
                    b_vals.append(b_row[metric_col].values[0] if len(b_row) else np.nan)
                rows.append(_paired_comparison(
                    np.array(a_vals), np.array(b_vals),
                    'wdmpnn', 'chemarch', f'{tkey}_{metric_col}'
                ))

    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(out_dir / 'statistical_comparisons.csv', index=False)
    print(f"  Step 11 (statistical) complete: {out_dir}")
    return stat_df
