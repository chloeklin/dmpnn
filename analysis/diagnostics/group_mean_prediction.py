"""Step 3: Group-mean prediction analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats as sp_stats

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS
from .data_loading import load_predictions_single
from .grouping import build_fold_df, group_level_stats, filter_matched_groups


def _group_mean_metrics(gstats: pd.DataFrame) -> dict:
    """Compute group-mean prediction quality metrics from group_level_stats output."""
    if len(gstats) < 3:
        return {k: np.nan for k in [
            'gm_r2', 'gm_mae', 'gm_rmse',
            'gm_slope', 'gm_intercept', 'n_groups',
        ]}

    yt = gstats['y_bar_true'].values
    yp = gstats['y_bar_pred'].values
    valid = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[valid], yp[valid]

    if len(yt) < 3 or np.std(yt) < 1e-10:
        return {k: np.nan for k in [
            'gm_r2', 'gm_mae', 'gm_rmse',
            'gm_slope', 'gm_intercept', 'n_groups',
        ]}

    slope, intercept, _, _, _ = sp_stats.linregress(yt, yp)
    return {
        'gm_r2': float(r2_score(yt, yp)),
        'gm_mae': float(mean_absolute_error(yt, yp)),
        'gm_rmse': float(np.sqrt(np.mean((yt - yp) ** 2))),
        'gm_slope': float(slope),
        'gm_intercept': float(intercept),
        'n_groups': len(yt),
    }


def run_group_mean_prediction(df: pd.DataFrame, meta: dict[str, list]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 3: Analyse group-mean prediction quality.
    Saves group_mean_metrics.csv and group_mean_predictions.parquet.
    """
    out_dir = STEP_DIRS['03_group_mean_prediction']
    metric_rows = []
    all_group_preds = []

    for split in SPLITS:
        meta_folds = meta[split]
        for tkey in TARGETS:
            for fold in range(N_FOLDS[split]):
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

                    gstats = group_level_stats(matched)
                    gstats['model'] = model
                    gstats['split'] = split
                    gstats['target'] = tkey
                    gstats['fold'] = fold
                    all_group_preds.append(gstats)

                    gm = _group_mean_metrics(gstats)
                    gm['model'] = model
                    gm['split'] = split
                    gm['target'] = tkey
                    gm['fold'] = fold
                    metric_rows.append(gm)

    # Save metrics
    gm_df = pd.DataFrame(metric_rows)
    col_order = ['model', 'split', 'target', 'fold'] + [
        c for c in gm_df.columns if c not in ('model', 'split', 'target', 'fold')
    ]
    gm_df = gm_df[col_order]
    gm_df.to_csv(out_dir / 'group_mean_metrics.csv', index=False)

    # Save group-level predictions
    if all_group_preds:
        gpred_df = pd.concat(all_group_preds, ignore_index=True)
        gpred_df.to_csv(out_dir / 'group_mean_predictions.csv', index=False)

    print(f"  Step 3 (group-mean prediction) complete: {out_dir}")
    return gm_df, gpred_df if all_group_preds else pd.DataFrame()
