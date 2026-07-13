"""Step 5: Architecture-deviation calibration analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats as sp_stats

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS
from .data_loading import load_predictions_single
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def _calibration_metrics(matched: pd.DataFrame) -> dict:
    """
    Compute calibration metrics for architecture deviations.
    Input must already be matched and have group means added.
    """
    dt = matched['delta_true'].values
    dp = matched['delta_pred'].values

    if len(dt) < 10 or np.std(dt) < 1e-10:
        return {k: np.nan for k in [
            'delta_r2', 'delta_mae', 'delta_rmse',
            'delta_slope', 'delta_intercept',
            'pearson_r', 'dispersion_ratio', 'n_samples',
        ]}

    slope, intercept, r_val, _, _ = sp_stats.linregress(dt, dp)
    disp_ratio = float(np.std(dp) / np.std(dt))

    return {
        'delta_r2': float(r2_score(dt, dp)),
        'delta_mae': float(mean_absolute_error(dt, dp)),
        'delta_rmse': float(np.sqrt(np.mean((dt - dp) ** 2))),
        'delta_slope': float(slope),
        'delta_intercept': float(intercept),
        'pearson_r': float(r_val),
        'dispersion_ratio': disp_ratio,
        'n_samples': len(dt),
    }


def run_calibration(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Step 5: Architecture-deviation calibration analysis.
    Saves calibration_metrics.csv.
    """
    out_dir = STEP_DIRS['04_architecture_calibration']
    rows = []

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
                    matched = add_group_means(matched)

                    cal = _calibration_metrics(matched)
                    cal['model'] = model
                    cal['split'] = split
                    cal['target'] = tkey
                    cal['fold'] = fold
                    rows.append(cal)

    cal_df = pd.DataFrame(rows)
    col_order = ['model', 'split', 'target', 'fold'] + [
        c for c in cal_df.columns if c not in ('model', 'split', 'target', 'fold')
    ]
    cal_df = cal_df[col_order]
    cal_df.to_csv(out_dir / 'calibration_metrics.csv', index=False)
    print(f"  Step 5 (calibration) complete: {out_dir}")
    return cal_df
