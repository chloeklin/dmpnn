"""Step 9: Target-distribution shift analysis for monomer-heldout."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .config import TARGETS, N_FOLDS, STEP_DIRS, FOLD_MONOMER_NAMES
from .data_loading import load_predictions_single, get_global_indices
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def run_target_shift(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Step 9: Target distribution shift analysis for monomer-heldout.
    Saves target_shift_metrics.csv.
    """
    out_dir = STEP_DIRS['08_target_shift']
    split = 'monomer_heldout'
    meta_folds = meta[split]
    n_folds = N_FOLDS[split]
    all_indices = set(range(len(df)))

    rows = []
    for fold in range(n_folds):
        test_idx = get_global_indices(fold, meta_folds)
        train_idx = np.array(sorted(all_indices - set(test_idx.tolist())), dtype=int)

        for tkey, tcol in TARGETS.items():
            y_train = df.iloc[train_idx][tcol].values.astype(float)
            y_test = df.iloc[test_idx][tcol].values.astype(float)

            # Remove NaN
            y_train = y_train[~np.isnan(y_train)]
            y_test = y_test[~np.isnan(y_test)]

            if len(y_test) == 0 or len(y_train) == 0:
                continue

            # Basic stats
            row = {
                'fold': fold,
                'target': tkey,
                'monomer_name': FOLD_MONOMER_NAMES.get(fold, ''),
                'train_mean': float(y_train.mean()),
                'train_std': float(y_train.std()),
                'train_median': float(np.median(y_train)),
                'train_min': float(y_train.min()),
                'train_max': float(y_train.max()),
                'train_p5': float(np.percentile(y_train, 5)),
                'train_p95': float(np.percentile(y_train, 95)),
                'train_n': len(y_train),
                'test_mean': float(y_test.mean()),
                'test_std': float(y_test.std()),
                'test_median': float(np.median(y_test)),
                'test_min': float(y_test.min()),
                'test_max': float(y_test.max()),
                'test_p5': float(np.percentile(y_test, 5)),
                'test_p95': float(np.percentile(y_test, 95)),
                'test_n': len(y_test),
            }

            # Shift metrics
            row['mean_shift'] = row['test_mean'] - row['train_mean']
            row['std_ratio'] = row['test_std'] / row['train_std'] if row['train_std'] > 1e-10 else np.nan

            # Wasserstein distance
            try:
                row['wasserstein'] = float(sp_stats.wasserstein_distance(y_train, y_test))
            except Exception:
                row['wasserstein'] = np.nan

            # KS statistic
            try:
                ks_stat, ks_p = sp_stats.ks_2samp(y_train, y_test)
                row['ks_statistic'] = float(ks_stat)
                row['ks_pvalue'] = float(ks_p)
            except Exception:
                row['ks_statistic'] = np.nan
                row['ks_pvalue'] = np.nan

            # Also compute for group means and deltas
            fdf_test = build_fold_df(
                df, y_test, y_test, test_idx  # dummy pred = true for structure
            )
            matched_test = filter_matched_groups(fdf_test)
            if len(matched_test) > 5:
                matched_test = add_group_means(matched_test)
                gm_test = matched_test.groupby('group_key')['y_true'].mean().values
                delta_test = matched_test['delta_true'].values
                row['test_gm_std'] = float(np.std(gm_test))
                row['test_delta_std'] = float(np.std(delta_test))
                row['test_mean_abs_delta'] = float(np.abs(delta_test).mean())
            else:
                row['test_gm_std'] = np.nan
                row['test_delta_std'] = np.nan
                row['test_mean_abs_delta'] = np.nan

            rows.append(row)

    ts_df = pd.DataFrame(rows)
    ts_df.to_csv(out_dir / 'target_shift_metrics.csv', index=False)
    print(f"  Step 9 (target shift) complete: {out_dir}")
    return ts_df
