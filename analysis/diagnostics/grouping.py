"""Group-level operations: group means, deviations, matched groups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_fold_df(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                  global_idx: np.ndarray) -> pd.DataFrame:
    """
    Build a working dataframe for one fold's test predictions.

    Returns DataFrame with columns:
        global_idx, group_key, poly_type, smilesA, smilesB, fracA, fracB,
        y_true, y_pred
    """
    rows = df.iloc[global_idx].copy()
    rows = rows.reset_index(drop=True)
    rows['global_idx'] = global_idx
    rows['y_true'] = y_true
    rows['y_pred'] = y_pred
    cols = ['global_idx', 'group_key', 'poly_type', 'smilesA', 'smilesB',
            'fracA', 'fracB', 'y_true', 'y_pred']
    return rows[cols].copy()


def add_group_means(fdf: pd.DataFrame) -> pd.DataFrame:
    """
    Add group-mean columns and deviation columns.

    Adds:
        y_bar_true  - true group mean
        y_bar_pred  - predicted group mean
        delta_true  - y_true - y_bar_true
        delta_pred  - y_pred - y_bar_pred
    """
    fdf = fdf.copy()
    fdf['y_bar_true'] = fdf.groupby('group_key')['y_true'].transform('mean')
    fdf['y_bar_pred'] = fdf.groupby('group_key')['y_pred'].transform('mean')
    fdf['delta_true'] = fdf['y_true'] - fdf['y_bar_true']
    fdf['delta_pred'] = fdf['y_pred'] - fdf['y_bar_pred']
    return fdf


def filter_matched_groups(fdf: pd.DataFrame, min_archs: int = 2) -> pd.DataFrame:
    """
    Keep only samples from groups with >= min_archs distinct architectures.
    """
    n_arch = fdf.groupby('group_key')['poly_type'].nunique()
    valid_groups = n_arch[n_arch >= min_archs].index
    return fdf[fdf['group_key'].isin(valid_groups)].copy()


def group_level_stats(fdf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one row per group with group-level statistics.

    Returns DataFrame with columns:
        group_key, n_group, n_archs, y_bar_true, y_bar_pred,
        group_mean_error, group_mean_abs_error, group_mean_sq_error,
        delta_true_std, delta_pred_std
    """
    fdf_m = add_group_means(fdf)
    grp = fdf_m.groupby('group_key')

    out = pd.DataFrame({
        'group_key':    grp['group_key'].first(),
        'n_group':      grp.size(),
        'n_archs':      grp['poly_type'].nunique(),
        'y_bar_true':   grp['y_true'].mean(),
        'y_bar_pred':   grp['y_pred'].mean(),
        'delta_true_std': grp['delta_true'].std(),
        'delta_pred_std': grp['delta_pred'].std(),
    }).reset_index(drop=True)

    out['group_mean_error']     = out['y_bar_pred'] - out['y_bar_true']
    out['group_mean_abs_error'] = out['group_mean_error'].abs()
    out['group_mean_sq_error']  = out['group_mean_error'] ** 2
    return out
