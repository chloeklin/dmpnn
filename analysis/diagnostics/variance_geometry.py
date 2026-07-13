"""Step 2: True variance geometry and Step 4: model error decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS, MODEL_DISPLAY
from .data_loading import load_predictions_single
from .grouping import build_fold_df, add_group_means, filter_matched_groups


def _variance_decomposition(fdf: pd.DataFrame) -> dict:
    """
    Exact SS decomposition for matched-group samples.
    Returns dict with SS_total, SS_between, SS_within, fractions, and descriptive stats.
    """
    matched = filter_matched_groups(fdf)
    if len(matched) == 0:
        return {k: np.nan for k in [
            'SS_total', 'SS_between', 'SS_within',
            'frac_between', 'frac_within',
            'sd_group_means', 'sd_delta_y',
            'ratio_sd_group_over_delta',
            'mean_abs_delta', 'median_abs_delta',
            'p5_abs_delta', 'p95_abs_delta', 'max_abs_delta',
            'n_samples', 'n_groups',
        ]}

    matched = add_group_means(matched)
    y = matched['y_true'].values
    global_mean = y.mean()
    y_bar_g = matched['y_bar_true'].values
    delta_y = matched['delta_true'].values

    SS_total = np.sum((y - global_mean) ** 2)
    grp_info = matched.groupby('group_key')['y_true'].agg(['mean', 'count'])
    SS_between = float((grp_info['count'] * (grp_info['mean'] - global_mean) ** 2).sum())
    SS_within = np.sum(delta_y ** 2)

    # Group-level SDs
    group_means = matched.groupby('group_key')['y_true'].mean().values
    sd_gm = float(np.std(group_means))
    sd_dy = float(np.std(delta_y))

    abs_delta = np.abs(delta_y)

    return {
        'SS_total': float(SS_total),
        'SS_between': float(SS_between),
        'SS_within': float(SS_within),
        'frac_between': float(SS_between / SS_total) if SS_total > 0 else np.nan,
        'frac_within': float(SS_within / SS_total) if SS_total > 0 else np.nan,
        'sd_group_means': sd_gm,
        'sd_delta_y': sd_dy,
        'ratio_sd_group_over_delta': sd_gm / sd_dy if sd_dy > 1e-12 else np.nan,
        'mean_abs_delta': float(abs_delta.mean()),
        'median_abs_delta': float(np.median(abs_delta)),
        'p5_abs_delta': float(np.percentile(abs_delta, 5)),
        'p95_abs_delta': float(np.percentile(abs_delta, 95)),
        'max_abs_delta': float(abs_delta.max()),
        'n_samples': len(matched),
        'n_groups': matched['group_key'].nunique(),
    }


def run_variance_geometry(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Step 2: Compute variance geometry for each split/target/fold.
    Saves variance_geometry.csv.
    """
    out_dir = STEP_DIRS['02_variance_geometry']
    rows = []

    for split in SPLITS:
        meta_folds = meta[split]
        for tkey in TARGETS:
            for fold in range(N_FOLDS[split]):
                # Use any available model to get test indices
                pred = load_predictions_single('frac', tkey, split, fold, meta_folds)
                if pred is None:
                    pred = load_predictions_single('wdmpnn', tkey, split, fold, meta_folds)
                if pred is None:
                    continue

                fdf = build_fold_df(df, pred['y_true'], pred['y_pred'], pred['global_idx'])
                vd = _variance_decomposition(fdf)
                vd['split'] = split
                vd['target'] = tkey
                vd['fold'] = fold
                rows.append(vd)

    vg_df = pd.DataFrame(rows)
    col_order = ['split', 'target', 'fold'] + [
        c for c in vg_df.columns if c not in ('split', 'target', 'fold')
    ]
    vg_df = vg_df[col_order]
    vg_df.to_csv(out_dir / 'variance_geometry.csv', index=False)
    print(f"  Step 2 (variance geometry) complete: {out_dir}")
    return vg_df


def _error_decomposition(fdf: pd.DataFrame) -> dict:
    """
    Decompose a model's prediction error into between-group and within-group.
    Requires fdf to have y_true, y_pred columns.
    """
    matched = filter_matched_groups(fdf)
    if len(matched) == 0:
        return {k: np.nan for k in [
            'SSE_total', 'SSE_between', 'SSE_within',
            'frac_SSE_between', 'frac_SSE_within',
            'MSE_between', 'MSE_within',
            'n_samples', 'n_groups',
        ]}

    matched = add_group_means(matched)
    yt = matched['y_true'].values
    yp = matched['y_pred'].values
    y_bar_t = matched['y_bar_true'].values
    y_bar_p = matched['y_bar_pred'].values
    delta_t = matched['delta_true'].values
    delta_p = matched['delta_pred'].values

    SSE_total = float(np.sum((yt - yp) ** 2))

    # Between-group SSE: sum_g n_g * (y_bar_g_true - y_bar_g_pred)^2
    grp_stats = matched.groupby('group_key').agg(
        n_g=('y_true', 'size'),
        mean_true=('y_true', 'mean'),
        mean_pred=('y_pred', 'mean'),
    )
    SSE_between = float(
        (grp_stats['n_g'] * (grp_stats['mean_true'] - grp_stats['mean_pred']) ** 2).sum()
    )

    # Within-group SSE: sum (delta_t - delta_p)^2
    SSE_within = float(np.sum((delta_t - delta_p) ** 2))

    n = len(matched)

    return {
        'SSE_total': SSE_total,
        'SSE_between': SSE_between,
        'SSE_within': SSE_within,
        'frac_SSE_between': SSE_between / SSE_total if SSE_total > 0 else np.nan,
        'frac_SSE_within': SSE_within / SSE_total if SSE_total > 0 else np.nan,
        'MSE_between': SSE_between / n if n > 0 else np.nan,
        'MSE_within': SSE_within / n if n > 0 else np.nan,
        'n_samples': n,
        'n_groups': int(matched['group_key'].nunique()),
    }


def run_error_decomposition(df: pd.DataFrame, meta: dict[str, list]) -> pd.DataFrame:
    """
    Step 4: Decompose each model's error into between-group and within-group.
    Saves model_error_decomposition.csv.
    """
    out_dir = STEP_DIRS['02_variance_geometry']
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
                    ed = _error_decomposition(fdf)
                    ed['model'] = model
                    ed['split'] = split
                    ed['target'] = tkey
                    ed['fold'] = fold
                    rows.append(ed)

    ed_df = pd.DataFrame(rows)
    col_order = ['model', 'split', 'target', 'fold'] + [
        c for c in ed_df.columns if c not in ('model', 'split', 'target', 'fold')
    ]
    ed_df = ed_df[col_order]
    ed_df.to_csv(out_dir / 'model_error_decomposition.csv', index=False)
    print(f"  Step 4 (error decomposition) complete: {out_dir}")
    return ed_df
