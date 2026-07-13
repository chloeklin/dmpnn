"""Step 6: Within-group architecture ordering analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from itertools import combinations
import warnings

from .config import MODELS, SPLITS, TARGETS, N_FOLDS, STEP_DIRS
from .data_loading import load_predictions_single
from .grouping import build_fold_df, filter_matched_groups


def _group_ordering_metrics(group_df: pd.DataFrame) -> dict:
    """Compute ordering metrics for a single group.

    Tie-handling policy (statistically defensible):
    - Ties in y_true:  pair is uninformative → score 0.5 (no information lost)
    - Ties in y_pred:  model cannot order → score 0.5 (expected accuracy under
                       random breaking of pred tie)
    - Constant y_pred: scipy returns NaN for spearman/kendall; we propagate NaN
                       so that NaN-aware medians skip these groups rather than
                       penalising the model for being uninformative.
    - n=2 fallback:    use the same sign-product logic, but treat a pred-tie as
                       NaN (not −1) to match scipy's constant-array behaviour.
    """
    yt = group_df['y_true'].values
    yp = group_df['y_pred'].values
    n = len(yt)

    if n < 2:
        return {'pairwise_acc': np.nan, 'spearman': np.nan,
                'kendall': np.nan, 'exact_rank': np.nan, 'n_arch': n}

    # ── Pairwise ordering accuracy ─────────────────────────────────────────
    # Score per pair:
    #   concordant (correct direction)  → 1
    #   discordant (wrong direction)    → 0
    #   truth tied                      → 0.5 (pair is uninformative)
    #   pred tied, truth not tied       → 0.5 (model cannot order this pair)
    correct, total = 0, 0
    for i, j in combinations(range(n), 2):
        total += 1
        truth_diff = yt[i] - yt[j]
        pred_diff  = yp[i] - yp[j]
        if abs(truth_diff) < 1e-10:
            correct += 0.5                            # tie in truth
        elif abs(pred_diff) < 1e-10:
            correct += 0.5                            # tie in pred (FIX: was 0)
        elif truth_diff * pred_diff > 0:
            correct += 1                              # concordant
        # else discordant → 0
    pairwise_acc = correct / total if total > 0 else np.nan

    # ── Full ranking accuracy ──────────────────────────────────────────────
    true_order = np.argsort(yt)
    pred_order = np.argsort(yp)
    exact_rank = float(np.array_equal(true_order, pred_order))

    # ── Spearman and Kendall ───────────────────────────────────────────────
    # scipy handles ties natively and returns NaN when the input is constant.
    # For n=2 scipy is not called; we replicate its NaN-for-tied behaviour.
    if n >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            spearman = sp_stats.spearmanr(yt, yp).statistic
            kendall  = sp_stats.kendalltau(yt, yp).statistic
    else:
        # n==2: sign-product heuristic, but pred-tie → NaN (FIX: was -1)
        truth_diff = yt[0] - yt[1]
        pred_diff  = yp[0] - yp[1]
        if abs(pred_diff) < 1e-10:
            spearman = np.nan
            kendall  = np.nan
        elif truth_diff * pred_diff > 0:
            spearman = 1.0
            kendall  = 1.0
        else:
            spearman = -1.0
            kendall  = -1.0

    return {
        'pairwise_acc': float(pairwise_acc),
        'spearman': float(spearman) if not np.isnan(spearman) else np.nan,
        'kendall':  float(kendall)  if not np.isnan(kendall)  else np.nan,
        'exact_rank': exact_rank,
        'n_arch': n,
    }


def run_ordering(df: pd.DataFrame, meta: dict[str, list]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 6: Architecture ordering analysis.
    Saves ordering_metrics.csv and group_level_ordering.csv.
    """
    out_dir = STEP_DIRS['05_architecture_ordering']
    agg_rows = []
    group_rows = []

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

                    # Per-group ordering
                    group_metrics = []
                    for gkey, gdf in matched.groupby('group_key'):
                        gm = _group_ordering_metrics(gdf)
                        gm['group_key'] = gkey
                        gm['model'] = model
                        gm['split'] = split
                        gm['target'] = tkey
                        gm['fold'] = fold
                        group_metrics.append(gm)
                        group_rows.append(gm)

                    if not group_metrics:
                        continue

                    gm_arr = pd.DataFrame(group_metrics)
                    sp_vals  = gm_arr['spearman'].values.astype(float)
                    kd_vals  = gm_arr['kendall'].values.astype(float)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        med_sp = float(np.nanmedian(sp_vals))
                        med_kd = float(np.nanmedian(kd_vals))
                    agg_rows.append({
                        'model': model,
                        'split': split,
                        'target': tkey,
                        'fold': fold,
                        'exact_rank_acc': float(gm_arr['exact_rank'].mean()),
                        'pairwise_acc': float(gm_arr['pairwise_acc'].mean()),
                        'median_spearman': med_sp,
                        'median_kendall': med_kd,
                        'n_valid_groups': len(gm_arr),
                        'n_spearman_nan': int(gm_arr['spearman'].isna().sum()),
                    })

    agg_df = pd.DataFrame(agg_rows)
    grp_df = pd.DataFrame(group_rows)

    if len(agg_df) > 0:
        col_order = ['model', 'split', 'target', 'fold'] + [
            c for c in agg_df.columns if c not in ('model', 'split', 'target', 'fold')
        ]
        agg_df = agg_df[col_order]
    agg_df.to_csv(out_dir / 'ordering_metrics.csv', index=False)
    grp_df.to_csv(out_dir / 'group_level_ordering.csv', index=False)
    print(f"  Step 6 (ordering) complete: {out_dir}")
    return agg_df, grp_df
