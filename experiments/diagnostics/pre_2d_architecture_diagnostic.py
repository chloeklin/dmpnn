#!/usr/bin/env python3
"""Pre-2D architecture diagnostic: evaluate whether polymer architecture
(alternating / random / block) explains meaningful EA/IP variance beyond
monomer identity and composition.

Usage:
    python pre_2d_architecture_diagnostic.py --data path/to/data.csv --out pre_2d_diagnostics
    python pre_2d_architecture_diagnostic.py --data path/to/data.csv --out pre_2d_diagnostics --plot
    python pre_2d_architecture_diagnostic.py --data path/to/data.csv --frac-pred preds.csv --out pre_2d_diagnostics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


# ══════════════════════════════════════════════════════════════════════════════
# Column name inference
# ══════════════════════════════════════════════════════════════════════════════

# Map of canonical name -> list of acceptable alternatives (case-insensitive)
COLUMN_ALIASES = {
    "smiles_A": ["smiles_a", "monomer_a", "a_smiles", "smilesA", "smilesa"],
    "smiles_B": ["smiles_b", "monomer_b", "b_smiles", "smilesB", "smilesb"],
    "fracA": ["fraca", "frac_a", "fraction_a", "x_a"],
    "fracB": ["fracb", "frac_b", "fraction_b", "x_b"],
    "poly_type": ["poly_type", "polymer_type", "architecture", "arch", "polytype"],
    "EA": ["ea", "ea vs she (ev)", "ea_vs_she", "ea vs she"],
    "IP": ["ip", "ip vs she (ev)", "ip_vs_she", "ip vs she"],
    "fold": ["fold", "cv_fold", "split", "Fold"],
}


def infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Infer canonical column names from the dataframe.

    Returns a dict mapping canonical_name -> actual column name in df.
    Raises ValueError if a required column cannot be found.
    """
    col_lower_map = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        # First try exact match (case-insensitive)
        for alias in aliases:
            if alias.lower() in col_lower_map:
                found = col_lower_map[alias.lower()]
                break
        # Also try the canonical name itself
        if found is None and canonical.lower() in col_lower_map:
            found = col_lower_map[canonical.lower()]
        mapping[canonical] = found

    # Validate required columns
    required = ["smiles_A", "smiles_B", "fracA", "fracB", "poly_type", "EA", "IP"]
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise ValueError(
            f"Could not infer required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected aliases: {[(k, COLUMN_ALIASES[k]) for k in missing]}"
        )

    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 1 — Matched architecture group detection
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic1_matched_groups(df: pd.DataFrame, col_map: Dict[str, str]) -> Dict:
    """Group rows by (monomer_A, monomer_B, fracA, fracB) and count architecture coverage."""
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]

    group_cols = [sA, sB, fA, fB]
    grouped = df.groupby(group_cols)[arch].nunique().reset_index(name="n_arch")

    total_rows = len(df)
    total_groups = len(grouped)
    groups_ge2 = int((grouped["n_arch"] >= 2).sum())
    groups_all3 = int((grouped["n_arch"] >= 3).sum())

    # Count matched rows (rows belonging to groups with >= 2 architectures)
    matched_group_keys = grouped[grouped["n_arch"] >= 2][group_cols]
    df_matched = df.merge(matched_group_keys, on=group_cols, how="inner")
    matched_rows = len(df_matched)
    pct_covered = 100.0 * matched_rows / total_rows if total_rows > 0 else 0.0

    results = {
        "total_rows": total_rows,
        "total_groups": total_groups,
        "groups_with_ge2_arch": groups_ge2,
        "groups_with_all3_arch": groups_all3,
        "matched_rows": matched_rows,
        "pct_matched": pct_covered,
    }

    # Save group counts
    grouped.to_csv(OUT_DIR / "matched_group_counts.csv", index=False)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 2 — Within-group architecture variance
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic2_within_group_variance(df: pd.DataFrame, col_map: Dict[str, str]) -> Dict:
    """Compute within-group variance of EA/IP across architectures."""
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]

    group_cols = [sA, sB, fA, fB]

    # Filter to groups with >= 2 architectures
    arch_counts = df.groupby(group_cols)[arch].nunique()
    matched_keys = arch_counts[arch_counts >= 2].index
    df_matched = df.set_index(group_cols).loc[matched_keys].reset_index()

    if len(df_matched) == 0:
        return {
            "EA_mean_within_var": np.nan, "EA_median_within_var": np.nan,
            "EA_total_var": np.nan, "EA_arch_var_ratio": np.nan,
            "IP_mean_within_var": np.nan, "IP_median_within_var": np.nan,
            "IP_total_var": np.nan, "IP_arch_var_ratio": np.nan,
        }

    results = {}
    for target, t_col in [("EA", ea_col), ("IP", ip_col)]:
        # Within-group variance (variance of target across architectures within each group)
        within_vars = df_matched.groupby(group_cols)[t_col].var(ddof=0)
        # Drop groups with only 1 sample (var = NaN or 0)
        within_vars = within_vars.dropna()

        mean_wv = float(within_vars.mean())
        median_wv = float(within_vars.median())
        total_var = float(df[t_col].var(ddof=0))
        ratio = mean_wv / total_var if total_var > 0 else np.nan

        results[f"{target}_mean_within_var"] = mean_wv
        results[f"{target}_median_within_var"] = median_wv
        results[f"{target}_total_var"] = total_var
        results[f"{target}_arch_var_ratio"] = ratio

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 3 — Architecture effect size using residual model
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic3_architecture_effect(df: pd.DataFrame, col_map: Dict[str, str]) -> Dict:
    """Compare baseline (monomer_pair + frac) vs architecture-augmented Ridge models."""
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]

    # Create monomer pair identifier (order-invariant)
    pair_id = df.apply(
        lambda row: "|".join(sorted([str(row[sA]), str(row[sB])])), axis=1
    )

    # One-hot encode monomer pair
    pair_enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_pair = pair_enc.fit_transform(pair_id.values.reshape(-1, 1))

    # Fraction features
    X_frac = df[[fA, fB]].values

    # Architecture one-hot
    arch_enc = OneHotEncoder(sparse_output=True, drop="first", handle_unknown="ignore")
    X_arch = arch_enc.fit_transform(df[[arch]].values)

    # Baseline: monomer_pair + frac
    from scipy.sparse import hstack as sp_hstack
    X_baseline = sp_hstack([X_pair, X_frac])

    # Architecture model: monomer_pair + frac + architecture
    X_full = sp_hstack([X_pair, X_frac, X_arch])

    results = {}
    for target, t_col in [("EA", ea_col), ("IP", ip_col)]:
        y = df[t_col].values

        # Baseline Ridge
        reg_base = Ridge(alpha=1.0)
        reg_base.fit(X_baseline, y)
        y_pred_base = reg_base.predict(X_baseline)
        r2_base = r2_score(y, y_pred_base)
        rmse_base = np.sqrt(mean_squared_error(y, y_pred_base))

        # Architecture Ridge
        reg_full = Ridge(alpha=1.0)
        reg_full.fit(X_full, y)
        y_pred_full = reg_full.predict(X_full)
        r2_full = r2_score(y, y_pred_full)
        rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))

        results[f"{target}_baseline_R2"] = r2_base
        results[f"{target}_arch_R2"] = r2_full
        results[f"{target}_delta_R2"] = r2_full - r2_base
        results[f"{target}_baseline_RMSE"] = rmse_base
        results[f"{target}_arch_RMSE"] = rmse_full
        results[f"{target}_delta_RMSE"] = rmse_base - rmse_full

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 4 — Fold-aware analysis (optional)
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic4_fold_aware(df: pd.DataFrame, col_map: Dict[str, str]) -> Optional[pd.DataFrame]:
    """If a fold column exists, repeat Diagnostic 3 per fold."""
    fold_col = col_map.get("fold")
    if fold_col is None:
        return None

    folds = sorted(df[fold_col].unique())
    rows = []
    for f in folds:
        df_fold = df[df[fold_col] == f].reset_index(drop=True)
        if len(df_fold) < 50:
            continue
        try:
            res = diagnostic3_architecture_effect(df_fold, col_map)
            res["fold"] = f
            rows.append(res)
        except Exception:
            continue

    if not rows:
        return None
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 5 — Architecture-specific paired differences
# ══════════════════════════════════════════════════════════════════════════════

def diagnostic5_paired_differences(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """Compute pairwise absolute property differences between architecture types
    within matched groups."""
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]

    group_cols = [sA, sB, fA, fB]
    pairs = [("alternating", "random"), ("alternating", "block"), ("random", "block")]

    rows = []
    for arch1, arch2 in pairs:
        df1 = df[df[arch] == arch1].set_index(group_cols)
        df2 = df[df[arch] == arch2].set_index(group_cols)
        common = df1.index.intersection(df2.index)

        if len(common) == 0:
            for target in ["EA", "IP"]:
                rows.append({
                    "pair": f"{arch1} vs {arch2}",
                    "target": target,
                    "count": 0,
                    "mean_abs_diff": np.nan,
                    "median_abs_diff": np.nan,
                    "std_abs_diff": np.nan,
                })
            continue

        for target, t_col in [("EA", ea_col), ("IP", ip_col)]:
            # For groups with multiple rows per architecture, take the mean
            vals1 = df1.loc[common, t_col].groupby(level=list(range(len(group_cols)))).mean()
            vals2 = df2.loc[common, t_col].groupby(level=list(range(len(group_cols)))).mean()
            # Align
            common_idx = vals1.index.intersection(vals2.index)
            diffs = np.abs(vals1.loc[common_idx].values - vals2.loc[common_idx].values)

            rows.append({
                "pair": f"{arch1} vs {arch2}",
                "target": target,
                "count": len(diffs),
                "mean_abs_diff": float(np.mean(diffs)) if len(diffs) > 0 else np.nan,
                "median_abs_diff": float(np.median(diffs)) if len(diffs) > 0 else np.nan,
                "std_abs_diff": float(np.std(diffs)) if len(diffs) > 0 else np.nan,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 6 — Decision rule
# ══════════════════════════════════════════════════════════════════════════════

def interpret_results(d3_results: Dict, d2_results: Dict) -> List[str]:
    """Generate interpretation lines based on decision rules."""
    lines = []
    for target in ["EA", "IP"]:
        dr2 = d3_results[f"{target}_delta_R2"]
        var_ratio = d2_results[f"{target}_arch_var_ratio"]
        lines.append(f"\n  {target}:")
        lines.append(f"    delta_R2 = {dr2:.6f}, arch_variance_ratio = {var_ratio:.6f}")

        if dr2 < 0.01 and var_ratio < 0.01:
            lines.append(
                "    → Architecture signal is likely too small to justify Stage 2D for this target."
            )
        elif 0.01 <= dr2 < 0.03:
            lines.append(
                "    → Architecture signal is weak but potentially diagnosable; "
                "use targeted within-group metrics, not global RMSE."
            )
        elif dr2 >= 0.03:
            lines.append(
                "    → Architecture signal may justify Stage 2D modeling."
            )
        else:
            # dr2 >= 0.01 but var_ratio >= 0.01
            lines.append(
                "    → Architecture signal is weak but potentially diagnosable; "
                "use targeted within-group metrics, not global RMSE."
            )

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC — FRAC + ARCHITECTURE ORACLE HEADROOM
# ══════════════════════════════════════════════════════════════════════════════

# Prediction column aliases for robust inference from CSV
PRED_COLUMN_ALIASES = {
    "EA_pred": ["ea_pred", "pred_ea", "ea_predicted", "ea vs she (ev)_pred", "ea_vs_she_pred"],
    "IP_pred": ["ip_pred", "pred_ip", "ip_predicted", "ip vs she (ev)_pred", "ip_vs_she_pred"],
}


def _infer_pred_columns(pred_df: pd.DataFrame) -> Dict[str, str]:
    """Infer prediction column names from the prediction CSV."""
    col_lower_map = {c.lower().strip(): c for c in pred_df.columns}
    mapping = {}
    for canonical, aliases in PRED_COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias.lower() in col_lower_map:
                found = col_lower_map[alias.lower()]
                break
        if found is None and canonical.lower() in col_lower_map:
            found = col_lower_map[canonical.lower()]
        mapping[canonical] = found
    return mapping


def load_frac_predictions(
    frac_pred_path: Optional[str],
    n_rows: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Load Frac model predictions from a CSV file.

    The CSV must have columns mappable to EA_pred and IP_pred,
    with rows aligned to the main dataset (same ordering, same length).

    Returns dict {"EA": array, "IP": array} or None if unavailable.
    """
    if frac_pred_path is None:
        return None

    path = Path(frac_pred_path)
    if not path.exists():
        print(f"  WARNING: --frac-pred file not found: {path}")
        return None

    pred_df = pd.read_csv(path)
    if len(pred_df) != n_rows:
        print(f"  WARNING: Prediction CSV has {len(pred_df)} rows but dataset has {n_rows} rows. Skipping oracle.")
        return None

    col_map = _infer_pred_columns(pred_df)
    if col_map["EA_pred"] is None or col_map["IP_pred"] is None:
        print(f"  WARNING: Could not find EA_pred/IP_pred columns in {path}.")
        print(f"  Available columns: {list(pred_df.columns)}")
        return None

    return {
        "EA": pred_df[col_map["EA_pred"]].values,
        "IP": pred_df[col_map["IP_pred"]].values,
    }


def load_frac_predictions_from_npz(
    predictions_dir: Path,
    splits_json: Path,
    n_rows: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Auto-load Frac predictions from per-fold .npz files.

    Uses fold assignments to reconstruct full-dataset prediction arrays.
    Returns dict {"EA": array, "IP": array} or None.
    """
    import json

    if not predictions_dir.exists() or not splits_json.exists():
        return None

    with open(splits_json) as f:
        splits = json.load(f)

    targets_map = {
        "EA": "EA vs SHE (eV)",
        "IP": "IP vs SHE (eV)",
    }
    pattern = "ea_ip__{target}__copoly_mix_meta__poly_type__a_held_out__split{fold}.npz"

    preds = {}
    for short, target_full in targets_map.items():
        arr = np.full(n_rows, np.nan)
        all_found = True
        for fold_info in splits["folds"]:
            fold_idx = fold_info["fold"]
            fname = pattern.format(target=target_full, fold=fold_idx)
            fpath = predictions_dir / fname
            if not fpath.exists():
                all_found = False
                break
            data = np.load(fpath, allow_pickle=True)
            y_pred = data["y_pred"].squeeze()
            test_indices = np.array(fold_info["test_indices"])
            if len(y_pred) != len(test_indices):
                all_found = False
                break
            arr[test_indices] = y_pred

        if not all_found:
            return None
        preds[short] = arr

    # Check coverage
    if np.isnan(preds["EA"]).any() or np.isnan(preds["IP"]).any():
        # Some rows not covered by any fold (shouldn't happen with full 5-fold)
        n_missing = int(np.isnan(preds["EA"]).sum())
        print(f"  WARNING: {n_missing} rows not covered by fold predictions.")
        return None

    return preds


def _compute_oracle_correction(
    residuals: np.ndarray,
    group_ids: np.ndarray,
    arch_labels: np.ndarray,
) -> np.ndarray:
    """Compute oracle-corrected predictions by adding group×architecture mean residual.

    For each (group, architecture) combination, computes the mean residual
    and applies it as an additive correction.

    Returns the correction array (same length as inputs).
    """
    df_temp = pd.DataFrame({
        "group": group_ids,
        "arch": arch_labels,
        "residual": residuals,
    })
    # Vectorized: compute group×arch means and map back via transform
    corrections = df_temp.groupby(["group", "arch"])["residual"].transform("mean").values
    return corrections


def _compute_oracle_correction_fold_safe(
    residuals: np.ndarray,
    group_ids: np.ndarray,
    arch_labels: np.ndarray,
    fold_assignments: np.ndarray,
) -> np.ndarray:
    """Compute oracle corrections using leave-one-fold-out to avoid leakage.

    For each fold's test set, corrections are computed using only
    training-fold data (all other folds).
    """
    corrections = np.zeros_like(residuals)
    folds = np.unique(fold_assignments)

    df_temp = pd.DataFrame({
        "group": group_ids,
        "arch": arch_labels,
        "residual": residuals,
        "fold": fold_assignments,
    })

    for fold in folds:
        train_mask = df_temp["fold"] != fold
        test_mask = df_temp["fold"] == fold

        # Compute group×arch mean corrections from training folds only
        train_means = (
            df_temp[train_mask]
            .groupby(["group", "arch"])["residual"]
            .mean()
            .reset_index()
            .rename(columns={"residual": "_correction"})
        )

        # Merge corrections onto test rows (unmatched get NaN -> fill with 0)
        test_df = df_temp[test_mask][["group", "arch"]].copy()
        test_df = test_df.merge(train_means, on=["group", "arch"], how="left")
        corrections[test_mask.values] = test_df["_correction"].fillna(0.0).values

    return corrections


def diagnostic_oracle_headroom(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    frac_preds: Dict[str, np.ndarray],
    d2_results: Dict,
) -> Dict:
    """Estimate maximum recoverable architecture-aware improvement beyond Frac.

    The Frac model predicts identical values for all architectures within a
    composition group because h_poly = Σ f_i·h_i. Any architecture-conditioned
    variance appears entirely in the residuals.

    This diagnostic computes:
    - Optimistic oracle: group×arch mean correction (all data)
    - Fold-safe oracle: corrections computed on training folds only
    """
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch_col = col_map["poly_type"]
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]
    fold_col = col_map.get("fold")

    # Build group IDs for efficient lookup
    group_ids = (
        df[sA].astype(str) + "|" +
        df[sB].astype(str) + "|" +
        df[fA].astype(str) + "|" +
        df[fB].astype(str)
    ).values
    arch_labels = df[arch_col].values

    results = {}
    oracle_preds_all = {}  # Store for saving
    group_offsets_rows = []

    target_cols = {"EA": ea_col, "IP": ip_col}

    for short, t_col in target_cols.items():
        y_true = df[t_col].values
        y_frac = frac_preds[short]

        # Step 1: Residuals
        residuals = y_true - y_frac

        # Frac metrics
        frac_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        frac_r2 = float(1.0 - np.sum(residuals ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        residual_var = float(np.var(residuals))

        # Step 3A: Optimistic oracle correction (uses all data)
        corrections_opt = _compute_oracle_correction(residuals, group_ids, arch_labels)
        y_oracle_opt = y_frac + corrections_opt
        residuals_opt = y_true - y_oracle_opt
        oracle_rmse_opt = float(np.sqrt(np.mean(residuals_opt ** 2)))
        oracle_r2_opt = float(1.0 - np.sum(residuals_opt ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        residual_var_after_opt = float(np.var(residuals_opt))

        # Headroom metrics (optimistic)
        abs_rmse_reduction_opt = frac_rmse - oracle_rmse_opt
        rel_rmse_reduction_opt = 100.0 * abs_rmse_reduction_opt / frac_rmse if frac_rmse > 0 else 0.0
        var_removed_opt = 1.0 - (residual_var_after_opt / residual_var) if residual_var > 0 else 0.0

        results[f"{short}_frac_RMSE"] = frac_rmse
        results[f"{short}_frac_R2"] = frac_r2
        results[f"{short}_oracle_opt_RMSE"] = oracle_rmse_opt
        results[f"{short}_oracle_opt_R2"] = oracle_r2_opt
        results[f"{short}_abs_RMSE_reduction_opt"] = abs_rmse_reduction_opt
        results[f"{short}_rel_RMSE_reduction_opt"] = rel_rmse_reduction_opt
        results[f"{short}_residual_var_before"] = residual_var
        results[f"{short}_residual_var_after_opt"] = residual_var_after_opt
        results[f"{short}_var_removed_opt"] = var_removed_opt

        # Architecture share of residual
        arch_var_ratio = d2_results.get(f"{short}_arch_var_ratio", 0.0)
        arch_share = arch_var_ratio / (1.0 - frac_r2) if frac_r2 < 1.0 else 0.0
        results[f"{short}_arch_share_of_residual"] = arch_share

        # Step 3B: Fold-safe oracle (if fold column available)
        if fold_col is not None:
            fold_assignments = df[fold_col].values
            corrections_safe = _compute_oracle_correction_fold_safe(
                residuals, group_ids, arch_labels, fold_assignments
            )
            y_oracle_safe = y_frac + corrections_safe
            residuals_safe = y_true - y_oracle_safe
            oracle_rmse_safe = float(np.sqrt(np.mean(residuals_safe ** 2)))
            oracle_r2_safe = float(
                1.0 - np.sum(residuals_safe ** 2) / np.sum((y_true - y_true.mean()) ** 2)
            )
            residual_var_after_safe = float(np.var(residuals_safe))
            abs_rmse_reduction_safe = frac_rmse - oracle_rmse_safe
            rel_rmse_reduction_safe = 100.0 * abs_rmse_reduction_safe / frac_rmse if frac_rmse > 0 else 0.0
            var_removed_safe = 1.0 - (residual_var_after_safe / residual_var) if residual_var > 0 else 0.0

            results[f"{short}_oracle_safe_RMSE"] = oracle_rmse_safe
            results[f"{short}_oracle_safe_R2"] = oracle_r2_safe
            results[f"{short}_abs_RMSE_reduction_safe"] = abs_rmse_reduction_safe
            results[f"{short}_rel_RMSE_reduction_safe"] = rel_rmse_reduction_safe
            results[f"{short}_var_removed_safe"] = var_removed_safe

            oracle_preds_all[f"{short}_oracle_safe"] = y_oracle_safe
        else:
            results[f"{short}_oracle_safe_RMSE"] = np.nan
            results[f"{short}_oracle_safe_R2"] = np.nan
            results[f"{short}_abs_RMSE_reduction_safe"] = np.nan
            results[f"{short}_rel_RMSE_reduction_safe"] = np.nan
            results[f"{short}_var_removed_safe"] = np.nan

        oracle_preds_all[f"{short}_true"] = y_true
        oracle_preds_all[f"{short}_frac_pred"] = y_frac
        oracle_preds_all[f"{short}_oracle_opt"] = y_oracle_opt

        # Collect group offsets
        df_offsets = pd.DataFrame({
            "group": group_ids,
            "arch": arch_labels,
            "residual": residuals,
        })
        offset_means = df_offsets.groupby(["group", "arch"])["residual"].agg(["mean", "count"]).reset_index()
        offset_means.columns = ["group", "arch", f"{short}_mean_offset", f"{short}_count"]
        if len(group_offsets_rows) == 0:
            group_offsets_rows.append(offset_means)
        else:
            group_offsets_rows[0] = group_offsets_rows[0].merge(
                offset_means, on=["group", "arch"], how="outer"
            )

    # Save outputs
    # Oracle predictions CSV
    pred_df = pd.DataFrame(oracle_preds_all)
    pred_df.to_csv(OUT_DIR / "oracle_predictions.csv", index=False)

    # Group offsets CSV
    if group_offsets_rows:
        group_offsets_rows[0].to_csv(OUT_DIR / "oracle_group_offsets.csv", index=False)

    # Summary metrics CSV
    pd.DataFrame([results]).to_csv(OUT_DIR / "oracle_summary_metrics.csv", index=False)

    return results


def print_oracle_report(results: Dict) -> List[str]:
    """Print and return formatted oracle headroom report."""
    lines = []

    # Table 1: RMSE comparison
    lines.append("\n  ┌──────────┬────────────┬─────────────────┬─────────────────┬───────────┬───────────┬──────────────────────────┐")
    lines.append("  │ Target   │ Frac RMSE  │ Oracle RMSE     │ Oracle RMSE     │ Abs ΔRMSE │ Rel ΔRMSE │ Residual Var Removed     │")
    lines.append("  │          │            │ (optimistic)    │ (fold-safe)     │ (opt)     │ (opt)     │ (optimistic)             │")
    lines.append("  ├──────────┼────────────┼─────────────────┼─────────────────┼───────────┼───────────┼──────────────────────────┤")
    for short in ["EA", "IP"]:
        frac_rmse = results[f"{short}_frac_RMSE"]
        oracle_opt = results[f"{short}_oracle_opt_RMSE"]
        oracle_safe = results.get(f"{short}_oracle_safe_RMSE", np.nan)
        abs_dr = results[f"{short}_abs_RMSE_reduction_opt"]
        rel_dr = results[f"{short}_rel_RMSE_reduction_opt"]
        var_rem = results[f"{short}_var_removed_opt"]
        safe_str = f"{oracle_safe:.6f}" if not np.isnan(oracle_safe) else "N/A"
        lines.append(
            f"  │ {short:<8} │ {frac_rmse:.6f}   │ {oracle_opt:.6f}        │ {safe_str:<15} │ {abs_dr:.6f}  │ {rel_dr:5.1f}%    │ {var_rem:.4f} ({var_rem*100:.1f}%)            │"
        )
    lines.append("  └──────────┴────────────┴─────────────────┴─────────────────┴───────────┴───────────┴──────────────────────────┘")

    # Table 2: Architecture share
    lines.append("\n  ┌──────────┬─────────────────────┬──────────────────┬─────────────────────────────┐")
    lines.append("  │ Target   │ Arch Variance Ratio │ Residual Var     │ Arch Share of Residual      │")
    lines.append("  ├──────────┼─────────────────────┼──────────────────┼─────────────────────────────┤")
    for short in ["EA", "IP"]:
        res_var = results[f"{short}_residual_var_before"]
        arch_share = results[f"{short}_arch_share_of_residual"]
        # Get arch var ratio from stored results
        lines.append(
            f"  │ {short:<8} │ —                   │ {res_var:.8f}       │ {arch_share:.4f} ({arch_share*100:.1f}%)                │"
        )
    lines.append("  └──────────┴─────────────────────┴──────────────────┴─────────────────────────────┘")

    # Interpretation
    lines.append("\n  Interpretation:")
    for short in ["EA", "IP"]:
        arch_share = results[f"{short}_arch_share_of_residual"]
        rel_reduction_opt = results[f"{short}_rel_RMSE_reduction_opt"]
        oracle_safe_rmse = results.get(f"{short}_oracle_safe_RMSE", np.nan)
        oracle_opt_rmse = results[f"{short}_oracle_opt_RMSE"]

        lines.append(f"\n    {short}:")

        if arch_share > 0.3:
            lines.append(
                "      → Architecture contributes a substantial fraction of remaining Frac error."
            )
        elif arch_share > 0.1:
            lines.append(
                "      → Architecture contributes a moderate fraction of remaining Frac error."
            )
        else:
            lines.append(
                "      → Architecture contributes a minor fraction of remaining Frac error."
            )

        if rel_reduction_opt > 20:
            lines.append(
                "      → Architecture-aware modeling may provide meaningful global RMSE improvement."
            )
        elif rel_reduction_opt > 5:
            lines.append(
                "      → Architecture-aware modeling could provide modest RMSE improvement."
            )
        else:
            lines.append(
                "      → Architecture-aware modeling unlikely to provide large global RMSE gains."
            )

        # Fold-safe vs optimistic collapse check
        if not np.isnan(oracle_safe_rmse) and oracle_opt_rmse > 0:
            safe_reduction = results[f"{short}_abs_RMSE_reduction_safe"]
            opt_reduction = results[f"{short}_abs_RMSE_reduction_opt"]
            if opt_reduction > 0:
                collapse_ratio = safe_reduction / opt_reduction
                if collapse_ratio < 0.3:
                    lines.append(
                        f"      → ⚠️  Fold-safe oracle collapses strongly vs optimistic "
                        f"({collapse_ratio:.0%} retained). Architecture effects may be difficult "
                        f"to generalize under monomer-disjoint evaluation."
                    )
                elif collapse_ratio < 0.7:
                    lines.append(
                        f"      → Fold-safe oracle retains {collapse_ratio:.0%} of optimistic gain. "
                        f"Partial generalization expected."
                    )
                else:
                    lines.append(
                        f"      → Fold-safe oracle retains {collapse_ratio:.0%} of optimistic gain. "
                        f"Architecture effects generalize well across folds."
                    )

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Plotting (optional, only with --plot)
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(df: pd.DataFrame, col_map: Dict[str, str]):
    """Generate boxplots of pairwise architecture differences for EA and IP."""
    import matplotlib.pyplot as plt

    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]

    group_cols = [sA, sB, fA, fB]
    pairs = [("alternating", "random"), ("alternating", "block"), ("random", "block")]

    for target, t_col in [("EA", ea_col), ("IP", ip_col)]:
        diff_data = {}
        for arch1, arch2 in pairs:
            df1 = df[df[arch] == arch1].set_index(group_cols)
            df2 = df[df[arch] == arch2].set_index(group_cols)
            common = df1.index.intersection(df2.index)
            if len(common) == 0:
                diff_data[f"{arch1}\nvs\n{arch2}"] = []
                continue
            vals1 = df1.loc[common, t_col].groupby(level=list(range(len(group_cols)))).mean()
            vals2 = df2.loc[common, t_col].groupby(level=list(range(len(group_cols)))).mean()
            common_idx = vals1.index.intersection(vals2.index)
            diffs = np.abs(vals1.loc[common_idx].values - vals2.loc[common_idx].values)
            diff_data[f"{arch1}\nvs\n{arch2}"] = diffs

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(diff_data.keys())
        data = [diff_data[k] for k in labels]
        # Filter out empty
        valid = [(l, d) for l, d in zip(labels, data) if len(d) > 0]
        if valid:
            ax.boxplot([d for _, d in valid], tick_labels=[l for l, _ in valid],
                       patch_artist=True,
                       boxprops=dict(facecolor="#42a5f5", alpha=0.7))
        ax.set_ylabel(f"|Δ{target}| (eV)")
        ax.set_title(f"Pairwise architecture differences — {target}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"pairwise_boxplot_{target}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {OUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description="Pre-2D architecture diagnostic")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default="pre_2d_diagnostics", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate boxplots")
    parser.add_argument(
        "--frac-pred", type=str, default=None,
        help="Path to Frac prediction CSV with EA_pred/IP_pred columns (aligned to data rows)"
    )
    parser.add_argument(
        "--auto-npz", action="store_true",
        help="Auto-load Frac predictions from per-fold .npz files in predictions/DMPNN/"
    )
    args = parser.parse_args()

    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Shape: {df.shape}")

    # Infer columns
    col_map = infer_columns(df)
    print(f"  Column mapping:")
    for k, v in col_map.items():
        if v is not None:
            print(f"    {k} -> '{v}'")

    # Drop rows with missing targets
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]
    n_before = len(df)
    df = df.dropna(subset=[ea_col, ip_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing targets")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PRE-2D ARCHITECTURE DIAGNOSTIC REPORT")
    report_lines.append("=" * 70)

    # ── Diagnostic 1 ──
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1 — Matched architecture group detection")
    print("=" * 70)
    d1 = diagnostic1_matched_groups(df, col_map)
    for k, v in d1.items():
        val_str = f"{v:.1f}%" if "pct" in k else str(v)
        print(f"  {k}: {val_str}")
    report_lines.append("\nDIAGNOSTIC 1 — Matched architecture group detection")
    for k, v in d1.items():
        report_lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Diagnostic 2 ──
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2 — Within-group architecture variance")
    print("=" * 70)
    d2 = diagnostic2_within_group_variance(df, col_map)
    for k, v in d2.items():
        print(f"  {k}: {v:.8f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
    report_lines.append("\nDIAGNOSTIC 2 — Within-group architecture variance")
    for k, v in d2.items():
        report_lines.append(f"  {k}: {v:.8f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")

    # ── Diagnostic 3 ──
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3 — Architecture effect size (Ridge regression)")
    print("=" * 70)
    d3 = diagnostic3_architecture_effect(df, col_map)
    for k, v in d3.items():
        print(f"  {k}: {v:.6f}")
    report_lines.append("\nDIAGNOSTIC 3 — Architecture effect size (Ridge regression)")
    for k, v in d3.items():
        report_lines.append(f"  {k}: {v:.6f}")

    # ── Diagnostic 4 ──
    d4_df = diagnostic4_fold_aware(df, col_map)
    if d4_df is not None:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 4 — Fold-aware analysis")
        print("=" * 70)
        print(d4_df.to_string(index=False))
        d4_df.to_csv(OUT_DIR / "fold_aware_results.csv", index=False)
        report_lines.append("\nDIAGNOSTIC 4 — Fold-aware analysis")
        report_lines.append(d4_df.to_string(index=False))

    # ── Diagnostic 5 ──
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 5 — Architecture-specific paired differences")
    print("=" * 70)
    d5 = diagnostic5_paired_differences(df, col_map)
    print(d5.to_string(index=False))
    d5.to_csv(OUT_DIR / "pairwise_architecture_differences.csv", index=False)
    report_lines.append("\nDIAGNOSTIC 5 — Architecture-specific paired differences")
    report_lines.append(d5.to_string(index=False))

    # ── Diagnostic 6 — Interpretation ──
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 6 — Decision rule")
    print("=" * 70)
    interp = interpret_results(d3, d2)
    for line in interp:
        print(line)
    report_lines.append("\nDIAGNOSTIC 6 — Decision rule")
    report_lines.extend(interp)

    # ── Oracle headroom diagnostic (if predictions available) ──
    frac_preds = None

    # Try loading from CSV first
    if args.frac_pred:
        print(f"\n  Loading Frac predictions from: {args.frac_pred}")
        frac_preds = load_frac_predictions(args.frac_pred, len(df))

    # Try auto-loading from npz if requested or CSV not provided
    if frac_preds is None and (args.auto_npz or args.frac_pred is None):
        # Auto-detect paths relative to project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parents[1]
        predictions_dir = project_root / "predictions" / "DMPNN"
        splits_json = project_root / "results" / "splits" / "ea_ip_aheldout_seed42.json"

        if predictions_dir.exists() and splits_json.exists():
            print(f"\n  Auto-loading Frac predictions from .npz files...")
            frac_preds = load_frac_predictions_from_npz(predictions_dir, splits_json, len(df))
            if frac_preds is not None:
                print(f"    Successfully loaded predictions for {len(df)} samples.")

    if frac_preds is not None:
        # If no fold column in data, construct one from splits JSON for fold-safe oracle
        if col_map.get("fold") is None:
            import json
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parents[1]
            splits_json = project_root / "results" / "splits" / "ea_ip_aheldout_seed42.json"
            if splits_json.exists():
                with open(splits_json) as f:
                    splits_data = json.load(f)
                fold_arr = np.full(len(df), -1, dtype=int)
                for fold_info in splits_data["folds"]:
                    test_idx = np.array(fold_info["test_indices"])
                    fold_arr[test_idx] = fold_info["fold"]
                if (fold_arr >= 0).all():
                    df["_fold"] = fold_arr
                    col_map["fold"] = "_fold"
                    print(f"  Constructed fold assignments from splits JSON ({splits_data['n_folds']} folds)")

        print("\n" + "=" * 70)
        print("DIAGNOSTIC — FRAC + ARCHITECTURE ORACLE HEADROOM")
        print("=" * 70)
        oracle_results = diagnostic_oracle_headroom(df, col_map, frac_preds, d2)
        oracle_lines = print_oracle_report(oracle_results)
        for line in oracle_lines:
            print(line)
        report_lines.append("\nDIAGNOSTIC — FRAC + ARCHITECTURE ORACLE HEADROOM")
        report_lines.extend(oracle_lines)

        # Save oracle report
        with open(OUT_DIR / "oracle_report.txt", "w") as f:
            f.write("\n".join(oracle_lines))
    else:
        print("\n  Skipping oracle headroom diagnostic (no Frac predictions available).")
        print("  Use --frac-pred <csv> or --auto-npz to enable.")

    # ── Save summary metrics ──
    summary = {**d1, **d2, **d3}
    if frac_preds is not None:
        summary.update(oracle_results)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    # ── Save report ──
    with open(OUT_DIR / "diagnostic_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nAll outputs saved to: {OUT_DIR}/")

    # ── Optional plots ──
    if args.plot:
        print("\nGenerating plots...")
        generate_plots(df, col_map)


# Allow OUT_DIR to be set by main()
OUT_DIR = Path("pre_2d_diagnostics")

if __name__ == "__main__":
    main()
