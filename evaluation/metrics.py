"""
Canonical metric functions for EA/IP copolymer predictions.

Key design choices
------------------
* ``compute_archdev_r2`` always uses **global** dataframe indices.
  The caller must supply the true global integer row positions of each
  test sample; local/positional indices (0 … n_test-1) must NOT be
  passed in unless they happen to equal the global indices for fold 0.
* Group key = (monomer_A, monomer_B, f_A, f_B) — architecture excluded.
* Δy_true = y_true − group_mean(y_true)  (mean from y_true only)
* Δy_pred = y_pred − group_mean(y_pred)  (mean from y_pred only)
* Only groups with ≥ 2 distinct architectures contribute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Sequence


# ── Overall metrics ───────────────────────────────────────────────────────────

def compute_overall_r2(y_true: Sequence, y_pred: Sequence) -> float:
    """R² between y_true and y_pred (float64)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(r2_score(yt, yp))


def compute_overall_mae(y_true: Sequence, y_pred: Sequence) -> float:
    """MAE between y_true and y_pred."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(mean_absolute_error(yt, yp))


# ── Architecture-deviation metric ────────────────────────────────────────────

def compute_archdev_r2(
    df: pd.DataFrame,
    y_true: Sequence,
    y_pred: Sequence,
    global_indices: Sequence[int],
    group_cols: list[str] | None = None,
    arch_col: str = "poly_type",
    min_samples: int = 20,
) -> dict:
    """Compute architecture-deviation R² using correct global df indices.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset dataframe (all rows).  Must contain ``arch_col`` and
        the columns used to build the group key.
    y_true, y_pred : array-like
        Predictions for the test split (one value per test sample).
    global_indices : array-like of int
        **Global** row positions in ``df`` for each test sample.
        These must be absolute ``df.iloc[i]`` positions, NOT local
        positional indices within the test split.
    group_cols : list of str, optional
        Columns that together uniquely identify a (monomer pair,
        composition) group.  Architecture must NOT be in this list.
        Defaults to ``["smilesA", "smilesB", "fracA", "fracB"]`` if the
        df contains those columns, else tries ``["smiles_A", "smiles_B",
        "frac_A", "frac_B"]`` and finally falls back to ``"group_key"``.
    arch_col : str
        Column containing the architecture label (default ``"poly_type"``).
    min_samples : int
        Minimum number of valid samples required; returns NaN if fewer.

    Returns
    -------
    dict with keys:
        r2          – architecture-deviation R² (float or nan)
        mae         – architecture-deviation MAE (float or nan)
        n_samples   – number of samples used
        n_groups    – number of matched groups (≥2 architectures)
        avg_group_size – mean group size across matched groups
        mean_dt     – mean(Δy_true), should be ≈ 0
        mean_dp     – mean(Δy_pred)
        std_dt      – std(Δy_true)
    """
    yt  = np.asarray(y_true,  dtype=np.float64)
    yp  = np.asarray(y_pred,  dtype=np.float64)
    idx = np.asarray(global_indices, dtype=int)

    _nan_result = dict(r2=np.nan, mae=np.nan, n_samples=0,
                       n_groups=0, avg_group_size=np.nan,
                       mean_dt=np.nan, mean_dp=np.nan, std_dt=np.nan)

    if len(idx) < min_samples:
        return _nan_result

    # ── Validate global indices ───────────────────────────────────────
    if idx.min() < 0 or idx.max() >= len(df):
        raise IndexError(
            f"global_indices out of range [0, {len(df)-1}]: "
            f"min={idx.min()}, max={idx.max()}"
        )

    rows = df.iloc[idx]

    # ── Resolve group column(s) ───────────────────────────────────────
    if group_cols is None:
        if "group_key" in df.columns:
            group_cols = ["group_key"]
        elif "smilesA" in df.columns:
            group_cols = ["smilesA", "smilesB", "fracA", "fracB"]
        elif "smiles_A" in df.columns and "fracA" in df.columns:
            group_cols = ["smiles_A", "smiles_B", "fracA", "fracB"]
        elif "smiles_A" in df.columns:
            group_cols = ["smiles_A", "smiles_B", "frac_A", "frac_B"]
        else:
            raise ValueError(
                "Cannot infer group columns from df. "
                "Pass group_cols explicitly or ensure df has 'group_key'."
            )

    missing = [c for c in group_cols + [arch_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from df: {missing}")

    # ── Build working dataframe ───────────────────────────────────────
    if len(group_cols) == 1:
        groups = rows[group_cols[0]].values
    else:
        groups = [
            "||".join(str(rows[c].iloc[i]) for c in group_cols)
            for i in range(len(rows))
        ]
    arch = rows[arch_col].values

    gdf = pd.DataFrame({"yt": yt, "yp": yp, "group": groups, "arch": arch})

    # Keep only groups with ≥ 2 distinct architectures
    n_arch = gdf.groupby("group")["arch"].nunique()
    multi  = n_arch[n_arch >= 2].index
    sub    = gdf[gdf["group"].isin(multi)].copy()

    if len(sub) < min_samples:
        return _nan_result

    # ── Compute Δy (per-group mean subtraction) ───────────────────────
    sub["gmt"] = sub.groupby("group")["yt"].transform("mean")
    sub["gmp"] = sub.groupby("group")["yp"].transform("mean")
    dt = (sub["yt"] - sub["gmt"]).values
    dp = (sub["yp"] - sub["gmp"]).values

    if dt.std() < 1e-10:
        return _nan_result

    return dict(
        r2            = float(r2_score(dt, dp)),
        mae           = float(mean_absolute_error(dt, dp)),
        n_samples     = len(sub),
        n_groups      = len(multi),
        avg_group_size= float(len(sub) / len(multi)),
        mean_dt       = float(dt.mean()),
        mean_dp       = float(dp.mean()),
        std_dt        = float(dt.std()),
    )


def compute_archdev_mae(
    df: pd.DataFrame,
    y_true: Sequence,
    y_pred: Sequence,
    global_indices: Sequence[int],
    **kwargs,
) -> float:
    """Convenience wrapper returning just the architecture-deviation MAE."""
    return compute_archdev_r2(df, y_true, y_pred, global_indices, **kwargs)["mae"]


# ── Architecture variance diagnostics ────────────────────────────────────────

def architecture_variance_ratio(
    df: pd.DataFrame,
    y: Sequence,
    global_indices: Sequence[int],
    group_cols: list[str] | None = None,
    arch_col: str = "poly_type",
) -> float:
    """Fraction of total variance that is within-group (architecture-driven).

    architecture_variance_ratio = mean(within-group var) / total var
    """
    yt  = np.asarray(y, dtype=np.float64)
    idx = np.asarray(global_indices, dtype=int)
    rows = df.iloc[idx]

    if group_cols is None:
        if "group_key" in df.columns:
            group_cols = ["group_key"]
        elif "smilesA" in df.columns:
            group_cols = ["smilesA", "smilesB", "fracA", "fracB"]
        elif "smiles_A" in df.columns and "fracA" in df.columns:
            group_cols = ["smiles_A", "smiles_B", "fracA", "fracB"]
        else:
            group_cols = ["smiles_A", "smiles_B", "frac_A", "frac_B"]

    if len(group_cols) == 1:
        groups = rows[group_cols[0]].values
    else:
        groups = [
            "||".join(str(rows[c].iloc[i]) for c in group_cols)
            for i in range(len(rows))
        ]

    gdf = pd.DataFrame({"y": yt, "group": groups})
    n_arch = gdf.groupby("group")["y"].count()
    multi  = n_arch[n_arch >= 2].index
    sub    = gdf[gdf["group"].isin(multi)]

    within_var = sub.groupby("group")["y"].var().mean()
    total_var  = np.var(yt)
    return float(within_var / total_var) if total_var > 1e-12 else np.nan


def residual_architecture_share(
    arch_var_ratio: float,
    r2_frac: float,
) -> float:
    """Fraction of residual variance (after Frac baseline) explained by arch.

    residual_architecture_share = arch_var_ratio / (1 - R²_Frac)
    """
    denom = 1.0 - r2_frac
    if abs(denom) < 1e-10:
        return np.nan
    return float(arch_var_ratio / denom)


# ── Scale sanity check ────────────────────────────────────────────────────────

def scale_sanity_check(
    y_true: Sequence,
    y_pred: Sequence,
    target_name: str = "",
    expected_mean_range: tuple[float, float] = (-4.0, 1.0),
    expected_std_range:  tuple[float, float] = (0.1, 2.0),
) -> dict:
    """Check whether y_true / y_pred look like physical units (eV).

    Returns a dict with statistics and a 'warnings' list.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    warnings = []
    lo_m, hi_m = expected_mean_range
    lo_s, hi_s = expected_std_range

    if not (lo_m <= float(yt.mean()) <= hi_m):
        warnings.append(
            f"y_true mean={yt.mean():.3f} outside expected range {expected_mean_range}"
        )
    if not (lo_s <= float(yt.std()) <= hi_s):
        warnings.append(
            f"y_true std={yt.std():.3f} outside expected range {expected_std_range}"
        )
    if abs(yt.std() - 1.0) < 0.05:
        warnings.append(
            "y_true std ≈ 1 — may be normalised rather than physical units"
        )
    if abs(yp.std() - 1.0) < 0.05 and abs(yt.std() - 1.0) > 0.2:
        warnings.append(
            "y_pred std ≈ 1 but y_true std is not — possible scale mismatch"
        )

    return dict(
        target          = target_name,
        yt_mean         = float(yt.mean()),
        yt_std          = float(yt.std()),
        yt_min          = float(yt.min()),
        yt_max          = float(yt.max()),
        yp_mean         = float(yp.mean()),
        yp_std          = float(yp.std()),
        yp_min          = float(yp.min()),
        yp_max          = float(yp.max()),
        n               = len(yt),
        warnings        = warnings,
        likely_physical = len(warnings) == 0,
    )


# ── Validation checks ─────────────────────────────────────────────────────────

def validate_prediction_inputs(
    df: pd.DataFrame,
    y_true: Sequence,
    y_pred: Sequence,
    global_indices: Sequence[int],
    split_name: str = "",
    arch_col: str = "poly_type",
) -> None:
    """Raise a clear error if any sanity check fails.

    Checks
    ------
    1. y_true, y_pred, global_indices all have the same length.
    2. global_indices are within [0, len(df)-1].
    3. df contains the architecture column.
    4. global_indices are NOT a contiguous 0…n-1 sequence starting from 0
       unless the caller is explicitly evaluating fold 0 (we cannot enforce
       this automatically, but we emit a warning).
    """
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError(
            f"y_true (n={n}) and y_pred (n={len(y_pred)}) have different lengths."
        )
    idx = np.asarray(global_indices, dtype=int)
    if len(idx) != n:
        raise ValueError(
            f"global_indices (n={len(idx)}) length does not match "
            f"y_true/y_pred (n={n})."
        )
    if idx.min() < 0 or idx.max() >= len(df):
        raise IndexError(
            f"[{split_name}] global_indices out of range [0, {len(df)-1}]: "
            f"min={idx.min()}, max={idx.max()}.  "
            f"These must be GLOBAL df row indices, not local test-split positions."
        )
    if arch_col not in df.columns:
        raise ValueError(
            f"Architecture column {arch_col!r} not found in df. "
            f"Available: {list(df.columns)}"
        )
    # Warn if indices look like 0…n-1 (local positional pattern)
    if np.array_equal(idx, np.arange(n)):
        import warnings
        warnings.warn(
            f"[{split_name}] global_indices appear to be 0…{n-1} (local positional). "
            "For monomer-heldout folds > 0 this would assign wrong group_key/poly_type. "
            "Ensure these are global df row indices.",
            UserWarning,
            stacklevel=3,
        )
