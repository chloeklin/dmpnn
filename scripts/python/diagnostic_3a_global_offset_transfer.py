#!/usr/bin/env python3
"""Diagnostic 3A — Global Architecture Offset Transfer Test.

Tests whether Δy_arch ≈ c_arch, where c_arch is a global per-architecture
mean offset learned from training folds and applied to test folds.

This is the simplest possible architecture-aware correction:
  c_alt  = mean(delta_EA for alternating rows in TRAIN)
  c_rand = mean(delta_EA for random rows in TRAIN)
  c_block= mean(delta_EA for block rows in TRAIN)
  pred   = c_arch[architecture_label]

Evaluated under monomer-disjoint (leave-one-fold-out) cross-validation.
Also compares against a zero predictor (no architecture correction).

Usage:
    python diagnostic_3a_global_offset_transfer.py --data data.csv --out diagnostic_3a_out
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# Column inference (same aliases as pre_2d_architecture_diagnostic.py)
# ══════════════════════════════════════════════════════════════════════════════

COLUMN_ALIASES = {
    "smiles_A": ["smiles_a", "monomer_a", "a_smiles", "smilesA", "smilesa"],
    "smiles_B": ["smiles_b", "monomer_b", "b_smiles", "smilesB", "smilesb"],
    "fracA":    ["fraca", "frac_a", "fraction_a", "x_a"],
    "fracB":    ["fracb", "frac_b", "fraction_b", "x_b"],
    "poly_type": ["poly_type", "polymer_type", "architecture", "arch", "polytype"],
    "EA": ["ea", "ea vs she (ev)", "ea_vs_she", "ea vs she"],
    "IP": ["ip", "ip vs she (ev)", "ip_vs_she", "ip vs she"],
    "fold": ["fold", "cv_fold", "split", "Fold"],
}


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    col_lower_map = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias.lower() in col_lower_map:
                found = col_lower_map[alias.lower()]
                break
        if found is None and canonical.lower() in col_lower_map:
            found = col_lower_map[canonical.lower()]
        mapping[canonical] = found

    required = ["smiles_A", "smiles_B", "fracA", "fracB", "poly_type", "EA", "IP"]
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise ValueError(
            f"Could not infer required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build matched groups
# ══════════════════════════════════════════════════════════════════════════════

def build_matched_groups(df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
    """Keep only rows belonging to groups (sA, sB, fA, fB) with >=2 architectures.

    Adds a '_group_id' integer column for efficient lookups.
    """
    sA = col_map["smiles_A"]
    sB = col_map["smiles_B"]
    fA = col_map["fracA"]
    fB = col_map["fracB"]
    arch = col_map["poly_type"]
    group_cols = [sA, sB, fA, fB]

    arch_counts = df.groupby(group_cols)[arch].nunique()
    matched_keys = arch_counts[arch_counts >= 2].reset_index()[group_cols]
    df_matched = df.merge(matched_keys, on=group_cols, how="inner").reset_index(drop=True)

    # Assign integer group IDs
    group_labels = (
        df_matched[sA].astype(str) + "|" +
        df_matched[sB].astype(str) + "|" +
        df_matched[fA].astype(str) + "|" +
        df_matched[fB].astype(str)
    )
    group_cat = pd.Categorical(group_labels)
    df_matched["_group_id"] = group_cat.codes
    return df_matched


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Compute architecture deviations
# ══════════════════════════════════════════════════════════════════════════════

def compute_deviations(df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
    """delta_y = y - group_mean(y) for EA and IP."""
    ea_col = col_map["EA"]
    ip_col = col_map["IP"]

    df = df.copy()
    for out_col, src_col in [("delta_EA", ea_col), ("delta_IP", ip_col)]:
        group_means = df.groupby("_group_id")[src_col].transform("mean")
        df[out_col] = df[src_col] - group_means

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 & 4 — Fold assignments + global offset prediction
# ══════════════════════════════════════════════════════════════════════════════

def get_fold_assignments(df: pd.DataFrame, col_map: Dict) -> np.ndarray:
    """Return integer fold array. Uses existing fold column or GroupKFold on monomer_A."""
    fold_col = col_map.get("fold")
    if fold_col and fold_col in df.columns:
        raw = df[fold_col].values
        # Normalise to 0-based integer
        cats = pd.Categorical(raw)
        return cats.codes.astype(int)

    # Fallback: GroupKFold on monomer_A
    from sklearn.model_selection import GroupKFold
    sA_col = col_map["smiles_A"]
    groups = df[sA_col].values
    gkf = GroupKFold(n_splits=5)
    fold_arr = np.zeros(len(df), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(gkf.split(df, groups=groups)):
        fold_arr[test_idx] = fold_idx
    return fold_arr


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute R², RMSE, MAE."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def run_diagnostic_3a(
    df: pd.DataFrame,
    col_map: Dict,
    fold_arr: np.ndarray,
    arch_labels_all: List[str],
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the global architecture offset transfer test.

    For each fold:
      - Compute c_alt, c_rand, c_block from TRAIN rows only
      - Predict TEST rows using c_arch[architecture label]
      - Evaluate vs true delta_y and vs zero predictor

    Returns:
      fold_metrics_df   — per-fold metrics
      summary_df        — mean metrics across folds
      offsets_df        — per-fold learned offset values
    """
    arch_col = col_map["poly_type"]
    folds = sorted(np.unique(fold_arr))
    targets = {"EA": "delta_EA", "IP": "delta_IP"}

    fold_rows = []
    offset_rows = []

    for fold in folds:
        train_mask = fold_arr != fold
        test_mask = fold_arr == fold

        df_train = df[train_mask]
        df_test = df[test_mask]

        for target_short, delta_col in targets.items():
            y_test = df_test[delta_col].values

            # Step 3: Compute per-architecture mean offsets from train only
            offsets = {}
            for arch in arch_labels_all:
                arch_train_vals = df_train[df_train[arch_col] == arch][delta_col].values
                if len(arch_train_vals) > 0:
                    offsets[arch] = float(arch_train_vals.mean())
                else:
                    offsets[arch] = 0.0  # fallback if architecture missing in train

            # Step 4: Predict test using architecture label
            global_mean_offset = np.mean([offsets[a] for a in arch_labels_all])
            y_pred = df_test[arch_col].map(offsets).fillna(global_mean_offset).values

            # Zero predictor
            y_zero = np.zeros(len(y_test))

            # Step 5: Evaluate
            metrics_offset = _eval_metrics(y_test, y_pred)
            metrics_zero = _eval_metrics(y_test, y_zero)

            fold_rows.append({
                "target": target_short,
                "fold": fold,
                "predictor": "global_arch_offset",
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                **metrics_offset,
            })
            fold_rows.append({
                "target": target_short,
                "fold": fold,
                "predictor": "zero",
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                **metrics_zero,
            })

            # Record offsets for this fold
            for arch in arch_labels_all:
                offset_rows.append({
                    "target": target_short,
                    "fold": fold,
                    "architecture": arch,
                    "train_count": int((df_train[arch_col] == arch).sum()),
                    "train_mean_delta": offsets[arch],
                })

    fold_df = pd.DataFrame(fold_rows)

    # Summary: mean across folds for each target × predictor
    summary_rows = []
    for target_short in ["EA", "IP"]:
        for predictor in ["global_arch_offset", "zero"]:
            sub = fold_df[
                (fold_df["target"] == target_short) &
                (fold_df["predictor"] == predictor)
            ]
            summary_rows.append({
                "target": target_short,
                "predictor": predictor,
                "mean_R2": sub["R2"].mean(),
                "std_R2": sub["R2"].std(),
                "mean_RMSE": sub["RMSE"].mean(),
                "std_RMSE": sub["RMSE"].std(),
                "mean_MAE": sub["MAE"].mean(),
                "std_MAE": sub["MAE"].std(),
                "n_folds": len(sub),
            })
    summary_df = pd.DataFrame(summary_rows)
    offsets_df = pd.DataFrame(offset_rows)

    return fold_df, summary_df, offsets_df


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    df_matched: pd.DataFrame,
    col_map: Dict,
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    offsets_df: pd.DataFrame,
    n_folds: int,
    args,
) -> List[str]:
    arch_col = col_map["poly_type"]
    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC 3A — GLOBAL ARCHITECTURE OFFSET TRANSFER TEST")
    lines.append("=" * 70)
    lines.append(f"\nDataset:      {args.data}")
    lines.append(f"Total rows:   {len(df_matched)}")
    lines.append(f"Groups:       {df_matched['_group_id'].nunique()}")
    lines.append(f"Folds:        {n_folds}")
    lines.append(f"Architectures: {sorted(df_matched[arch_col].unique())}")

    # Offset summary across folds
    lines.append("\n\n" + "=" * 70)
    lines.append("LEARNED GLOBAL ARCHITECTURE OFFSETS (mean ± std across folds)")
    lines.append("=" * 70)

    for target_short in ["EA", "IP"]:
        lines.append(f"\n  Δ{target_short}:")
        sub = offsets_df[offsets_df["target"] == target_short]
        for arch in sorted(sub["architecture"].unique()):
            arch_sub = sub[sub["architecture"] == arch]["train_mean_delta"]
            lines.append(
                f"    c_{arch:<12} = {arch_sub.mean():+.6f}  ±  {arch_sub.std():.6f}"
                f"  (avg train n = {sub[sub['architecture'] == arch]['train_count'].mean():.0f})"
            )

    # Fold-wise results
    lines.append("\n\n" + "=" * 70)
    lines.append("FOLD-WISE PERFORMANCE")
    lines.append("=" * 70)

    for target_short in ["EA", "IP"]:
        lines.append(f"\n  Δ{target_short}:")
        lines.append(f"  {'Fold':<6} {'Predictor':<25} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
        lines.append(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*10} {'-'*10}")
        sub = fold_df[fold_df["target"] == target_short].sort_values(["fold", "predictor"])
        for _, row in sub.iterrows():
            lines.append(
                f"  {int(row['fold']):<6} {row['predictor']:<25} "
                f"{row['R2']:>8.4f} {row['RMSE']:>10.6f} {row['MAE']:>10.6f}"
            )

    # Summary
    lines.append("\n\n" + "=" * 70)
    lines.append("MEAN PERFORMANCE ACROSS FOLDS")
    lines.append("=" * 70)

    for target_short in ["EA", "IP"]:
        lines.append(f"\n  Δ{target_short}:")
        lines.append(f"  {'Predictor':<25} {'R²':>8} {'±std':>7} {'RMSE':>10} {'±std':>8} {'MAE':>10}")
        lines.append(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*10} {'-'*8} {'-'*10}")
        sub = summary_df[summary_df["target"] == target_short].sort_values("mean_R2", ascending=False)
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['predictor']:<25} {row['mean_R2']:>8.4f} {row['std_R2']:>7.4f} "
                f"{row['mean_RMSE']:>10.6f} {row['std_RMSE']:>8.6f} {row['mean_MAE']:>10.6f}"
            )

    # Interpretation
    lines.append("\n\n" + "=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)

    for target_short in ["EA", "IP"]:
        lines.append(f"\n  Δ{target_short}:")
        offset_row = summary_df[
            (summary_df["target"] == target_short) &
            (summary_df["predictor"] == "global_arch_offset")
        ].iloc[0]
        zero_row = summary_df[
            (summary_df["target"] == target_short) &
            (summary_df["predictor"] == "zero")
        ].iloc[0]

        r2_offset = offset_row["mean_R2"]
        r2_zero = zero_row["mean_R2"]
        rmse_gain = zero_row["mean_RMSE"] - offset_row["mean_RMSE"]

        # Offset consistency
        sub = offsets_df[offsets_df["target"] == target_short]
        std_by_arch = sub.groupby("architecture")["train_mean_delta"].std()
        max_offset_std = float(std_by_arch.max())

        lines.append(f"    Global offset R²:     {r2_offset:.4f}")
        lines.append(f"    Zero predictor R²:    {r2_zero:.4f}")
        lines.append(f"    RMSE improvement:     {rmse_gain:+.6f} (offset vs zero)")
        lines.append(f"    Max offset fold-std:  {max_offset_std:.6f}")

        if r2_offset > 0.05:
            lines.append(
                f"    → Architecture offsets transfer meaningfully (R²={r2_offset:.4f} > 0.05)."
            )
            lines.append(
                "    → Δy_arch ≈ c_arch holds with moderate transferability."
            )
        elif r2_offset > 0.01:
            lines.append(
                f"    → Architecture offsets show weak but non-zero transfer (R²={r2_offset:.4f})."
            )
            lines.append(
                "    → Δy_arch ≈ c_arch is a weak approximation."
            )
        else:
            lines.append(
                f"    → Architecture offsets do NOT transfer reliably (R²={r2_offset:.4f} ≈ 0)."
            )
            lines.append(
                "    → Δy_arch is not well-approximated by a global c_arch."
            )
            lines.append(
                "    → Architecture effects are highly monomer-system-specific."
            )

        if max_offset_std > 0.01:
            lines.append(
                f"    → High fold-to-fold offset variability (max std={max_offset_std:.4f})."
            )
            lines.append(
                "    → Offset estimates are unstable — small dataset or heterogeneous signal."
            )
        else:
            lines.append(
                f"    → Offset estimates are stable across folds (max std={max_offset_std:.4f})."
            )

    lines.append("\n\n" + "=" * 70)
    lines.append("RELATIONSHIP TO OTHER DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("""
  Diagnostic 3 (pre_2d):
    Used an in-sample Ridge with monomer identity + frac + arch one-hot.
    That test asked: does arch improve R² at all (global, no fold split)?

  Diagnostic Oracle (pre_2d):
    Fold-safe oracle showed zero RMSE improvement under monomer-disjoint eval.
    That used per-(group, arch) offsets, which are monomer-specific — NOT global.

  Diagnostic 3A (this test):
    Uses only GLOBAL per-architecture offsets (not monomer-specific).
    This is the minimal-information test: can architecture type alone predict Δy?
    A high R² here means arch type is sufficient.
    A low R² here (but high in feature_conditioned_transfer arch_only) would mean
    the signal exists but is not transferable as a global constant.
""")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic 3A — Global architecture offset transfer test"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default="diagnostic_3a_out", help="Output directory")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DIAGNOSTIC 3A — GLOBAL ARCHITECTURE OFFSET TRANSFER TEST")
    print("=" * 70)

    # Load data
    print(f"\nLoading: {args.data}")
    df = pd.read_csv(args.data)
    print(f"  Shape: {df.shape}")

    col_map = infer_columns(df)
    print("  Column mapping:")
    for k, v in col_map.items():
        if v is not None:
            print(f"    {k} -> '{v}'")

    ea_col = col_map["EA"]
    ip_col = col_map["IP"]
    df = df.dropna(subset=[ea_col, ip_col]).reset_index(drop=True)
    print(f"  After dropping missing targets: {len(df)} rows")

    # Step 1: Matched groups
    print("\nStep 1 — Building matched groups...")
    df_matched = build_matched_groups(df, col_map)
    n_groups = df_matched["_group_id"].nunique()
    print(f"  Matched rows: {len(df_matched)} / {len(df)}")
    print(f"  Matched groups (≥2 architectures): {n_groups}")

    arch_col = col_map["poly_type"]
    arch_labels = sorted(df_matched[arch_col].unique())
    print(f"  Architecture labels: {arch_labels}")

    # Step 2: Compute deviations
    print("\nStep 2 — Computing architecture deviations Δy = y - group_mean...")
    df_matched = compute_deviations(df_matched, col_map)
    print(f"  delta_EA: mean={df_matched['delta_EA'].mean():+.6f}, "
          f"std={df_matched['delta_EA'].std():.6f}, "
          f"var={df_matched['delta_EA'].var():.8f}")
    print(f"  delta_IP: mean={df_matched['delta_IP'].mean():+.6f}, "
          f"std={df_matched['delta_IP'].std():.6f}, "
          f"var={df_matched['delta_IP'].var():.8f}")

    # Fold assignments
    fold_arr = get_fold_assignments(df_matched, col_map)
    n_folds = len(np.unique(fold_arr))
    fold_source = "existing fold column" if (col_map.get("fold") and col_map["fold"] in df_matched.columns) else "GroupKFold on monomer_A"
    print(f"\nFold assignments: {n_folds} folds ({fold_source})")

    # Step 3-5: Run
    print("\nSteps 3–5 — Computing offsets and evaluating per fold...")
    fold_df, summary_df, offsets_df = run_diagnostic_3a(
        df_matched, col_map, fold_arr, arch_labels, args.random_seed
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for target_short in ["EA", "IP"]:
        print(f"\n  Δ{target_short}:")
        sub = summary_df[summary_df["target"] == target_short].sort_values("mean_R2", ascending=False)
        print(f"  {'Predictor':<25} {'R²':>8} {'±std':>7} {'RMSE':>10} {'MAE':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*10} {'-'*10}")
        for _, row in sub.iterrows():
            print(f"  {row['predictor']:<25} {row['mean_R2']:>8.4f} {row['std_R2']:>7.4f} "
                  f"{row['mean_RMSE']:>10.6f} {row['mean_MAE']:>10.6f}")

    # Print offsets
    print("\n" + "=" * 70)
    print("LEARNED GLOBAL OFFSETS (mean across folds)")
    print("=" * 70)
    for target_short in ["EA", "IP"]:
        print(f"\n  Δ{target_short}:")
        sub = offsets_df[offsets_df["target"] == target_short]
        for arch in sorted(sub["architecture"].unique()):
            vals = sub[sub["architecture"] == arch]["train_mean_delta"]
            print(f"    c_{arch:<12} = {vals.mean():+.6f}  (std={vals.std():.6f})")

    # Save outputs
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    fold_df.to_csv(out_dir / "diagnostic3a_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "diagnostic3a_metrics.csv", index=False)
    offsets_df.to_csv(out_dir / "diagnostic3a_offsets.csv", index=False)

    report_lines = generate_report(
        df_matched, col_map, fold_df, summary_df, offsets_df, n_folds, args
    )
    with open(out_dir / "diagnostic3a_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"  diagnostic3a_metrics.csv")
    print(f"  diagnostic3a_fold_metrics.csv")
    print(f"  diagnostic3a_offsets.csv")
    print(f"  diagnostic3a_report.txt")
    print(f"\n  All saved to: {out_dir}/")


if __name__ == "__main__":
    main()
