#!/usr/bin/env python3
"""
Stage 2D Post-Rerun Analysis
==============================
Comprehensive analysis after fixing the dead-initialization bug and
rerunning 2d1_fixed and 2d1_arch.

Phases:
  1. Verify rerun worked (alpha extraction + prediction diff)
  2. Regenerate master results table
  3. Architecture-deviation analysis
  4. Identify best 2D0 and best 2D1
  5. Direct 2D0 vs 2D1 comparison
  6. Per-fold significance analysis
  7. Paper-level interpretation
  8. Final recommendation
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_DIR = PROJECT_ROOT / "checkpoints" / "HPG2Stage"
DATA_PATH = PROJECT_ROOT / "data" / "ea_ip.csv"
PRED_DIR = PROJECT_ROOT / "predictions" / "HPG2Stage"
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "postrerun"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
TARGET_SHORT = {"EA vs SHE (eV)": "EA", "IP vs SHE (eV)": "IP"}
VARIANTS = ["frac", "2d0_fixed", "2d0_arch", "2d0_gate",
            "2d1_fixed", "2d1_arch", "2d1_gate"]
N_SPLITS = 5

VARIANT_DISPLAY = {
    "frac": "Frac", "2d0_fixed": "2D0-fixed", "2d0_arch": "2D0-arch",
    "2d0_gate": "2D0-gate", "2d1_fixed": "2D1-fixed",
    "2d1_arch": "2D1-arch", "2d1_gate": "2D1-gate",
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_normalization_params():
    """Estimate per-split train_mean, train_std via linear regression on frac predictions."""
    train_stats = {t: [] for t in TARGETS}
    for target in TARGETS:
        for split_idx in range(N_SPLITS):
            fname = f"ea_ip__{target}__copoly_stage2d_frac__a_held_out__split{split_idx}.npz"
            fpath = PRED_DIR / fname
            if not fpath.exists():
                train_stats[target].append((0.0, 1.0))
                continue
            npz = np.load(fpath, allow_pickle=True)
            y_true = npz["y_true"].flatten()
            y_pred = npz["y_pred"].flatten()
            slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
            train_stats[target].append((intercept, slope))
    return train_stats


def load_predictions_corrected(variant, target, train_stats):
    """Load predictions for a variant/target, apply inverse transform, return per-split."""
    per_split = []
    for split_idx in range(N_SPLITS):
        fname = f"ea_ip__{target}__copoly_stage2d_{variant}__a_held_out__split{split_idx}.npz"
        fpath = PRED_DIR / fname
        if not fpath.exists():
            continue
        npz = np.load(fpath, allow_pickle=True)
        y_true = npz["y_true"].flatten()
        y_pred_norm = npz["y_pred"].flatten()
        est_mean, est_std = train_stats[target][split_idx]
        y_pred = y_pred_norm * est_std + est_mean
        per_split.append((y_true, y_pred, split_idx))
    return per_split


def concat_splits(per_split):
    """Concatenate per-split data."""
    if not per_split:
        return np.array([]), np.array([])
    y_true = np.concatenate([s[0] for s in per_split])
    y_pred = np.concatenate([s[1] for s in per_split])
    return y_true, y_pred


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: VERIFY RERUN WORKED
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_verify_rerun(train_stats):
    """Verify alpha ≠ 0 and predictions differ from frac."""
    print("=" * 70)
    print("PHASE 1: VERIFY RERUN WORKED")
    print("=" * 70)

    lines = ["# Rerun Verification Report\n"]

    # ── Extract alpha from checkpoints ──
    lines.append("## Alpha Parameter Extraction\n")

    for variant in ["2d1_fixed", "2d1_arch"]:
        lines.append(f"### {VARIANT_DISPLAY[variant]}\n")
        alphas_all = []

        for target in TARGETS:
            tshort = TARGET_SHORT[target]
            for rep in range(N_SPLITS):
                ckpt_dir = CKPT_DIR / f"ea_ip__{target}__copoly_stage2d_{variant}__a_held_out__rep{rep}"
                ckpt_files = list((ckpt_dir / "logs" / "checkpoints").glob("*.ckpt")) if (ckpt_dir / "logs" / "checkpoints").exists() else []
                if not ckpt_files:
                    continue
                ckpt_path = ckpt_files[0]
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    sd = ckpt["state_dict"]
                    alpha_key = None
                    for k in sd:
                        if "stage2_aggregator.alpha" in k:
                            alpha_key = k
                            break
                    if alpha_key is None:
                        continue
                    alpha_tensor = sd[alpha_key]
                    if "fixed" in variant:
                        val = alpha_tensor.item()
                        alphas_all.append(val)
                        lines.append(f"- {tshort} rep{rep}: alpha = {val:.6f}")
                    else:
                        vals = alpha_tensor.tolist()
                        alphas_all.extend(vals)
                        lines.append(f"- {tshort} rep{rep}: alpha_alt={vals[0]:.6f}, alpha_rand={vals[1]:.6f}, alpha_block={vals[2]:.6f}")
                except Exception as e:
                    lines.append(f"- {tshort} rep{rep}: ERROR: {e}")

        alphas_arr = np.array(alphas_all)
        lines.append(f"\n**Summary**: mean={alphas_arr.mean():.6f}, std={alphas_arr.std():.6f}, "
                     f"min={alphas_arr.min():.6f}, max={alphas_arr.max():.6f}")
        all_zero = np.allclose(alphas_arr, 0.0, atol=1e-6)
        lines.append(f"**alpha ≠ 0**: {'⚠️ FAIL — still zero!' if all_zero else '✅ PASS'}")

        if "arch" in variant:
            # Check per-arch alphas are not all identical
            # Group by position (0,1,2 repeating)
            alt_vals = alphas_arr[0::3]
            rand_vals = alphas_arr[1::3]
            block_vals = alphas_arr[2::3]
            all_identical = np.allclose(alt_vals.mean(), rand_vals.mean(), atol=1e-4) and \
                           np.allclose(rand_vals.mean(), block_vals.mean(), atol=1e-4)
            lines.append(f"**Per-arch alphas differ**: {'⚠️ FAIL — all identical' if all_identical else '✅ PASS'}")
            lines.append(f"  alt mean={alt_vals.mean():.6f}, rand mean={rand_vals.mean():.6f}, block mean={block_vals.mean():.6f}")

        lines.append("")

    # ── Prediction difference vs frac ──
    lines.append("## Prediction Difference vs Frac\n")

    for variant in ["2d1_fixed", "2d1_arch"]:
        lines.append(f"### {VARIANT_DISPLAY[variant]}\n")
        for target in TARGETS:
            tshort = TARGET_SHORT[target]
            frac_splits = load_predictions_corrected("frac", target, train_stats)
            var_splits = load_predictions_corrected(variant, target, train_stats)
            if not frac_splits or not var_splits:
                continue
            frac_true, frac_pred = concat_splits(frac_splits)
            var_true, var_pred = concat_splits(var_splits)
            # Predictions are on same test sets per split, so lengths should match
            if len(frac_pred) == len(var_pred):
                mad = np.abs(frac_pred - var_pred).mean()
                max_diff = np.abs(frac_pred - var_pred).max()
                lines.append(f"- {tshort}: Mean Abs Diff = {mad:.6f}, Max Abs Diff = {max_diff:.6f}")
                if mad < 1e-4:
                    lines.append(f"  ⚠️ Near-zero difference — rerun may have failed!")
                else:
                    lines.append(f"  ✅ Predictions differ meaningfully from Frac")
            else:
                lines.append(f"- {tshort}: Length mismatch (frac={len(frac_pred)}, {variant}={len(var_pred)})")
        lines.append("")

    outpath = OUT_DIR / "rerun_verification.md"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  → Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: REGENERATE MASTER RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_master_results(train_stats):
    """Compute R², MAE, RMSE for all valid models."""
    print("\n" + "=" * 70)
    print("PHASE 2: REGENERATE MASTER RESULTS TABLE")
    print("=" * 70)

    rows = []
    per_fold_data = defaultdict(lambda: defaultdict(list))  # variant -> target -> [(r2, mae, rmse)]

    for variant in VARIANTS:
        for target in TARGETS:
            tshort = TARGET_SHORT[target]
            splits = load_predictions_corrected(variant, target, train_stats)
            if not splits:
                continue

            # Overall metrics (pooled across folds)
            y_t, y_p = concat_splits(splits)
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))

            rows.append({
                "variant": variant, "target": tshort,
                "R2": r2, "MAE": mae, "RMSE": rmse, "N": len(y_t),
            })

            # Per-fold metrics
            for yt_s, yp_s, si in splits:
                per_fold_data[variant][tshort].append({
                    "split": si,
                    "R2": r2_score(yt_s, yp_s),
                    "MAE": mean_absolute_error(yt_s, yp_s),
                    "RMSE": np.sqrt(mean_squared_error(yt_s, yp_s)),
                })

            print(f"  {variant:12s} | {tshort}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stage2d_master_results.csv", index=False)

    # Markdown version
    md = ["# Stage 2D Master Results\n"]
    md.append("| Variant | EA R² | EA MAE | EA RMSE | IP R² | IP MAE | IP RMSE |")
    md.append("|---------|-------|--------|---------|-------|--------|---------|")
    for v in VARIANTS:
        ea = df[(df["variant"] == v) & (df["target"] == "EA")]
        ip = df[(df["variant"] == v) & (df["target"] == "IP")]
        if ea.empty or ip.empty:
            continue
        md.append(f"| {VARIANT_DISPLAY[v]} | {ea.iloc[0]['R2']:.4f} | {ea.iloc[0]['MAE']:.4f} | "
                  f"{ea.iloc[0]['RMSE']:.4f} | {ip.iloc[0]['R2']:.4f} | {ip.iloc[0]['MAE']:.4f} | "
                  f"{ip.iloc[0]['RMSE']:.4f} |")

    with open(OUT_DIR / "stage2d_master_results.md", "w") as f:
        f.write("\n".join(md))
    print(f"  → Saved: stage2d_master_results.csv + .md")

    return df, per_fold_data


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ARCHITECTURE-DEVIATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_architecture_deviations(train_stats):
    """Compute architecture-deviation metrics for all models."""
    print("\n" + "=" * 70)
    print("PHASE 3: ARCHITECTURE-DEVIATION ANALYSIS")
    print("=" * 70)

    df_dataset = pd.read_csv(DATA_PATH)

    # Build lookup: round(y_true,6) → row index
    lookups = {}
    for target in TARGETS:
        tshort = TARGET_SHORT[target]
        vals = df_dataset[target].values
        lookup = {}
        for idx, v in enumerate(vals):
            if np.isfinite(v):
                key = round(float(v), 6)
                lookup[key] = idx
        lookups[tshort] = lookup

    rows = []
    per_fold_dev = defaultdict(lambda: defaultdict(list))  # variant -> target -> [r2_dev per fold]

    for variant in VARIANTS:
        for target in TARGETS:
            tshort = TARGET_SHORT[target]
            splits = load_predictions_corrected(variant, target, train_stats)
            if not splits:
                continue

            all_y_true, all_y_pred, all_meta_idx = [], [], []
            for yt_s, yp_s, si in splits:
                lookup = lookups[tshort]
                for i in range(len(yt_s)):
                    key = round(float(yt_s[i]), 6)
                    if key in lookup:
                        all_y_true.append(yt_s[i])
                        all_y_pred.append(yp_s[i])
                        all_meta_idx.append(lookup[key])

            if not all_y_true:
                continue

            pred_df = pd.DataFrame({
                "y_true": all_y_true, "y_pred": all_y_pred, "dataset_idx": all_meta_idx,
            })
            meta_cols = ["smiles_A", "smiles_B", "fracA", "poly_type"]
            pred_df = pred_df.merge(
                df_dataset[meta_cols].reset_index().rename(columns={"index": "dataset_idx"}),
                on="dataset_idx", how="left",
            )
            pred_df["group"] = pred_df.apply(
                lambda r: f"{r['smiles_A']}|{r['smiles_B']}|{r['fracA']}", axis=1
            )

            group_means_true = pred_df.groupby("group")["y_true"].transform("mean")
            group_means_pred = pred_df.groupby("group")["y_pred"].transform("mean")
            group_sizes = pred_df.groupby("group")["y_true"].transform("count")

            pred_df["delta_true"] = pred_df["y_true"] - group_means_true
            pred_df["delta_pred"] = pred_df["y_pred"] - group_means_pred

            df_multi = pred_df[group_sizes > 1]
            if len(df_multi) < 10:
                continue

            dt = df_multi["delta_true"].values
            dp = df_multi["delta_pred"].values

            r2_dev = r2_score(dt, dp)
            mae_dev = mean_absolute_error(dt, dp)

            rows.append({
                "variant": variant, "target": tshort,
                "R2_dev": r2_dev, "MAE_dev": mae_dev,
                "n_multi": len(df_multi),
            })

            # Per-fold deviations
            for yt_s, yp_s, si in splits:
                fold_idx = []
                fold_yt, fold_yp = [], []
                lookup = lookups[tshort]
                for i in range(len(yt_s)):
                    key = round(float(yt_s[i]), 6)
                    if key in lookup:
                        fold_idx.append(lookup[key])
                        fold_yt.append(yt_s[i])
                        fold_yp.append(yp_s[i])
                if len(fold_yt) < 10:
                    continue
                fold_df = pd.DataFrame({
                    "y_true": fold_yt, "y_pred": fold_yp, "dataset_idx": fold_idx,
                })
                fold_df = fold_df.merge(
                    df_dataset[meta_cols].reset_index().rename(columns={"index": "dataset_idx"}),
                    on="dataset_idx", how="left",
                )
                fold_df["group"] = fold_df.apply(
                    lambda r: f"{r['smiles_A']}|{r['smiles_B']}|{r['fracA']}", axis=1
                )
                gmt = fold_df.groupby("group")["y_true"].transform("mean")
                gmp = fold_df.groupby("group")["y_pred"].transform("mean")
                gsz = fold_df.groupby("group")["y_true"].transform("count")
                fold_df["dt"] = fold_df["y_true"] - gmt
                fold_df["dp"] = fold_df["y_pred"] - gmp
                fm = fold_df[gsz > 1]
                if len(fm) >= 10:
                    per_fold_dev[variant][tshort].append(r2_score(fm["dt"].values, fm["dp"].values))

            print(f"  {variant:12s} | {tshort}: R²(Δy)={r2_dev:.4f}, MAE(Δy)={mae_dev:.4f}")

    df_dev = pd.DataFrame(rows)
    df_dev.to_csv(OUT_DIR / "stage2d_architecture_deviation_results.csv", index=False)
    print(f"  → Saved: stage2d_architecture_deviation_results.csv")
    return df_dev, per_fold_dev


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: IDENTIFY BEST 2D0 AND BEST 2D1
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_identify_best(df_master, df_dev):
    """Identify best 2D0 and best 2D1 by target."""
    print("\n" + "=" * 70)
    print("PHASE 4: IDENTIFY BEST 2D0 AND BEST 2D1")
    print("=" * 70)

    best = {}
    for family, members in [("2D0", ["2d0_fixed", "2d0_arch", "2d0_gate"]),
                             ("2D1", ["2d1_fixed", "2d1_arch", "2d1_gate"])]:
        for tshort in ["EA", "IP"]:
            sub = df_master[(df_master["variant"].isin(members)) & (df_master["target"] == tshort)]
            if sub.empty:
                continue
            best_row = sub.loc[sub["R2"].idxmax()]
            key = f"best_{family}_{tshort}"
            best[key] = {
                "variant": best_row["variant"],
                "R2": best_row["R2"],
                "MAE": best_row["MAE"],
                "RMSE": best_row["RMSE"],
            }
            # Also get deviation R²
            dev_row = df_dev[(df_dev["variant"] == best_row["variant"]) & (df_dev["target"] == tshort)]
            if not dev_row.empty:
                best[key]["R2_dev"] = dev_row.iloc[0]["R2_dev"]
                best[key]["MAE_dev"] = dev_row.iloc[0]["MAE_dev"]

            print(f"  {key}: {VARIANT_DISPLAY[best_row['variant']]} "
                  f"(R²={best_row['R2']:.4f})")

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: DIRECT 2D0 VS 2D1 COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_head_to_head(best, df_master, df_dev):
    """Head-to-head comparison of best 2D0 vs best 2D1."""
    print("\n" + "=" * 70)
    print("PHASE 5: DIRECT 2D0 VS 2D1 COMPARISON")
    print("=" * 70)

    lines = ["# Stage 2D Head-to-Head: Best 2D0 vs Best 2D1\n"]

    for tshort in ["EA", "IP"]:
        b0 = best.get(f"best_2D0_{tshort}")
        b1 = best.get(f"best_2D1_{tshort}")
        if not b0 or not b1:
            continue

        lines.append(f"## {tshort}\n")
        lines.append(f"| Metric | Best 2D0 ({VARIANT_DISPLAY[b0['variant']]}) | Best 2D1 ({VARIANT_DISPLAY[b1['variant']]}) | Δ (2D1 − 2D0) |")
        lines.append(f"|--------|------|------|------|")

        for metric in ["R2", "MAE", "RMSE"]:
            v0 = b0[metric]
            v1 = b1[metric]
            delta = v1 - v0
            sign = "+" if delta > 0 else ""
            lines.append(f"| Overall {metric} | {v0:.4f} | {v1:.4f} | {sign}{delta:.4f} |")

        if "R2_dev" in b0 and "R2_dev" in b1:
            for metric, label in [("R2_dev", "Arch-dev R²"), ("MAE_dev", "Arch-dev MAE")]:
                v0 = b0[metric]
                v1 = b1[metric]
                delta = v1 - v0
                sign = "+" if delta > 0 else ""
                lines.append(f"| {label} | {v0:.4f} | {v1:.4f} | {sign}{delta:.4f} |")

        lines.append("")

        dr2 = b1["R2"] - b0["R2"]
        print(f"  {tshort}: ΔR²(overall) = {dr2:+.4f}")
        if "R2_dev" in b0 and "R2_dev" in b1:
            dr2_dev = b1["R2_dev"] - b0["R2_dev"]
            print(f"  {tshort}: ΔR²(arch-dev) = {dr2_dev:+.4f}")

    outpath = OUT_DIR / "stage2d_head_to_head.md"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  → Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: PER-FOLD SIGNIFICANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def phase6_significance(best, per_fold_data, per_fold_dev):
    """Paired t-test on per-fold R² between best 2D0 and best 2D1."""
    print("\n" + "=" * 70)
    print("PHASE 6: PER-FOLD SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    lines = ["# Significance Analysis: Best 2D0 vs Best 2D1\n"]

    for tshort in ["EA", "IP"]:
        b0 = best.get(f"best_2D0_{tshort}")
        b1 = best.get(f"best_2D1_{tshort}")
        if not b0 or not b1:
            continue

        lines.append(f"## {tshort}\n")

        # Overall R² per fold
        lines.append("### Overall R² (per fold)\n")
        folds_0 = per_fold_data[b0["variant"]][tshort]
        folds_1 = per_fold_data[b1["variant"]][tshort]

        if folds_0 and folds_1 and len(folds_0) == len(folds_1):
            r2_0 = np.array([f["R2"] for f in folds_0])
            r2_1 = np.array([f["R2"] for f in folds_1])

            lines.append(f"| Fold | {VARIANT_DISPLAY[b0['variant']]} R² | {VARIANT_DISPLAY[b1['variant']]} R² | Diff |")
            lines.append(f"|------|------|------|------|")
            for i in range(len(r2_0)):
                lines.append(f"| {i} | {r2_0[i]:.4f} | {r2_1[i]:.4f} | {r2_1[i]-r2_0[i]:+.4f} |")

            diff = r2_1 - r2_0
            t_stat, p_val = stats.ttest_rel(r2_1, r2_0)
            lines.append(f"\n- Mean difference: {diff.mean():+.4f} ± {diff.std():.4f}")
            lines.append(f"- Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                winner = "2D1" if diff.mean() > 0 else "2D0"
                lines.append(f"- **Significant at p<0.05**: {winner} is better")
            else:
                lines.append(f"- **Not significant at p<0.05**: Differences are not statistically reliable")

            print(f"  {tshort} overall R²: Δ={diff.mean():+.4f}±{diff.std():.4f}, p={p_val:.4f}")
        else:
            lines.append("Insufficient per-fold data.")

        # Architecture-deviation R² per fold
        lines.append("\n### Architecture-Deviation R² (per fold)\n")
        dev_0 = per_fold_dev.get(b0["variant"], {}).get(tshort, [])
        dev_1 = per_fold_dev.get(b1["variant"], {}).get(tshort, [])

        if dev_0 and dev_1 and len(dev_0) == len(dev_1):
            dev_0 = np.array(dev_0)
            dev_1 = np.array(dev_1)

            lines.append(f"| Fold | {VARIANT_DISPLAY[b0['variant']]} R²(Δ) | {VARIANT_DISPLAY[b1['variant']]} R²(Δ) | Diff |")
            lines.append(f"|------|------|------|------|")
            for i in range(len(dev_0)):
                lines.append(f"| {i} | {dev_0[i]:.4f} | {dev_1[i]:.4f} | {dev_1[i]-dev_0[i]:+.4f} |")

            diff_dev = dev_1 - dev_0
            t_stat_d, p_val_d = stats.ttest_rel(dev_1, dev_0)
            lines.append(f"\n- Mean difference: {diff_dev.mean():+.4f} ± {diff_dev.std():.4f}")
            lines.append(f"- Paired t-test: t={t_stat_d:.4f}, p={p_val_d:.4f}")
            if p_val_d < 0.05:
                winner = "2D1" if diff_dev.mean() > 0 else "2D0"
                lines.append(f"- **Significant at p<0.05**: {winner} has better arch-deviation R²")
            else:
                lines.append(f"- **Not significant at p<0.05**")

            print(f"  {tshort} arch-dev R²: Δ={diff_dev.mean():+.4f}±{diff_dev.std():.4f}, p={p_val_d:.4f}")
        else:
            lines.append("Insufficient per-fold data.")

        lines.append("")

    # Also compare all 2D variants against Frac
    lines.append("## All Variants vs Frac (Overall R²)\n")
    for tshort in ["EA", "IP"]:
        frac_folds = per_fold_data["frac"][tshort]
        if not frac_folds:
            continue
        frac_r2 = np.array([f["R2"] for f in frac_folds])
        lines.append(f"### {tshort}\n")
        lines.append(f"| Variant | Mean R² | Δ vs Frac | p-value |")
        lines.append(f"|---------|---------|-----------|---------|")
        for v in VARIANTS:
            v_folds = per_fold_data[v][tshort]
            if not v_folds or len(v_folds) != len(frac_folds):
                continue
            v_r2 = np.array([f["R2"] for f in v_folds])
            diff = v_r2 - frac_r2
            if v == "frac":
                lines.append(f"| {VARIANT_DISPLAY[v]} | {v_r2.mean():.4f} | — | — |")
            else:
                _, p = stats.ttest_rel(v_r2, frac_r2)
                lines.append(f"| {VARIANT_DISPLAY[v]} | {v_r2.mean():.4f} | {diff.mean():+.4f} | {p:.4f} |")
        lines.append("")

    outpath = OUT_DIR / "significance_analysis.md"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  → Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: PAPER-LEVEL INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

def phase7_interpretation(df_master, df_dev, best, per_fold_data, per_fold_dev):
    """Generate paper-level interpretation."""
    print("\n" + "=" * 70)
    print("PHASE 7: PAPER-LEVEL INTERPRETATION")
    print("=" * 70)

    lines = ["# Stage 2D: Paper-Level Interpretation\n"]

    # Q1: Does architecture improve over Frac?
    lines.append("## 1. Does architecture improve over Frac?\n")
    frac_ea = df_master[(df_master["variant"]=="frac") & (df_master["target"]=="EA")]["R2"].values
    frac_ip = df_master[(df_master["variant"]=="frac") & (df_master["target"]=="IP")]["R2"].values

    any_improvement = False
    for v in VARIANTS[1:]:
        for tshort, frac_val in [("EA", frac_ea), ("IP", frac_ip)]:
            v_val = df_master[(df_master["variant"]==v) & (df_master["target"]==tshort)]["R2"].values
            if len(v_val) > 0 and len(frac_val) > 0 and v_val[0] > frac_val[0] + 0.001:
                any_improvement = True

    if any_improvement:
        lines.append("**YES.** Multiple architecture-aware variants improve over the Frac baseline.\n")
    else:
        lines.append("**NO.** No architecture-aware variant meaningfully improves over Frac.\n")

    # Show improvement table
    lines.append("| Variant | EA R² | ΔEA | IP R² | ΔIP |")
    lines.append("|---------|-------|-----|-------|-----|")
    for v in VARIANTS:
        ea_r2 = df_master[(df_master["variant"]==v) & (df_master["target"]=="EA")]["R2"].values
        ip_r2 = df_master[(df_master["variant"]==v) & (df_master["target"]=="IP")]["R2"].values
        if len(ea_r2) == 0 or len(ip_r2) == 0:
            continue
        d_ea = ea_r2[0] - frac_ea[0] if len(frac_ea) > 0 else 0
        d_ip = ip_r2[0] - frac_ip[0] if len(frac_ip) > 0 else 0
        lines.append(f"| {VARIANT_DISPLAY[v]} | {ea_r2[0]:.4f} | {d_ea:+.4f} | {ip_r2[0]:.4f} | {d_ip:+.4f} |")
    lines.append("")

    # Q2: Is 2D0 sufficient?
    lines.append("## 2. Is a global architecture model (2D0) sufficient?\n")
    b0_ea = best.get("best_2D0_EA", {})
    b0_ip = best.get("best_2D0_IP", {})
    b1_ea = best.get("best_2D1_EA", {})
    b1_ip = best.get("best_2D1_IP", {})

    if b0_ea and b1_ea:
        d_ea = b1_ea["R2"] - b0_ea["R2"]
        d_ip = b1_ip["R2"] - b0_ip["R2"] if b0_ip and b1_ip else 0
        if abs(d_ea) < 0.003 and abs(d_ip) < 0.003:
            lines.append(f"**YES.** 2D0 and 2D1 achieve nearly identical overall R² "
                        f"(ΔR²_EA={d_ea:+.4f}, ΔR²_IP={d_ip:+.4f}).\n")
        elif d_ea > 0.003 or d_ip > 0.003:
            lines.append(f"**POSSIBLY NOT.** 2D1 shows meaningful improvement "
                        f"(ΔR²_EA={d_ea:+.4f}, ΔR²_IP={d_ip:+.4f}).\n")
        else:
            lines.append(f"**YES.** 2D0 matches or exceeds 2D1 "
                        f"(ΔR²_EA={d_ea:+.4f}, ΔR²_IP={d_ip:+.4f}).\n")

    # Q3: Does chemistry-conditioned architecture (2D1) help?
    lines.append("## 3. Does chemistry-conditioned architecture modeling (2D1) help?\n")
    if b0_ea and b1_ea:
        dr2_ea = b1_ea["R2"] - b0_ea["R2"]
        dr2_ip = b1_ip["R2"] - b0_ip["R2"] if b0_ip and b1_ip else 0

        # Check arch-deviation
        dr2_dev_ea = (b1_ea.get("R2_dev", 0) - b0_ea.get("R2_dev", 0)) if "R2_dev" in b1_ea and "R2_dev" in b0_ea else None
        dr2_dev_ip = (b1_ip.get("R2_dev", 0) - b0_ip.get("R2_dev", 0)) if b1_ip and b0_ip and "R2_dev" in b1_ip and "R2_dev" in b0_ip else None

        lines.append(f"Overall R²: ΔEA={dr2_ea:+.4f}, ΔIP={dr2_ip:+.4f}")
        if dr2_dev_ea is not None:
            lines.append(f"Arch-deviation R²: ΔEA={dr2_dev_ea:+.4f}" + (f", ΔIP={dr2_dev_ip:+.4f}" if dr2_dev_ip else ""))
        lines.append("")

        if (dr2_dev_ea and dr2_dev_ea > 0.01) or (dr2_dev_ip and dr2_dev_ip > 0.01):
            lines.append("**The chemistry-conditioned model (2D1) better captures architecture-specific effects.**\n")
        elif abs(dr2_ea) < 0.003 and abs(dr2_ip) < 0.003:
            lines.append("**No meaningful improvement from chemistry conditioning. The simpler 2D0 is sufficient.**\n")
        else:
            lines.append("**Mixed results. See detailed metrics above.**\n")

    # Q4: Which mechanism?
    lines.append("## 4. Which architecture mechanism is supported?\n")
    lines.append("Two candidate mechanisms:\n")
    lines.append("- **(A)** Δy_arch = g(arch) — global architecture offset")
    lines.append("- **(B)** Δy_arch = g(arch) + h(h_A, h_B, f_A, f_B, arch) — chemistry-conditioned\n")

    if b0_ea and b1_ea:
        if (b1_ea.get("R2_dev", 0) > b0_ea.get("R2_dev", 0) + 0.01) or \
           (b1_ip and b0_ip and b1_ip.get("R2_dev", 0) > b0_ip.get("R2_dev", 0) + 0.01):
            lines.append("**Evidence favors (B)**: Chemistry-conditioned model captures more architecture variation.\n")
        else:
            lines.append("**Evidence favors (A)**: A global architecture offset is sufficient; "
                        "chemistry conditioning does not meaningfully improve architecture-deviation prediction.\n")

    # Q5: Is extra complexity justified?
    lines.append("## 5. Is the extra complexity of 2D1 justified?\n")
    if b0_ea and b1_ea:
        if abs(b1_ea["R2"] - b0_ea["R2"]) < 0.003 and \
           (not b0_ip or not b1_ip or abs(b1_ip["R2"] - b0_ip["R2"]) < 0.003):
            lines.append("**No.** The additional parameters and computational cost of the 2D1 interaction MLP "
                        "do not yield a meaningful improvement over 2D0.\n")
        else:
            lines.append("**Depends on use case.** See metrics for details.\n")

    # Q6: Final model selection
    lines.append("## 6. What model should be selected as the final Stage 2D architecture?\n")

    # Determine overall best
    all_arch = df_master[df_master["variant"] != "frac"]
    if not all_arch.empty:
        # Score by mean R² across targets
        scores = {}
        for v in VARIANTS[1:]:
            ea_r2 = df_master[(df_master["variant"]==v) & (df_master["target"]=="EA")]["R2"].values
            ip_r2 = df_master[(df_master["variant"]==v) & (df_master["target"]=="IP")]["R2"].values
            if len(ea_r2) > 0 and len(ip_r2) > 0:
                scores[v] = (ea_r2[0] + ip_r2[0]) / 2
        if scores:
            best_overall = max(scores, key=scores.get)
            lines.append(f"**Recommended: {VARIANT_DISPLAY[best_overall]}** "
                        f"(mean R² across EA+IP = {scores[best_overall]:.4f})\n")

            # Also mention runner-up
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            if len(sorted_scores) > 1:
                runner = sorted_scores[1]
                lines.append(f"Runner-up: {VARIANT_DISPLAY[runner[0]]} "
                            f"(mean R² = {runner[1]:.4f}, Δ = {runner[1]-scores[best_overall]:+.4f})\n")

    # Q7: What to report
    lines.append("## 7. What should be reported in the paper?\n")
    lines.append("Recommended reporting:\n")
    lines.append("1. Frac baseline establishes performance without architecture information")
    lines.append("2. All 2D0 and 2D1 variants show [improvement/no improvement] over Frac")
    lines.append("3. Best overall model and supporting metrics")
    lines.append("4. Architecture-deviation R² demonstrates ability to capture architecture-specific effects")
    lines.append("5. Statistical significance of improvements (paired t-test across folds)")
    lines.append("6. Conclusion on whether chemistry-conditioned architecture modeling is necessary")
    lines.append("")

    outpath = OUT_DIR / "final_stage2d_interpretation.md"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  → Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: FINAL RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

def phase8_final_recommendation(df_master, df_dev, best, per_fold_data, per_fold_dev):
    """Produce executive summary."""
    print("\n" + "=" * 70)
    print("PHASE 8: FINAL RECOMMENDATION")
    print("=" * 70)

    lines = ["# Stage 2D Final Recommendation\n"]
    lines.append("## Executive Summary\n")

    # Determine best model
    scores = {}
    for v in VARIANTS[1:]:
        ea_r2 = df_master[(df_master["variant"]==v) & (df_master["target"]=="EA")]["R2"].values
        ip_r2 = df_master[(df_master["variant"]==v) & (df_master["target"]=="IP")]["R2"].values
        if len(ea_r2) > 0 and len(ip_r2) > 0:
            scores[v] = (ea_r2[0] + ip_r2[0]) / 2

    best_variant = max(scores, key=scores.get) if scores else "frac"
    frac_ea = df_master[(df_master["variant"]=="frac") & (df_master["target"]=="EA")]["R2"].values[0]
    frac_ip = df_master[(df_master["variant"]=="frac") & (df_master["target"]=="IP")]["R2"].values[0]

    best_ea = df_master[(df_master["variant"]==best_variant) & (df_master["target"]=="EA")]["R2"].values[0]
    best_ip = df_master[(df_master["variant"]==best_variant) & (df_master["target"]=="IP")]["R2"].values[0]

    lines.append(f"### Selected Final Model: **{VARIANT_DISPLAY[best_variant]}**\n")

    # Why selected
    lines.append("### Why It Was Selected\n")
    lines.append(f"- Highest mean R² across both targets ({scores[best_variant]:.4f})")
    lines.append(f"- EA R² = {best_ea:.4f} (Frac baseline: {frac_ea:.4f}, Δ = {best_ea-frac_ea:+.4f})")
    lines.append(f"- IP R² = {best_ip:.4f} (Frac baseline: {frac_ip:.4f}, Δ = {best_ip-frac_ip:+.4f})")
    lines.append("")

    # Supporting metrics table
    lines.append("### Supporting Metrics\n")
    lines.append("| Variant | EA R² | IP R² | Mean R² |")
    lines.append("|---------|-------|-------|---------|")
    sorted_variants = sorted(scores.items(), key=lambda x: -x[1])
    lines.append(f"| Frac (baseline) | {frac_ea:.4f} | {frac_ip:.4f} | {(frac_ea+frac_ip)/2:.4f} |")
    for v, s in sorted_variants:
        v_ea = df_master[(df_master["variant"]==v) & (df_master["target"]=="EA")]["R2"].values[0]
        v_ip = df_master[(df_master["variant"]==v) & (df_master["target"]=="IP")]["R2"].values[0]
        marker = " ←" if v == best_variant else ""
        lines.append(f"| {VARIANT_DISPLAY[v]} | {v_ea:.4f} | {v_ip:.4f} | {s:.4f} |{marker}")
    lines.append("")

    # Architecture-deviation evidence
    lines.append("### Architecture-Deviation Evidence\n")
    lines.append("| Variant | EA R²(Δ) | IP R²(Δ) |")
    lines.append("|---------|----------|----------|")
    frac_dev_ea = df_dev[(df_dev["variant"]=="frac") & (df_dev["target"]=="EA")]["R2_dev"].values
    frac_dev_ip = df_dev[(df_dev["variant"]=="frac") & (df_dev["target"]=="IP")]["R2_dev"].values
    frac_dev_ea_v = frac_dev_ea[0] if len(frac_dev_ea) > 0 else float("nan")
    frac_dev_ip_v = frac_dev_ip[0] if len(frac_dev_ip) > 0 else float("nan")
    lines.append(f"| Frac (baseline) | {frac_dev_ea_v:.4f} | {frac_dev_ip_v:.4f} |")
    for v, _ in sorted_variants:
        dev_ea = df_dev[(df_dev["variant"]==v) & (df_dev["target"]=="EA")]["R2_dev"].values
        dev_ip = df_dev[(df_dev["variant"]==v) & (df_dev["target"]=="IP")]["R2_dev"].values
        ea_str = f"{dev_ea[0]:.4f}" if len(dev_ea) > 0 else "—"
        ip_str = f"{dev_ip[0]:.4f}" if len(dev_ip) > 0 else "—"
        lines.append(f"| {VARIANT_DISPLAY[v]} | {ea_str} | {ip_str} |")
    lines.append("")

    # 2D1 scientific justification
    lines.append("### Is 2D1 Scientifically Justified?\n")
    b0_ea = best.get("best_2D0_EA", {})
    b1_ea = best.get("best_2D1_EA", {})
    b0_ip = best.get("best_2D0_IP", {})
    b1_ip = best.get("best_2D1_IP", {})

    if b0_ea and b1_ea:
        dr2_ea = b1_ea["R2"] - b0_ea["R2"]
        dr2_ip = b1_ip["R2"] - b0_ip["R2"] if b0_ip and b1_ip else 0
        dr2_dev_ea = (b1_ea.get("R2_dev", 0) - b0_ea.get("R2_dev", 0))
        dr2_dev_ip = (b1_ip.get("R2_dev", 0) - b0_ip.get("R2_dev", 0)) if b0_ip and b1_ip else 0

        lines.append(f"- Overall R² improvement: EA={dr2_ea:+.4f}, IP={dr2_ip:+.4f}")
        lines.append(f"- Arch-deviation R² improvement: EA={dr2_dev_ea:+.4f}, IP={dr2_dev_ip:+.4f}")
        lines.append("")

        if abs(dr2_ea) < 0.003 and abs(dr2_ip) < 0.003 and abs(dr2_dev_ea) < 0.01 and abs(dr2_dev_ip) < 0.01:
            lines.append("**Conclusion**: The chemistry-conditioned interaction MLP (2D1) does NOT provide "
                        "meaningful improvement over the simpler global architecture embedding (2D0). "
                        "The additional complexity is not justified.")
        elif dr2_dev_ea > 0.01 or dr2_dev_ip > 0.01:
            lines.append("**Conclusion**: 2D1 provides better architecture-deviation prediction, "
                        "suggesting that chemistry-conditioned architecture modeling captures "
                        "real effects. The complexity may be justified for applications where "
                        "architecture-specific predictions matter.")
        else:
            lines.append("**Conclusion**: See detailed metrics above for nuanced assessment.")
    lines.append("")

    # Caveats
    lines.append("### Remaining Caveats\n")
    lines.append("1. Predictions were in normalized space due to the UnscaleTransform bug; "
                "all metrics use post-hoc inverse transform")
    lines.append("2. Architecture-deviation analysis requires matching predictions to metadata "
                "via y_true lookup, which may miss some samples")
    lines.append("3. Only 3 architecture types (alternating, random, block) with unequal representation")
    lines.append("4. 5-fold cross-validation provides limited statistical power for paired comparisons")
    lines.append("5. The dead-initialization fix (alpha_init=0.1) was applied only to 2d1_fixed and 2d1_arch; "
                "original 2d0 variants used alpha_init=0.0 but did not suffer from the dead-branch issue "
                "because their arch_embedding was randomly initialized (non-zero)")
    lines.append("")

    outpath = OUT_DIR / "stage2d_final_recommendation.md"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  → Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Stage 2D Post-Rerun Analysis")
    print("=" * 70)

    # Estimate normalization parameters
    print("Estimating normalization parameters...")
    train_stats = estimate_normalization_params()
    for target in TARGETS:
        tshort = TARGET_SHORT[target]
        means = [s[0] for s in train_stats[target]]
        stds = [s[1] for s in train_stats[target]]
        print(f"  {tshort}: mean={np.mean(means):.4f}, std={np.mean(stds):.4f}")

    # Phase 1
    phase1_verify_rerun(train_stats)

    # Phase 2
    df_master, per_fold_data = phase2_master_results(train_stats)

    # Phase 3
    df_dev, per_fold_dev = phase3_architecture_deviations(train_stats)

    # Phase 4
    best = phase4_identify_best(df_master, df_dev)

    # Phase 5
    phase5_head_to_head(best, df_master, df_dev)

    # Phase 6
    phase6_significance(best, per_fold_data, per_fold_dev)

    # Phase 7
    phase7_interpretation(df_master, df_dev, best, per_fold_data, per_fold_dev)

    # Phase 8
    phase8_final_recommendation(df_master, df_dev, best, per_fold_data, per_fold_dev)

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)
