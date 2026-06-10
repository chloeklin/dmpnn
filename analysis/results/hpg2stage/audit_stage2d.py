"""Full audit of HPG2Stage Stage 2D evaluation pipeline.

Tasks 1-8: Verify R², alignment, architecture deviations, compare with Diagnostic 3B.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = PROJECT_ROOT / "predictions" / "HPG2Stage"
RESULTS_DIR = PROJECT_ROOT / "results" / "HPG2Stage"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "audit_output"
AUDIT_PLOTS_DIR = OUT_DIR / "audit_plots"

VARIANT_PATTERN = re.compile(r"copoly_(stage2d_\w+?)__")
TARGET_RAW = {"EA": "EA vs SHE (eV)", "IP": "IP vs SHE (eV)"}
TARGET_SHORT = {v: k for k, v in TARGET_RAW.items()}

MODEL_KEYS = [
    "stage2d_frac",
    "stage2d_2d0_fixed",
    "stage2d_2d0_arch",
    "stage2d_2d0_gate",
    "stage2d_2d1_fixed",
    "stage2d_2d1_arch",
    "stage2d_2d1_gate",
]

MODEL_DISPLAY = {
    "stage2d_frac":      "Frac",
    "stage2d_2d0_fixed": "2D0-fixed",
    "stage2d_2d0_arch":  "2D0-arch",
    "stage2d_2d0_gate":  "2D0-gate",
    "stage2d_2d1_fixed": "2D1-fixed",
    "stage2d_2d1_arch":  "2D1-arch",
    "stage2d_2d1_gate":  "2D1-gate",
}


def load_predictions_by_split():
    """Load all .npz files -> dict[model][target][split] = (y_true, y_pred, ids)."""
    data = {}
    for npz_path in sorted(PRED_DIR.glob("ea_ip*stage2d*.npz")):
        m = VARIANT_PATTERN.search(npz_path.stem)
        if not m:
            continue
        variant = m.group(1)

        target = None
        for raw, short in TARGET_SHORT.items():
            if raw in npz_path.stem:
                target = short
                break
        if target is None:
            continue

        split_m = re.search(r"split(\d+)", npz_path.stem)
        split_idx = int(split_m.group(1)) if split_m else -1

        d = np.load(npz_path, allow_pickle=True)
        yt = d["y_true"].flatten()
        yp = d["y_pred"].flatten()
        ids = d["test_ids"] if "test_ids" in d else None
        meta = d["metadata"].item() if "metadata" in d else {}

        data.setdefault(variant, {}).setdefault(target, {})[split_idx] = {
            "y_true": yt, "y_pred": yp, "ids": ids, "metadata": meta, "path": npz_path.name
        }
    return data


def load_result_csvs():
    """Load result CSVs -> dict[model][target] = DataFrame(fold, mae, rmse, r2)."""
    data = {}
    for csv_path in sorted(RESULTS_DIR.glob("*stage2d*.csv")):
        m = VARIANT_PATTERN.search(csv_path.stem)
        if not m:
            continue
        variant = m.group(1)

        df = pd.read_csv(csv_path)
        col_map = {"test/rmse": "rmse", "test/r2": "r2", "test/mae": "mae", "split": "fold"}
        df = df.rename(columns=col_map)

        if "target" in df.columns:
            target_raw = df["target"].iloc[0]
            target = TARGET_SHORT.get(target_raw, target_raw)
        else:
            target = None
            for raw, short in TARGET_SHORT.items():
                if raw in csv_path.stem:
                    target = short
                    break

        if target is None:
            continue

        data.setdefault(variant, {})[target] = df
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: Verify overall R² computation
# ═══════════════════════════════════════════════════════════════════════════════

def task1_verify_r2(preds, result_csvs):
    print("=" * 70)
    print("TASK 1: VERIFY OVERALL R² COMPUTATION")
    print("=" * 70)

    lines = []
    lines.append("TASK 1: Overall R² Audit")
    lines.append("=" * 50)

    for model in MODEL_KEYS:
        if model not in preds:
            print(f"\n  {model}: NO PREDICTIONS FOUND")
            continue

        for target in ["EA", "IP"]:
            if target not in preds[model]:
                continue

            # Pool across splits
            all_yt = []
            all_yp = []
            per_split_r2 = {}
            for split_idx, sd in sorted(preds[model][target].items()):
                yt = sd["y_true"]
                yp = sd["y_pred"]
                all_yt.append(yt)
                all_yp.append(yp)

                sse_s = np.sum((yt - yp) ** 2)
                sst_s = np.sum((yt - yt.mean()) ** 2)
                r2_s = 1 - sse_s / sst_s if sst_s > 0 else float("nan")
                per_split_r2[split_idx] = r2_s

            yt_all = np.concatenate(all_yt)
            yp_all = np.concatenate(all_yp)

            n = len(yt_all)
            mean_yt = yt_all.mean()
            mean_yp = yp_all.mean()
            var_yt = yt_all.var()
            var_yp = yp_all.var()
            sse = np.sum((yt_all - yp_all) ** 2)
            sst = np.sum((yt_all - mean_yt) ** 2)
            r2_manual = 1 - sse / sst if sst > 0 else float("nan")
            r2_sklearn = r2_score(yt_all, yp_all)
            corr = np.corrcoef(yt_all, yp_all)[0, 1]

            # Get CSV-reported R² (mean across folds)
            csv_r2 = float("nan")
            if model in result_csvs and target in result_csvs[model]:
                csv_df = result_csvs[model][target]
                csv_r2 = csv_df["r2"].mean()

            msg = textwrap.dedent(f"""
            --- {MODEL_DISPLAY.get(model, model)} / {target} ---
            n_samples (pooled)  : {n}
            mean(y_true)        : {mean_yt:.6f}
            mean(y_pred)        : {mean_yp:.6f}
            var(y_true)         : {var_yt:.6f}
            var(y_pred)         : {var_yp:.6f}
            SSE                 : {sse:.4f}
            SST                 : {sst:.4f}
            SSE/SST             : {sse / sst:.6f}
            Manual R² (pooled)  : {r2_manual:.6f}
            sklearn R² (pooled) : {r2_sklearn:.6f}
            Pearson corr        : {corr:.6f}
            CSV mean R² (folds) : {csv_r2:.6f}
            Per-split R²        : {per_split_r2}
            """)
            print(msg)
            lines.append(msg)

            # Key diagnostic: is y_pred on different scale?
            if abs(mean_yt - mean_yp) > 2 * np.sqrt(var_yt):
                warn = f"  *** WARNING: mean(y_pred) is {abs(mean_yt - mean_yp):.4f} away from mean(y_true) — possible normalization mismatch ***"
                print(warn)
                lines.append(warn)

            if var_yp < 0.01 * var_yt:
                warn = f"  *** WARNING: var(y_pred) << var(y_true) — predictions may be near-constant ***"
                print(warn)
                lines.append(warn)

    with open(OUT_DIR / "evaluation_audit_r2.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Written: evaluation_audit_r2.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: Verify prediction/target alignment
# ═══════════════════════════════════════════════════════════════════════════════

def task2_verify_alignment(preds):
    print("\n" + "=" * 70)
    print("TASK 2: VERIFY PREDICTION/TARGET ALIGNMENT")
    print("=" * 70)

    rng = np.random.RandomState(42)

    for model in MODEL_KEYS:
        if model not in preds:
            continue

        for target in ["EA", "IP"]:
            if target not in preds[model]:
                continue

            # Use first available split
            split_idx = sorted(preds[model][target].keys())[0]
            sd = preds[model][target][split_idx]
            yt = sd["y_true"]
            yp = sd["y_pred"]
            ids = sd["ids"]

            n = len(yt)
            sample_idx = rng.choice(n, size=min(20, n), replace=False)
            sample_idx = np.sort(sample_idx)

            print(f"\n  --- {MODEL_DISPLAY.get(model, model)} / {target} / split {split_idx} ---")
            print(f"  {'idx':>6s}  {'test_id':>10s}  {'y_true':>10s}  {'y_pred':>10s}  {'residual':>10s}")
            for i in sample_idx:
                tid = ids[i] if ids is not None else "N/A"
                print(f"  {i:6d}  {str(tid):>10s}  {yt[i]:10.4f}  {yp[i]:10.4f}  {yt[i] - yp[i]:10.4f}")

            # Correlation check
            corr = np.corrcoef(yt, yp)[0, 1]
            r2 = r2_score(yt, yp)
            print(f"  Pearson r       : {corr:.6f}")
            print(f"  R² (sklearn)    : {r2:.6f}")

            # Shuffle test: if we shuffle predictions, does R² change much?
            yp_shuffled = yp.copy()
            rng.shuffle(yp_shuffled)
            r2_shuffled = r2_score(yt, yp_shuffled)
            print(f"  R² (shuffled)   : {r2_shuffled:.6f}")
            if r2 < r2_shuffled:
                print(f"  *** WARNING: shuffled R² > actual R² — predictions may be misaligned ***")

            # Break after first model for brevity in output
        break  # only one model for detailed alignment check

    # Summary across all models
    print("\n  Summary (all models, pooled across splits):")
    print(f"  {'Model':>15s}  {'Target':>4s}  {'corr':>8s}  {'R²':>10s}  {'MAE':>8s}")
    for model in MODEL_KEYS:
        if model not in preds:
            continue
        for target in ["EA", "IP"]:
            if target not in preds[model]:
                continue
            yt_all = np.concatenate([sd["y_true"] for sd in preds[model][target].values()])
            yp_all = np.concatenate([sd["y_pred"] for sd in preds[model][target].values()])
            corr = np.corrcoef(yt_all, yp_all)[0, 1]
            r2 = r2_score(yt_all, yp_all)
            mae = mean_absolute_error(yt_all, yp_all)
            print(f"  {MODEL_DISPLAY.get(model, model):>15s}  {target:>4s}  {corr:8.4f}  {r2:10.4f}  {mae:8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: Verify architecture-deviation computation
# ═══════════════════════════════════════════════════════════════════════════════

def task3_verify_arch_deviation(preds, dataset_df):
    print("\n" + "=" * 70)
    print("TASK 3: VERIFY ARCHITECTURE-DEVIATION COMPUTATION")
    print("=" * 70)

    ea_col = "EA vs SHE (eV)"
    ip_col = "IP vs SHE (eV)"
    target_col_map = {"EA": ea_col, "IP": ip_col}

    # Build lookup: y_true → (smiles_A, smiles_B, fracA, poly_type)
    print("\n  Building y_true → metadata lookup...")

    ea_lookup = {}
    ip_lookup = {}
    for _, row in dataset_df.iterrows():
        group_key = f"{row['smiles_A']}|{row['smiles_B']}|{row['fracA']:.3f}"
        arch = row["poly_type"]
        if pd.notna(row[ea_col]):
            ea_lookup[round(float(row[ea_col]), 6)] = {"group": group_key, "arch": arch}
        if pd.notna(row[ip_col]):
            ip_lookup[round(float(row[ip_col]), 6)] = {"group": group_key, "arch": arch}

    print(f"  EA lookup size: {len(ea_lookup)} (dataset: {len(dataset_df)})")
    print(f"  IP lookup size: {len(ip_lookup)} (dataset: {len(dataset_df)})")

    # Check for collisions (same y_true mapping to different metadata)
    # Count unique y_true values
    ea_vals = dataset_df[ea_col].dropna().round(6)
    ip_vals = dataset_df[ip_col].dropna().round(6)
    print(f"  EA unique y_true values: {ea_vals.nunique()} / {len(ea_vals)} "
          f"(collision rate: {1 - ea_vals.nunique() / len(ea_vals):.2%})")
    print(f"  IP unique y_true values: {ip_vals.nunique()} / {len(ip_vals)} "
          f"(collision rate: {1 - ip_vals.nunique() / len(ip_vals):.2%})")

    lookups = {"EA": ea_lookup, "IP": ip_lookup}

    # Verify deviation computation for one model
    model = "stage2d_2d0_arch"  # pick a representative model
    if model not in preds:
        model = next((m for m in MODEL_KEYS if m in preds and m != "stage2d_frac"), None)
    if model is None:
        print("  No model predictions available for verification.")
        return

    print(f"\n  Detailed verification for: {MODEL_DISPLAY.get(model, model)}")

    for target in ["EA", "IP"]:
        if target not in preds[model]:
            continue

        lut = lookups[target]

        # Pool all splits
        yt_all = np.concatenate([sd["y_true"] for sd in preds[model][target].values()])
        yp_all = np.concatenate([sd["y_pred"] for sd in preds[model][target].values()])

        # Match to metadata
        groups = []
        archs = []
        matched_count = 0
        for yt_val in yt_all:
            meta = lut.get(round(float(yt_val), 6), None)
            if meta:
                groups.append(meta["group"])
                archs.append(meta["arch"])
                matched_count += 1
            else:
                groups.append(None)
                archs.append(None)

        match_rate = matched_count / len(yt_all)
        print(f"\n  {target}: matched {matched_count}/{len(yt_all)} ({match_rate:.1%})")

        df = pd.DataFrame({
            "y_true": yt_all.astype(float),
            "y_pred": yp_all.astype(float),
            "group": groups,
            "arch": archs,
        })
        df_matched = df.dropna(subset=["group"])

        # Group stats
        group_counts = df_matched.groupby("group").size()
        print(f"  Total groups: {len(group_counts)}")
        print(f"  Groups with >1 member: {(group_counts > 1).sum()}")
        print(f"  Samples in multi-member groups: {group_counts[group_counts > 1].sum()}")

        # Compute deviations
        group_mean_true = df_matched.groupby("group")["y_true"].transform("mean")
        group_mean_pred = df_matched.groupby("group")["y_pred"].transform("mean")
        df_matched = df_matched.copy()
        df_matched["delta_true"] = df_matched["y_true"].values - group_mean_true.values
        df_matched["delta_pred"] = df_matched["y_pred"].values - group_mean_pred.values

        multi = df_matched[df_matched.groupby("group")["y_true"].transform("count") > 1]

        if len(multi) < 2:
            print(f"  Not enough multi-member groups for deviation analysis")
            continue

        dt = multi["delta_true"].values
        dp = multi["delta_pred"].values

        r2_dev_manual = 1 - np.sum((dt - dp) ** 2) / np.sum((dt - dt.mean()) ** 2)
        r2_dev_sklearn = r2_score(dt, dp)

        print(f"  Deviation R² (manual) : {r2_dev_manual:.6f}")
        print(f"  Deviation R² (sklearn): {r2_dev_sklearn:.6f}")
        print(f"  Deviation MAE         : {np.mean(np.abs(dt - dp)):.6f}")
        print(f"  mean(delta_true)      : {dt.mean():.6f}")
        print(f"  std(delta_true)       : {dt.std():.6f}")
        print(f"  mean(delta_pred)      : {dp.mean():.6f}")
        print(f"  std(delta_pred)       : {dp.std():.6f}")

        # Show some example groups
        print(f"\n  Example groups (first 3 multi-member):")
        shown = 0
        for gname, gdf in df_matched.groupby("group"):
            if len(gdf) < 2:
                continue
            if shown >= 3:
                break
            print(f"    Group: {gname[:60]}... ({len(gdf)} members)")
            for _, r in gdf.iterrows():
                print(f"      arch={r['arch']:>12s}  y_true={r['y_true']:8.4f}  y_pred={r['y_pred']:8.4f}  "
                      f"Δtrue={r.get('delta_true', float('nan')):8.4f}  Δpred={r.get('delta_pred', float('nan')):8.4f}")
            shown += 1


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4: Compare with Diagnostic 3B
# ═══════════════════════════════════════════════════════════════════════════════

def task4_compare_diagnostic3b():
    print("\n" + "=" * 70)
    print("TASK 4: COMPARE WITH DIAGNOSTIC 3B")
    print("=" * 70)

    # Find Diagnostic 3B code
    analysis_dir = PROJECT_ROOT / "analysis"
    diag3b_candidates = list(analysis_dir.rglob("*diagnostic*3*")) + \
                        list(analysis_dir.rglob("*diag*3b*")) + \
                        list(analysis_dir.rglob("*phase3*")) + \
                        list(analysis_dir.rglob("*3B*"))
    print("\n  Searching for Diagnostic 3B code...")
    for f in diag3b_candidates:
        print(f"    Found: {f.relative_to(PROJECT_ROOT)}")

    # Also check for HPG phase3 code
    hpg_dir = analysis_dir / "results" / "hpg"
    if hpg_dir.exists():
        phase3_files = list(hpg_dir.glob("*phase3*")) + list(hpg_dir.glob("*3b*"))
        for f in phase3_files:
            print(f"    HPG: {f.relative_to(PROJECT_ROOT)}")

    return diag3b_candidates


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5: Verify matched group construction
# ═══════════════════════════════════════════════════════════════════════════════

def task5_verify_groups(dataset_df):
    print("\n" + "=" * 70)
    print("TASK 5: VERIFY MATCHED GROUP CONSTRUCTION")
    print("=" * 70)

    # Build composition groups
    dataset_df = dataset_df.copy()
    dataset_df["group"] = dataset_df.apply(
        lambda r: f"{r['smiles_A']}|{r['smiles_B']}|{r['fracA']:.3f}", axis=1
    )

    total_polymers = len(dataset_df)
    total_groups = dataset_df["group"].nunique()

    print(f"\n  Total polymers: {total_polymers}")
    print(f"  Total composition groups: {total_groups}")

    # Architecture distribution per group
    group_arch = dataset_df.groupby("group")["poly_type"].apply(set)

    # Count by number of architectures
    n_arch_counts = group_arch.apply(len).value_counts().sort_index()
    print(f"\n  Groups by number of architectures:")
    for n, count in n_arch_counts.items():
        print(f"    {n} architecture(s): {count} groups")

    # Detailed breakdown
    arch_combo_counts = group_arch.apply(lambda s: "+".join(sorted(s))).value_counts()
    print(f"\n  Groups by architecture combination:")
    for combo, count in arch_combo_counts.items():
        print(f"    {combo}: {count}")

    # Samples in multi-architecture groups
    multi_arch_groups = group_arch[group_arch.apply(len) > 1].index
    samples_in_multi = dataset_df[dataset_df["group"].isin(multi_arch_groups)]
    print(f"\n  Samples in multi-architecture groups: {len(samples_in_multi)} / {total_polymers} ({len(samples_in_multi)/total_polymers:.1%})")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 6: Verify monomer-disjoint split
# ═══════════════════════════════════════════════════════════════════════════════

def task6_verify_split(preds, dataset_df):
    print("\n" + "=" * 70)
    print("TASK 6: VERIFY MONOMER-DISJOINT SPLIT")
    print("=" * 70)

    # Check what split is being used from metadata
    model = "stage2d_frac"
    if model not in preds:
        model = next(iter(preds.keys()), None)
    if model is None:
        print("  No predictions available.")
        return

    for target in ["EA", "IP"]:
        if target not in preds[model]:
            continue

        for split_idx, sd in sorted(preds[model][target].items()):
            meta = sd["metadata"]
            print(f"\n  {model}/{target}/split{split_idx} metadata:")
            for k, v in meta.items():
                print(f"    {k}: {v}")

            # Match test samples to dataset
            yt = sd["y_true"]
            target_col = TARGET_RAW[target]

            # Find which rows in dataset match test y_true
            ea_lookup = {}
            for idx, row in dataset_df.iterrows():
                if pd.notna(row[target_col]):
                    ea_lookup[round(float(row[target_col]), 6)] = idx

            test_indices = []
            for val in yt:
                idx = ea_lookup.get(round(float(val), 6), None)
                if idx is not None:
                    test_indices.append(idx)

            test_df = dataset_df.iloc[test_indices] if test_indices else pd.DataFrame()
            if test_df.empty:
                print(f"    Could not match test samples to dataset")
                continue

            # Get train samples (everything not in test)
            train_df = dataset_df.drop(index=test_df.index, errors="ignore")

            # Check monomer overlap
            test_monomers_a = set(test_df["smiles_A"].unique())
            train_monomers_a = set(train_df["smiles_A"].unique())
            test_monomers_b = set(test_df["smiles_B"].unique())
            train_monomers_b = set(train_df["smiles_B"].unique())

            overlap_a = test_monomers_a & train_monomers_a
            overlap_b = test_monomers_b & train_monomers_b

            print(f"\n    Test monomers A: {len(test_monomers_a)}, Train monomers A: {len(train_monomers_a)}")
            print(f"    Overlap A: {len(overlap_a)} monomers")
            print(f"    Test monomers B: {len(test_monomers_b)}, Train monomers B: {len(train_monomers_b)}")
            print(f"    Overlap B: {len(overlap_b)} monomers")

            if overlap_a:
                print(f"    *** Split is NOT monomer-A-disjoint (overlap: {len(overlap_a)}) ***")
            else:
                print(f"    Split IS monomer-A-disjoint ✓")

            # Architecture distribution in test set
            if "poly_type" in test_df.columns:
                arch_dist = test_df["poly_type"].value_counts()
                print(f"\n    Test set architecture distribution:")
                for arch, count in arch_dist.items():
                    print(f"      {arch}: {count}")

            break  # Only check first split per target
        break  # Only check first target


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 7: Verify Δy prediction quality directly
# ═══════════════════════════════════════════════════════════════════════════════

def task7_verify_delta_y(preds, dataset_df):
    print("\n" + "=" * 70)
    print("TASK 7: VERIFY Δy PREDICTION QUALITY")
    print("=" * 70)

    ea_col = "EA vs SHE (eV)"
    ip_col = "IP vs SHE (eV)"

    # Build lookups
    ea_lookup = {}
    ip_lookup = {}
    for _, row in dataset_df.iterrows():
        group_key = f"{row['smiles_A']}|{row['smiles_B']}|{row['fracA']:.3f}"
        arch = row["poly_type"]
        if pd.notna(row[ea_col]):
            ea_lookup[round(float(row[ea_col]), 6)] = {"group": group_key, "arch": arch}
        if pd.notna(row[ip_col]):
            ip_lookup[round(float(row[ip_col]), 6)] = {"group": group_key, "arch": arch}

    lookups = {"EA": ea_lookup, "IP": ip_lookup}

    print(f"\n  {'Model':>15s}  {'Tgt':>3s}  {'R²_dev(man)':>12s}  {'R²_dev(skl)':>12s}  {'MAE_dev':>10s}  {'n_multi':>8s}  {'n_matched':>10s}")

    for model in MODEL_KEYS:
        if model not in preds:
            continue
        if model == "stage2d_frac":
            pass  # include frac for comparison

        for target in ["EA", "IP"]:
            if target not in preds[model]:
                continue

            lut = lookups[target]
            yt_all = np.concatenate([sd["y_true"] for sd in preds[model][target].values()])
            yp_all = np.concatenate([sd["y_pred"] for sd in preds[model][target].values()])

            groups = []
            for yt_val in yt_all:
                meta = lut.get(round(float(yt_val), 6), None)
                groups.append(meta["group"] if meta else None)

            df = pd.DataFrame({
                "y_true": yt_all.astype(float),
                "y_pred": yp_all.astype(float),
                "group": groups,
            })
            df_matched = df.dropna(subset=["group"])

            group_mean_true = df_matched.groupby("group")["y_true"].transform("mean")
            group_mean_pred = df_matched.groupby("group")["y_pred"].transform("mean")
            df_matched = df_matched.copy()
            df_matched["delta_true"] = df_matched["y_true"].values - group_mean_true.values
            df_matched["delta_pred"] = df_matched["y_pred"].values - group_mean_pred.values

            multi = df_matched[df_matched.groupby("group")["y_true"].transform("count") > 1]

            if len(multi) < 2:
                print(f"  {MODEL_DISPLAY.get(model, model):>15s}  {target:>3s}  {'N/A':>12s}  {'N/A':>12s}  {'N/A':>10s}  {0:>8d}  {len(df_matched):>10d}")
                continue

            dt = multi["delta_true"].values
            dp = multi["delta_pred"].values

            sst = np.sum((dt - dt.mean()) ** 2)
            sse = np.sum((dt - dp) ** 2)
            r2_manual = 1 - sse / sst if sst > 0 else float("nan")
            r2_sklearn = r2_score(dt, dp)
            mae = mean_absolute_error(dt, dp)

            print(f"  {MODEL_DISPLAY.get(model, model):>15s}  {target:>3s}  {r2_manual:12.6f}  {r2_sklearn:12.6f}  {mae:10.6f}  {len(multi):>8d}  {len(df_matched):>10d}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 8: Visual diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def task8_visual_diagnostics(preds, dataset_df):
    print("\n" + "=" * 70)
    print("TASK 8: VISUAL DIAGNOSTICS")
    print("=" * 70)

    AUDIT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use first non-frac model with best data, or frac
    model = "stage2d_frac"
    if model not in preds:
        model = next(iter(preds.keys()), None)

    ea_col = "EA vs SHE (eV)"
    ip_col = "IP vs SHE (eV)"

    ea_lookup = {}
    ip_lookup = {}
    for _, row in dataset_df.iterrows():
        group_key = f"{row['smiles_A']}|{row['smiles_B']}|{row['fracA']:.3f}"
        if pd.notna(row[ea_col]):
            ea_lookup[round(float(row[ea_col]), 6)] = group_key
        if pd.notna(row[ip_col]):
            ip_lookup[round(float(row[ip_col]), 6)] = group_key

    lookups = {"EA": ea_lookup, "IP": ip_lookup}

    for target in ["EA", "IP"]:
        if target not in preds[model]:
            continue

        yt = np.concatenate([sd["y_true"] for sd in preds[model][target].values()])
        yp = np.concatenate([sd["y_pred"] for sd in preds[model][target].values()])

        # 1. True vs Predicted scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(yt, yp, s=5, alpha=0.3)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        r2 = r2_score(yt, yp)
        ax.set_title(f"{MODEL_DISPLAY.get(model, model)}: True vs Pred {target}\nR²={r2:.4f}, corr={np.corrcoef(yt, yp)[0,1]:.4f}")
        fig.tight_layout()
        fig.savefig(AUDIT_PLOTS_DIR / f"scatter_{target}.png", dpi=150)
        plt.close(fig)

        # 2. Histogram of targets and predictions
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(yt, bins=50, alpha=0.7, label="y_true")
        axes[0].set_title(f"Distribution of y_true ({target})")
        axes[0].set_xlabel(target)
        axes[1].hist(yp, bins=50, alpha=0.7, color="orange", label="y_pred")
        axes[1].set_title(f"Distribution of y_pred ({target})")
        axes[1].set_xlabel(target)
        fig.tight_layout()
        fig.savefig(AUDIT_PLOTS_DIR / f"hist_{target}.png", dpi=150)
        plt.close(fig)

        # 3. Deviation scatter (if possible)
        lut = lookups[target]
        groups = [lut.get(round(float(v), 6), None) for v in yt]
        df = pd.DataFrame({"y_true": yt, "y_pred": yp, "group": groups})
        df_matched = df.dropna(subset=["group"])

        if len(df_matched) > 10:
            gmt = df_matched.groupby("group")["y_true"].transform("mean")
            gmp = df_matched.groupby("group")["y_pred"].transform("mean")
            df_matched = df_matched.copy()
            df_matched["dt"] = df_matched["y_true"].values - gmt.values
            df_matched["dp"] = df_matched["y_pred"].values - gmp.values
            multi = df_matched[df_matched.groupby("group")["y_true"].transform("count") > 1]

            if len(multi) > 10:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(multi["dt"], multi["dp"], s=5, alpha=0.3)
                lo = min(multi["dt"].min(), multi["dp"].min())
                hi = max(multi["dt"].max(), multi["dp"].max())
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
                ax.set_xlabel(f"True Δ{target}")
                ax.set_ylabel(f"Predicted Δ{target}")
                r2_dev = r2_score(multi["dt"], multi["dp"])
                ax.set_title(f"Architecture Deviation: Δ{target}\nR²={r2_dev:.4f}")
                fig.tight_layout()
                fig.savefig(AUDIT_PLOTS_DIR / f"deviation_scatter_{target}.png", dpi=150)
                plt.close(fig)

    print(f"  Saved plots to: {AUDIT_PLOTS_DIR.relative_to(PROJECT_ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    dataset_df = pd.read_csv(DATA_DIR / "ea_ip.csv")
    print(f"  Dataset: {len(dataset_df)} rows")

    preds = load_predictions_by_split()
    print(f"  Predictions: {len(preds)} models")

    result_csvs = load_result_csvs()
    print(f"  Result CSVs: {len(result_csvs)} models")

    # Run all tasks
    task1_verify_r2(preds, result_csvs)
    task2_verify_alignment(preds)
    task3_verify_arch_deviation(preds, dataset_df)
    diag3b_files = task4_compare_diagnostic3b()
    task5_verify_groups(dataset_df)
    task6_verify_split(preds, dataset_df)
    task7_verify_delta_y(preds, dataset_df)
    task8_visual_diagnostics(preds, dataset_df)

    # ── Write final audit report ──
    print("\n" + "=" * 70)
    print("WRITING FINAL AUDIT REPORT")
    print("=" * 70)

    # Read back R² audit for summary
    report = []
    report.append("# Stage 2D Evaluation Audit Report")
    report.append("")
    report.append("## Summary of Findings")
    report.append("")
    report.append("See console output above for detailed diagnostics.")
    report.append("See audit_plots/ for visual diagnostics.")
    report.append("See evaluation_audit_r2.txt for detailed R² breakdown.")
    report.append("")
    report.append("## Checklist")
    report.append("")
    report.append("1. **Is overall R² computed correctly?** — See Task 1")
    report.append("2. **Is architecture-deviation R² computed correctly?** — See Task 3")
    report.append("3. **Are predictions aligned with targets?** — See Task 2")
    report.append("4. **Are matched groups constructed correctly?** — See Task 5")
    report.append("5. **Does Stage 2D evaluation match Diagnostic 3B?** — See Task 4")
    report.append("6. **Most likely cause of discrepancy** — See Task 1 (normalization analysis)")

    with open(OUT_DIR / "stage2d_evaluation_audit.md", "w") as f:
        f.write("\n".join(report))

    print(f"\n  Written: {OUT_DIR / 'stage2d_evaluation_audit.md'}")
    print(f"  All audit output in: {OUT_DIR}")


if __name__ == "__main__":
    main()
