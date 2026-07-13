"""
Evaluate EA/IP prediction files using canonical naming and correct global indices.

Loads prediction files from:
    predictions/ea_ip_group/   (group_disjoint)
    predictions/ea_ip_pair/    (pair_disjoint)
    predictions/ea_ip_lomo/    (monomer_heldout)

Computes per fold:
    - Overall R²  (and MAE)
    - Architecture-deviation R²  (ΔEA / ΔIP)

For monomer_heldout: also reports mean ± SD across folds.

Uses:
    metadata/splits/{split}.json   for correct global df indices
    evaluation/metrics.py          for the canonical metric functions

Usage:
    python scripts/evaluate_ea_ip_predictions.py
        [--split monomer_heldout] [--split group_disjoint] [--split pair_disjoint]
        [--target "EA vs SHE (eV)"] [--target "IP vs SHE (eV)"]
        [--model frac] [--model wdmpnn] [--model globalarch] [--model chemarch]
        [--pooled]          report pooled R² instead of fold-wise
        [--scale-check]     print scale sanity statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.naming import (
    CANONICAL_MODELS, CANONICAL_SPLITS, CANONICAL_TARGETS,
    make_prediction_path, target_from_token, standard_target_token,
)
from evaluation.metrics import (
    compute_overall_r2, compute_overall_mae,
    compute_archdev_r2,
    scale_sanity_check,
    validate_prediction_inputs,
)

DATA_PATH  = ROOT / "data" / "ea_ip.csv"
META_DIR   = ROOT / "metadata" / "splits"
PRED_ROOT  = ROOT / "predictions"


# ── Dataset / metadata loaders ────────────────────────────────────────────────

def load_df() -> pd.DataFrame:
    """Load and canonicalize the dataset (adds group_key, smilesA, smilesB)."""
    sys.path.insert(0, str(ROOT / "experiments" / "hpg2stage" / "scripts"))
    try:
        import generate_stage2d_paper_outputs as g
        g._NORM_PARAMS = g.estimate_normalization_params()
        g._NORM_PARAMS_ORIG = g.estimate_normalization_params_orig()
        df = g.load_dataset()
        g._ensure_value_map(df)
        return df
    except Exception:
        return pd.read_csv(DATA_PATH)


def load_split_metadata(split: str) -> list[dict]:
    meta_path = META_DIR / f"{split}.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Split metadata not found: {meta_path}. "
            f"Run  python scripts/generate_split_metadata.py  first."
        )
    with open(meta_path) as f:
        data = json.load(f)
    return data["folds"]


def get_global_indices(split: str, fold: int, meta_folds: list[dict]) -> np.ndarray:
    """Return global df row indices for a fold from metadata."""
    for rec in meta_folds:
        if rec["fold"] == fold:
            return np.asarray(rec["global_test_indices"], dtype=int)
    raise KeyError(f"Fold {fold} not found in {split} metadata")


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_fold(
    df: pd.DataFrame,
    split: str,
    fold: int,
    target: str,
    model: str,
    meta_folds: list[dict],
    scale_check: bool = False,
) -> dict | None:
    """Evaluate one prediction file. Returns a metrics dict or None if file missing."""
    fpath = make_prediction_path(PRED_ROOT, target, model, split, fold)
    if not fpath.exists():
        return None

    p      = np.load(fpath, allow_pickle=True)
    y_true = p["y_true"].flatten().astype(np.float64)
    y_pred = p["y_pred"].flatten().astype(np.float64)

    global_idx = get_global_indices(split, fold, meta_folds)

    # ── Validation ──
    validate_prediction_inputs(df, y_true, y_pred, global_idx,
                                split_name=f"{split}/fold{fold}/{model}")

    # ── Scale sanity ──
    if scale_check:
        sc = scale_sanity_check(y_true, y_pred, target_name=target)
        print(f"  [{model} fold{fold}] y_true: mean={sc['yt_mean']:.3f} "
              f"std={sc['yt_std']:.3f} [{sc['yt_min']:.3f},{sc['yt_max']:.3f}]  "
              f"y_pred: mean={sc['yp_mean']:.3f} std={sc['yp_std']:.3f}")
        for w in sc["warnings"]:
            print(f"    ⚠ {w}")

    # ── Metrics ──
    r2_overall = compute_overall_r2(y_true, y_pred)
    mae_overall = compute_overall_mae(y_true, y_pred)
    archdev = compute_archdev_r2(df, y_true, y_pred, global_idx)

    return dict(
        split=split, fold=fold, target=target, model=model,
        r2_overall=r2_overall,
        mae_overall=mae_overall,
        r2_archdev=archdev["r2"],
        mae_archdev=archdev["mae"],
        n_samples=archdev["n_samples"],
        n_groups=archdev["n_groups"],
        mean_dt=archdev["mean_dt"],
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   N/A  "
    return f"{v:+.4f}"


def report_results(results: list[dict], pooled: bool) -> None:
    if not results:
        print("No results to report.")
        return

    splits  = sorted({r["split"]  for r in results})
    targets = sorted({r["target"] for r in results})
    models  = sorted({r["model"]  for r in results})

    for split in splits:
        for target in targets:
            print(f"\n{'='*72}")
            print(f"  {split.upper()}  |  {target}")
            print(f"{'='*72}")
            print(f"{'Model':<12}  {'Fold':>5}  {'R²':>8}  {'MAE':>8}  "
                  f"{'ΔR²':>8}  {'ΔMAE':>8}  {'n_grp':>6}")
            print("-" * 72)

            for model in models:
                sub = [r for r in results
                       if r["split"] == split and r["target"] == target
                       and r["model"] == model]
                if not sub:
                    continue

                folds_sorted = sorted(sub, key=lambda r: r["fold"])
                for r in folds_sorted:
                    if not pooled:
                        print(
                            f"{model:<12}  {r['fold']:>5}  "
                            f"{_fmt(r['r2_overall'])}  {_fmt(r['mae_overall'])}  "
                            f"{_fmt(r['r2_archdev'])}  {_fmt(r['mae_archdev'])}  "
                            f"{r['n_groups']:>6}"
                        )

                # Summary row
                r2_vals   = [r["r2_overall"] for r in folds_sorted if not np.isnan(r["r2_overall"])]
                dr2_vals  = [r["r2_archdev"] for r in folds_sorted
                             if r["r2_archdev"] is not None and not np.isnan(r["r2_archdev"])]
                if r2_vals:
                    r2_m  = np.mean(r2_vals);  r2_s  = np.std(r2_vals)
                    dr2_m = np.mean(dr2_vals) if dr2_vals else np.nan
                    dr2_s = np.std(dr2_vals)  if dr2_vals else np.nan
                    print(
                        f"{model:<12}  {'MEAN':>5}  "
                        f"{r2_m:+.4f}  {'±'+f'{r2_s:.4f}':>8}  "
                        f"{dr2_m:+.4f}  {'±'+f'{dr2_s:.4f}':>8}"
                    )
                print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split",  dest="splits",  action="append",
                   choices=CANONICAL_SPLITS, metavar="SPLIT",
                   help="Split(s) to evaluate (default: all)")
    p.add_argument("--target", dest="targets", action="append",
                   choices=CANONICAL_TARGETS, metavar="TARGET",
                   help="Target(s) to evaluate (default: all)")
    p.add_argument("--model",  dest="models",  action="append",
                   choices=CANONICAL_MODELS, metavar="MODEL",
                   help="Model(s) to evaluate (default: all)")
    p.add_argument("--pooled", action="store_true",
                   help="Report only pooled (mean) metrics, not per-fold rows")
    p.add_argument("--scale-check", action="store_true",
                   help="Print scale sanity statistics for each prediction file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    splits  = args.splits  or CANONICAL_SPLITS
    targets = args.targets or CANONICAL_TARGETS
    models  = args.models  or CANONICAL_MODELS

    print("Loading dataset...")
    df = load_df()
    print(f"  {len(df)} rows")

    results = []
    for split in splits:
        print(f"\nLoading {split} metadata...")
        meta_folds = load_split_metadata(split)
        folds = [r["fold"] for r in meta_folds]

        for target in targets:
            for model in models:
                for fold in folds:
                    r = evaluate_fold(
                        df, split, fold, target, model,
                        meta_folds, scale_check=args.scale_check,
                    )
                    if r is not None:
                        results.append(r)
                    else:
                        pass  # missing file, silently skip

    report_results(results, pooled=args.pooled)

    # Summary count
    found = len(results)
    total = len(splits) * len(targets) * len(models) * max(
        len(load_split_metadata(s)) for s in splits
    )
    print(f"\nEvaluated {found} prediction files")


if __name__ == "__main__":
    main()
