#!/usr/bin/env python3
"""
Main runner for the model diagnostics pipeline.

Usage:
    python -m analysis.diagnostics.run_all_diagnostics
    # or
    python analysis/diagnostics/run_all_diagnostics.py

Generates all diagnostics into analysis/model_diagnostics/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.diagnostics.config import ensure_dirs, OUT_ROOT, STEP_DIRS
from analysis.diagnostics.data_loading import load_dataset, load_all_meta
from analysis.diagnostics.validation import run_validation
from analysis.diagnostics.variance_geometry import run_variance_geometry, run_error_decomposition
from analysis.diagnostics.group_mean_prediction import run_group_mean_prediction
from analysis.diagnostics.calibration import run_calibration
from analysis.diagnostics.ordering import run_ordering
from analysis.diagnostics.effect_magnitude import run_effect_magnitude
from analysis.diagnostics.novelty import run_novelty
from analysis.diagnostics.target_shift import run_target_shift
from analysis.diagnostics.case_studies import run_case_studies
from analysis.diagnostics.statistical import run_statistical
from analysis.diagnostics.summary import run_summary
from analysis.diagnostics.plotting import (
    plot_variance_decomposition,
    plot_error_decomposition,
    plot_group_mean_scatter,
    plot_calibration,
    plot_ordering,
    plot_effect_magnitude,
    plot_novelty,
    plot_target_shift,
)

import pandas as pd


def main():
    t0 = time.time()
    print("=" * 72)
    print("MODEL DIAGNOSTICS PIPELINE")
    print("=" * 72)
    print(f"Output directory: {OUT_ROOT}\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    ensure_dirs()

    print("Loading dataset...")
    df = load_dataset()
    print(f"  Dataset: {len(df)} rows, {df.columns.tolist()}")

    print("Loading split metadata...")
    meta = load_all_meta()
    for split, folds in meta.items():
        print(f"  {split}: {len(folds)} folds")
    print()

    # ── Step 1: Validation ────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 1: Validation")
    print("─" * 72)
    inv_df = run_validation(df, meta)
    n_passed = inv_df['validation_passed'].sum()
    n_total = len(inv_df)
    n_missing = (inv_df['n_test'] == 0).sum()
    print(f"  {n_passed}/{n_total} passed, {n_missing} missing")

    # Check for critical failures
    available = inv_df[inv_df['n_test'] > 0]
    failures = available[~available['validation_passed']]
    if len(failures) > 0:
        print(f"\n  WARNING: {len(failures)} validations FAILED:")
        for _, row in failures.iterrows():
            print(f"    {row['model']}/{row['target']}/{row['split']}/fold{row['fold']}")
        print("  Continuing with available data...\n")
    print()

    # ── Step 2: Variance Geometry ─────────────────────────────────────────────
    print("─" * 72)
    print("STEP 2: Variance Geometry")
    print("─" * 72)
    vg_df = run_variance_geometry(df, meta)
    plot_variance_decomposition(vg_df)
    print()

    # ── Step 3: Group-Mean Prediction ─────────────────────────────────────────
    print("─" * 72)
    print("STEP 3: Group-Mean Prediction")
    print("─" * 72)
    gm_df, gpred_df = run_group_mean_prediction(df, meta)
    plot_group_mean_scatter(gm_df, gpred_df)
    print()

    # ── Step 4: Error Decomposition ───────────────────────────────────────────
    print("─" * 72)
    print("STEP 4: Error Decomposition")
    print("─" * 72)
    ed_df = run_error_decomposition(df, meta)
    plot_error_decomposition(ed_df)
    print()

    # ── Step 5: Architecture-Deviation Calibration ────────────────────────────
    print("─" * 72)
    print("STEP 5: Architecture-Deviation Calibration")
    print("─" * 72)
    cal_df = run_calibration(df, meta)
    plot_calibration(cal_df, df, meta)
    print()

    # ── Step 6: Architecture Ordering ─────────────────────────────────────────
    print("─" * 72)
    print("STEP 6: Architecture Ordering")
    print("─" * 72)
    ord_df, grp_ord_df = run_ordering(df, meta)
    plot_ordering(ord_df)
    print()

    # ── Step 7: Effect Magnitude ──────────────────────────────────────────────
    print("─" * 72)
    print("STEP 7: Effect Magnitude")
    print("─" * 72)
    em_df = run_effect_magnitude(df, meta)
    plot_effect_magnitude(em_df)
    print()

    # ── Step 8: Chemical Novelty ──────────────────────────────────────────────
    print("─" * 72)
    print("STEP 8: Chemical Novelty")
    print("─" * 72)
    nov_df = run_novelty(df, meta)
    plot_novelty(nov_df)
    print()

    # ── Step 9: Target-Distribution Shift ─────────────────────────────────────
    print("─" * 72)
    print("STEP 9: Target-Distribution Shift")
    print("─" * 72)
    ts_df = run_target_shift(df, meta)
    plot_target_shift(ts_df)
    print()

    # ── Step 10: Per-Fold Case Studies ────────────────────────────────────────
    print("─" * 72)
    print("STEP 10: Per-Fold Case Studies")
    print("─" * 72)
    run_case_studies(df, meta)
    print()

    # ── Step 11: Statistical Comparisons ──────────────────────────────────────
    print("─" * 72)
    print("STEP 11: Statistical Comparisons")
    print("─" * 72)
    stat_df = run_statistical(df, meta, cal_df=cal_df, ord_df=ord_df)
    print()

    # ── Step 12: Final Summary ────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 12: Final Summary Report")
    print("─" * 72)
    run_summary(vg_df, ed_df, gm_df, cal_df, ord_df, stat_df, nov_df, ts_df)
    print()

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 72)
    print(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
    print("=" * 72)
    print(f"\nAll outputs saved to: {OUT_ROOT}")
    print("\nGenerated files:")
    for step_name, step_dir in STEP_DIRS.items():
        files = sorted(step_dir.glob('*'))
        if files:
            print(f"\n  {step_name}/")
            for f in files:
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    print(f"    {f.name}  ({size_kb:.1f} KB)")
                elif f.is_dir():
                    n_files = sum(1 for _ in f.rglob('*') if _.is_file())
                    print(f"    {f.name}/  ({n_files} files)")


if __name__ == '__main__':
    main()
