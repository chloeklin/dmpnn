"""Runner for the three follow-up diagnostics (Parts 1-3) + final summary.

Usage:
    python -m analysis.diagnostics.run_followup_diagnostics [--parts 1 2 3]

Parts:
    1 - pathological folds (Part 1)
    2 - residual correlation (Part 2)
    3 - ChemArch residual ablation (Part 3)
    summary - final followup summary (auto-run after all parts)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["1", "2", "3"],
                        help="Which parts to run: 1, 2, 3 (default: all)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip Part 3 (requires checkpoint loading, may be slow)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Active seed for prediction files (default: 42)")
    args = parser.parse_args()

    parts = set(args.parts)
    if args.skip_ablation:
        parts.discard("3")

    from analysis.diagnostics import config as diag_config
    diag_config.set_active_seed(args.seed)

    from analysis.diagnostics.data_loading import load_dataset, load_all_meta

    (diag_config.OUT_ROOT / "11_pathological_folds").mkdir(parents=True, exist_ok=True)
    (diag_config.OUT_ROOT / "12_residual_correlation").mkdir(parents=True, exist_ok=True)
    (diag_config.OUT_ROOT / "13_chemarch_residual_ablation").mkdir(parents=True, exist_ok=True)

    print("Loading dataset and metadata...")
    df   = load_dataset()
    meta = load_all_meta()

    path_df = corr_df = abla_df = None

    # ── Part 1 ────────────────────────────────────────────────────────────────
    if "1" in parts:
        print("\n" + "="*60)
        print("PART 1: Pathological fold absolute errors")
        print("="*60)
        t0 = time.time()
        from analysis.diagnostics.pathological_folds import run_pathological_folds
        path_df = run_pathological_folds()
        print(f"  Part 1 done in {time.time()-t0:.1f}s")

    # ── Part 2 ────────────────────────────────────────────────────────────────
    if "2" in parts:
        print("\n" + "="*60)
        print("PART 2: Residual correlation between models")
        print("="*60)
        t0 = time.time()
        from analysis.diagnostics.residual_correlation import run_residual_correlation
        corr_df, _ = run_residual_correlation(df, meta)
        print(f"  Part 2 done in {time.time()-t0:.1f}s")

    # ── Part 3 ────────────────────────────────────────────────────────────────
    if "3" in parts:
        print("\n" + "="*60)
        print("PART 3: ChemArch backbone vs full residual ablation")
        print("="*60)
        t0 = time.time()
        from analysis.diagnostics.chemarch_residual_ablation import run_chemarch_residual_ablation
        abla_df = run_chemarch_residual_ablation(df)
        print(f"  Part 3 done in {time.time()-t0:.1f}s")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    from analysis.diagnostics.followup_summary import run_followup_summary
    run_followup_summary(path_df, corr_df, abla_df)

    print("\nAll done. Outputs under:", diag_config.OUT_ROOT)


if __name__ == "__main__":
    main()
