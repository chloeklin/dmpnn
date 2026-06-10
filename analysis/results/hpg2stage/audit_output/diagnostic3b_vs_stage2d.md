# Diagnostic 3B vs Stage 2D: Line-by-Line Comparison

## Split Definition

| Aspect           | Diagnostic 3B (Phase 3) | Stage 2D           | Same? |
|------------------|------------------------|--------------------|-------|
| Split type       | `a_held_out`           | `a_held_out`       | **SAME** |
| # folds          | 5                      | 5                  | **SAME** |
| Monomer-disjoint | No (a_held_out)        | No (a_held_out)    | **SAME** |
| Group definition | GroupShuffleSplit on (A,B) pairs | GroupShuffleSplit on (A,B) pairs | **SAME** |

## Group Definition for Architecture Deviation

| Aspect          | Diagnostic 3B         | Stage 2D                           | Same? |
|-----------------|----------------------|-------------------------------------|-------|
| Group key       | Not computed directly | (smiles_A, smiles_B, fracA)         | **N/A** |
| Architecture col| N/A                  | poly_type (alternating/random/block)| **N/A** |

**Diagnostic 3B does NOT compute architecture-deviation R².** It computes overall ΔR² and
ΔRMSE vs the frac baseline from per-fold CSV results. Architecture-deviation analysis
is unique to Stage 2D.

## Architecture Deviation Definition

| Aspect            | Diagnostic 3B  | Stage 2D                                | Same? |
|-------------------|---------------|------------------------------------------|-------|
| Definition        | Not computed  | Δy = y - mean(y within composition group)| **DIFFERENT** |
| True deviations   | N/A           | From raw y_true                          | N/A |
| Predicted devs    | N/A           | From normalized y_pred (BUG)             | N/A |

## Evaluation Metric

| Aspect            | Diagnostic 3B            | Stage 2D                         | Same? |
|-------------------|-------------------------|-----------------------------------|-------|
| R² source         | CSV (test/r2 from trainer) | Recomputed from .npz predictions | **DIFFERENT** |
| R² computation    | Per-fold, then averaged   | Pooled across all folds           | **DIFFERENT** |
| Normalization     | Correct (via UnscaleTransform in predictor) | **BUG** — missing unscale for Stage2D | **DIFFERENT** |
| Target space      | Raw eV                   | Mismatch: y_true raw, y_pred normalized | **DIFFERENT** |

## Summary

| Component                   | Verdict     |
|-----------------------------|-------------|
| Split definition            | **SAME**    |
| Group definition            | **DIFFERENT** (3B doesn't use groups) |
| Architecture deviation      | **DIFFERENT** (3B doesn't compute it) |
| Evaluation metric source    | **DIFFERENT** (CSV vs .npz) |
| Normalization handling       | **DIFFERENT** (3B correct, 2D has bug) |

The architecture-deviation R² values (EA ≈ 0.61, IP ≈ 0.72) cited from "Diagnostic 3B" likely
came from a separate manual analysis — possibly a notebook or one-off script — not from
`plot_phase3.py`. That script only computes standard R²/RMSE/MAE from CSVs and ΔR²/ΔRMSE
relative to the frac baseline.
