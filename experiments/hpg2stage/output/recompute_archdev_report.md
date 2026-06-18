# Recomputed Architecture-Deviation Metrics Report

## Objective

Recompute architecture-deviation R² for 100% learning-curve checkpoints using the **EXACT** same metric function from `analyze_pair_disjoint_transfer.py` to determine whether the discrepancy with final Stage 2D values is due to metric computation differences or genuinely different predictions.

## Method

- **Metric function**: `compute_archdev_metrics()` copied exactly from `analyze_pair_disjoint_transfer.py` (lines 167-194)
- **Group definition**: 4-part key (`smiles_A || smiles_B || fracA || fracB`)
- **Filter**: Groups with ≥ 2 unique architectures (poly_type)
- **Aggregation**: Per-fold (each fold computed independently, then averaged)
- **Inverse transform**: linregress-based (same as `analyze_pair_disjoint_transfer.py` line 142) — applied because LC predictions are stored in normalized space
- **Sources compared**:
  - `HPG2Stage_LC/` — 100% learning-curve checkpoints (a_held_out split)
  - `HPG2Stage/` — Original final checkpoints (a_held_out split)
  - `HPG2Stage_Gen/` — Generalization checkpoints (group_disjoint and pair_disjoint splits)

## Results

### Full Three-Way Comparison

Using the **identical** metric function for all sources:

#### EA (EA vs SHE (eV))

| Source | Split | 2d0_arch R²(Δy) | 2d1_arch R²(Δy) |
|--------|-------|-----------------|-----------------|
| LC(100%) | a_held_out | 0.4742 ± 0.30 | 0.3498 ± 0.55 |
| Original | a_held_out | 0.8438 ± 0.02 | 0.8626 ± 0.01 |
| Gen | group_disjoint | 0.8868 ± 0.01 | 0.9381 ± 0.01 |
| Gen | pair_disjoint | 0.8862 ± 0.005 | 0.9346 ± 0.01 |

#### IP (IP vs SHE (eV))

| Source | Split | 2d0_arch R²(Δy) | 2d1_arch R²(Δy) |
|--------|-------|-----------------|-----------------|
| LC(100%) | a_held_out | 0.3984 ± 0.27 | 0.5318 ± 0.42 |
| Original | a_held_out | 0.9064 ± 0.02 | 0.9135 ± 0.02 |
| Gen | group_disjoint | 0.9423 ± 0.003 | 0.9649 ± 0.002 |
| Gen | pair_disjoint | 0.9406 ± 0.002 | 0.9637 ± 0.003 |

### Averages Across All Models

| Source | Mean R²(Δy) |
|--------|-------------|
| LC(100%) [a_held_out] | **0.44** |
| Original [a_held_out] | **0.88** |
| Gen [group_disjoint] | **0.93** |
| Gen [pair_disjoint] | **0.93** |

## Critical Findings

### 1. The Discrepancy REMAINS

Using identical metric computation, the LC(100%) checkpoints produce R²(Δy) ≈ 0.35–0.53, while Original/Gen checkpoints produce R²(Δy) ≈ 0.84–0.96.

### 2. LC and Original Are Different Checkpoints

Despite both using `a_held_out` splits, the LC(100%) and Original predictions differ:
- **Different test set sizes**: LC fold 0 has 9548 samples; Original fold 0 has 8596
- **Different predictions**: y_pred distributions do not match
- **Conclusion**: These are different training runs with different split assignments

### 3. Split Type Has Moderate Effect (~0.05 gap)

Comparing Original (a_held_out) vs Gen (group/pair_disjoint) — **same model quality, different split**:
- Original a_held_out: R²(Δy) ≈ 0.88
- Gen group_disjoint: R²(Δy) ≈ 0.93
- Gap from split type alone: ~0.05

### 4. The Dominant Factor Is the Model Checkpoint (~0.44 gap)

The LC(100%) checkpoint produces dramatically worse architecture-deviation predictions even compared to the Original checkpoint trained on the same split type:
- LC(100%) a_held_out: R²(Δy) ≈ 0.44
- Original a_held_out: R²(Δy) ≈ 0.88
- **Gap from different checkpoint: ~0.44**

## Root Cause Decomposition

| Factor | Contribution | Evidence |
|--------|-------------|----------|
| Model checkpoint quality | **~0.44 R²** | LC(100%) vs Original, same split type |
| Split type | ~0.05 R² | Original vs Gen, different split type |
| Metric computation | 0.00 R² | Same function used for all |

## Why LC(100%) Checkpoints Are Worse

The LC experiment trains models at multiple fractions (5%, 10%, 20%, 40%, 60%, 80%, 100%). The 100% fraction checkpoint likely differs from the Original because:

1. **Different split assignment** — test sizes differ (9548 vs 8596), confirming different GroupKFold assignments or train/val/test ratios
2. **Different training pipeline** — the LC script may use a simplified training loop (fixed epochs, no full hyperparameter optimization) optimized for quick iteration across fractions
3. **Linregress inverse transform limitation** — a single linear correction across ALL test samples cannot recover per-group deviations if the underlying predictions are poor at the group level
4. **High fold-to-fold variance** (std ≈ 0.3–0.55) — suggests some folds are reasonable while others are very poor, indicating an under-trained or unstable model

## Conclusion

**The discrepancy is NOT a metric computation artifact.** Using the identical `compute_archdev_metrics()` function from `analyze_pair_disjoint_transfer.py`:

- LC(100%) checkpoints: R²(Δy) ≈ **0.44** (high variance)
- Original a_held_out checkpoints: R²(Δy) ≈ **0.88** (low variance)
- Gen group/pair_disjoint checkpoints: R²(Δy) ≈ **0.93** (low variance)

The dominant source of discrepancy (~0.44 R² gap) is that the LC experiment produced different, lower-quality checkpoints compared to the Original training run. A secondary ~0.05 R² contribution comes from the split type (a_held_out vs group/pair_disjoint).

---

*Generated: June 18, 2026*
*Scripts: `recompute_lc_archdev.py`, `_full_comparison.py`*
