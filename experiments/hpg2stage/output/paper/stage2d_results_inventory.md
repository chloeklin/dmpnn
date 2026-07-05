# Stage 2D Results Inventory

Generated from existing outputs. Project root: `/Users/u6788552/Desktop/experiments/dmpnn`

**Primary split: LOMO (Leave-One-Monomer-Out)** — directory: `HPG2Stage_LOMAO/`, split token: `a_held_out`, folds: 9

## 1. Final Stage 2D (LOMO — Leave-One-Monomer-Out)

| Model | Target | Folds | Status |
|-------|--------|-------|--------|
| frac | EA | 9/9 | complete |
| frac | IP | 9/9 | complete |
| 2d0_arch | EA | 9/9 | complete |
| 2d0_arch | IP | 9/9 | complete |
| 2d1_arch | EA | 9/9 | complete |
| 2d1_arch | IP | 9/9 | complete |

## 2. wDMPNN (LOMO)

| Target | Folds | Status |
|--------|-------|--------|
| EA | 9/9 | complete |
| IP | 9/9 | complete |

## 3. Generalization Experiments (group_disjoint, pair_disjoint)

| Model | Split | Target | Folds | Status |
|-------|-------|--------|-------|--------|
| frac | group_disjoint | EA | 5/5 | complete |
| frac | group_disjoint | IP | 5/5 | complete |
| frac | pair_disjoint | EA | 5/5 | complete |
| frac | pair_disjoint | IP | 5/5 | complete |
| 2d0_arch | group_disjoint | EA | 5/5 | complete |
| 2d0_arch | group_disjoint | IP | 5/5 | complete |
| 2d0_arch | pair_disjoint | EA | 5/5 | complete |
| 2d0_arch | pair_disjoint | IP | 5/5 | complete |
| 2d1_arch | group_disjoint | EA | 5/5 | complete |
| 2d1_arch | group_disjoint | IP | 5/5 | complete |
| 2d1_arch | pair_disjoint | EA | 5/5 | complete |
| 2d1_arch | pair_disjoint | IP | 5/5 | complete |

## 4. wDMPNN Generalization

| Target | Split | Folds | Status |
|--------|-------|-------|--------|
| EA | group_disjoint | 5/5 | complete |
| EA | pair_disjoint | 5/5 | complete |
| IP | group_disjoint | 5/5 | complete |
| IP | pair_disjoint | 5/5 | complete |

## 5. Learning Curve (Final Pipeline)

| Model | Fraction | Target | Folds | Status |
|-------|----------|--------|-------|--------|
| 2d0_arch | 25% | EA | 5/5 | complete |
| 2d0_arch | 25% | IP | 5/5 | complete |
| 2d0_arch | 50% | EA | 5/5 | complete |
| 2d0_arch | 50% | IP | 5/5 | complete |
| 2d0_arch | 75% | EA | 5/5 | complete |
| 2d0_arch | 75% | IP | 5/5 | complete |
| 2d0_arch | 100% | EA | 5/5 | complete |
| 2d0_arch | 100% | IP | 5/5 | complete |
| 2d1_arch | 25% | EA | 5/5 | complete |
| 2d1_arch | 25% | IP | 5/5 | complete |
| 2d1_arch | 50% | EA | 5/5 | complete |
| 2d1_arch | 50% | IP | 5/5 | complete |
| 2d1_arch | 75% | EA | 5/5 | complete |
| 2d1_arch | 75% | IP | 5/5 | complete |
| 2d1_arch | 100% | EA | 5/5 | complete |
| 2d1_arch | 100% | IP | 5/5 | complete |

## 6. Pre-Stage-2D Diagnostics

| Diagnostic | Output File | Status |
|------------|-------------|--------|
| Variance decomposition | `output/bottleneck/architecture_variance_table.csv` | complete |
| Diagnostic 3A (global offset) | `diagnostics/diagnostic_3a/diagnostic3a_metrics.csv` | complete |
| Diagnostic 3B (feature-conditioned) | `diagnostics/feature_conditioned_transfer/transfer_metrics.csv` | complete |

## 7. Missing Items

- No critical outputs missing. wDMPNN generalization results are now included, with pair-disjoint values provisional until the final fold completes.
