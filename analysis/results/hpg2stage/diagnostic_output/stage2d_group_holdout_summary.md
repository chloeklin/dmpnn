# Experiment 4: Held-Out Matched-Group Generalization

## Split Design

The existing `a_held_out` split holds out ALL polymers sharing a given monomer A.
Since there are only 9 unique monomers A, each fold holds out ~2 monomers.
**This means every test-set matched group comes from a monomer A the model has NEVER seen.**

This is a **stronger generalization test** than merely holding out matched groups, because the model must generalize to entirely novel chemistry, not just unseen architecture variants of seen monomers.

## Results (per-fold, averaged)

### Overall R²

| Model | EA R² | IP R² |
|-------|-------|-------|
| Frac | 0.9741 ± 0.0048 | 0.9642 ± 0.0105 |
| 2D0-arch | 0.9810 ± 0.0057 | 0.9789 ± 0.0091 |
| 2D1-arch | 0.9820 ± 0.0045 | 0.9794 ± 0.0110 |

### Architecture-Deviation R² (on held-out groups)

| Model | EA R²(Δ) | IP R²(Δ) |
|-------|----------|----------|
| Frac | -0.0035 ± 0.0027 | -0.0028 ± 0.0016 |
| 2D0-arch | 0.8434 ± 0.0222 | 0.9060 ± 0.0268 |
| 2D1-arch | 0.8626 ± 0.0163 | 0.9140 ± 0.0222 |

### Test Set Matched-Group Coverage

| Fold | Test samples | Multi-arch groups | Samples in multi-arch groups |
|------|-------------|-------------------|-----------------------------|
| 0 | 8596 | 3657 | 8499 |
| 1 | 8596 | 3652 | 8492 |
| 2 | 8624 | 3667 | 8522 |
| 3 | 8575 | 3642 | 8456 |
| 4 | 8575 | 3636 | 8457 |

## Interpretation

**2D1-arch achieves arch-dev R² = 0.8626 on held-out monomer groups.** This demonstrates genuine generalization — the model learns transferable chemistry × architecture interactions, not merely memorized group corrections.

Frac baseline has arch-dev R² = -0.0035 (negative), confirming it cannot predict architecture-dependent variation, as expected.

2D1-arch (0.8626) outperforms 2D0-arch (0.8434) on held-out groups, suggesting chemistry-conditioned architecture modeling provides better generalization.
