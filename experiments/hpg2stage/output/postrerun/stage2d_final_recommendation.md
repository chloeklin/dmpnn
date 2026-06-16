# Stage 2D Final Recommendation

## Executive Summary

### Selected Final Model: **2D1-arch**

### Why It Was Selected

- Highest mean R² across both targets (0.9805)
- EA R² = 0.9822 (Frac baseline: 0.9743, Δ = +0.0079)
- IP R² = 0.9788 (Frac baseline: 0.9639, Δ = +0.0150)

### Supporting Metrics

| Variant | EA R² | IP R² | Mean R² |
|---------|-------|-------|---------|
| Frac (baseline) | 0.9743 | 0.9639 | 0.9691 |
| 2D1-arch | 0.9822 | 0.9788 | 0.9805 | ←
| 2D1-fixed | 0.9808 | 0.9799 | 0.9804 |
| 2D1-gate | 0.9806 | 0.9793 | 0.9800 |
| 2D0-arch | 0.9812 | 0.9784 | 0.9798 |
| 2D0-fixed | 0.9806 | 0.9775 | 0.9790 |
| 2D0-gate | 0.9807 | 0.9773 | 0.9790 |

### Architecture-Deviation Evidence

| Variant | EA R²(Δ) | IP R²(Δ) |
|---------|----------|----------|
| Frac (baseline) | -0.0243 | -0.0280 |
| 2D1-arch | 0.8475 | 0.8910 |
| 2D1-fixed | 0.8476 | 0.8970 |
| 2D1-gate | 0.8519 | 0.8849 |
| 2D0-arch | 0.8240 | 0.8851 |
| 2D0-fixed | 0.8208 | 0.8811 |
| 2D0-gate | 0.8165 | 0.8813 |

### Is 2D1 Scientifically Justified?

- Overall R² improvement: EA=+0.0010, IP=+0.0015
- Arch-deviation R² improvement: EA=+0.0235, IP=+0.0119

**Conclusion**: 2D1 provides better architecture-deviation prediction, suggesting that chemistry-conditioned architecture modeling captures real effects. The complexity may be justified for applications where architecture-specific predictions matter.

### Remaining Caveats

1. Predictions were in normalized space due to the UnscaleTransform bug; all metrics use post-hoc inverse transform
2. Architecture-deviation analysis requires matching predictions to metadata via y_true lookup, which may miss some samples
3. Only 3 architecture types (alternating, random, block) with unequal representation
4. 5-fold cross-validation provides limited statistical power for paired comparisons
5. The dead-initialization fix (alpha_init=0.1) was applied only to 2d1_fixed and 2d1_arch; original 2d0 variants used alpha_init=0.0 but did not suffer from the dead-branch issue because their arch_embedding was randomly initialized (non-zero)
