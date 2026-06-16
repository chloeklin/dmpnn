# Stage 2D: Paper-Level Interpretation

## 1. Does architecture improve over Frac?

**YES.** Multiple architecture-aware variants improve over the Frac baseline.

| Variant | EA R² | ΔEA | IP R² | ΔIP |
|---------|-------|-----|-------|-----|
| Frac | 0.9743 | +0.0000 | 0.9639 | +0.0000 |
| 2D0-fixed | 0.9806 | +0.0063 | 0.9775 | +0.0136 |
| 2D0-arch | 0.9812 | +0.0069 | 0.9784 | +0.0145 |
| 2D0-gate | 0.9807 | +0.0064 | 0.9773 | +0.0134 |
| 2D1-fixed | 0.9808 | +0.0065 | 0.9799 | +0.0160 |
| 2D1-arch | 0.9822 | +0.0079 | 0.9788 | +0.0150 |
| 2D1-gate | 0.9806 | +0.0064 | 0.9793 | +0.0155 |

## 2. Is a global architecture model (2D0) sufficient?

**YES.** 2D0 and 2D1 achieve nearly identical overall R² (ΔR²_EA=+0.0010, ΔR²_IP=+0.0015).

## 3. Does chemistry-conditioned architecture modeling (2D1) help?

Overall R²: ΔEA=+0.0010, ΔIP=+0.0015
Arch-deviation R²: ΔEA=+0.0235, ΔIP=+0.0119

**The chemistry-conditioned model (2D1) better captures architecture-specific effects.**

## 4. Which architecture mechanism is supported?

Two candidate mechanisms:

- **(A)** Δy_arch = g(arch) — global architecture offset
- **(B)** Δy_arch = g(arch) + h(h_A, h_B, f_A, f_B, arch) — chemistry-conditioned

**Evidence favors (B)**: Chemistry-conditioned model captures more architecture variation.

## 5. Is the extra complexity of 2D1 justified?

**No.** The additional parameters and computational cost of the 2D1 interaction MLP do not yield a meaningful improvement over 2D0.

## 6. What model should be selected as the final Stage 2D architecture?

**Recommended: 2D1-arch** (mean R² across EA+IP = 0.9805)

Runner-up: 2D1-fixed (mean R² = 0.9804, Δ = -0.0002)

## 7. What should be reported in the paper?

Recommended reporting:

1. Frac baseline establishes performance without architecture information
2. All 2D0 and 2D1 variants show [improvement/no improvement] over Frac
3. Best overall model and supporting metrics
4. Architecture-deviation R² demonstrates ability to capture architecture-specific effects
5. Statistical significance of improvements (paired t-test across folds)
6. Conclusion on whether chemistry-conditioned architecture modeling is necessary
