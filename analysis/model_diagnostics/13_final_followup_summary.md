# Final Follow-up Diagnostic Summary

This report synthesises findings from three supplementary diagnostics:

- **Part 1**: Absolute errors on pathological folds (11_pathological_folds)
- **Part 2**: Residual correlation between models (12_residual_correlation)
- **Part 3**: ChemArch backbone-only vs full residual ablation (13_chemarch_residual_ablation)

---

## Q1: What causes ChemArch's catastrophic LOMO failures?

ChemArch has **3** model-fold-target combinations with R² < 0 out of 18 total on monomer-heldout.

R²–TargetStd correlation: EA = 0.621,  IP = 0.461
Median MAE in R² < 0 cases: 0.2745 eV
Median TargetStd in R² < 0 cases: 0.2428 eV

**Primary driver: genuinely large absolute errors** on held-out monomer chemistry.

### Evidence from residual ablation (Part 3)

**EA** — mean across folds:
  - Full ChemArch:    R² = -1.219,  MAE = 0.2915 eV
  - Backbone only:    R² = -4.030,  MAE = 0.7958 eV
  - ΔR² (full−backbone): +2.811
  → The residual head **helps** on monomer-heldout for EA.

**IP** — mean across folds:
  - Full ChemArch:    R² = 0.215,  MAE = 0.2185 eV
  - Backbone only:    R² = -7.712,  MAE = 0.7651 eV
  - ΔR² (full−backbone): +7.927
  → The residual head **helps** on monomer-heldout for IP.

---

## Q2: Is the LOMO failure in the chemistry baseline, the residual head, or architecture prediction?

### EA

| Fold | Held-out monomer | Backbone R² | Full R² | ΔR² |
|------|-----------------|-------------|---------|-----|
| 0 | spiro-bifluorene | -0.670 | 0.764 | +1.434 |
| 1 | dibenzothiophene sulfone | -12.519 | 0.533 | +13.052 |
| 2 | difluorobenzene diboronic acid | -0.533 | 0.846 | +1.379 |
| 3 | DTT fused trithiophene | -1.232 | 0.915 | +2.148 |
| 4 | pyrene diboronic acid | -2.316 | 0.808 | +3.124 |
| 5 | bithiophene diboronic acid | -4.785 | 0.862 | +5.647 |
| 6 | benzothiadiazole diboronic aci | -7.418 | -16.735 | -9.316 |
| 7 | benzene-1,4-diboronic acid | -4.972 | 0.541 | +5.514 |
| 8 | carbazole diboronic acid | -1.822 | 0.499 | +2.321 |

- Folds where **backbone bad, residual rescues** (R²_back<0, R²_full≥0): 8
- Folds where **both bad** (R²<0 for both): 1
- Folds where **residual hurts** (backbone OK, full bad): 0
- Folds where **both OK**: 0

**Conclusion:** The **residual head rescues a poor backbone** on most folds. The chemistry backbone alone cannot extrapolate; the architecture-conditioned correction substantially recovers performance.

### IP

| Fold | Held-out monomer | Backbone R² | Full R² | ΔR² |
|------|-----------------|-------------|---------|-----|
| 0 | spiro-bifluorene | -4.249 | 0.416 | +4.665 |
| 1 | dibenzothiophene sulfone | -1.126 | 0.856 | +1.981 |
| 2 | difluorobenzene diboronic acid | -1.218 | -0.036 | +1.182 |
| 3 | DTT fused trithiophene | -13.991 | 0.644 | +14.635 |
| 4 | pyrene diboronic acid | -3.768 | 0.779 | +4.547 |
| 5 | bithiophene diboronic acid | -11.534 | -2.068 | +9.466 |
| 6 | benzothiadiazole diboronic aci | -10.262 | 0.540 | +10.802 |
| 7 | benzene-1,4-diboronic acid | -3.847 | 0.840 | +4.686 |
| 8 | carbazole diboronic acid | -19.414 | -0.034 | +19.380 |

- Folds where **backbone bad, residual rescues** (R²_back<0, R²_full≥0): 6
- Folds where **both bad** (R²<0 for both): 3
- Folds where **residual hurts** (backbone OK, full bad): 0
- Folds where **both OK**: 0

**Conclusion:** The **residual head rescues a poor backbone** on most folds. The chemistry backbone alone cannot extrapolate; the architecture-conditioned correction substantially recovers performance.

---

## Q3: Are ChemArch and wDMPNN making complementary mistakes?

**overall** residuals — ChemArch vs wDMPNN:
  - Mean Pearson r = 0.396
  - Mean Spearman ρ = 0.353
  - Mean covariance = 0.0011
  → **partially complementary** (moderate correlation)

**group_mean** residuals — ChemArch vs wDMPNN:
  - Mean Pearson r = 0.343
  - Mean Spearman ρ = 0.311
  - Mean covariance = 0.0007
  → **partially complementary** (moderate correlation)

**arch_deviation** residuals — ChemArch vs wDMPNN:
  - Mean Pearson r = 0.567
  - Mean Spearman ρ = 0.509
  - Mean covariance = 0.0004
  → **partially complementary** (moderate correlation)

On **monomer-heldout** specifically: mean Pearson r = 0.251
The two models diverge more on held-out monomers, suggesting they encode different aspects of monomer chemistry.

---

## Q4: Does evidence support combining a graph encoder with an explicit chemistry-conditioned residual?

**Hypothesis:** "A graph-based chemistry encoder combined with an explicit chemistry-conditioned architecture residual head could potentially combine the strengths of both models."

### Supporting evidence

- ChemArch and wDMPNN residuals have low–moderate overall correlation (r=0.40), indicating they encode partially complementary information.
- EA: residual head improves arch-deviation R² by ΔR²=+0.456 — the arch-conditioned correction provides signal beyond the backbone.
- IP: residual head improves arch-deviation R² by ΔR²=+0.483 — the arch-conditioned correction provides signal beyond the backbone.
- wDMPNN (mean R²=0.91) substantially outperforms ChemArch (mean R²=-0.41) on LOMO — demonstrating that a better chemistry encoder would benefit the architecture-aware model.

### Contrary evidence

- No direct contrary evidence found.

### Conclusion

The diagnostics **support** the hypothesis. The key finding is that wDMPNN's stronger chemistry encoder generalises better to unseen monomers, while ChemArch's architecture residual provides ordering signal within-distribution. A model that combines a wDMPNN-quality chemistry encoder (or graph-based backbone) with an explicit architecture-conditioned correction head could, in principle, capture both strengths. This hypothesis is mechanistically plausible but would require experimental validation through retraining.

---

_Generated automatically by `analysis/diagnostics/followup_summary.py`_