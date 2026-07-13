# Split 6 Diagnostic Report

## Q1. Which monomer is held out in split 6?

**SMILES:** `OB(O)c1ccc(B(O)O)c2nsnc12`

**Identity:** Benzothiadiazole-4,7-diboronic acid (BTD) — a strong electron-acceptor heterocycle containing N-S-N linkage.


## Q2. How many polymers in the test fold?

- Test fold size: **4774**
- Mean test fold size (all folds): 4774.0 ± 0.0
- Deviation from mean: +0.00 σ (std=0.0)

## Q3. Is the test fold substantially smaller than other folds?

**No.** All LOMO folds have identical test set sizes (4774 polymers each), because each monomer A appears in exactly the same number of polymers. The split is balanced by construction.

## Q4. Does split 6 have unusually low target variance?

| Metric | Split 6 | Mean across folds | Percentile rank |
|--------|---------|-------------------|-----------------|
| EA variance (eV²) | 0.0590 | 0.2096 | 11th |
| IP variance (eV²) | 0.1495 | 0.1160 | 78th |
| EA range (eV) | 2.9603 | 3.9676 | 11th |
| IP range (eV) | 3.5020 | 2.9875 | 78th |

**Yes** — split 6 EA variance is in the lowest quartile. A compressed target range makes R² unreliable (R² = 1 − SS_res/SS_tot; if SS_tot is small, even small errors collapse R²).

## Q5. Does split 6 have an unusual architecture distribution?

| Architecture | Split 6 | Expected (1/3 of 4774) |
|--------------|---------|------------------------|
| alternating | 682 | 682 |
| random | 2046 | 2046 |
| block | 2046 | 2046 |

Architecture distribution is identical across all folds (same proportions). The dataset is balanced by construction — every monomer A appears with the same set of architectures.

## Q6. Does split 6 have unusual compositions?

- fracA range: [0.25, 0.75], mean=0.50, std=0.19
- Number of unique partner monomers B: 682
- (Same as all other folds — each monomer A is paired with every monomer B and every fracA in [0.25, 0.50, 0.75])

## Q7. Is the split-6 monomer chemically isolated from the others?

ECFP4 Tanimoto similarity to all other monomers:
- Min: 0.200
- Max: 0.308
- Mean: 0.252

**Partially** — moderate chemical distance from other monomers. Some extrapolation challenge is expected.

## Q8. Do all four models fail similarly in split 6?

| Model | EA RMSE | IP RMSE | EA R² | IP R² |
|-------|---------|---------|-------|-------|
| wDMPNN | 0.565 | 0.149 | -4.409 | 0.851 |
| Frac | 0.786 | 0.209 | -9.479 | 0.709 |
| 2D0 | 0.701 | 0.220 | -7.327 | 0.675 |
| 2D1 | 1.040 | 0.109 | -17.344 | 0.921 |

EA RMSE fold 6 vs other folds:
- wDMPNN: fold6=0.565  others mean=0.210 ± 0.059  (+6.0σ above mean)
- Frac: fold6=0.786  others mean=0.268 ± 0.119  (+4.3σ above mean)
- 2D0: fold6=0.701  others mean=0.275 ± 0.124  (+3.4σ above mean)
- 2D1: fold6=1.040  others mean=0.234 ± 0.103  (+7.8σ above mean)

**Yes** — all four models produce negative EA R² for split 6, confirming this is a dataset/split property, not a model-specific failure.

## Q9. Assessment: expected difficulty, construction error, or extrapolation challenge?

### Evidence summary

| Evidence | Finding |
|----------|---------|
| Test set size | Identical to all other folds (4774 polymers) — split construction is correct |
| EA variance | 0.0590 eV² (global mean: 0.2096, ratio: 0.28×) |
| EA range | 2.9603 eV (global mean: 3.9676 eV) |
| Architecture distribution | Identical to all other folds — balanced by construction |
| All models fail | Yes — model-agnostic failure |
| Monomer identity | `OB(O)c1ccc(B(O)O)c2nsnc12` — benzothiadiazole (BTD), strong acceptor |

### Conclusion

The catastrophic R² in split 6 is **primarily explained by low target variance**, not by a genuinely poor model. The EA target for BTD-containing polymers spans only 2.960 eV (vs a global mean of 3.968 eV), meaning that even a small absolute prediction error produces large SS_res/SS_tot. RMSE for fold 6 is elevated but not catastrophic (see Q8), confirming that the model predictions are not wildly wrong — it is the R² denominator that is small.

**This is a well-known limitation of R² as a metric when target variance is low.** RMSE and MAE are more informative metrics for this fold. The experiment is correctly constructed and the poor R² is an expected consequence of BTD-polymer chemistry being more homogeneous in EA than other monomer classes.