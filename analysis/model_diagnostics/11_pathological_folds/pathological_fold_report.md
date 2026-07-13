# Pathological Folds Report — Monomer-heldout


## Target: EA

### Per-fold summary (mean across models)

|   fold |      R2 |    MAE |   RMSE |   TargetStd |   TargetRange |
|-------:|--------:|-------:|-------:|------------:|--------------:|
|      0 |  0.7953 | 0.1527 | 0.2154 |      0.4971 |        4.2345 |
|      1 |  0.4933 | 0.1882 | 0.2328 |      0.3573 |        3.7458 |
|      2 |  0.7903 | 0.1861 | 0.2263 |      0.5253 |        4.5099 |
|      3 |  0.918  | 0.0747 | 0.1051 |      0.398  |        3.7823 |
|      4 |  0.7701 | 0.14   | 0.1846 |      0.4015 |        3.6813 |
|      5 |  0.9219 | 0.0786 | 0.0976 |      0.3666 |        3.5549 |
|      6 | -8.3246 | 0.6002 | 0.655  |      0.2428 |        2.9603 |
|      7 |  0.5339 | 0.3364 | 0.3751 |      0.617  |        4.8148 |
|      8 |  0.7514 | 0.2354 | 0.2666 |      0.5844 |        4.4248 |


### Worst fold per model (lowest R²)

| Model | Fold | Held-out monomer | R² | MAE | RMSE | TargetStd |
|-------|------|------------------|----|-----|------|-----------|
| Frac | 6 | benzothiadiazole diboronic acid | -9.479 | 0.6943 | 0.7860 | 0.2428 |
| wDMPNN | 6 | benzothiadiazole diboronic acid | 0.852 | 0.0703 | 0.0933 | 0.2428 |
| GlobalArch | 6 | benzothiadiazole diboronic acid | -7.327 | 0.6094 | 0.7007 | 0.2428 |
| ChemArch | 6 | benzothiadiazole diboronic acid | -17.344 | 1.0269 | 1.0400 | 0.2428 |

### R² vs Target Std (denominator collapse check)

Pearson correlation between R² and test TargetStd: **0.486**

Interpretation:
- Positive correlation suggests denominator collapse (small variance → low R²)
  even when absolute errors are comparable to other folds.

### Fold 6 Detail (benzothiadiazole held-out, EA)

| Model | R² | MAE | RMSE | MedianAE | P95AE | MaxAE | TargetStd |
|-------|-----|-----|------|----------|-------|-------|-----------|
| Frac | -9.479 | 0.6943 | 0.7860 | 0.7824 | 1.1886 | 1.3066 | 0.2428 |
| wDMPNN | 0.852 | 0.0703 | 0.0933 | 0.0537 | 0.1912 | 0.5670 | 0.2428 |
| GlobalArch | -7.327 | 0.6094 | 0.7007 | 0.7043 | 1.0575 | 1.1826 | 0.2428 |
| ChemArch | -17.344 | 1.0269 | 1.0400 | 1.0607 | 1.2416 | 1.4350 | 0.2428 |

TargetStd = 0.2428 eV  |  ChemArch MAE = 1.0269 eV

**TargetStd is 0.2428 eV — not trivially small.**
Negative R² reflects genuinely poor predictions.


## Target: IP

### Per-fold summary (mean across models)

|   fold |      R2 |    MAE |   RMSE |   TargetStd |   TargetRange |
|-------:|--------:|-------:|-------:|------------:|--------------:|
|      0 |  0.433  | 0.1646 | 0.2116 |      0.2852 |        2.7823 |
|      1 |  0.7574 | 0.1375 | 0.1701 |      0.3748 |        3.1627 |
|      2 |  0.4224 | 0.3149 | 0.36   |      0.5209 |        3.9525 |
|      3 |  0.4582 | 0.1322 | 0.1513 |      0.224  |        2.3917 |
|      4 |  0.5966 | 0.1015 | 0.1279 |      0.209  |        2.4052 |
|      5 | -0.0907 | 0.2275 | 0.2557 |      0.2641 |        2.4287 |
|      6 |  0.8216 | 0.1132 | 0.1476 |      0.3866 |        3.502  |
|      7 |  0.8891 | 0.1107 | 0.1296 |      0.4256 |        3.7807 |
|      8 |  0.3009 | 0.1577 | 0.1793 |      0.2376 |        2.482  |


### Worst fold per model (lowest R²)

| Model | Fold | Held-out monomer | R² | MAE | RMSE | TargetStd |
|-------|------|------------------|----|-----|------|-----------|
| Frac | 5 | bithiophene diboronic acid | -0.498 | 0.2792 | 0.3232 | 0.2641 |
| wDMPNN | 0 | spiro-bifluorene | 0.224 | 0.2298 | 0.2513 | 0.2852 |
| GlobalArch | 5 | bithiophene diboronic acid | -0.604 | 0.2931 | 0.3345 | 0.2641 |
| ChemArch | 5 | bithiophene diboronic acid | -0.171 | 0.2745 | 0.2858 | 0.2641 |

### R² vs Target Std (denominator collapse check)

Pearson correlation between R² and test TargetStd: **0.275**

Interpretation:
- Low or negative correlation: R² failures reflect genuinely large errors,
  not just denominator collapse.

---

## Diagnostic Answer

### Q: Is negative R² caused mainly by genuinely large errors, or amplified by small target variance?

**EA:** 3 model-fold combinations with R² < 0
- Median TargetStd for failing folds: 0.2428 eV
- Median MAE for failing folds: 0.6943 eV
- R²–TargetStd correlation: 0.486
- **Verdict: GENUINELY LARGE ERRORS are the primary driver.**
  Median MAE = 0.694 eV is large in absolute terms.

**IP:** 4 model-fold combinations with R² < 0
- Median TargetStd for failing folds: 0.2641 eV
- Median MAE for failing folds: 0.2769 eV
- R²–TargetStd correlation: 0.275
- **Verdict: MIXED — both denominator shrinkage and elevated errors contribute.**
