# Fusion Ablation — Consistency Report

All conclusions are based solely on paired fold analyses and statistical tests.
No model speculation or extrapolation beyond observed data.

## Q1: Does FiLM consistently outperform Additive for overall EA R²?

- FiLM wins EA R² in 7/9 folds; Additive wins 2/9 folds.
- All-fold: mean Δ=+0.6224, median Δ=+0.0965, Wilcoxon p=0.0547, Cohen's d=+0.3838
- Excl. fold 6: mean Δ=+0.0836, median Δ=+0.0730, Wilcoxon p=0.1094
- Per-fold EA R² for reference: Additive medians=0.6918, FiLM median=0.7883

## Q2: Does Additive consistently outperform fusion for EA ArchDev R²?

| Variant | Additive wins | Variant wins | Wilcoxon p (all) | Wilcoxon p (excl. fold 6) |
|---|---|---|---|---|
| FiLM | 8/9 | 1/9 | 0.0195 | 0.0391 |
| NLMix | 7/9 | 2/9 | 0.0391 | 0.0781 |
| FiLM+NLMix | 8/9 | 1/9 | 0.0195 | 0.0391 |

## Q3: Are the observed differences statistically significant?

Summary of Wilcoxon signed-rank tests (all 9 folds):

| Metric | Comparison | Mean Δ | Wilcoxon p | Significant (p<0.05)? |
|---|---|---|---|---|
| EA Overall R² | FiLM vs Additive | +0.6224 | 0.0547 | No |
| EA Overall R² | NLMix vs Additive | +0.0926 | 0.1289 | No |
| EA Overall R² | FiLM+NLMix vs Additive | +0.0724 | 0.4258 | No |
| IP Overall R² | FiLM vs Additive | +0.2536 | 0.0195 | Yes |
| IP Overall R² | NLMix vs Additive | +0.1087 | 0.2031 | No |
| IP Overall R² | FiLM+NLMix vs Additive | +0.2098 | 0.0117 | Yes |
| EA ArchDev R² | FiLM vs Additive | -0.5284 | 0.0195 | Yes |
| EA ArchDev R² | NLMix vs Additive | -0.2652 | 0.0391 | Yes |
| EA ArchDev R² | FiLM+NLMix vs Additive | -0.3899 | 0.0195 | Yes |
| IP ArchDev R² | FiLM vs Additive | -0.1722 | 0.0742 | No |
| IP ArchDev R² | NLMix vs Additive | -0.1484 | 0.1289 | No |
| IP ArchDev R² | FiLM+NLMix vs Additive | -0.1791 | 0.0977 | No |
| EA MAE | FiLM vs Additive | -0.0373 | 0.0977 | No |
| EA MAE | NLMix vs Additive | -0.0177 | 0.4258 | No |
| EA MAE | FiLM+NLMix vs Additive | -0.0120 | 0.7344 | No |
| IP MAE | FiLM vs Additive | -0.0398 | 0.0742 | No |
| IP MAE | NLMix vs Additive | -0.0117 | 0.4961 | No |
| IP MAE | FiLM+NLMix vs Additive | -0.0344 | 0.0391 | Yes |

## Q4: Are conclusions robust after excluding fold 6?

No comparison changes direction after excluding fold 6. The mean-delta sign is consistent in all cases.

## Q5: Is there evidence for a genuine trade-off between overall prediction and architecture recovery?

For each variant, compare direction of Δ for Overall R² vs ArchDev R²:

| Variant | EA Overall Δ (mean) | EA ArchDev Δ (mean) | IP Overall Δ (mean) | IP ArchDev Δ (mean) | Trade-off observed? |
|---|---|---|---|---|---|
| FiLM | +0.6224 | -0.5284 | +0.2536 | -0.1722 | Yes (EA) Yes (IP) |
| NLMix | +0.0926 | -0.2652 | +0.1087 | -0.1484 | Yes (EA) Yes (IP) |
| FiLM+NLMix | +0.0724 | -0.3899 | +0.2098 | -0.1791 | Yes (EA) Yes (IP) |

## Reference: Per-fold R² table

### EA Overall R²

| Fold | OOD? | Additive (2D1) | FiLM | NLMix | FiLM+NLMix |
|---|---|---|---|---|---|
| 0 | No | 0.7558 | 0.9702 | 0.9775 | 0.9750 |
| 1 | No | 0.3284 | 0.6102 | 0.5400 | 0.7246 |
| 2 | No | 0.7754 | 0.9407 | 0.9268 | 0.9086 |
| 3 | No | 0.9734 | 0.8161 | 0.9626 | 0.9501 |
| 4 | No | 0.6518 | 0.7013 | 0.5084 | 0.5642 |
| 5 | No | 0.9559 | 0.9756 | 0.8820 | 0.9552 |
| 6 | **YES** | -17.3442 | -12.4113 | -17.0213 | -17.2123 |
| 7 | No | 0.6918 | 0.7883 | 0.7220 | 0.6731 |
| 8 | No | 0.6187 | 0.6178 | 0.7428 | 0.5199 |

### IP Overall R²

| Fold | OOD? | Additive (2D1) | FiLM | NLMix | FiLM+NLMix |
|---|---|---|---|---|---|
| 0 | No | 0.2667 | 0.9151 | 0.7207 | 0.8805 |
| 1 | No | 0.8134 | 0.8987 | 0.9246 | 0.9590 |
| 2 | No | 0.4084 | 0.3503 | 0.3742 | 0.4204 |
| 3 | No | 0.4973 | 0.7382 | 0.7703 | 0.6189 |
| 4 | No | 0.6025 | 0.8708 | 0.7952 | 0.7990 |
| 5 | No | -0.1710 | 0.1359 | -0.5140 | -0.0858 |
| 6 | **YES** | 0.9210 | 0.9826 | 0.9585 | 0.9463 |
| 7 | No | 0.9863 | 0.9423 | 0.9092 | 0.9695 |
| 8 | No | -0.1371 | 0.6361 | 0.2268 | 0.5676 |

