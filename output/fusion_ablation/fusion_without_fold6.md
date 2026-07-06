# Fusion Ablation — Analysis Excluding Fold 6

Fold 6 is a hard OOD fold (held-out monomer: OB(O)c1ccc(B(O)O)c2nsnc12).
This document compares paired analyses with and without this fold.

## EA Overall R²

| Comparison | Subset | Mean Δ | Median Δ | Wilcoxon p | Cohen's d | Conclusion |
|---|---|---|---|---|---|---|
| FiLM | All 9 folds | +0.6224 | +0.0965 | 0.0547 | +0.3838 | variant better, n.s. |
| FiLM | Excl. fold 6 (n=8) | +0.0836 | +0.0730 | 0.1094 | +0.6051 | variant better, n.s. |
| NLMix | All 9 folds | +0.0926 | +0.1242 | 0.1289 | +0.6071 | variant better, n.s. |
| NLMix | Excl. fold 6 (n=8) | +0.0639 | +0.0772 | 0.2500 | +0.4747 | variant better, n.s. |
| FiLM+NLMix | All 9 folds | +0.0724 | -0.0007 | 0.4258 | +0.4452 | variant better, n.s. |
| FiLM+NLMix | Excl. fold 6 (n=8) | +0.0649 | -0.0097 | 0.7422 | +0.3772 | variant better, n.s. |

## IP Overall R²

| Comparison | Subset | Mean Δ | Median Δ | Wilcoxon p | Cohen's d | Conclusion |
|---|---|---|---|---|---|---|
| FiLM | All 9 folds | +0.2536 | +0.2410 | 0.0195 | +0.8700 | variant better, p<0.05 |
| FiLM | Excl. fold 6 (n=8) | +0.2776 | +0.2546 | 0.0391 | +0.9193 | variant better, p<0.05 |
| NLMix | All 9 folds | +0.1087 | +0.1112 | 0.2031 | +0.4426 | variant better, n.s. |
| NLMix | Excl. fold 6 (n=8) | +0.1176 | +0.1519 | 0.2500 | +0.4506 | variant better, n.s. |
| FiLM+NLMix | All 9 folds | +0.2098 | +0.1216 | 0.0117 | +0.7927 | variant better, p<0.05 |
| FiLM+NLMix | Excl. fold 6 (n=8) | +0.2328 | +0.1336 | 0.0234 | +0.8526 | variant better, p<0.05 |

## EA ArchDev R²

| Comparison | Subset | Mean Δ | Median Δ | Wilcoxon p | Cohen's d | Conclusion |
|---|---|---|---|---|---|---|
| FiLM | All 9 folds | -0.5284 | -0.2967 | 0.0195 | -0.6494 | additive better, p<0.05 |
| FiLM | Excl. fold 6 (n=8) | -0.2796 | -0.2153 | 0.0391 | -0.8075 | additive better, p<0.05 |
| NLMix | All 9 folds | -0.2652 | -0.1586 | 0.0391 | -0.6935 | additive better, p<0.05 |
| NLMix | Excl. fold 6 (n=8) | -0.1823 | -0.1459 | 0.0781 | -0.5871 | additive better, n.s. |
| FiLM+NLMix | All 9 folds | -0.3899 | -0.2126 | 0.0195 | -0.7611 | additive better, p<0.05 |
| FiLM+NLMix | Excl. fold 6 (n=8) | -0.2534 | -0.1881 | 0.0391 | -0.7698 | additive better, p<0.05 |

## IP ArchDev R²

| Comparison | Subset | Mean Δ | Median Δ | Wilcoxon p | Cohen's d | Conclusion |
|---|---|---|---|---|---|---|
| FiLM | All 9 folds | -0.1722 | -0.1578 | 0.0742 | -0.4685 | additive better, n.s. |
| FiLM | Excl. fold 6 (n=8) | -0.1740 | -0.1466 | 0.1094 | -0.4429 | additive better, n.s. |
| NLMix | All 9 folds | -0.1484 | -0.1404 | 0.1289 | -0.4004 | additive better, n.s. |
| NLMix | Excl. fold 6 (n=8) | -0.1494 | -0.1227 | 0.1953 | -0.3771 | additive better, n.s. |
| FiLM+NLMix | All 9 folds | -0.1791 | -0.1610 | 0.0977 | -0.4335 | additive better, n.s. |
| FiLM+NLMix | Excl. fold 6 (n=8) | -0.1938 | -0.1941 | 0.1484 | -0.4411 | additive better, n.s. |

## Per-fold EA Overall R² (all models)

| Fold | OOD? | Additive | FiLM | NLMix | FiLM+NLMix |
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

## Summary: Do conclusions change after excluding fold 6?

The following comparisons change direction or significance:

- **EA ArchDev R² / NLMix**: significance changes (sig → n.s.)
