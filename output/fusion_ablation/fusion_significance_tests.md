# Fusion Ablation — Paired Statistical Tests

Nine folds treated as paired observations (variant vs Additive 2D1).
**Δ = variant − additive** (positive = variant better for R², negative = variant better for MAE/RMSE).

## EA Overall R²

| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | Cohen's d | 95% CI (bootstrap) | Significant? |
|---|---|---|---|---|---|---|---|
| FiLM vs Additive | +0.6224 | +0.0965 | 0.2828 | 0.0547 | +0.3838 | [+0.0156, +1.7262] | No |
| NLMix vs Additive | +0.0926 | +0.1242 | 0.1061 | 0.1289 | +0.6071 | [-0.0024, +0.1842] | No |
| FiLM+NLMix vs Additive | +0.0724 | -0.0007 | 0.2184 | 0.4258 | +0.4452 | [-0.0218, +0.1830] | No |

## IP Overall R²

| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | Cohen's d | 95% CI (bootstrap) | Significant? |
|---|---|---|---|---|---|---|---|
| FiLM vs Additive | +0.2536 | +0.2410 | 0.0311 | 0.0195 | +0.8700 | [+0.0873, +0.4443] | Yes |
| NLMix vs Additive | +0.1087 | +0.1112 | 0.2209 | 0.2031 | +0.4426 | [-0.0418, +0.2544] | No |
| FiLM+NLMix vs Additive | +0.2098 | +0.1216 | 0.0447 | 0.0117 | +0.7927 | [+0.0654, +0.3870] | Yes |

## EA ArchDev R²

| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | Cohen's d | 95% CI (bootstrap) | Significant? |
|---|---|---|---|---|---|---|---|
| FiLM vs Additive | -0.5284 | -0.2967 | 0.0872 | 0.0195 | -0.6494 | [-1.0874, -0.1271] | Yes |
| NLMix vs Additive | -0.2652 | -0.1586 | 0.0711 | 0.0391 | -0.6935 | [-0.5155, -0.0495] | Yes |
| FiLM+NLMix vs Additive | -0.3899 | -0.2126 | 0.0518 | 0.0195 | -0.7611 | [-0.7333, -0.1115] | Yes |

## IP ArchDev R²

| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | Cohen's d | 95% CI (bootstrap) | Significant? |
|---|---|---|---|---|---|---|---|
| FiLM vs Additive | -0.1722 | -0.1578 | 0.1975 | 0.0742 | -0.4685 | [-0.3841, +0.0729] | No |
| NLMix vs Additive | -0.1484 | -0.1404 | 0.2640 | 0.1289 | -0.4004 | [-0.3767, +0.0946] | No |
| FiLM+NLMix vs Additive | -0.1791 | -0.1610 | 0.2296 | 0.0977 | -0.4335 | [-0.4273, +0.0960] | No |

## EA MAE

| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | Cohen's d | 95% CI (bootstrap) | Significant? |
|---|---|---|---|---|---|---|---|
| FiLM vs Additive | -0.0373 | -0.0290 | 0.1619 | 0.0977 | -0.5136 | [-0.0816, +0.0107] | No |
| NLMix vs Additive | -0.0177 | -0.0102 | 0.3679 | 0.4258 | -0.3181 | [-0.0522, +0.0156] | No |
| FiLM+NLMix vs Additive | -0.0120 | +0.0013 | 0.4946 | 0.7344 | -0.2385 | [-0.0440, +0.0174] | No |

## IP MAE

| Comparison | Mean Δ | Median Δ | t p-value | Wilcoxon p | Cohen's d | 95% CI (bootstrap) | Significant? |
|---|---|---|---|---|---|---|---|
| FiLM vs Additive | -0.0398 | -0.0392 | 0.0410 | 0.0742 | -0.8108 | [-0.0698, -0.0091] | No |
| NLMix vs Additive | -0.0117 | -0.0368 | 0.4446 | 0.4961 | -0.2680 | [-0.0360, +0.0177] | No |
| FiLM+NLMix vs Additive | -0.0344 | -0.0183 | 0.0349 | 0.0391 | -0.8456 | [-0.0600, -0.0103] | Yes |

