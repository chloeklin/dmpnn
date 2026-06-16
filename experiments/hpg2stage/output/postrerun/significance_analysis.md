# Significance Analysis: Best 2D0 vs Best 2D1

## EA

### Overall R² (per fold)

| Fold | 2D0-arch R² | 2D1-arch R² | Diff |
|------|------|------|------|
| 0 | 0.9838 | 0.9812 | -0.0026 |
| 1 | 0.9712 | 0.9747 | +0.0035 |
| 2 | 0.9841 | 0.9863 | +0.0022 |
| 3 | 0.9809 | 0.9849 | +0.0039 |
| 4 | 0.9850 | 0.9830 | -0.0021 |

- Mean difference: +0.0010 ± 0.0028
- Paired t-test: t=0.7205, p=0.5111
- **Not significant at p<0.05**: Differences are not statistically reliable

### Architecture-Deviation R² (per fold)

| Fold | 2D0-arch R²(Δ) | 2D1-arch R²(Δ) | Diff |
|------|------|------|------|
| 0 | 0.8276 | 0.8477 | +0.0200 |
| 1 | 0.8161 | 0.8473 | +0.0311 |
| 2 | 0.8454 | 0.8769 | +0.0315 |
| 3 | 0.8556 | 0.8590 | +0.0034 |
| 4 | 0.8722 | 0.8821 | +0.0099 |

- Mean difference: +0.0192 ± 0.0112
- Paired t-test: t=3.4255, p=0.0266
- **Significant at p<0.05**: 2D1 has better arch-deviation R²

## IP

### Overall R² (per fold)

| Fold | 2D0-arch R² | 2D1-fixed R² | Diff |
|------|------|------|------|
| 0 | 0.9825 | 0.9813 | -0.0013 |
| 1 | 0.9633 | 0.9650 | +0.0017 |
| 2 | 0.9826 | 0.9851 | +0.0026 |
| 3 | 0.9795 | 0.9817 | +0.0023 |
| 4 | 0.9867 | 0.9882 | +0.0015 |

- Mean difference: +0.0014 ± 0.0014
- Paired t-test: t=2.0030, p=0.1157
- **Not significant at p<0.05**: Differences are not statistically reliable

### Architecture-Deviation R² (per fold)

| Fold | 2D0-arch R²(Δ) | 2D1-fixed R²(Δ) | Diff |
|------|------|------|------|
| 0 | 0.9340 | 0.9425 | +0.0085 |
| 1 | 0.8616 | 0.8846 | +0.0231 |
| 2 | 0.9097 | 0.9292 | +0.0196 |
| 3 | 0.9094 | 0.9197 | +0.0102 |
| 4 | 0.9155 | 0.9267 | +0.0112 |

- Mean difference: +0.0145 ± 0.0057
- Paired t-test: t=5.0681, p=0.0071
- **Significant at p<0.05**: 2D1 has better arch-deviation R²

## All Variants vs Frac (Overall R²)

### EA

| Variant | Mean R² | Δ vs Frac | p-value |
|---------|---------|-----------|---------|
| Frac | 0.9741 | — | — |
| 2D0-fixed | 0.9805 | +0.0064 | 0.0020 |
| 2D0-arch | 0.9810 | +0.0069 | 0.0035 |
| 2D0-gate | 0.9805 | +0.0064 | 0.0067 |
| 2D1-fixed | 0.9806 | +0.0065 | 0.0060 |
| 2D1-arch | 0.9820 | +0.0079 | 0.0006 |
| 2D1-gate | 0.9804 | +0.0063 | 0.0360 |

### IP

| Variant | Mean R² | Δ vs Frac | p-value |
|---------|---------|-----------|---------|
| Frac | 0.9642 | — | — |
| 2D0-fixed | 0.9780 | +0.0138 | 0.0006 |
| 2D0-arch | 0.9789 | +0.0147 | 0.0006 |
| 2D0-gate | 0.9777 | +0.0135 | 0.0006 |
| 2D1-fixed | 0.9803 | +0.0161 | 0.0001 |
| 2D1-arch | 0.9794 | +0.0152 | 0.0002 |
| 2D1-gate | 0.9797 | +0.0155 | 0.0001 |
