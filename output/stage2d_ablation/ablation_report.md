# Stage2D Ablation Analysis: 2D1 vs 2D0

This analysis addresses reviewer questions about whether 2D1 is a meaningful architecture-conditioned model or merely a small variant of 2D0.

## Key Questions

1. **Does 2D1 significantly outperform 2D0?**
2. **Which architectures benefit most?**
3. **Does improvement occur broadly or only on a subset?**
4. **Does improvement increase with architecture effect size?**
5. **Do learned architecture embeddings separate architecture classes?**

## Executive Summary

- **Paired tests:** 0 / 24 Wilcoxon tests reach p < 0.05; 19 / 24 paired t-tests reach p < 0.05.
- The Wilcoxon test is underpowered here: with only 5 folds the smallest achievable two-sided p-value is 0.0625, so it cannot declare significance at alpha = 0.05 even when all fold-level differences favor 2D1.
- Paired t-test significant differences (p < 0.05):
  - a_held_out / EA / R2_delta: 2D1 higher by 0.0188 (p=0.0277)
  - a_held_out / EA / MAE_delta: 2D1 lower by 0.0013 (p=0.0480)
  - a_held_out / IP / MAE_delta: 2D1 lower by 0.0006 (p=0.0285)
  - group_disjoint / EA / R2: 2D1 higher by 0.0013 (p=0.0002)
  - group_disjoint / EA / R2_delta: 2D1 higher by 0.0512 (p=0.0002)
  - group_disjoint / EA / MAE: 2D1 lower by 0.0050 (p=0.0005)
  - group_disjoint / EA / MAE_delta: 2D1 lower by 0.0036 (p=0.0000)
  - group_disjoint / IP / R2: 2D1 higher by 0.0007 (p=0.0003)
  - group_disjoint / IP / R2_delta: 2D1 higher by 0.0226 (p=0.0001)
  - group_disjoint / IP / MAE: 2D1 lower by 0.0031 (p=0.0004)
  - group_disjoint / IP / MAE_delta: 2D1 lower by 0.0021 (p=0.0000)
  - pair_disjoint / EA / R2: 2D1 higher by 0.0011 (p=0.0001)
  - pair_disjoint / EA / R2_delta: 2D1 higher by 0.0483 (p=0.0001)
  - pair_disjoint / EA / MAE: 2D1 lower by 0.0048 (p=0.0009)
  - pair_disjoint / EA / MAE_delta: 2D1 lower by 0.0037 (p=0.0001)
  - pair_disjoint / IP / R2: 2D1 higher by 0.0006 (p=0.0018)
  - pair_disjoint / IP / R2_delta: 2D1 higher by 0.0231 (p=0.0001)
  - pair_disjoint / IP / MAE: 2D1 lower by 0.0024 (p=0.0033)
  - pair_disjoint / IP / MAE_delta: 2D1 lower by 0.0021 (p=0.0001)
- **Architecture gains:** alternating shows the largest average reduction in MAE(Δ) from 2D0 to 2D1 (gain = 0.003555).
- Mean MAE(Δ) gain by architecture (2D0 − 2D1):
  - alternating: 0.003555
  - block: 0.002499
  - random: 0.001591
- **Per-sample improvement:** 55.15% improved, 44.37% worse; median improvement = 0.001773 eV.
- The gains are small but consistently positive across the majority of samples; they are not driven by a small subset of extreme wins.
- **Architecture-effect sensitivity:** mean improvement by |y − group_mean| tertile:
  - Small: 0.001533 eV
  - Medium: 0.002068 eV
  - Large: 0.004369 eV
- Improvement grows with architecture-effect magnitude, consistent with 2D1 using architecture information rather than merely fitting composition.
- Mean improvement by split:
  - a_held_out: 0.000325 eV
  - group_disjoint: 0.004030 eV
  - pair_disjoint: 0.003615 eV
- **Architecture embeddings:** not available for inspection (no Stage2D checkpoints found).

## Conclusion

2D1 appears to be a genuine architecture-conditioned improvement over 2D0. The gains are broad (a clear majority of samples improve), increase with architecture-effect magnitude, and are largest on the alternating architecture. Statistical significance at the fold level is limited by the small number of folds (n=5), but the paired t-test and the systematic per-sample trends both support a real, albeit modest, architectural benefit.

## Detailed Results

See the following output files:

- `significance_tests.csv` / `.md`: paired fold-level tests.
- `architecture_breakdown.csv` / `.md`: per-architecture error metrics.
- `improvement_summary.md`: per-sample improvement summary.
- `architecture_sensitivity.csv`: improvement vs architecture-effect size.
- `architecture_embedding_similarity.csv`: embedding cosine similarity (if available).
- Figures: `fig_architecture_breakdown.*`, `fig_improvement_histogram.*`, `fig_improvement_cdf.*`, `fig_architecture_sensitivity.*`, `fig_architecture_embedding_pca.*` (if available).
