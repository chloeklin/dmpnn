# Fusion Ablation — Table Verification Notes

## 1. Median R² values are actual fold medians

Values in `fusion_ablation_table_values.csv` are computed as `np.median(per_fold_R2_values)` from `fusion_per_fold_metrics.csv`.
They are **not** median differences (Δ) from `fusion_significance_tests.csv`.

Cross-check:
  - Additive (2D1) EA Overall R²: median = 0.6918  (median of 9 fold R² values)
  - FiLM EA Overall R²:           median = 0.7883  (median of 9 fold R² values)
  - FiLM vs Additive EA Overall median Δ from significance tests: +0.0965  (this is a different quantity)
  - Difference: 0.7883 − 0.6918 = +0.0965  ≠ median Δ above (due to nonlinearity of median)

## 2. p-values are paired Wilcoxon signed-rank tests

Taken directly from `fusion_significance_tests.csv` column `Wilcoxon_pvalue`.
Nine folds were treated as paired observations.
Comparison in each row: variant fold_i − Additive fold_i.

## 3. Robustness: effect of excluding fold 6

Fold 6 held-out monomer (OB(O)c1ccc(B(O)O)c2nsnc12) causes catastrophic failure in ALL models (EA R² ≈ −12 to −17).

**No comparison changes direction after excluding fold 6.**
The sign of the mean difference is identical for all 12 model–metric pairs with and without fold 6. The conclusions in the table are robust to this outlier.

## 4. Win counts

Win counts = number of the 9 folds where variant R² > Additive R² (strict inequality). Computed directly from `fusion_per_fold_metrics.csv`.

## 5. Best-in-column bolding

Bold values in the Markdown and LaTeX tables indicate the model with the highest median R² in each column.

  - EA Overall: best = FiLM
  - IP Overall: best = FiLM
  - EA ArchDev: best = Additive (2D1)
  - IP ArchDev: best = Additive (2D1)
