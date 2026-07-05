# Stage 2D Paper Figure Manifest

## Figures

| Figure | Filename | Caption | Data Source | Section |
|--------|----------|---------|-------------|---------|
| A | `fig_A_variance_decomposition.pdf` | Residual variance decomposition showing composition dominates (>98%) with architecture contributing ~1-1.5% | `output/bottleneck/architecture_variance_table.csv` | 8.2 Diagnostics |
| B | `fig_B_architecture_transfer_diagnostics.pdf` | Architecture transfer diagnostics: global offsets are small; chemistry-conditioned features dramatically improve transfer R²(Δy) | `diagnostics/diagnostic_3a/diagnostic3a_offsets.csv`, `diagnostics/diagnostic_3a/diagnostic3a_metrics.csv`, `diagnostics/feature_conditioned_transfer/transfer_metrics.csv` | 8.2 Diagnostics |
| D | `fig_D_overall_vs_architecture_recovery.pdf` | LOMO comparison: overall R²(y) and architecture-deviation R²(Δy) for Frac, wDMPNN, 2D0-arch, 2D1-arch | HPG2Stage_LOMAO predictions | 8.3 Stage 2D Models |
| F | `fig_F_generalization_performance.pdf` | Generalization: arch-dev R² across group-disjoint, pair-disjoint, LOMO splits (ordered by difficulty) | HPG2Stage_LOMAO + HPG2Stage_Gen + wDMPNN_Gen predictions | 8.4 Generalization |
| G | `fig_G_learning_curve.pdf` | Learning curve: arch-dev R² vs training group fraction (25%-100%) for 2D0 and 2D1 | HPG2Stage_LC_Final predictions | 8.5 Learning Curve |

## Tables

| Table | Filename | Caption | Section |
|-------|----------|---------|---------|
| 1 | `table1_overall_performance.csv` | Overall EA/IP R² and MAE for all models (LOMO) | 8.3 Stage 2D Models |
| 2 | `table2_architecture_performance.csv` | Architecture-deviation R²(Δ) and MAE(Δ) for all models (LOMO) | 8.3 Stage 2D Models |
| 3 | `table3_generalization_comparison.csv` | R² and R²(Δ) across all three split types | 8.4 Generalization |
