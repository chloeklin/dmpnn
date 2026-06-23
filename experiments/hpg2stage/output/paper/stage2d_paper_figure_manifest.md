# Stage 2D Paper Figure Manifest

## Figures

| Figure | Filename | Caption | Data Source | Section |
|--------|----------|---------|-------------|---------|
| A | `fig_A_variance_decomposition.pdf` | Residual variance decomposition showing composition dominates (>98%) with architecture contributing ~1-1.5% | `output/bottleneck/architecture_variance_table.csv` | 8.2 Diagnostics |
| B | `fig_B_diagnostic_3a_global_transfer.pdf` | Global architecture offset transfer: per-architecture mean correction achieves R²≈0.25-0.32 on held-out folds | `diagnostics/diagnostic_3a/diagnostic3a_fold_metrics.csv` | 8.2 Diagnostics |
| C | `fig_C_diagnostic_3b_feature_ablation.pdf` | Feature-conditioned transfer: chemistry features double transfer R² vs architecture-only | `diagnostics/feature_conditioned_transfer/transfer_metrics.csv` | 8.2 Diagnostics |
| D | `fig_D_model_comparison_overall.pdf` | Overall EA/IP R² comparison: wDMPNN, Frac, 2D0-arch, 2D1-arch under a_held_out split | HPG2Stage + wDMPNN predictions | 8.3 Stage 2D Models |
| E | `fig_E_model_comparison_archdev.pdf` | Architecture-deviation R²: 2D0/2D1 achieve R²(Δ)≈0.85-0.91 vs near-zero for Frac/wDMPNN | HPG2Stage + wDMPNN predictions | 8.3 Stage 2D Models |
| F | `fig_F_generalization_performance.pdf` | Generalization: arch-dev R² maintained across a_held_out, group-disjoint, pair-disjoint splits | HPG2Stage + HPG2Stage_Gen predictions | 8.4 Generalization |
| G | `fig_G_learning_curve.pdf` | Learning curve: arch-dev R² vs training group fraction (25%-100%) for 2D0 and 2D1 | HPG2Stage_LC_Final predictions | 8.5 Learning Curve |

## Tables

| Table | Filename | Caption | Section |
|-------|----------|---------|---------|
| 1 | `table1_overall_performance.csv` | Overall EA/IP R² and MAE for all models (a_held_out) | 8.3 Stage 2D Models |
| 2 | `table2_architecture_performance.csv` | Architecture-deviation R²(Δ) and MAE(Δ) for all models | 8.3 Stage 2D Models |
| 3 | `table3_generalization_comparison.csv` | R² and R²(Δ) across all three split types | 8.4 Generalization |
