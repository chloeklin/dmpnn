# Experiments

Scripts for exploratory analysis, model comparison plotting, and architecture-aware
(Stage 2D) experiments.

## Directory Structure

```text
experiments/
├── eda/            # Exploratory data analysis
├── tabular/        # Tabular model comparison plots
├── hpg/            # HPG graph model phase plots
├── diagnostics/    # Pre-Stage-2D feasibility diagnostics
└── hpg2stage/      # Stage 2D architecture-aware models (see hpg2stage/README.md)
```

## EDA (`eda/`)

| Script | Purpose |
|--------|---------|
| `plot_target_distributions.py` | Target value histograms per dataset |
| `plot_feature_correlations.py` | Pearson correlation heatmaps |
| `feature_space_analysis.py` | PCA / UMAP / variance analysis (tabular features) |
| `graph_feature_space_analysis.py` | Graph-level feature space analysis |
| `check_zero_variance.py` | Flag zero-variance features |
| `analyze_ab_features.py` | A/B block feature analysis for copolymers |
| `dataset_config.yaml` | Dataset paths and target column config |

## Tabular Model Plots (`tabular/`)

| Script | Purpose |
|--------|---------|
| `plot_ea_ip_comparison.py` | Overall EA/IP model comparison |
| `plot_ea_ip_polytype_effect.py` | Effect of polytype encoding on EA/IP |
| `plot_ea_ip_strategy_effect.py` | Effect of copolymer encoding strategy |
| `plot_fusion_comparison.py` | Fusion method comparison (late_concat, FiLM, none) |
| `plot_mix_pair_comparison.py` | Mix vs. pair model comparison |
| `plot_repr_comparison.py` | Representation comparison (DMPNN/GAT/GIN) |
| `plot_attn_comparison.py` | Attention model comparison |
| `plot_tabular_ablation.py` | Tabular feature-set ablation |
| `compare_tabular_vs_graph.py` | Tabular vs. graph model comparison |
| `model_feature_interaction.py` | Feature interaction analysis |
| `best_model_feature_importance.py` | Feature importance (SHAP) for best models |

## HPG Graph Model Plots (`hpg/`)

| Script | Purpose |
|--------|---------|
| `plot_hpg_phase1.py` | Phase 1: baseline → frac → frac_polytype ablation |
| `plot_hpg_ablation.py` | Cross-variant ablation summary (all phases) |
| `plot_phase2a.py` | Phase 2A: relational message passing variants |
| `plot_phase2b.py` | Phase 2B: edge-typed variants |
| `plot_phase2b_archGraph.py` | Phase 2B: archGraph variant |
| `plot_phase3.py` | Phase 3: multi-monomer extensions |
| `plot_phase4.py` | Phase 4: full model comparison |
| `plot_phase4_diagnostics.py` | Phase 4: diagnostic/residual plots |

## Diagnostics (`diagnostics/`)

Pre-Stage-2D feasibility analyses that motivated the architecture-aware models.

| Script | Purpose |
|--------|---------|
| `pre_2d_architecture_diagnostic.py` | ANOVA / variance decomposition of architecture effect on EA/IP |
| `pre2C_diagnostics.py` | Typed interaction (BB/BF/FF) feasibility test |
| `diagnostic_3a_global_offset_transfer.py` | Global per-architecture offset transfer test |
| `feature_conditioned_architecture_transfer.py` | Feature-conditioned transfer (justified Stage 2D-1) |

## HPG2Stage (`hpg2stage/`)

See [`hpg2stage/README.md`](hpg2stage/README.md) for full details on Stage 2D
architecture-aware models, learning curve, and generalization experiments.
