# Analysis

Scripts and figures for exploring data and visualising experiment results.

## Directory structure

```
analysis/
  eda/                      exploratory data analysis scripts
  results/
    tabular/                plotting scripts for tabular model experiments
    hpg/                    plotting scripts for HPG graph model experiments
  figures/
    tabular_ea_ip/          EA/IP tabular experiment figures + reports
    tabular_ablation/       tabular feature-set ablation figures
    tabular_fusion/         fusion strategy comparison figures
    tabular_repr/           representation comparison figures
    hpg_phase1/             HPG Phase 1 ablation figures (baseline → frac_polytype)
    hpg_phase1d/            HPG Phase 1D figures (frac_edgeTyped)
    hpg_phase1e/            HPG Phase 1E figures (frac_archAware)
    hpg_phase2/             HPG Phase 2 figures (relMsg, …) — generated on run
```

## EDA scripts (`eda/`)

| Script | Purpose |
|---|---|
| `plot_target_distributions.py` | Target value histograms per dataset |
| `plot_feature_correlations.py` | Pearson correlation heatmaps |
| `feature_space_analysis.py` | PCA / UMAP / variance analysis (tabular) |
| `graph_feature_space_analysis.py` | Graph-level feature space analysis |
| `check_zero_variance.py` | Flag zero-variance features |
| `analyze_ab_features.py` | A/B block feature analysis |
| `dataset_config.yaml` | Dataset paths and target column config |

## Results scripts (`results/tabular/`)

| Script | Purpose |
|---|---|
| `plot_ea_ip_comparison.py` | Overall EA/IP model comparison |
| `plot_ea_ip_polytype_effect.py` | Effect of polytype on EA/IP predictions |
| `plot_ea_ip_strategy_effect.py` | Effect of copolymer encoding strategy |
| `plot_fusion_comparison.py` | Fusion method comparison |
| `plot_mix_pair_comparison.py` | Mix vs. pair model comparison |
| `plot_repr_comparison.py` | Representation comparison (DMPNN/GAT/GIN) |
| `plot_attn_comparison.py` | Attention model comparison |
| `plot_tabular_ablation.py` | Tabular feature-set ablation |
| `compare_tabular_vs_graph.py` | Tabular vs. graph model comparison |
| `model_feature_interaction.py` | Feature interaction analysis |
| `best_model_feature_importance.py` | Feature importance for best models |

## Results scripts (`results/hpg/`)

| Script | Purpose |
|---|---|
| `plot_hpg_phase1.py` | HPG Phase 1 ablation plots |
| `plot_hpg_ablation.py` | HPG cross-variant ablation summary (Phase 2+) |
