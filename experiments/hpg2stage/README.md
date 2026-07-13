# HPG2Stage Experiments

Architecture-aware polymer property prediction using Stage 2D models.

## Structure

```text
hpg2stage/
├── scripts/                              # Analysis, training & plotting scripts
├── stage2d_learning_curve_final_configs/ # YAML configs for LC experiment
└── output/
    ├── postrerun/          # Main model comparison (7 variants, 5-fold)
    ├── bottleneck/         # Bottleneck diagnostic results
    ├── generalization/     # Group/pair-disjoint generalization results
    ├── pair_transfer/      # Pair-disjoint transfer analysis
    ├── learning_curve_final/  # Final LC metadata & group selections
    └── diagnostics/        # Diagnostic experiment outputs
```

## Scripts

### Core Analysis

| Script | Purpose |
|--------|---------|
| `stage2d_postrerun_analysis.py` | Master analysis: overall R², arch-dev R², significance tests, best-model selection |
| `analyze_bottleneck.py` | Signal magnitude + matched-group saturation assessment |
| `analyze_stage2d_generalization.py` | Group-disjoint and pair-disjoint generalization metrics |
| `analyze_pair_disjoint_transfer.py` | Pair-held-out architecture transfer analysis (Experiment C) |
| `stage2d_diagnostic_experiments.py` | Bottleneck diagnostics: data-limited vs model-limited |

### Learning Curve (Final Pipeline)

| Script | Purpose |
|--------|---------|
| `run_stage2d_learning_curve_final.py` | Training script — replicates exact `train_graph.py` pipeline with matched-group subsampling |
| `evaluate_stage2d_learning_curve_final.py` | Evaluation — computes overall + arch-dev metrics, generates figures |

### Plotting & Reporting

| Script | Purpose |
|--------|---------|
| `plot_stage2d_analysis.py` | Publication figures: Frac → 2D0 → 2D1 progression |
| `plot_hpg2stage_comparison.py` | Quick comparison barplots across all HPG2Stage variants |
| `build_hpg2stage_comparison_table.py` | Markdown comparison table from aggregate CSVs |
| `generate_stage2d_presentation.py` | Supervisor presentation: slides, figures, summary |

## Key Results

- **Best model**: 2D1-arch (R²=0.982 EA, 0.979 IP)
- **Arch-dev R²**: 2D1-arch significantly better than 2D0-arch (p<0.03)
- **Learning curve**: See `output/learning_curve_final/` after training

## Running

```bash
# Main analysis (requires predictions in predictions/HPG2Stage/)
python experiments/hpg2stage/scripts/stage2d_postrerun_analysis.py

# Learning curve — dry run to verify splits
python experiments/hpg2stage/scripts/run_stage2d_learning_curve_final.py --dry_run

# Learning curve — submit to PBS cluster
bash scripts/shell/submit_learning_curve_final.sh

# Learning curve — evaluate after training
python experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py
```
