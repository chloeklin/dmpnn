# HPG2Stage Experiments

Architecture-aware polymer property prediction using Stage 2D models.

## Structure

```
hpg2stage/
├── scripts/          # Analysis & plotting scripts
└── output/           # Generated results
    ├── postrerun/    # Main model comparison (7 variants, 5-fold)
    ├── diagnostics/  # Bottleneck diagnostic experiments
    └── learning_curve/  # Training-side learning curve
```

## Scripts

| Script | Purpose |
|--------|---------|
| `stage2d_postrerun_analysis.py` | Main analysis: overall R², arch-dev R², significance tests |
| `plot_stage2d_analysis.py` | Publication figures for model comparison |
| `stage2d_diagnostic_experiments.py` | Bottleneck diagnostics (group audit, effect size, generalization) |
| `analyze_stage2d_learning_curve.py` | Learning curve analysis (after training completes) |
| `build_hpg2stage_comparison_table.py` | Comparison table across all HPG2Stage variants |
| `plot_hpg2stage_comparison.py` | Quick comparison plots |
| `generate_stage2d_presentation.py` | Generate presentation materials |
| `audit_stage2d.py` | Checkpoint/prediction audit |
| `stage2d_prerun_audit.py` | Pre-training readiness checks |
| `stage2d_recovery_audit.py` | Post-failure recovery diagnostics |

## Key Results

- **Best model**: 2D1-arch (R²=0.982 EA, 0.979 IP)
- **Arch-dev R²**: 2D1-arch significantly better than 2D0-arch (p<0.03)
- **Learning curve**: See `output/learning_curve/` after training

## Running

```bash
# Main analysis (requires predictions in predictions/HPG2Stage/)
python experiments/hpg2stage/scripts/stage2d_postrerun_analysis.py

# Learning curve (requires training first)
python scripts/python/run_stage2d_learning_curve.py --dry_run
# Then submit jobs via scripts/shell/submit_learning_curve.sh
# Then analyze:
python experiments/hpg2stage/scripts/analyze_stage2d_learning_curve.py
```
