# Stage 2D Learning Curve — Final Pipeline Plan

## Why Rerun?

The previous learning-curve checkpoints (`HPG2Stage_LC/`) are **invalid** for
saturation analysis because they used a different pipeline from the final
Stage 2D models:

| Parameter | Old LC Script | Final train_graph.py | Impact |
|-----------|--------------|---------------------|--------|
| EPOCHS | 100 | **300** | Under-trained models |
| PATIENCE | 15 | **30** | Premature early stopping |
| Split generation | Raw `df['smiles_A']` (all rows) | **Filtered `valid_smiles_A`** | Different fold assignments |
| Test set sizes | ~9548 per fold | ~8596 per fold | Incompatible predictions |
| 100% R²(Δ) | ~0.44 | ~0.88 | 50% gap proves invalidity |

## Pipeline Files

### Final Stage 2D (reference)

- **Training**: `scripts/python/train_graph.py` (copolymer branch, lines 700-1260)
- **Config**: `scripts/python/train_config.yaml` (EPOCHS=300, PATIENCE=30)
- **Batch spec**: `scripts/shell/batch_experiments.yaml` (lines 1768-1806)
- **Split function**: `scripts/python/utils.py::generate_a_held_out_splits()`
- **Data function**: `scripts/python/utils.py::create_copolymer_data()`
- **Model builder**: `scripts/python/utils.py::build_copolymer_model_and_trainer()`
- **Arch-dev metric**: `experiments/hpg2stage/scripts/analyze_pair_disjoint_transfer.py::compute_archdev_metrics()`
- **Predictions**: `predictions/HPG2Stage/`
- **Checkpoints**: `checkpoints/HPG2Stage/`

### New Learning Curve (this run)

- **Training**: `experiments/hpg2stage/scripts/run_stage2d_learning_curve_final.py`
- **Evaluation**: `experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py`
- **Configs**: `experiments/hpg2stage/stage2d_learning_curve_final_configs/*.yaml`
- **PBS script**: `scripts/shell/submit_learning_curve_final.sh`
- **Predictions**: `predictions/HPG2Stage_LC_Final/`
- **Checkpoints**: `checkpoints/HPG2Stage_LC_Final/`
- **Output**: `experiments/hpg2stage/output/learning_curve_final/`

## What Changed (New vs Old LC)

Four changes, all to match `train_graph.py`:

1. **EPOCHS=300, PATIENCE=30** (was 100/15)
2. **Pair canonicalization**: swap A↔B if B < A alphabetically, matching
   `load_and_preprocess_data()`. Without this, GroupKFold sees 9 groups
   instead of 653, producing completely different fold assignments.
3. **Fraction normalization**: fracA + fracB = 1.0 exactly
4. **Saves original-df row indices in predictions** (for direct group_key lookup)

Everything else is identical: same `build_copolymer_model_and_trainer()`,
same `create_copolymer_data()`, same `generate_a_held_out_splits()`,
same `CopolymerDataset`, same `normalize_targets()`, same `trainer.predict()`.

## Verification Protocol

### Step 1: Run 100% fraction only

```bash
# On cluster:
./scripts/shell/submit_learning_curve_final.sh
# Submits 10 jobs: 2 models × 5 folds × 100% only
```

### Step 2: Evaluate 100% predictions

```bash
python experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py \
    --fractions 100
```

Expected output:

| Model | Target | R²(Δ) | Reference | Tolerance |
|-------|--------|-------|-----------|-----------|
| 2D0-arch | EA | ≈ 0.84 | 0.84 | ±0.05 |
| 2D0-arch | IP | ≈ 0.91 | 0.91 | ±0.05 |
| 2D1-arch | EA | ≈ 0.86 | 0.86 | ±0.05 |
| 2D1-arch | IP | ≈ 0.91 | 0.91 | ±0.05 |

**If ANY check fails: STOP and debug. Do NOT proceed to 25/50/75%.**

### Step 3: Run remaining fractions

```bash
./scripts/shell/submit_learning_curve_final.sh --all
# Submits 30 additional jobs: 2 models × 5 folds × 3 fractions
```

### Step 4: Full evaluation

```bash
python experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py
```

## Outputs

After all runs complete:

| File | Description |
|------|-------------|
| `stage2d_learning_curve_final_metrics.csv` | Per-fold metrics (model, fold, fraction, ea/ip r2/mae/rmse/arch_r2/arch_mae) |
| `stage2d_learning_curve_final_summary.md` | Per-fold mean±std table + verification report |
| `fig_final_learning_curve_EA_overall.png/pdf` | EA R² vs fraction |
| `fig_final_learning_curve_IP_overall.png/pdf` | IP R² vs fraction |
| `fig_final_learning_curve_EA_archdev.png/pdf` | EA R²(Δ) vs fraction |
| `fig_final_learning_curve_IP_archdev.png/pdf` | IP R²(Δ) vs fraction |
| `selected_groups_fold{f}_frac{p}.json` | Group keys selected for each fold×fraction |
| `metadata_fold{f}_frac{p}.json` | Subsampling metadata |

## Run Commands Summary

```bash
# 1. Generate group subsets (dry run, no training)
python experiments/hpg2stage/scripts/run_stage2d_learning_curve_final.py --dry_run

# 2. Local test (single fold, single fraction)
python experiments/hpg2stage/scripts/run_stage2d_learning_curve_final.py \
    --models 2d0_arch --folds 0 --fractions 100

# 3. Cluster: Phase 1 (100% only)
./scripts/shell/submit_learning_curve_final.sh

# 4. Evaluate Phase 1
python experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py \
    --fractions 100

# 5. Cluster: Phase 2 (all fractions)
./scripts/shell/submit_learning_curve_final.sh --all

# 6. Full evaluation
python experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py
```

---

*Created: June 18, 2026*
