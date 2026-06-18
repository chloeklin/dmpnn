# Architecture-Deviation R² Metric Audit

## Executive Summary

This audit examines four scripts that compute architecture-deviation R² (R²(Δy)) to understand why learning curve values (~0.35-0.54) differ from final Stage 2D values (~0.86-0.96).

---

## Script 1: analyze_stage2d_learning_curve.py

| Attribute | Value |
|-----------|-------|
| **Filename** | `experiments/hpg2stage/scripts/analyze_stage2d_learning_curve.py` |
| **Function** | `compute_archdev_r2(y_true, y_pred, row_indices)` (lines 85-122) |
| **Group Definition** | `smiles_A || smiles_B || fracA || fracB` (4-part key) |
| **Per-fold vs Pooled** | **Per-file** (processes each training fraction independently) |
| **Inverse Transform** | **YES** - Applies `check_needs_inverse_transform()` + `apply_inverse_transform()` per file (lines 179-180) |
| **Reduced vs Full Training** | **Reduced-training checkpoints** (frac ∈ {5%, 10%, 20%, 40%, 60%, 80%, 100%}) |
| **Group Means** | **Per-file pooled** (group means computed within each fraction's predictions) |
| **Truly R²(Δy)** | ✅ YES - Computes `r2_score(delta_true, delta_pred)` where `Δy = y - group_mean(y)` |

**Key Details:**
- Loads predictions from: `ea_ip__{target}__stage2d_{model}__fold{fold}__frac{frac_pct}__split{fold}.npz`
- Uses `test_indices` from NPZ or falls back to y_true matching
- Group key includes ALL 4 components: smiles_A, smiles_B, fracA, fracB
- Predictions may be in normalized space → applies inverse transform per file

---

## Script 2: analyze_stage2d_generalization.py

| Attribute | Value |
|-----------|-------|
| **Filename** | `experiments/hpg2stage/scripts/analyze_stage2d_generalization.py` |
| **Function** | `compute_archdev_metrics(per_split, df, target)` (lines 166-224) |
| **Group Definition** | `smiles_A | smiles_B | fracA` (3-part key, **NO fracB!**) |
| **Per-fold vs Pooled** | **Pooled across all folds** (concatenates all 5 folds before computing) |
| **Inverse Transform** | **NO** - Assumes predictions are in real scale (post-UnscaleTransform fix) |
| **Reduced vs Full Training** | **Full-training checkpoints** (100% training data) |
| **Group Means** | **Pooled across all folds** (population mean) |
| **Truly R²(Δy)** | ✅ YES - Computes `r2_score(delta_true, delta_pred)` where `Δy = y - group_mean(y)` |

**Key Details:**
- Loads predictions from: `predictions/stage2d_generalization/`
- Concatenates ALL fold predictions: `y_true_all = np.concatenate([s[0] for s in per_split])`
- Group key uses ONLY 3 components (excludes fracB) - broader groups
- Per-fold metrics available via `compute_per_fold_metrics()` but main result is pooled

---

## Script 3: stage2d_postrerun_analysis.py

| Attribute | Value |
|-----------|-------|
| **Filename** | `experiments/hpg2stage/scripts/stage2d_postrerun_analysis.py` |
| **Function** | `phase3_architecture_deviations(train_stats)` (lines 269-384) |
| **Group Definition** | `smiles_A | smiles_B | fracA` (3-part key, **NO fracB!**) |
| **Per-fold vs Pooled** | **Both** - Computes pooled R² AND per-fold R² (stored in `per_fold_dev`) |
| **Inverse Transform** | **NO** - Uses corrected predictions in real scale |
| **Reduced vs Full Training** | **Full-training checkpoints** (100% training data, post-rerun) |
| **Group Means** | **Pooled** for main metric, **per-fold** for `per_fold_dev` |
| **Truly R²(Δy)** | ✅ YES - Computes `r2_score(dt, dp)` where `dt = y_true - group_mean(y_true)` |

**Key Details:**
- Loads predictions via `load_predictions_corrected()` 
- Reports: `"R²(Δy)={r2_dev:.4f}"`
- Also computes per-fold deviations for statistical testing

---

## Script 4: analyze_pair_disjoint_transfer.py

| Attribute | Value |
|-----------|-------|
| **Filename** | `experiments/hpg2stage/scripts/analyze_pair_disjoint_transfer.py` |
| **Function** | `compute_archdev_metrics(y_true, y_pred, row_indices, df)` (lines 167-194) |
| **Group Definition** | `smiles_A || smiles_B || fracA || fracB` (4-part key) |
| **Per-fold vs Pooled** | **Per-fold** (processes each fold's predictions separately) |
| **Inverse Transform** | Not explicitly applied (assumes real scale) |
| **Reduced vs Full Training** | **Full-training checkpoints** (100% training data) |
| **Group Means** | **Per-fold** (computed within each fold's test set) |
| **Truly R²(Δy)** | ✅ YES - Computes `r2_score(dt, dp)` where `dt = y_true - group_mean(y_true)` |

**Key Details:**
- Uses `df['group_key']` with 4-part key
- Filters for groups with ≥2 architectures: `ga[ga >= 2].index`

---

## Key Differences Summary

| Dimension | Learning Curve | Stage 2D Generalization |
|-----------|----------------|------------------------|
| **Training Data** | Reduced (5%-100%) | Full (100%) |
| **Group Key** | 4-part (A,B,fracA,fracB) | 3-part (A,B,fracA) |
| **Aggregation** | Per-file | Pooled across folds |
| **Inverse Transform** | YES (per-file) | NO (already real scale) |
| **Checkpoint Type** | Learning curve fractions | Full training |

---

## Critical Finding: Group Definition Mismatch

**Learning curve scripts** use:
```python
df['group_key'] = (df['smiles_A'].astype(str) + '||' +
                   df['smiles_B'].astype(str) + '||' +
                   df['fracA'].astype(str) + '||' +
                   df['fracB'].astype(str))  # ← INCLUDES fracB
```

**Stage 2D generalization scripts** use:
```python
pred_df['group'] = (pred_df['smiles_A'].astype(str) + '|' +
                    pred_df['smiles_B'].astype(str) + '|' +
                    pred_df['fracA'].astype(str))  # ← EXCLUDES fracB
```

This means:
- Learning curve: groups are (A, B, fracA, fracB) - more granular
- Stage 2D: groups are (A, B, fracA) - broader (fracB is redundant since fracB = 1 - fracA)

---

*Generated: June 18, 2026*
