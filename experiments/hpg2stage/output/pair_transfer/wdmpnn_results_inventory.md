# wDMPNN Results Inventory — ea_ip Dataset

## Summary

| Split Type | Metric CSV | Prediction .npz | Checkpoints (.ckpt) | Status |
|---|---|---|---|---|
| **random** | ❌ None | ✅ 2 files (split0 only, EA+IP) | ✅ 10 dirs (EA×5, IP×3 complete) | Partial |
| **a_held_out** | ✅ 1 file (5 folds × 2 targets) | ❌ None | ✅ 10 dirs (EA×5, IP×5, all complete) | **Metrics only** |
| **group_disjoint** | ❌ None | ❌ None | ❌ None | **MISSING** |
| **pair_disjoint** | ❌ None | ❌ None | ❌ None | **MISSING** |

---

## 1. Which wDMPNN Results Exist

### A. Results CSV (metrics only, no y_true/y_pred)

| File | Split | Contents |
|---|---|---|
| `results/wDMPNN/ea_ip__a_held_out_results.csv` | a_held_out | 5 folds × 2 targets (EA, IP); columns: test/mae, test/rmse, test/r2, split, target |

**Values from CSV:**
```
split  target          MAE       RMSE      R²
0      IP vs SHE (eV)  0.0067    0.0113    0.9993
1      IP vs SHE (eV)  0.0782    0.1402    0.9247
2      IP vs SHE (eV)  0.0540    0.0887    0.9660
3      IP vs SHE (eV)  0.0596    0.1037    0.9563
4      IP vs SHE (eV)  0.0506    0.0856    0.9683
0      EA vs SHE (eV)  0.0097    0.0145    0.9994
1      EA vs SHE (eV)  0.0764    0.1195    0.9596
2      EA vs SHE (eV)  0.0607    0.1018    0.9735
3      EA vs SHE (eV)  0.0553    0.0945    0.9739
4      EA vs SHE (eV)  0.0618    0.1170    0.9618
```

Mean ± std (from HPG2Stage comparison table):
- EA: RMSE = 0.0895 ± 0.0432, R² = 0.9736 ± 0.0158
- IP: RMSE = 0.0859 ± 0.0470, R² = 0.9630 ± 0.0268

### B. Prediction Files (.npz with y_true/y_pred)

| File | Split | Fold | Contains y_true/y_pred? | n_test |
|---|---|---|---|---|
| `predictions/wDMPNN/ea_ip__EA vs SHE (eV)__split0.npz` | random (10-fold) | 0 | ✅ Yes | 4297 |
| `predictions/wDMPNN/ea_ip__IP vs SHE (eV)__split0.npz` | random (10-fold) | 0 | ✅ Yes | 4297 |

**NPZ structure:**
- Keys: `y_true`, `y_pred`, `metadata`, `test_ids`
- `y_true`: shape (4297,), float64
- `y_pred`: shape (4297, 1), float32
- `metadata`: dict with `model`, `split`, `polymer_type`, `fusion_mode`, `aux_task`
- `test_ids`: shape (4297,), object (string indices like "idx_0", "idx_1", ...)
- No `test_indices` key (unlike HPG2Stage_Gen predictions)

**Note:** n_test = 4297 = 42966/10. This appears to be fold 0 of a **random 10-fold split**, NOT the a_held_out split (which has ~8596–9548 test samples per fold). Only 1 fold saved, only for the random split.

### C. Checkpoints

**a_held_out (complete, all 10 dirs have TRAINING_COMPLETE + local .ckpt files):**
```
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__a_held_out__rep0/  (4 .ckpt files)
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__a_held_out__rep1/  (2 .ckpt files)
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__a_held_out__rep2/  (8 .ckpt files)
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__a_held_out__rep3/  (4 .ckpt files)
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__a_held_out__rep4/  (6 .ckpt files)
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__a_held_out__rep0/  (5 .ckpt files)
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__a_held_out__rep1/  (2 .ckpt files)
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__a_held_out__rep2/  (8 .ckpt files)
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__a_held_out__rep3/  (4 .ckpt files)
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__a_held_out__rep4/  (6 .ckpt files)
```

**random split (partial — some incomplete):**
```
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__rep0/  DONE
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__rep1/  DONE
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__rep2/  DONE
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__rep3/  DONE
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__rep4/  INCOMPLETE
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__rep0/  DONE
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__rep1/  INCOMPLETE
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__rep2/  INCOMPLETE
```

**Note:** `best.json` in a_held_out checkpoints points to paths on the Gadi HPC cluster (`/scratch/um09/hl4138/dmpnn/checkpoints/wDMPNN/...`), but local .ckpt files also exist in the `logs/checkpoints/` subdirectory.

---

## 2. Which Splits Are Missing

| Split Type | What's Missing | Impact |
|---|---|---|
| **group_disjoint** | Everything (no metrics, no predictions, no checkpoints) | Must train from scratch |
| **pair_disjoint** | Everything (no metrics, no predictions, no checkpoints) | Must train from scratch |
| **a_held_out** | Prediction .npz files (y_true/y_pred per fold) | Checkpoints exist → can regenerate predictions without retraining |
| **random** | Only fold 0 prediction saved; 3 checkpoints incomplete | Not needed for current analysis |

---

## 3. Exact File Paths Found

### Predictions
```
predictions/wDMPNN/ea_ip__EA vs SHE (eV)__split0.npz
predictions/wDMPNN/ea_ip__IP vs SHE (eV)__split0.npz
```

### Results CSV
```
results/wDMPNN/ea_ip__a_held_out_results.csv
```

### Checkpoints (ea_ip only)
```
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__a_held_out__rep{0,1,2,3,4}/
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__a_held_out__rep{0,1,2,3,4}/
checkpoints/wDMPNN/ea_ip__EA vs SHE (eV)__rep{0,1,2,3,4}/
checkpoints/wDMPNN/ea_ip__IP vs SHE (eV)__rep{0,1,2}/
```

---

## 4. Whether Files Contain y_true/y_pred Per Fold

| Source | y_true/y_pred? | Per fold? | Notes |
|---|---|---|---|
| `predictions/wDMPNN/ea_ip__*__split0.npz` | ✅ Yes | ❌ Only fold 0 | Random split only, not a_held_out |
| `results/wDMPNN/ea_ip__a_held_out_results.csv` | ❌ No (metrics only) | ✅ All 5 folds | Has R², MAE, RMSE but no raw predictions |
| Checkpoints (a_held_out) | Can regenerate | ✅ All 5 folds available | .ckpt files present locally, can re-run inference |

---

## 5. Relevance to Stage 2D Comparison

The existing HPG2Stage comparison table includes wDMPNN for a_held_out:
- EA R² = 0.9736 ± 0.0158
- IP R² = 0.9630 ± 0.0268

This is **worse** than HPG-2Stage-Frac (EA 0.9832, IP 0.9796) and substantially worse than Stage 2D arch models.

**To include wDMPNN in the pair-disjoint architecture transfer analysis:**
- group_disjoint and pair_disjoint models must be trained from scratch
- a_held_out predictions can be regenerated from existing checkpoints (if architecture-deviation R² is needed)

---

## 6. Naming Convention

The wDMPNN model in this codebase uses:
- Model name: `wDMPNN` 
- Copolymer mode: `mix` (internally uses `copoly_mix` — late-concat fusion of monomer embeddings)
- Training: via `train_graph.py --model_name wDMPNN`
- Results subdir: `wDMPNN/`
- Predictions subdir: `wDMPNN/`
- Checkpoints subdir: `wDMPNN/`
- YAML config: `batch_experiments.yaml` has commented-out wDMPNN entries for ea_ip (both random and a_held_out)
