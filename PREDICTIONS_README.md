# EA/IP Prediction Files

## Folder Structure

```
predictions/
├── ea_ip_group/     ← group_disjoint split (5 folds)
├── ea_ip_pair/      ← pair_disjoint split  (5 folds)
└── ea_ip_lomo/      ← monomer_heldout split (9 folds, LOMAO)
```

## Filename Convention

```
ea_ip__{target}__{model}__{split}__fold{N}.npz
```

### Target tokens

| Token            | Physical column          |
|------------------|--------------------------|
| `EA_vs_SHE_eV`   | `EA vs SHE (eV)`         |
| `IP_vs_SHE_eV`   | `IP vs SHE (eV)`         |

### Model tokens

| Token        | Internal name(s)                                     |
|--------------|------------------------------------------------------|
| `frac`       | `stage2d_frac`, `copoly_stage2d_frac`               |
| `wdmpnn`     | `wDMPNN`, HPG2Stage wDMPNN (no copolymer_mode)      |
| `globalarch` | `stage2d_2d0_arch`, `copoly_stage2d_2d0_arch`       |
| `chemarch`   | `stage2d_2d1_arch`, `copoly_stage2d_2d1_arch`       |

### Split tokens

| Token              | Old names                               |
|--------------------|-----------------------------------------|
| `group_disjoint`   | `group_disjoint`                        |
| `pair_disjoint`    | `pair_disjoint`                         |
| `monomer_heldout`  | `a_held_out`, `lomo`, `LOMO`, `lomao`   |

### Examples

```
ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_heldout__fold0.npz
ea_ip__IP_vs_SHE_eV__chemarch__group_disjoint__fold3.npz
ea_ip__EA_vs_SHE_eV__frac__pair_disjoint__fold1.npz
```

## Prediction Scale

**All saved files use physical units (eV).**

Files include a `prediction_scale` field in their saved arrays (new files) or
metadata dict (legacy lomo files). If `prediction_scale` is absent, check
`y_true` statistics: mean ≈ −2.5 eV, std ≈ 0.5 eV for EA; mean ≈ 5.8 eV for IP.

A normalised file would show std ≈ 1.0.

## File Contents

New-format files (group/pair splits) contain:

| Key              | Description                                              |
|------------------|----------------------------------------------------------|
| `y_true`         | True target values in physical units (eV)                |
| `y_pred`         | Predicted values in physical units (eV)                  |
| `test_indices`   | **Global** df row indices for each test sample           |
| `split_type`     | Canonical split token                                    |
| `model`          | Canonical model token                                    |
| `target`         | Canonical target token                                   |
| `fold`           | Fold index                                               |
| `n_train/val/test` | Sample counts                                          |
| `prediction_scale` | `"physical_units"`                                     |

Legacy lomo files contain `y_true`, `y_pred`, `metadata` (dict), `test_ids`
(local `idx_0..idx_N`).  Global indices must be reconstructed from the split
metadata files (see below).

## Split Metadata

Machine-readable split definitions are stored in:

```
metadata/splits/
├── monomer_heldout.json
├── group_disjoint.json
└── pair_disjoint.json
```

Each file contains per-fold records including:
- `global_test_indices`  — global df row indices (use for `compute_archdev_r2`)
- `held_out_monomer_A`   — the held-out monomer (monomer_heldout only)
- `leakage_check_passed` — True if held-out monomer absent from training set
- Source file hash for reproducibility

To regenerate:

```bash
python scripts/generate_split_metadata.py
```

## Regenerating Metrics

```bash
# All splits, all models, all targets (fold-level)
python scripts/evaluate_ea_ip_predictions.py

# Only monomer_heldout, EA, with scale sanity check
python scripts/evaluate_ea_ip_predictions.py \
    --split monomer_heldout \
    --target "EA vs SHE (eV)" \
    --scale-check

# Show only pooled (mean ± SD) across folds
python scripts/evaluate_ea_ip_predictions.py --pooled
```

## Shared Utilities

| Module                       | Purpose                                           |
|------------------------------|---------------------------------------------------|
| `evaluation/naming.py`       | `make_prediction_filename`, `standard_model_name`, `standard_split_name` |
| `evaluation/metrics.py`      | `compute_overall_r2`, `compute_archdev_r2`, `scale_sanity_check`, `validate_prediction_inputs` |
| `scripts/migrate_prediction_filenames.py` | Rename old files to canonical convention |
| `scripts/generate_split_metadata.py`      | Write `metadata/splits/*.json`           |

## Migration Log

`predictions/migration_log.json` records every file renamed by the migration
script (old_name → new_name, action, date).

## Important Notes

- **Always use global df row indices** when calling `compute_archdev_r2`.
  The legacy lomo files store only local positional indices (`idx_0..idx_N`);
  use the `global_test_indices` from `metadata/splits/monomer_heldout.json`.
- **Do not mix normalised and physical-unit files** when comparing models.
- The migration script copies (does not delete) old files. Old filenames remain
  for backward compatibility with older analysis scripts.
