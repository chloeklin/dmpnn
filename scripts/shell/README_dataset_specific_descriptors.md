# Dataset-Specific Descriptor Configuration

The learning curve script generator now supports dataset-specific descriptor configuration with model-based script type detection, allowing you to specify which datasets have descriptors available and should generate descriptor variants.

## Configuration

For each dataset in your `learning_curve_config.yaml`, you can add:

```yaml
datasets:
  your_dataset:
    targets:
      - your_target
    train_sizes:
      - 250
      - 500
    models:
      - tabular
      - DMPNN
      - AttentiveFP  # Mix of tabular and graph models
    walltime: "02:00:00"
    
    # Dataset-specific descriptor settings
    has_descriptors: true      # Set to true if dataset has descriptors
    descriptor_variants:
      use_descriptors: true    # Generate scripts with --incl_desc
      no_descriptors: true    # Generate scripts without --incl_desc
```

## Model-Based Script Type Detection

Script type is now automatically determined based on the model:

- **`tabular`** model → `script_type: tabular`
- **All other models** (DMPNN, AttentiveFP, etc.) → `script_type: graph`

This allows datasets to have both tabular and graph models in the same configuration.

## Field Descriptions

- **`has_descriptors`**: Boolean indicating if the dataset has additional descriptors available
  - `true`: Dataset has descriptors (e.g., insulator, htpmd)
  - `false`: Dataset doesn't have descriptors (e.g., opv_camb3lyf, tc)
  - **Important**: When `false`, the script won't read `descriptor_variants` settings

- **`descriptor_variants`**: Controls which descriptor variants to generate (only read if `has_descriptors: true`)
  - **`use_descriptors`**: Generate scripts with `--incl_desc` flag
  - **`no_descriptors`**: Generate scripts without `--incl_desc` flag

## Examples

### Dataset WITH Descriptors (e.g., Insulator)
```yaml
insulator:
  targets:
    - bandgap_chain
  train_sizes:
    - 256
    - 512
  models:
    - tabular
    - DMPNN
    - AttentiveFP
  walltime: "02:00:00"
  has_descriptors: true
  descriptor_variants:
    use_descriptors: true
    no_descriptors: true
```

**Generated Scripts:**
- Tabular: baseline, +desc, +RDKit, +desc+RDKit (4 variants)
- Graph: baseline, +RDKit (2 variants each)

### Dataset WITHOUT Descriptors (e.g., OPV)
```yaml
opv_camb3lyf:
  targets:
    - gap
    - homo
  train_sizes:
    - 250
    - 500
  models:
    - tabular
    - DMPNN
  walltime: "02:00:00"
  has_descriptors: false
  # No descriptor_variants section needed
```

**Generated Scripts:**
- Tabular: baseline, +RDKit (2 variants)
- Graph: baseline, +RDKit (2 variants each)

## Default Behavior

- If `has_descriptors` is not specified or is `null`, it defaults to `false`
- If `has_descriptors` is `false`, descriptor variant settings are not read
- If `has_descriptors` is `true` but variant settings are missing, they default to:
  - `use_descriptors: false`
  - `no_descriptors: true`

## Script Naming

- **Tabular scripts**: `train_dataset_tabular_target[_desc][_rdkit]_sizeN_lc.sh`
- **Graph scripts**: `train_dataset_model_target[_rdkit]_sizeN_lc.sh`

The `_desc` suffix only appears for tabular scripts when descriptors are used.

## Mixed Model Support

A single dataset can now include both tabular and graph models:

```yaml
datasets:
  mixed_dataset:
    models:
      - tabular      # Uses train_tabular.py
      - DMPNN        # Uses train_graph.py
      - AttentiveFP   # Uses train_attentivefp.py
```

Each model type gets the appropriate training script and variant generation logic.
