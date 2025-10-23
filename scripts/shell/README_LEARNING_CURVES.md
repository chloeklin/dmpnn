# Learning Curve Script Generation System

## Overview

This system provides two ways to generate learning curve training scripts:

1. **YAML-based (Recommended)**: `generate_learning_curve_scripts_from_config.sh` - Reads all settings from `learning_curve_config.yaml`
2. **Command-line based**: `generate_learning_curve_scripts.sh` - Specify all settings via command-line arguments

## YAML-Based Generation (Recommended)

### Quick Start

```bash
# Generate all scripts defined in config
./generate_learning_curve_scripts_from_config.sh

# Preview what would be generated
./generate_learning_curve_scripts_from_config.sh --dry-run

# Generate only for specific datasets
./generate_learning_curve_scripts_from_config.sh --datasets opv_camb3lyp insulator

# Generate only DMPNN scripts (exclude wDMPNN)
./generate_learning_curve_scripts_from_config.sh --models DMPNN

# Generate and auto-submit
./generate_learning_curve_scripts_from_config.sh --datasets insulator --submit
```

### Graph vs Tabular Scripts

The system supports both graph-based and tabular training:

| Feature | Graph Scripts | Tabular Scripts |
|---------|---------------|-----------------|
| **Training script** | `train_graph.py` | `train_tabular.py` |
| **--model_name** | ✅ Required (DMPNN, wDMPNN) | ❌ Not used |
| **--save_predictions** | ✅ Enabled | ❌ Not used |
| **--export_embeddings** | ✅ Enabled | ❌ Not used |
| **Job name** | `{MODEL}_{dataset}_{target}...` | `tabular_{dataset}_{target}...` |
| **Script name** | `train_{dataset}_{MODEL}_{target}...` | `train_{dataset}_tabular_{target}...` |

### Configuration File: `learning_curve_config.yaml`

The YAML file stores all dataset-specific settings:

```yaml
datasets:
  opv_camb3lyp:
    targets:
      - optical_lumo
      - gap
      - homo
    train_sizes:
      - 250
      - 500
      - 1000
      - 2000
    models:
      - DMPNN
      - wDMPNN
    script_type: graph  # or tabular
    walltime: "04:00:00"
```

### Key Features

✅ **Dataset-specific train sizes** - Each dataset has its own train size list  
✅ **Model variants** - Specify which models to generate (DMPNN, wDMPNN, etc.)  
✅ **Script type** - Support for both graph and tabular training scripts  
✅ **Global settings** - PBS settings, module paths, etc. defined once  
✅ **Variant control** - Enable/disable RDKit and batch norm variants globally  
✅ **Easy maintenance** - Update config file instead of modifying scripts  

### Adding a New Dataset

Edit `learning_curve_config.yaml`:

**For graph-based models (DMPNN, wDMPNN):**
```yaml
datasets:
  my_new_dataset:
    targets:
      - target1
      - target2
    train_sizes:
      - 100
      - 500
      - 1000
    models:
      - DMPNN
      - wDMPNN
    script_type: graph
    walltime: "02:00:00"
```

**For tabular models:**
```yaml
datasets:
  my_tabular_dataset:
    targets:
      - target1
      - target2
    train_sizes:
      - 100
      - 500
      - 1000
    models:
      - tabular  # Model name is ignored for tabular
    script_type: tabular
    walltime: "01:00:00"
```

Then generate:

```bash
./generate_learning_curve_scripts_from_config.sh --datasets my_new_dataset
```

### Command-Line Options

```
--config FILE         Path to YAML config (default: scripts/shell/learning_curve_config.yaml)
--output-dir DIR      Output directory (default: ./)
--datasets DS...      Only generate for specific datasets
--models MODEL...     Only generate for specific models
--dry-run             Preview without creating files
--submit              Auto-submit to PBS queue
-h, --help            Show help
```

## Command-Line Based Generation

For one-off or custom configurations:

```bash
./generate_learning_curve_scripts.sh \
    --dataset opv_camb3lyp \
    --targets optical_lumo gap homo \
    --train-sizes 250 500 1000 \
    --model DMPNN \
    --batch-norm
```

See the script's `--help` for full options.

## Configuration Structure

### Dataset Settings

Each dataset in the YAML config has:

- **targets**: List of target columns to train on
- **train_sizes**: List of training set sizes for learning curve
- **models**: List of models to generate scripts for
- **script_type**: `graph` or `tabular`
- **walltime**: PBS walltime for this dataset

### Global Settings

Shared across all datasets:

- **PBS settings**: queue, project, resources
- **Module settings**: Python, CUDA, venv paths
- **Training flags**: save_predictions, export_embeddings, etc.
- **Variants**: Control RDKit and batch norm generation

### Variant Generation

Control which variants are generated globally:

```yaml
global:
  variants:
    rdkit: true          # Generate with RDKit descriptors
    no_rdkit: true       # Generate without RDKit descriptors
    batch_norm: false    # Generate with batch normalization
```

This generates:
- `rdkit: true, no_rdkit: true` → Both RDKit and non-RDKit variants
- `rdkit: true, no_rdkit: false` → Only RDKit variants
- `batch_norm: true` → Doubles the number of scripts (with/without batch norm)

## Examples

### Example 1: Generate All Scripts

```bash
# Preview all scripts that would be generated
./generate_learning_curve_scripts_from_config.sh --dry-run

# Generate all scripts
./generate_learning_curve_scripts_from_config.sh --output-dir ./lc_scripts
```

### Example 2: Generate for Specific Dataset

```bash
# Only OPV dataset
./generate_learning_curve_scripts_from_config.sh \
    --datasets opv_camb3lyp \
    --output-dir ./opv_scripts
```

### Example 3: Generate Only DMPNN (Not wDMPNN)

```bash
# Filter by model
./generate_learning_curve_scripts_from_config.sh \
    --models DMPNN \
    --output-dir ./dmpnn_only
```

### Example 4: Multiple Datasets, Specific Models

```bash
# OPV and insulator, only DMPNN
./generate_learning_curve_scripts_from_config.sh \
    --datasets opv_camb3lyp insulator \
    --models DMPNN \
    --output-dir ./selected_scripts
```

### Example 5: Generate and Submit

```bash
# Generate and immediately submit to PBS queue
./generate_learning_curve_scripts_from_config.sh \
    --datasets insulator \
    --submit
```

## Output Files

Generated scripts follow the naming convention:

**Graph scripts:**
```
train_{DATASET}_{MODEL}_{TARGET}[_rdkit][_batch_norm][_size{SIZE}]_lc.sh
```

**Tabular scripts:**
```
train_{DATASET}_tabular_{TARGET}[_rdkit][_batch_norm][_size{SIZE}]_lc.sh
```

Examples:
- `train_opv_camb3lyp_DMPNN_gap_size1000_lc.sh`
- `train_insulator_wDMPNN_bandgap_chain_rdkit_size512_lc.sh`
- `train_htpmd_tabular_Conductivity_rdkit_size400_lc.sh`

## Workflow

### 1. Configure

Edit `learning_curve_config.yaml` to define your datasets and settings.

### 2. Preview

```bash
./generate_learning_curve_scripts_from_config.sh --dry-run
```

### 3. Generate

```bash
./generate_learning_curve_scripts_from_config.sh --output-dir ./lc_scripts
```

### 4. Submit

```bash
cd lc_scripts
for script in train_*_lc.sh; do
    qsub "$script"
done
```

Or use `--submit` flag in step 3.

### 5. Monitor

```bash
qstat -u $USER
```

### 6. Analyze Results

After completion:

```bash
# Combine target-specific results
python3 scripts/python/combine_target_results.py results/DMPNN

# Plot learning curves
python3 plot_opv_learning_curves.py
```

## Comparison: YAML vs Command-Line

| Feature | YAML-based | Command-line |
|---------|------------|--------------|
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Reproducibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Maintenance** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Best for** | Standard workflows | One-off experiments |

**Recommendation**: Use YAML-based for standard learning curve studies. Use command-line for quick tests or non-standard configurations.

## Troubleshooting

### Error: 'yq' command not found

The YAML-based script requires `yq` for parsing YAML files.

**Install:**
- macOS: `brew install yq`
- Python: `pip install yq`
- Linux: Download from https://github.com/mikefarah/yq

### Error: Dataset not found in config

Make sure the dataset name matches exactly (case-sensitive):

```bash
# Check available datasets
yq eval '.datasets | keys' scripts/shell/learning_curve_config.yaml
```

### Too many scripts generated

Use filters to reduce:

```bash
# Only specific datasets
--datasets opv_camb3lyp

# Only specific models
--models DMPNN

# Disable variants in config
variants:
  rdkit: true
  no_rdkit: false  # Disable non-RDKit variant
```

### Scripts not executable

The script automatically makes generated files executable. If needed:

```bash
chmod +x *.sh
```

## Tips

1. **Always use `--dry-run` first** to preview what will be generated
2. **Start small** - Test with one dataset before generating all
3. **Use filters** - `--datasets` and `--models` to control scope
4. **Version control** - Commit `learning_curve_config.yaml` to track experiment settings
5. **Document changes** - Add comments in YAML file explaining modifications

## Requirements

- **bash** 4.0+
- **yq** (for YAML parsing)
- **PBS/Torque** (for job submission)

## Migration from Old Scripts

Old dataset-specific scripts (`generate_opv_learning_curve_scripts.sh`, etc.) are replaced by:

1. Add dataset to `learning_curve_config.yaml`
2. Run `generate_learning_curve_scripts_from_config.sh`

See individual dataset sections in the config file for examples.
