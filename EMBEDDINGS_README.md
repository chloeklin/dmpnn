# Embeddings-Only Training Scripts

This directory contains scripts for running DMPNN model training and stopping after embedding extraction (no evaluation step).

## Overview

The embeddings-only workflow allows you to:
1. Train DMPNN models on your datasets
2. Automatically extract and save GNN embeddings
3. Stop after embedding export (skip evaluation)
4. Use saved embeddings for downstream analysis

## Available Scripts

### 1. Batch Script Generator (PBS Jobs)
**File**: `scripts/shell/batch_generate_embeddings_scripts.sh`

Generates PBS job scripts for HPC clusters that run training until embedding extraction, then stop.

```bash
# Usage examples
./scripts/shell/batch_generate_embeddings_scripts.sh                                    # Use default config
./scripts/shell/batch_generate_embeddings_scripts.sh --no-submit                        # Generate without submitting
./scripts/shell/batch_generate_embeddings_scripts.sh --model DMPNN                      # Filter by model
./scripts/shell/batch_generate_embeddings_scripts.sh custom.yaml --no-submit           # Custom config
```

### 2. Individual Script Generator
**File**: `scripts/shell/generate_embeddings_script.sh`

Generates a single PBS job script for one experiment configuration.

```bash
# Usage examples
./scripts/shell/generate_embeddings_script.sh insulator DMPNN 2:00:00
./scripts/shell/generate_embeddings_script.sh htpmd wDMPNN 4:00:00 incl_rdkit incl_desc
./scripts/shell/generate_embeddings_script.sh polyinfo DMPNN 3:00:00 incl_rdkit multi --no-submit
```

### 3. Local Python Runner
**File**: `scripts/python/run_embeddings_only.py`

Runs embeddings-only training locally (without PBS). Good for testing or local execution.

```bash
# Single experiment
python3 scripts/python/run_embeddings_only.py --dataset insulator --model DMPNN --incl_rdkit --target bandgap_chain

# Batch from config
python3 scripts/python/run_embeddings_only.py --config scripts/shell/embeddings_experiments.yaml --model_filter DMPNN
```

## Configuration Files

### Example Config: `scripts/shell/embeddings_experiments.yaml`

```yaml
experiments:
  - dataset: insulator
    model: DMPNN
    walltime: "2:00:00"
    incl_rdkit: true
    task_type: reg
    targets:
      - bandgap_chain
      - bandgap_mol
  
  - dataset: opv_camb3lyp
    model: wDMPNN
    walltime: "3:00:00"
    incl_rdkit: true
    batch_norm: true
    task_type: reg
    targets:
      - gap
      - homo
```

### Supported Configuration Options

- **dataset**: Dataset name (required)
- **model**: Model name - DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP, PPG (required)
- **walltime**: PBS walltime limit (required)
- **task_type**: reg, binary, multi (default: reg)
- **incl_rdkit**: Include RDKit descriptors (default: false)
- **incl_desc**: Include dataset-specific descriptors (default: false)
- **incl_ab**: Include atom/bond pooled features (default: false)
- **batch_norm**: Use batch normalization (default: false)
- **pretrain_monomer**: Train multitask monomer model (default: false)
- **train_size**: Training size for learning curves (e.g., "500", "full")
- **targets**: List of targets (generates separate scripts per target)
- **target**: Single target (generates one script)

## Output

### Embeddings Location
Embeddings are saved to: `results/embeddings/`

### File Naming Pattern
```
{dataset}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}__X_{split}_split_{i}.npy
```

Examples:
- `insulator__bandgap_chain__rdkit__X_train_split_0.npy`
- `opv_camb3lyp__gap__rdkit__X_val_split_1.npy`
- `htpmd__Conductivity__desc__rdkit__batch_norm__X_test_split_0.npy`

### Feature Masks
For each embedding set, a feature mask is saved:
- `{prefix}__feature_mask_split_{i}.npy`

This mask indicates which embedding dimensions were kept after low-variance filtering.

## Workflow Examples

### 1. Generate and Submit PBS Jobs
```bash
# Generate scripts for all experiments in config and submit
./scripts/shell/batch_generate_embeddings_scripts.sh

# Generate scripts but don't submit (for review)
./scripts/shell/batch_generate_embeddings_scripts.sh --no-submit
```

### 2. Run Single Experiment Locally
```bash
# Test a single configuration
python3 scripts/python/run_embeddings_only.py \
    --dataset insulator \
    --model DMPNN \
    --incl_rdkit \
    --target bandgap_chain
```

### 3. Batch Run with Model Filter
```bash
# Run only DMPNN experiments from config
python3 scripts/python/run_embeddings_only.py \
    --config scripts/shell/embeddings_experiments.yaml \
    --model_filter DMPNN
```

## Using Extracted Embeddings

Once embeddings are extracted, you can use them with:

### 1. Evaluation Script
```bash
# Use pre-computed embeddings for faster evaluation
python3 scripts/python/evaluate_model.py \
    --dataset insulator \
    --model DMPNN \
    --target bandgap_chain \
    --incl_rdkit \
    --export_embeddings  # Loads existing embeddings
```

### 2. Custom Analysis
```python
import numpy as np

# Load embeddings
X_train = np.load('results/embeddings/insulator__bandgap_chain__rdkit__X_train_split_0.npy')
feature_mask = np.load('results/embeddings/insulator__bandgap_chain__rdkit__feature_mask_split_0.npy')

print(f"Training embeddings shape: {X_train.shape}")
print(f"Features kept: {feature_mask.sum()} / {len(feature_mask)}")
```

## Key Features

✅ **Automatic Embedding Export**: Always includes `--export_embeddings` flag  
✅ **Early Termination**: Stops after embedding extraction (no evaluation)  
✅ **PBS Integration**: Generates HPC-ready job scripts  
✅ **Local Execution**: Python runner for local/testing use  
✅ **Batch Processing**: YAML config for multiple experiments  
✅ **Model Filtering**: Generate/run specific model types only  
✅ **Target Support**: Per-target or multi-target training  
✅ **Consistent File Naming**: Matches evaluation script expectations  

## Notes

- Only graph models (DMPNN, wDMPNN, etc.) support embeddings - tabular models are automatically skipped
- Embeddings are filtered to remove low-variance features (eps=1e-8) for consistency with evaluation
- The same preprocessing pipeline is used as in full training to ensure reproducibility
- Generated scripts include comprehensive logging and progress tracking
