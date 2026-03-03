# Copolymer Training Guide

This guide explains how to train models on copolymer datasets (e.g., `ea_ip`, `block_new`) using the integrated copolymer support.

## Quick Start

### Option 1: Direct Command Line

```bash
# Basic copolymer training with mix mode
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode mix \
  --model_name DMPNN \
  --task_type reg

# Copolymer with interact mode and descriptors
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode interact \
  --model_name DMPNN \
  --incl_rdkit \
  --export_embeddings
```

### Option 2: Using Batch Scripts (Recommended for HPC)

1. **Edit the YAML configuration** (`scripts/shell/batch_experiments.yaml`):

```yaml
eaip_base: &eaip_base
  dataset: ea_ip
  task_type: reg
  polymer_type: copolymer
  export_embeddings: true

eaip_targets: &eaip_targets
  - EA vs SHE (eV)
  - IP vs SHE (eV)

experiments:
  # Copolymer with mix mode
  - <<: *eaip_base
    model: DMPNN
    walltime: "2:00:00"
    copolymer_mode: mix
    targets: *eaip_targets
  
  # Copolymer with interact mode
  - <<: *eaip_base
    model: DMPNN
    walltime: "2:00:00"
    copolymer_mode: interact
    targets: *eaip_targets
  
  # Copolymer with descriptors
  - <<: *eaip_base
    model: DMPNN
    walltime: "2:00:00"
    copolymer_mode: mix
    incl_rdkit: true
    targets: *eaip_targets
```

2. **Generate and submit jobs**:

```bash
cd scripts/shell
./batch_generate_scripts.sh batch_experiments.yaml
```

Or generate without submitting:

```bash
./batch_generate_scripts.sh batch_experiments.yaml --no-submit
```

## Dataset Requirements

Copolymer datasets must have the following columns:

- **Monomer SMILES**: `smilesA`/`smiles_A` and `smilesB`/`smiles_B`
- **Composition fractions**: `fracA` and `fracB`
- **Target columns**: One or more regression/classification targets

Example CSV structure:
```csv
smiles_A,smiles_B,fracA,fracB,EA vs SHE (eV),IP vs SHE (eV)
OB(O)c1cc(F)c(B(O)O)cc1F,Oc1cc(O)c(Br)c(O)c1Br,0.5,0.5,-3.406,1.808
...
```

## Copolymer Modes

### `mix` Mode (Level 1)
- **Formula**: `z = fracA * z_A + fracB * z_B`
- **Final embedding**: `[z || fracA || fracB || descriptors]`
- **Use case**: Simple weighted combination of monomer embeddings
- **FFN input dim**: `d_mp + 2 + d_desc`

### `interact` Mode (Level 2)
- **Formula**: Concatenates multiple interaction features
- **Final embedding**: `[z_A || z_B || |z_A - z_B| || (z_A ⊙ z_B) || fracA || fracB || descriptors]`
- **Use case**: Captures complex monomer interactions
- **FFN input dim**: `4*d_mp + 2 + d_desc`

## Supported Models

The following models support copolymer mode:
- ✅ **DMPNN** (recommended)
- ✅ **DMPNN_DiffPool**
- ✅ **GIN**, **GIN0**, **GINE**
- ✅ **GAT**, **GATv2**
- ❌ **wDMPNN** (not supported - use small-molecule encoders)
- ❌ **PPG** (not supported)

## Command Line Arguments

### Required
- `--dataset_name`: Dataset name (e.g., `ea_ip`, `block_new`)
- `--polymer_type copolymer`: Activates copolymer mode
- `--copolymer_mode {mix,interact}`: Integration mode

### Optional
- `--incl_desc`: Include dataset-specific descriptors
- `--incl_rdkit`: Include RDKit 2D descriptors
- `--batch_norm`: Enable batch normalization
- `--export_embeddings`: Save embeddings (z_A, z_B, z_final)
- `--save_checkpoint`: Save model checkpoints
- `--train_size N`: Subsample training data to N samples
- `--target "Target Name"`: Train on single target only

## Embedding Export

When using `--export_embeddings`, the following are saved:

```
results/embeddings/
├── {dataset}__{model}__{target}__copoly_{mode}__z_A_train_split_0.npy
├── {dataset}__{model}__{target}__copoly_{mode}__z_B_train_split_0.npy
├── {dataset}__{model}__{target}__copoly_{mode}__z_final_train_split_0.npy
├── {dataset}__{model}__{target}__copoly_{mode}__z_A_test_split_0.npy
├── {dataset}__{model}__{target}__copoly_{mode}__z_B_test_split_0.npy
└── {dataset}__{model}__{target}__copoly_{mode}__z_final_test_split_0.npy
```

- **z_A**: Monomer A embeddings (shape: `[N, d_mp]`)
- **z_B**: Monomer B embeddings (shape: `[N, d_mp]`)
- **z_final**: Combined embeddings after mode integration (shape: `[N, d_final]`)

## Data Splitting

Copolymer datasets use **group-based splitting** to prevent data leakage:

- If `group_key` column exists: Uses GroupKFold/GroupShuffleSplit
- Otherwise: Falls back to standard random splitting

This ensures that copolymer pairs (A, B) with the same composition are not split across train/test sets.

## Example Workflows

### 1. Compare Mix vs Interact Modes

```bash
# Mix mode
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode mix \
  --model_name DMPNN \
  --export_embeddings

# Interact mode
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode interact \
  --model_name DMPNN \
  --export_embeddings
```

### 2. Train with Different GNN Architectures

```bash
# DMPNN
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode mix \
  --model_name DMPNN

# GIN
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode mix \
  --model_name GIN

# GAT
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --polymer_type copolymer \
  --copolymer_mode mix \
  --model_name GAT
```

### 3. Learning Curve Analysis

```bash
for size in 100 200 500 1000 full; do
  python scripts/python/train_graph.py \
    --dataset_name ea_ip \
    --polymer_type copolymer \
    --copolymer_mode mix \
    --model_name DMPNN \
    --train_size $size \
    --export_embeddings
done
```

## Results

Results are saved to:
```
results/DMPNN/
└── ea_ip__DMPNN__copolymer__mix_results.csv
```

With columns:
- `split`: Cross-validation fold index
- `target`: Target name
- `test/rmse`, `test/mae`, `test/r2`: Test metrics
- Additional metrics as configured

## Troubleshooting

### Issue: "polymer_type must be 'copolymer' for datasets with smiles_A/smiles_B"
**Solution**: Add `--polymer_type copolymer` to your command

### Issue: "copolymer_mode is required when polymer_type='copolymer'"
**Solution**: Add `--copolymer_mode mix` or `--copolymer_mode interact`

### Issue: Model not supported for copolymer
**Solution**: Use DMPNN, GIN, GAT, or other small-molecule models (not wDMPNN or PPG)

### Issue: Missing columns in dataset
**Solution**: Ensure your CSV has `smiles_A`, `smiles_B`, `fracA`, `fracB` columns

## Implementation Details

- **Shared encoder**: Both monomers use the same GNN weights
- **Mean pooling**: Graph-level embeddings via mean aggregation
- **Backward compatible**: Homopolymer datasets unaffected
- **5-fold CV**: Default cross-validation strategy
- **Early stopping**: Monitors validation loss
- **Checkpointing**: Optional via `--save_checkpoint`

## Citation

If you use this copolymer implementation, please cite the original DMPNN paper and acknowledge the copolymer extension.
