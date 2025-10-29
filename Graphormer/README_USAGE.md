# Graphormer Training Guide

## Overview

This Graphormer implementation follows the NeurIPS 2021 paper and integrates seamlessly with your existing `train_graph.py` and `train_tabular.py` workflows.

## Key Features

✅ **Identical splitting logic** - Uses same `generate_data_splits()` as other scripts  
✅ **All replicates trained** - Trains all 5 replicates (or N splits for CV)  
✅ **Configurable metrics** - Automatically selects metrics based on task type  
✅ **Proper logging** - Saves results in same format as train_graph.py  
✅ **Paper-compliant** - Gradient clipping, proper loss functions, implicit hydrogens

## Quick Start

### Basic Regression (Default)

```bash
python Graphormer/main.py \
  --csv_path data/my_dataset.csv \
  --task_type reg \
  --seed 42 \
  --batch_size 16
```

### With Same Settings as train_graph.py

If you run train_graph.py with:
```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --seed 42
```

Run Graphormer with:
```bash
python Graphormer/main.py \
  --csv_path data/htpmd.csv \
  --task_type reg \
  --seed 42 \
  --batch_size 16 \
  --replicates 5
```

**The splits will be identical!** ✓

## Command-Line Arguments

### Dataset Arguments
- `--csv_path` (required): Path to CSV file
- `--smiles_col`: SMILES column (default: "0")
- `--descriptor_cols`: Comma-separated descriptor columns (optional)
- `--target_cols`: Comma-separated target columns (optional)

### Splitting Arguments (Match train_graph.py)
- `--task_type`: "reg", "binary", or "multi" (default: "reg")
- `--n_splits`: Number of CV folds (default: 1 for holdout)
- `--replicates`: Number of replicates for holdout (default: 5)
- `--seed`: Random seed (default: 1)

### Model Architecture (Paper Defaults)
- `--num_layers`: Transformer layers (default: 12)
- `--hidden_dim`: Hidden dimension (default: 768)
- `--num_heads`: Attention heads (default: 32)
- `--dropout`: Dropout rate (default: 0.1)

### Training Arguments
- `--num_epochs`: Training epochs (default: 16)
- `--lr`: Learning rate (default: 2e-4)
- `--batch_size`: Batch size (default: 16)
- `--weight_decay`: Weight decay (default: 0.0)
- `--num_workers`: DataLoader workers (default: 4)

### Output Arguments
- `--results_dir`: Results directory (default: "results")

## Metrics by Task Type

### Regression (`--task_type reg`)
- **MAE** (Mean Absolute Error) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **R2Score** (R² Score)
- **Loss function**: L1Loss (MAE)

### Binary Classification (`--task_type binary`)
- **BinaryAccuracy**
- **BinaryF1Score**
- **BinaryAUROC** (if both classes present)
- **Loss function**: BCEWithLogitsLoss

### Multi-class Classification (`--task_type multi`)
- **MulticlassAccuracy** (macro-averaged)
- **MulticlassF1Score** (macro-averaged)
- **MulticlassAUROC** (macro-averaged)
- **Loss function**: CrossEntropyLoss

## Output Format

Results are saved to: `results/Graphormer/{dataset_name}_results.csv`

### CSV Structure
```
split,test_MAE,test_RMSE,test_R2Score,val_MAE,val_RMSE,val_R2Score,train_MAE,train_RMSE,train_R2Score
0,0.1234,0.2345,0.8765,0.1123,0.2234,0.8876,0.0987,0.1876,0.9123
1,0.1245,0.2356,0.8754,...
...
```

### Summary Statistics
The script automatically prints:
- Mean ± Std for each metric across all splits
- Full descriptive statistics (min, max, quartiles)

## Example Workflows

### 1. Standard 5-Replicate Holdout (80/10/10)

```bash
python Graphormer/main.py \
  --csv_path data/htpmd.csv \
  --task_type reg \
  --seed 42 \
  --replicates 5 \
  --batch_size 32 \
  --num_epochs 20
```

### 2. 5-Fold Cross-Validation

```bash
python Graphormer/main.py \
  --csv_path data/htpmd.csv \
  --task_type reg \
  --seed 42 \
  --n_splits 5 \
  --batch_size 32
```

### 3. Binary Classification

```bash
python Graphormer/main.py \
  --csv_path data/molhiv.csv \
  --task_type binary \
  --seed 42 \
  --batch_size 128 \
  --num_epochs 8
```

### 4. With Global Descriptors

```bash
python Graphormer/main.py \
  --csv_path data/my_dataset.csv \
  --task_type reg \
  --descriptor_count 10 \
  --seed 42
```

### 5. Smaller Model (Faster Training)

```bash
python Graphormer/main.py \
  --csv_path data/zinc.csv \
  --task_type reg \
  --num_layers 6 \
  --hidden_dim 512 \
  --num_heads 16 \
  --batch_size 64
```

## Paper-Specific Settings

### PCQM4M-LSC (Large-scale pre-training)
```bash
python Graphormer/main.py \
  --csv_path data/pcqm4m.csv \
  --task_type reg \
  --num_layers 12 \
  --hidden_dim 768 \
  --num_heads 32 \
  --batch_size 1024 \
  --lr 2e-4 \
  --num_epochs 300
```

### ZINC (Benchmarking-GNN)
```bash
python Graphormer/main.py \
  --csv_path data/zinc.csv \
  --task_type reg \
  --num_layers 12 \
  --hidden_dim 80 \
  --num_heads 8 \
  --batch_size 256 \
  --weight_decay 0.01
```

## Implementation Details

### What's Implemented from the Paper

1. **Centrality Encoding** ✓
   - Degree-based node importance encoding
   - Separate in-degree and out-degree embeddings

2. **Spatial Encoding** ✓
   - Shortest path distance (SPD) as attention bias
   - Learnable bias terms for each distance

3. **Edge Encoding** ✓
   - Average of edge features along shortest paths
   - Integrated into attention mechanism

4. **Special [VNode] Token** ✓
   - Virtual node for graph-level prediction
   - Distinct spatial encoding for virtual connections

5. **Pre-LayerNorm Architecture** ✓
   - LayerNorm before attention and FFN
   - Better optimization stability

6. **Gradient Clipping** ✓
   - Norm clipping at 5.0 (paper specification)

7. **Implicit Hydrogens** ✓
   - No explicit H atoms added (paper default)

### Differences from Paper

- **No FLAG augmentation** (only needed for fine-tuning small datasets)
- **No pre-training** (trains from scratch)
- **Fixed path length** (5 hops, configurable in code)

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--num_layers` or `--hidden_dim`
- Reduce `--num_workers`

### Slow Training
- Increase `--batch_size` (if memory allows)
- Increase `--num_workers`
- Use smaller model (`--num_layers 6 --hidden_dim 512`)

### Poor Performance
- Increase `--num_epochs`
- Try different `--lr` (1e-4 to 3e-4)
- Check task_type matches your data
- Verify CSV format is correct

## Comparison with train_graph.py

| Feature | train_graph.py | Graphormer |
|---------|---------------|------------|
| Splitting | ✓ Same | ✓ Same |
| Metrics | ✓ Same | ✓ Same |
| Results format | ✓ CSV | ✓ CSV |
| Seed control | ✓ | ✓ |
| All replicates | ✓ | ✓ |
| Model type | DMPNN/wDMPNN | Transformer |
| Architecture | Message passing | Self-attention |

## Citation

If you use this implementation, please cite the Graphormer paper:

```bibtex
@inproceedings{ying2021graphormer,
  title={Do Transformers Really Perform Bad for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  booktitle={NeurIPS},
  year={2021}
}
```
