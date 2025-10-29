# Graphormer Integration - Quick Start Guide

This guide shows how to use the official DGL Graphormer implementation with your existing dataset infrastructure.

## Overview

The new `train_graphormer.py` script integrates the official DGL Graphormer example with your existing training patterns from `train_graph.py` and `train_tabular.py`.

## Key Features

✅ **Uses Official DGL Implementation**: Leverages `Graphormer/dgl/` code (official DGL example)  
✅ **Your Dataset Format**: Works with your CSV files in `data/`  
✅ **Your Training Setup**: Follows same patterns as train_graph.py (splits, replicates, config)  
✅ **Your Config System**: Uses `scripts/python/train_config.yaml`  
✅ **Learning Curves**: Supports `--train_size` for learning curve experiments  
✅ **Results Format**: Saves to `results/Graphormer/` in same format as other models

## Usage

### Basic Training

```bash
# Train on insulator dataset
python train_graphormer.py --dataset_name insulator --model_name Graphormer

# Train on OPV dataset
python train_graphormer.py --dataset_name opv_camb3lyp --model_name Graphormer
```

### Learning Curves

```bash
# Train with 500 samples
python train_graphormer.py --dataset_name insulator --train_size 500

# Train with different sizes
for size in 128 256 512 1024 2048; do
    python train_graphormer.py --dataset_name insulator --train_size $size
done
```

### Model Configuration

```bash
# Small model (faster, fewer parameters)
python train_graphormer.py \
    --dataset_name insulator \
    --num_layers 6 \
    --hidden_dim 512 \
    --num_heads 16 \
    --batch_size 64

# Large model (better performance)
python train_graphormer.py \
    --dataset_name insulator \
    --num_layers 12 \
    --hidden_dim 768 \
    --num_heads 32 \
    --batch_size 32
```

## Command-Line Arguments

### Required
- `--dataset_name`: Dataset name (must match CSV file in `data/`)
- `--model_name`: Model name for config lookup (default: "Graphormer")

### Model Architecture
- `--num_layers`: Number of Graphormer layers (default: 12)
- `--hidden_dim`: Hidden dimension (default: 768)
- `--num_heads`: Number of attention heads (default: 32)
- `--dropout`: Dropout rate (default: 0.1)

### Training
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 2e-4)
- `--weight_decay`: Weight decay (default: 0.0)
- `--train_size`: Training size, e.g., "500" or "full" (default: "full")
- `--task_type`: "regression" or "classification" (default: "regression")

## How It Works

### 1. Dataset Loading
- Reads CSV from `data/{dataset_name}.csv`
- Uses `smiles_column` from `train_config.yaml`
- Automatically detects target columns
- Applies same ignore rules as other models

### 2. Graph Preprocessing
- Uses `SimpleMoleculeMolGraphFeaturizer` (same as train_graph.py)
- Precomputes shortest path distances (SPD) and paths
- Stores in DGL graph format compatible with Graphormer

### 3. Training
- 5 replicates with random splits (80/10/10)
- Early stopping with patience=30
- Gradient clipping (norm=5.0)
- Polynomial decay learning rate schedule
- Uses Accelerate for distributed training

### 4. Results
- Saves to `results/Graphormer/{dataset}__size{N}_results.csv`
- Format matches other models (for plot_multi_model_learning_curves.py)
- Includes test/mae, test/rmse, test/r2 for regression

## Configuration

The script uses your existing `scripts/python/train_config.yaml`:

```yaml
models:
  Graphormer:
    smiles_column: smiles
    ignore_columns: [WDMPNN_Input]
```

## Comparison with Other Models

### vs train_graph.py (DMPNN, AttentiveFP)
- ✅ Same dataset loading
- ✅ Same split strategy
- ✅ Same results format
- ✅ Same train_size support
- ✅ Different model architecture (Transformer vs GNN)

### vs Graphormer/train.py (standalone)
- ✅ Uses official DGL Graphormer model
- ✅ Integrated with your infrastructure
- ✅ No need for separate config files
- ✅ Works with your existing datasets

## Adding to Learning Curve Plots

The results are automatically compatible with `plot_multi_model_learning_curves.py`:

```python
# In plot_multi_model_learning_curves.py, add:
MODEL_CONFIGS = {
    # ... existing models ...
    'Graphormer': {
        'dir': 'results/Graphormer',
        'pattern': '{dataset}*_results.csv',
        'color': '#e377c2',
        'marker': 'h',
        'linestyle': '-',
        'is_tabular': False
    },
}
```

Then run:
```bash
python plot_multi_model_learning_curves.py
```

## Model Variants

### Graphormer-Small (faster training)
```bash
python train_graphormer.py \
    --dataset_name insulator \
    --num_layers 6 \
    --hidden_dim 512 \
    --num_heads 16 \
    --batch_size 64 \
    --epochs 100
```

### Graphormer-Base (paper default)
```bash
python train_graphormer.py \
    --dataset_name insulator \
    --num_layers 12 \
    --hidden_dim 768 \
    --num_heads 32 \
    --batch_size 32 \
    --epochs 100
```

### Graphormer-Slim (< 500K parameters)
```bash
python train_graphormer.py \
    --dataset_name insulator \
    --num_layers 12 \
    --hidden_dim 80 \
    --num_heads 8 \
    --batch_size 128 \
    --epochs 100
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--hidden_dim` (try 512 or 256)
- Reduce `--num_layers` (try 6)

### Slow Training
- Increase `--batch_size` if memory allows
- Reduce `--num_layers`
- Use smaller model variant

### Poor Performance
- Increase model size (more layers/hidden_dim)
- Increase `--epochs`
- Try different `--lr` (1e-4 to 5e-4)
- Check if task_type is correct (regression vs classification)

## File Structure

```
dmpnn/
├── train_graphormer.py          # ← New integrated training script
├── Graphormer/
│   └── dgl/                     # ← Official DGL example (unchanged)
│       ├── model.py             # Graphormer model
│       ├── dataset.py           # MolHIV dataset (reference)
│       └── main.py              # Original example (reference)
├── data/
│   ├── insulator.csv            # Your datasets
│   └── opv_camb3lyp.csv
├── results/
│   └── Graphormer/              # Results saved here
└── scripts/python/
    └── train_config.yaml        # Config (Graphormer added)
```

## Next Steps

1. **Test on one dataset**:
   ```bash
   python train_graphormer.py --dataset_name insulator --epochs 10
   ```

2. **Run learning curves**:
   ```bash
   for size in 128 256 512 1024 2048 full; do
       python train_graphormer.py --dataset_name insulator --train_size $size
   done
   ```

3. **Compare with other models**:
   ```bash
   python plot_multi_model_learning_curves.py
   ```

4. **Tune hyperparameters** based on results

## Performance Tips

- **For small datasets (< 1K)**: Use Graphormer-Slim, higher dropout (0.3)
- **For medium datasets (1K-10K)**: Use Graphormer-Small
- **For large datasets (> 10K)**: Use Graphormer-Base

## Citation

If you use this implementation, cite both the Graphormer paper and DGL:

```bibtex
@inproceedings{ying2021transformers,
  title={Do transformers really perform bad for graph representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  booktitle={NeurIPS},
  year={2021}
}

@article{wang2019dgl,
  title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks},
  author={Wang, Minjie and Zheng, Da and Ye, Zihao and Gan, Quan and Li, Mufei and Song, Xiang and Zhou, Jinjing and Ma, Chao and Yu, Lingfan and Gai, Yu and others},
  journal={arXiv preprint arXiv:1909.01315},
  year={2019}
}
```
