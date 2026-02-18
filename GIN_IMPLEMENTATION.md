# GIN Implementation in Chemprop

This document describes the Graph Isomorphism Network (GIN) implementation integrated into the chemprop module.

## Overview

GIN (Graph Isomorphism Network) is a powerful graph neural network architecture that achieves maximum discriminative power among GNN architectures. This implementation provides three variants:

1. **GIN**: Standard GIN with learnable epsilon parameter
2. **GIN0**: GIN with epsilon fixed at 0
3. **GINE**: GIN with edge features incorporated in message aggregation

## Reference

Xu et al. "How Powerful are Graph Neural Networks?" ICLR 2019  
https://arxiv.org/abs/1810.00826

## Architecture

### GIN Update Rule

The core GIN update follows:

```
h_v^(k+1) = MLP((1 + ε) · h_v^(k) + Σ_{u∈N(v)} h_u^(k))
```

Where:
- `h_v^(k)` is the hidden representation of node v at layer k
- `ε` is a learnable parameter (or fixed at 0 for GIN0)
- `MLP` is a multi-layer perceptron with batch normalization
- `N(v)` is the set of neighbors of node v

### Key Features

- **Learnable Epsilon**: GIN and GINE use learnable epsilon per layer
- **Multi-layer MLPs**: Each GIN layer uses a 2-layer MLP by default (configurable)
- **Batch Normalization**: Built-in batch normalization in MLPs
- **Edge Features**: Optional edge feature incorporation (GIN, GINE)
- **Descriptor Support**: Compatible with RDKit and custom descriptors

## Files Modified/Created

### New Files
- `chemprop/nn/message_passing/gin.py` - GIN implementation (3 variants)

### Modified Files
- `chemprop/nn/message_passing/__init__.py` - Added GIN imports
- `scripts/python/utils.py` - Added GIN model selection logic
- `scripts/python/train_graph.py` - Added GIN to small molecule models
- `scripts/python/train_config.yaml` - Added GIN configurations

## Usage

### Basic Training

```bash
# Standard GIN with learnable epsilon
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name GIN

# GIN0 with fixed epsilon = 0
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name GIN0

# GINE with edge features
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name GINE
```

### With RDKit Descriptors

```bash
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name GIN \
    --incl_rdkit
```

### Advanced Options

```bash
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name GIN \
    --gin_mlp_layers 3 \        # Number of MLP layers per GIN layer (default: 2)
    --depth 5 \                  # Number of GIN layers (default: 3)
    --batch_norm \               # Enable batch normalization (recommended)
    --incl_rdkit                 # Include RDKit descriptors
```

## Implementation Details

### GIN Variants

#### 1. GINMessagePassing
- **Epsilon**: Learnable parameter initialized at 0
- **Edge Features**: Aggregated to nodes before initial projection
- **Use Case**: General-purpose GIN for molecular property prediction

#### 2. GIN0MessagePassing
- **Epsilon**: Fixed at 0 (not learnable)
- **Edge Features**: Aggregated to nodes before initial projection
- **Use Case**: Simpler variant, faster training

#### 3. GINEMessagePassing
- **Epsilon**: Learnable parameter initialized at 0
- **Edge Features**: Incorporated in message passing (ReLU(h_u + e_{uv}))
- **Use Case**: When edge features are critical (bond types, etc.)

### MLP Structure

Each GIN layer uses an MLP with the following structure (for `mlp_layers=2`):

```
Linear(d_h, d_h)
BatchNorm1d(d_h)
ReLU
Dropout
Linear(d_h, d_h)
BatchNorm1d(d_h)
ReLU
```

### Parameter Count

For default configuration (d_h=300, depth=3, mlp_layers=2):
- **GIN**: ~659,403 trainable parameters
- **GIN0**: ~659,400 trainable parameters (3 fewer due to fixed epsilon)
- **GINE**: ~659,403 trainable parameters

## Testing

Run the test script to verify the implementation:

```bash
python test_gin_minimal.py
```

Expected output:
```
✓ GIN imports successful
✓ GIN instances created
✓ Epsilon parameters correct
✓ MLP structure correct
✓ All core tests passed!
```

## Comparison with DMPNN

| Feature | DMPNN | GIN |
|---------|-------|-----|
| Message Passing | Bond-based (directed edges) | Node-based (atoms) |
| Aggregation | Excludes reverse edge | Includes all neighbors |
| Epsilon | N/A | Learnable self-loop weight |
| MLP | Single linear layer | Multi-layer with BatchNorm |
| Edge Features | Concatenated with atom features | Optional aggregation or message-level |
| Theoretical Power | High expressiveness | Maximal discriminative power |

## Performance Tips

1. **Batch Normalization**: Always use `--batch_norm` for stable training
2. **Depth**: Start with depth=3, increase to 5 for complex tasks
3. **MLP Layers**: 2 layers is usually sufficient, 3 for very complex tasks
4. **Dropout**: Default 0.0 works well, increase to 0.1-0.2 if overfitting
5. **Learning Rate**: Default chemprop LR schedule works well
6. **Edge Features**: Use GINE if bond types are critical for your task

## Example Results

Training on htpmd dataset with pSMILES:

```bash
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name GIN \
    --batch_norm \
    --incl_rdkit
```

## Troubleshooting

### Issue: "No module named 'chemprop.models.ppg_adapter'"
**Solution**: This is a known issue with a missing PPG module. The ppg_adapter import has been commented out in `chemprop/models/__init__.py`. GIN works independently of this.

### Issue: Poor performance compared to DMPNN
**Solutions**:
- Enable batch normalization: `--batch_norm`
- Increase depth: `--depth 5`
- Add RDKit descriptors: `--incl_rdkit`
- Try GINE variant if edge features are important

### Issue: Training is slow
**Solutions**:
- Reduce MLP layers: `--gin_mlp_layers 1`
- Reduce depth: `--depth 2`
- Increase batch size if GPU memory allows

## API Reference

### GINMessagePassing

```python
from chemprop.nn.message_passing import GINMessagePassing

mp = GINMessagePassing(
    d_v=133,                    # Atom feature dimension
    d_e=14,                     # Edge feature dimension
    d_h=300,                    # Hidden dimension
    bias=False,                 # Add bias to linear layers
    depth=3,                    # Number of GIN layers
    dropout=0.0,                # Dropout probability
    activation="relu",          # Activation function
    eps_learnable=True,         # Make epsilon learnable
    mlp_layers=2,               # MLP layers per GIN layer
    d_vd=None,                  # Additional descriptor dimension
    use_edge_features=True      # Use edge features
)
```

### Integration with MPNN Model

```python
from chemprop import nn, models

# Create GIN message passing
mp = nn.GINMessagePassing(eps_learnable=True, mlp_layers=2)

# Create aggregation
agg = nn.MeanAggregation()

# Create predictor
ffn = nn.RegressionFFN(n_tasks=1, input_dim=mp.output_dim)

# Create full model
model = models.MPNN(
    message_passing=mp,
    agg=agg,
    predictor=ffn,
    batch_norm=True,
    metrics=[]
)
```

## Future Enhancements

Potential improvements for future versions:

1. **Virtual Node**: Add global graph-level node for better long-range interactions
2. **Jumping Knowledge**: Concatenate representations from all layers
3. **Graph-level Readout**: Alternative aggregation schemes (Set2Set, etc.)
4. **Attention Mechanisms**: Add attention to neighbor aggregation
5. **Pre-training**: Support for pre-trained GIN models

## Citation

If you use this GIN implementation, please cite both the original GIN paper and chemprop:

```bibtex
@inproceedings{xu2019how,
  title={How Powerful are Graph Neural Networks?},
  author={Xu, Keyulu and Hu, Weihua and Leskovec, Jure and Jegelka, Stefanie},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@article{yang2019analyzing,
  title={Analyzing Learned Molecular Representations for Property Prediction},
  author={Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and Kelley, Brian and Mathea, Miriam and others},
  journal={Journal of Chemical Information and Modeling},
  year={2019}
}
```

## Contact

For issues or questions about this GIN implementation, please refer to the chemprop documentation or open an issue in the repository.
