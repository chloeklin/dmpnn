# PPG Integration Guide

This document explains how PPG (Polymer Property Graph) has been integrated into your DMPNN workflow.

## Overview

PPG uses **chemprop v1.4.0** while your DMPNN/wDMPNN models use **chemprop v2.2.0**. To enable PPG in your workflow without conflicts, we've created an **adapter layer** that wraps PPG's model to work with your Lightning-based training pipeline.

## Architecture

```
Your Workflow (v2.2.0)
    ↓
PPGAdapter (Lightning wrapper)
    ↓
PPG's MoleculeModel (v1.4.0)
    ↓
PPG's MPN + FFN
```

## Files Added/Modified

### New Files
1. **`chemprop/models/ppg_adapter.py`**
   - `PPGAdapter`: Lightning wrapper for PPG model
   - `create_ppg_args()`: Converts your args to PPG's format

2. **`test_ppg_integration.py`**
   - Test suite to verify integration

### Modified Files
1. **`chemprop/models/__init__.py`**
   - Added PPGAdapter and create_ppg_args exports

2. **`scripts/python/utils.py`**
   - Added PPG case in `build_model_and_trainer()`
   - Creates PPGAdapter when `args.model_name == "PPG"`

3. **`scripts/python/train_graph.py`**
   - PPG already included in `small_molecule_models` list
   - Uses `SimpleMoleculeMolGraphFeaturizer` for PPG

## Usage

### Training with PPG

```bash
python scripts/python/train_graph.py \
    --dataset_name your_dataset \
    --model_name PPG \
    --task_type reg \
    --incl_rdkit \
    --save_checkpoint
```

### Available Options

All your standard options work with PPG:
- `--incl_desc`: Include dataset-specific descriptors
- `--incl_rdkit`: Include RDKit descriptors
- `--train_size 500`: Subsample training data
- `--export_embeddings`: Export GNN embeddings
- `--save_predictions`: Save predictions for analysis
- `--target Tg`: Train on specific target

### Example Commands

**Basic training:**
```bash
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name PPG \
    --task_type reg
```

**With RDKit descriptors:**
```bash
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name PPG \
    --task_type reg \
    --incl_rdkit
```

**Learning curve experiment:**
```bash
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name PPG \
    --task_type reg \
    --train_size 500 \
    --save_predictions
```

## How It Works

### 1. Data Flow

```python
# Your v2.2.0 data format
MoleculeDatapoint → SimpleMoleculeMolGraphFeaturizer → BatchMolGraph

# PPG expects the same BatchMolGraph format
BatchMolGraph → PPG's MPN → PPG's FFN → Predictions
```

### 2. Model Creation

When you specify `--model_name PPG`:

```python
# In utils.py build_model_and_trainer()
if args.model_name == "PPG":
    # Convert your args to PPG format
    ppg_args = create_ppg_args(args, combined_descriptor_data, n_classes)
    
    # Create PPG adapter
    model = PPGAdapter(
        ppg_args=ppg_args,
        output_transform=output_transform,
        loss_function=loss_fn,
        metric_list=metric_list
    )
    
    # Return PPG model with Lightning trainer
    return model, trainer
```

### 3. Training Loop

PPGAdapter implements Lightning's training interface:
- `training_step()`: Forward pass + loss computation
- `validation_step()`: Validation metrics
- `test_step()`: Test predictions
- `predict_step()`: Inference

The Lightning Trainer handles:
- Optimization
- Early stopping
- Checkpointing
- Logging

## PPG Model Configuration

Default PPG settings (in `create_ppg_args()`):

```python
hidden_size = 300          # Message passing hidden dimension
depth = 3                  # Number of message passing layers
dropout = 0.0              # Dropout rate
activation = 'ReLU'        # Activation function
aggregation = 'mean'       # Atom aggregation method
ffn_num_layers = 2         # Feed-forward network layers
ffn_hidden_size = 300      # FFN hidden dimension
```

To customize, modify `create_ppg_args()` in `chemprop/models/ppg_adapter.py`.

## Comparison with DMPNN/wDMPNN

| Feature | DMPNN/wDMPNN | PPG |
|---------|--------------|-----|
| **Chemprop version** | v2.2.0 | v1.4.0 (wrapped) |
| **Architecture** | Modular (MP + Agg + FFN) | Integrated (MPN + FFN) |
| **Message passing** | Bond-level | Bond-level |
| **Aggregation** | Separate layer | Built into MPN |
| **Training** | Lightning native | Lightning wrapped |
| **Descriptors** | ✅ Supported | ✅ Supported |
| **Polymers** | ✅ wDMPNN | ✅ Designed for polymers |

## Testing

Run the test suite to verify integration:

```bash
python test_ppg_integration.py
```

Expected output:
```
============================================================
PPG Integration Test Suite
============================================================
Testing PPG adapter import...
✅ PPG adapter imported successfully

Testing PPG args creation...
✅ PPG args created successfully
   - Dataset type: regression
   - Hidden size: 300
   - Depth: 3
   - Features size: 5

Testing PPG model creation...
✅ PPG model created successfully
   - Model type: PPGAdapter
   - Task type: regression

Testing PPG in utils.py...
✅ PPG model built through utils successfully
   - Model type: PPGAdapter
   - Trainer type: Trainer

============================================================
Test Summary
============================================================
✅ PASS: Import PPG adapter
✅ PASS: Create PPG args
✅ PASS: Create PPG model
✅ PASS: Build PPG via utils

Total: 4/4 tests passed

🎉 All tests passed! PPG integration is working.
```

## Troubleshooting

### Import Error: "Could not import PPG's chemprop"

**Problem**: PPG directory not found or chemprop v1.4.0 not installed.

**Solution**:
```bash
# Check PPG directory exists
ls -la PPG/

# Install PPG's dependencies
cd PPG/
pip install -e .
cd ..
```

### Model Not Training

**Problem**: PPG model created but training fails.

**Solution**: Check that your data format is compatible:
- PPG expects SMILES strings (works with MoleculeDatapoint)
- Use `--model_name PPG` (not "ppg" - case sensitive)
- Ensure dataset has valid SMILES column

### Different Results from Original PPG

**Problem**: Predictions differ from PPG's original implementation.

**Possible causes**:
1. Different random seeds
2. Different preprocessing
3. Different hyperparameters

**Solution**: Match hyperparameters in `create_ppg_args()` to original PPG settings.

## Limitations

1. **No pretrained model loading**: Currently doesn't support loading PPG's pretrained `.pt` checkpoints (can be added if needed)

2. **Single task only**: Adapter configured for single-task learning (your workflow)

3. **No hyperparameter optimization**: Uses fixed PPG defaults (can be made configurable)

## Future Enhancements

Potential improvements:

1. **Pretrained model support**: Load PPG's pretrained polymer models
2. **Configurable hyperparameters**: Add PPG-specific args to command line
3. **Multi-task support**: Enable multi-task learning if needed
4. **Checkpoint conversion**: Convert between PPG and Lightning checkpoint formats

## Questions?

The integration is designed to be transparent - PPG works just like DMPNN/wDMPNN in your workflow. If you encounter issues, check:

1. PPG directory structure is intact
2. Test suite passes
3. Model name is exactly "PPG" (case-sensitive)
4. Data format is compatible (MoleculeDatapoint with SMILES)

## Summary

✅ **PPG is now integrated** into your workflow  
✅ **Use `--model_name PPG`** to train with PPG  
✅ **All standard options work** (descriptors, train_size, etc.)  
✅ **Same interface** as DMPNN/wDMPNN  
✅ **No conflicts** between chemprop versions  
