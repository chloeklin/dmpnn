# PPG Quick Start

## What Was Done

PPG (chemprop v1.4.0) has been integrated into your workflow (chemprop v2.2.0) via an adapter layer.

## Files Created

1. `chemprop/models/ppg_adapter.py` - Lightning wrapper for PPG
2. `test_ppg_integration.py` - Test suite
3. `PPG_INTEGRATION.md` - Full documentation

## Files Modified

1. `chemprop/models/__init__.py` - Added PPGAdapter export
2. `scripts/python/utils.py` - Added PPG model creation
3. `scripts/python/train_graph.py` - Already had PPG in small_molecule_models

## Quick Test

```bash
# Test the integration
python test_ppg_integration.py

# Should see: "🎉 All tests passed! PPG integration is working."
```

## Usage

```bash
# Train with PPG (same as DMPNN/wDMPNN)
python scripts/python/train_graph.py \
    --dataset_name htpmd \
    --model_name PPG \
    --task_type reg \
    --incl_rdkit
```

## How It Works

```
Your Data → MoleculeDatapoint → BatchMolGraph
                                      ↓
                                 PPGAdapter (Lightning)
                                      ↓
                              PPG Model (v1.4.0)
                                      ↓
                                 Predictions
```

## Key Points

✅ PPG works exactly like DMPNN/wDMPNN in your workflow  
✅ All your options work: --incl_rdkit, --train_size, --save_predictions, etc.  
✅ No conflicts between chemprop versions (isolated imports)  
✅ Uses Lightning training (same as your other models)  

## Next Steps

1. Run test suite: `python test_ppg_integration.py`
2. Try training on a small dataset
3. Compare results with DMPNN/wDMPNN
4. See `PPG_INTEGRATION.md` for full details
