# Normalization Audit Report

## Target Statistics (Full Dataset)

### EA (EA vs SHE (eV))
- N = 42966
- mean = -2.541360
- std = 0.600106
- min = -4.193988
- max = 0.630153

### IP (IP vs SHE (eV))
- N = 42966
- mean = 1.453407
- std = 0.481776
- min = -0.061787
- max = 3.975924

## Per-Split Normalization Parameters

Computed from .npz y_true values (which match test set rows):

### EA
  Split 0: n_test=8596, estimated train_mean=-2.532698, estimated train_std=0.599137, linreg_r=0.988189
  Split 1: n_test=8596, estimated train_mean=-2.529174, estimated train_std=0.605765, linreg_r=0.982726
  Split 2: n_test=8624, estimated train_mean=-2.566840, estimated train_std=0.605839, linreg_r=0.988684
  Split 3: n_test=8575, estimated train_mean=-2.564960, estimated train_std=0.612526, linreg_r=0.988207
  Split 4: n_test=8575, estimated train_mean=-2.541191, estimated train_std=0.606733, linreg_r=0.986935

### IP
  Split 0: n_test=8596, estimated train_mean=1.456457, estimated train_std=0.493693, linreg_r=0.981357
  Split 1: n_test=8596, estimated train_mean=1.448053, estimated train_std=0.492871, linreg_r=0.973252
  Split 2: n_test=8624, estimated train_mean=1.436528, estimated train_std=0.488017, linreg_r=0.983735
  Split 3: n_test=8575, estimated train_mean=1.454697, estimated train_std=0.467326, linreg_r=0.983550
  Split 4: n_test=8575, estimated train_mean=1.465507, estimated train_std=0.486465, linreg_r=0.987749

## Summary: Estimated Normalization Parameters

### EA
- train_mean (avg across splits) = -2.546973 ± 0.015951
- train_std  (avg across splits) = 0.606000 ± 0.004250

### IP
- train_mean (avg across splits) = 1.452248 ± 0.009636
- train_std  (avg across splits) = 0.485675 ± 0.009580

## Normalization State at Each Pipeline Stage

| Stage | y_true | y_pred | Status |
|-------|--------|--------|--------|
| Training (train_ds) | NORMALIZED | NORMALIZED (model output) | CORRECT |
| Training (val_ds) | NORMALIZED | NORMALIZED (model output, UnscaleTransform inactive in training mode) | CORRECT |
| test_step (test_ds) | RAW | NORMALIZED (Stage2D bypasses UnscaleTransform) | **BUG** |
| predict_step (.npz) | RAW | NORMALIZED (same path as forward()) | **BUG** |
| CSV metrics (test/r2 etc.) | RAW | NORMALIZED | **BUG** |
| Val metrics (val/r2 etc.) | NORMALIZED | NORMALIZED | CORRECT |

## Root Cause

`CopolymerMPNN.forward()` for Stage2D models calls `forward_stage2d()` which
returns raw MLP outputs from `Stage2Aggregator`. These bypass the `RegressionFFN`
predictor which contains `UnscaleTransform`. During eval mode, the transform
should apply `y_pred * scale + mean` to get raw-space predictions, but it's
never called for Stage2D.

## Fix: Apply Inverse Transform Post-Hoc

Since `y_pred_normalized = (y_true - mean) / std`, we can recover:
```
y_pred_corrected = y_pred_normalized * train_std + train_mean
```