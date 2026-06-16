# Rerun Verification Report

## Alpha Parameter Extraction

### 2D1-fixed

- EA rep0: alpha = 0.025626
- EA rep1: alpha = 0.000000
- EA rep2: alpha = 0.025861
- EA rep3: alpha = 0.000000
- EA rep4: alpha = 0.000000
- IP rep0: alpha = 0.030343
- IP rep1: alpha = 0.031613
- IP rep2: alpha = 0.028433
- IP rep3: alpha = 0.029345
- IP rep4: alpha = 0.028989

**Summary**: mean=0.020021, std=0.013218, min=0.000000, max=0.031613
**alpha ≠ 0**: ✅ PASS

### 2D1-arch

- EA rep0: alpha_alt=0.000000, alpha_rand=0.000000, alpha_block=0.000000
- EA rep1: alpha_alt=0.041539, alpha_rand=0.044786, alpha_block=0.049981
- EA rep2: alpha_alt=0.033610, alpha_rand=0.032079, alpha_block=0.035637
- EA rep3: alpha_alt=0.037302, alpha_rand=0.036477, alpha_block=0.039310
- EA rep4: alpha_alt=0.000000, alpha_rand=0.000000, alpha_block=0.000000
- IP rep0: alpha_alt=0.000000, alpha_rand=0.000000, alpha_block=0.000000
- IP rep1: alpha_alt=0.041642, alpha_rand=0.042536, alpha_block=0.047778
- IP rep2: alpha_alt=0.036817, alpha_rand=0.036836, alpha_block=0.040701
- IP rep3: alpha_alt=0.042793, alpha_rand=0.043660, alpha_block=0.051349
- IP rep4: alpha_alt=0.036281, alpha_rand=0.036951, alpha_block=0.045404

**Summary**: mean=0.028449, std=0.019111, min=0.000000, max=0.051349
**alpha ≠ 0**: ✅ PASS
**Per-arch alphas differ**: ✅ PASS
  alt mean=0.026998, rand mean=0.027333, block mean=0.031016

## Prediction Difference vs Frac

### 2D1-fixed

- EA: Mean Abs Diff = 0.053862, Max Abs Diff = 0.903915
  ✅ Predictions differ meaningfully from Frac
- IP: Mean Abs Diff = 0.046728, Max Abs Diff = 0.507311
  ✅ Predictions differ meaningfully from Frac

### 2D1-arch

- EA: Mean Abs Diff = 0.054280, Max Abs Diff = 0.679120
  ✅ Predictions differ meaningfully from Frac
- IP: Mean Abs Diff = 0.047776, Max Abs Diff = 0.506127
  ✅ Predictions differ meaningfully from Frac
