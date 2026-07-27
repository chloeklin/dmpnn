# HPG-hier Gadi V100 noise floor

## Runtime environment

All six runs used accelerator `cuda`, device `Tesla V100-SXM2-32GB`, driver `580.126.20`, torch `2.8.0+cu128`, torch CUDA `12.8`, cuDNN `91002`, deterministic kernels requested `True`, deterministic algorithms enabled `False`, cuDNN deterministic `True`, and cuDNN benchmark `False`.

Every run used current code, A-heldout EA, seed 42, and changed only the independent process/repeat label. SD is the sample SD across three repeats.

## Pre-registered variability tier

- Reproducible: maximum absolute prediction difference no greater than 1.0e-06 eV in both folds.
- Practically equivalent: not reproducible to 1.0e-06 eV, but each fold's three-repeat MAE SD is below 0.005 eV.
- Materially variable: either fold's three-repeat MAE SD is at least 0.005 eV.

**Observed classification: Materially variable: MAE SD is at least 0.005 eV in one or both folds.**

## Direct fold-0 comparison with the canonical July-20 artifact

Canonical reference: overall R² 0.92411349; MAE 0.10680739 eV.

| repeat | max_abs_difference_eV | rmse_difference_eV | overall_r2 | mae_eV | accelerator | device | driver | torch | torch_cuda | deterministic_requested | deterministic_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.25836968 | 0.06651673 | 0.95798297 | 0.08380947 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 2 | 0.29338169 | 0.10582172 | 0.98019690 | 0.05459051 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 3 | 0.26115346 | 0.09267536 | 0.98320852 | 0.05210783 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |

## Primary fold-level spread

MAE SD, overall-R² SD, and delta-R² SD are the primary outputs.

| fold | mae_sd | overall_r2_sd | delta_r2_sd | max_pairwise_prediction_difference_eV |
| --- | --- | --- | --- | --- |
| 0 | 0.01763002 | 0.01377714 | 0.03377093 | 0.41858077 |
| 1 | 0.09106691 | 0.27051363 | 0.08276001 | 0.60612130 |

## Per-run metrics and V100 wall time

| fold | repeat | group_mean_r2 | delta_r2 | ordering | overall_r2 | mae | wall_time_seconds | accelerator | device | driver | torch | torch_cuda | deterministic_requested | deterministic_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.96202091 | 0.76442090 | 0.77973281 | 0.95798297 | 0.08380947 | 4283.84819483 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 0 | 2 | 0.98242197 | 0.82993524 | 0.80351906 | 0.98019690 | 0.05459051 | 3133.98861288 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 0 | 3 | 0.98550585 | 0.78295383 | 0.77093516 | 0.98320852 | 0.05210783 | 4718.19926075 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 1 | 1 | 0.78978591 | 0.73827110 | 0.69876181 | 0.78749832 | 0.14560965 | 5214.35993582 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 1 | 2 | 0.44970180 | 0.65505979 | 0.78983382 | 0.44059472 | 0.22635591 | 2866.05124666 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |
| 1 | 3 | 0.97815400 | 0.82057898 | 0.81997393 | 0.97359665 | 0.04459828 | 4358.51828299 | cuda | Tesla V100-SXM2-32GB | 580.126.20 | 2.8.0+cu128 | 12.8 | True | False |

## Full per-fold noise floor

| fold | group_mean_r2_mean | group_mean_r2_sd | delta_r2_mean | delta_r2_sd | ordering_mean | ordering_sd | overall_r2_mean | overall_r2_sd | mae_mean | mae_sd | max_pairwise_prediction_difference_eV | wall_time_mean_seconds | wall_time_sd_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.97664957 | 0.01276228 | 0.79243666 | 0.03377093 | 0.78472901 | 0.01685673 | 0.97379613 | 0.01377714 | 0.06350260 | 0.01763002 | 0.41858077 | 4045.34535615 | 818.59241827 |
| 1 | 0.73921390 | 0.26783125 | 0.73796996 | 0.08276001 | 0.76952319 | 0.06310694 | 0.73389656 | 0.27051363 | 0.13885461 | 0.09106691 | 0.60612130 | 4146.30982182 | 1188.44972470 |

## Octamer-versus-baseline EA MAE differences

**Finding: 8 of 9 recorded octamer-versus-baseline EA MAE differences fall inside ±2 SD.** The band is ±0.18213382 eV, using twice the larger of the measured fold-0 and fold-1 MAE SDs (0.09106691 eV).

This common conservative band provides the requested all-nine count; folds 2-8 do not have fold-matched repeat SDs, so the count extrapolates the larger measured SD rather than claiming nine fold-specific noise estimates.

Recorded signed differences: `+0.071, +0.183, +0.083, +0.010, -0.035, +0.025, +0.002, +0.100, +0.051` eV.

The measured SD columns are required context for future model-comparison tables.
