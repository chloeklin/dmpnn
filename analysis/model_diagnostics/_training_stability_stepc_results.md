# HPG-hier stability fixes: Step C results

All artifacts were verified as CUDA V100 runs. Representative environment: `Tesla V100-SXM2-32GB`, driver `580.126.20`, torch `2.8.0+cu128`, torch CUDA `12.8`, deterministic cuDNN `True`, global deterministic algorithms `False`.

> **Pending:** arm_c — artifacts not yet collected on Gadi V100.

## Primary comparison

| config | fold | mae_mean | mae_sd | group_mean_r2_mean | group_mean_r2_sd | delta_r2_mean | delta_r2_sd | ordering_mean | ordering_sd | overall_r2_mean | overall_r2_sd | checkpoint_mae_gap_mean | checkpoint_mae_gap_sd | wall_time_mean_seconds | fold1_mae_ok | fold1_delta_r2_ok | fold1_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_checkpoint | 0 | 0.04413612 | 0.01248756 | 0.98909917 | 0.00634024 | 0.73942101 | 0.03433510 | 0.75165635 | 0.04909377 | 0.98638229 | 0.00642246 | 0.05925105 | 0.05863891 | 4280.83203392 | nan | nan | nan |
| best_checkpoint | 1 | 0.08226540 | 0.02368001 | 0.93100001 | 0.04346962 | 0.66741590 | 0.22100932 | 0.82551320 | 0.01751442 | 0.92345244 | 0.03865242 | 0.03605258 | 0.00668441 | 4476.58443586 | 0.00000000 | 0.00000000 | 0.00000000 |
| current | 0 | 0.06350260 | 0.01763002 | 0.97664957 | 0.01276228 | 0.79243666 | 0.03377093 | 0.78472901 | 0.01685673 | 0.97379613 | 0.01377714 | nan | nan | 4045.34535615 | nan | nan | nan |
| current | 1 | 0.13885461 | 0.09106691 | 0.73921390 | 0.26783125 | 0.73796996 | 0.08276001 | 0.76952319 | 0.06310694 | 0.73389656 | 0.27051363 | nan | nan | 4146.30982182 | 0.00000000 | 1.00000000 | 0.00000000 |
| row_val_best | 0 | 0.07798517 | 0.03371061 | 0.95761156 | 0.04022371 | 0.75665513 | 0.00864694 | 0.72249375 | 0.04860693 | 0.95603362 | 0.03835381 | -0.00493615 | 0.02277028 | 7374.47035543 | nan | nan | nan |
| row_val_best | 1 | 0.04541099 | 0.01378909 | 0.97415960 | 0.01786806 | 0.84005195 | 0.01663573 | 0.83583143 | 0.01858449 | 0.97082783 | 0.01739567 | 0.00474622 | 0.01382059 | 9496.01539074 | 1.00000000 | 1.00000000 | 1.00000000 |

**Success criteria (co-equal):** fold-1 MAE SD < 0.02 eV AND fold-1 delta-R² SD < 0.083 (current baseline). Ordering SD is reported but has no fixed threshold. Successful fixes (both criteria met): row_val_best.

If `best_checkpoint` succeeds, retain the current single-monomer validation design. Use `row_val_best` only if the bug fix alone fails and row-level validation succeeds. `arm_c` (epoch floor 40, patience 30) targets runs that stopped at the first noisy minimum; it succeeds if both MAE SD and delta-R² SD fall below their respective thresholds. Fixed 60 epochs remains a fallback only if none of the above succeed.

## Validation loss versus test MAE

Pearson correlations use all three configurations (`current`, `best_checkpoint`, and `row_val_best`): nine runs per fold and 18 runs pooled.

| scope | runs | pearson_r | p_value |
| --- | --- | --- | --- |
| fold_0 | 9 | -0.52624867 | 0.14556058 |
| fold_1 | 9 | 0.60755588 | 0.08266800 |
| pooled | 18 | 0.43164892 | 0.07366727 |

**Validation-based model selection is uninformative under this design for fold_0:** the correlation is near zero or negative.
The pooled correlation is positive and not near zero, so the pooled 18-run diagnostic does not support the same conclusion.

## Three-repeat prediction averaging

For each configuration and fold, `y_pred` was averaged row-wise across the three repeats before recomputing every metric. `averaged_minus_individual_*` is the averaged-prediction metric minus the mean metric of the three individual runs; positive is better for R² and ordering, while negative is better for MAE.

| config | fold | repeats_averaged | averaged_mae | individual_mean_mae | averaged_minus_individual_mae | averaged_group_mean_r2 | individual_mean_group_mean_r2 | averaged_minus_individual_group_mean_r2 | averaged_delta_r2 | individual_mean_delta_r2 | averaged_minus_individual_delta_r2 | averaged_ordering | individual_mean_ordering | averaged_minus_individual_ordering | averaged_overall_r2 | individual_mean_overall_r2 | averaged_minus_individual_overall_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 0 | 3 | 0.05544518 | 0.06350260 | -0.00805742 | 0.98229569 | 0.97664957 | 0.00564611 | 0.84130561 | 0.79243666 | 0.04886895 | 0.80417074 | 0.78472901 | 0.01944173 | 0.98010874 | 0.97379613 | 0.00631260 |
| current | 1 | 3 | 0.13326957 | 0.13885461 | -0.00558504 | 0.82375086 | 0.73921390 | 0.08453696 | 0.86433637 | 0.73796996 | 0.12636641 | 0.84538938 | 0.76952319 | 0.07586619 | 0.82027471 | 0.73389656 | 0.08637814 |
| best_checkpoint | 0 | 3 | 0.03398528 | 0.04413612 | -0.01015084 | 0.99429850 | 0.98909917 | 0.00519933 | 0.80646726 | 0.73942101 | 0.06704625 | 0.77126100 | 0.75165635 | 0.01960465 | 0.99214531 | 0.98638229 | 0.00576302 |
| best_checkpoint | 1 | 3 | 0.06088814 | 0.08226540 | -0.02137726 | 0.96223731 | 0.93100001 | 0.03123729 | 0.80355237 | 0.66741590 | 0.13613647 | 0.85272076 | 0.82551320 | 0.02720756 | 0.95725117 | 0.92345244 | 0.03379873 |
| row_val_best | 0 | 3 | 0.07300328 | 0.07798517 | -0.00498189 | 0.96813810 | 0.95761156 | 0.01052654 | 0.81521970 | 0.75665513 | 0.05856458 | 0.72808732 | 0.72249375 | 0.00559357 | 0.96683083 | 0.95603362 | 0.01079721 |
| row_val_best | 1 | 3 | 0.03697465 | 0.04541099 | -0.00843634 | 0.98358186 | 0.97415960 | 0.00942226 | 0.88684987 | 0.84005195 | 0.04679792 | 0.86836103 | 0.83583143 | 0.03252960 | 0.98111775 | 0.97082783 | 0.01028992 |

Repeat averaging lowered MAE in 6/6 configuration-fold cells and improved the higher-is-better metrics in 24/24 comparisons. Averaging consistently recovers predictive performance that single-run validation-based selection does not reliably deliver.

## Small-sample caveat

Every SD in this report is estimated from n=3 repeats. The 95% confidence interval for the true population SD σ given an observed sample SD s is approximately [0.52 s, 6.28 s] (chi-squared distribution, 2 degrees of freedom). For example, an observed SD of 0.020 eV corresponds to a 95% CI of [0.010, 0.126] eV. **All threshold comparisons should be treated as provisional.**

## Best-checkpoint versus final-model MAE

Positive values mean the final patience-expired model was worse than its same-run best checkpoint.

| config | runs | final_minus_best_mae_mean | final_minus_best_mae_median | final_minus_best_mae_min | final_minus_best_mae_max |
| --- | --- | --- | --- | --- | --- |
| best_checkpoint | 6 | 0.04765182 | 0.03806371 | -0.00261461 | 0.11401666 |
| row_val_best | 6 | -0.00009497 | -0.00204371 | -0.01905861 | 0.02133193 |

## Nine-fold paired sign-test power

Exact two-sided sign-test power for nine folds, a common same-sign effect of 0.030 eV per fold, independent equal per-model run SD, and alpha 0.05 (rejection at at least 8 of 9 signs):

| per_run_sd_eV | runs_per_model_cell | exact_power | jobs_for_144_cells |
| --- | --- | --- | --- |
| 0.00500000 | 1 | 1.00000000 | 144 |
| 0.01000000 | 1 | 0.99044751 | 144 |
| 0.01500000 | 1 | 0.84600354 | 144 |
| 0.02000000 | 2 | 0.88252279 | 288 |
| 0.03000000 | 4 | 0.84600354 | 576 |

## Per-run diagnostics

| config | fold | repeat | mae | group_mean_r2 | delta_r2 | ordering | overall_r2 | epochs | best_epoch | best_val_loss | final_model_mae | final_minus_best_mae | wall_time_seconds | validation_curve_epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 0 | 1 | 0.08380947 | 0.96202091 | 0.76442090 | 0.77973281 | 0.95798297 | 38 | nan | 0.00681657 | nan | nan | 4283.84819483 | nan |
| current | 0 | 2 | 0.05459051 | 0.98242197 | 0.82993524 | 0.80351906 | 0.98019690 | 28 | nan | 0.00741245 | nan | nan | 3133.98861288 | nan |
| current | 0 | 3 | 0.05210783 | 0.98550585 | 0.78295383 | 0.77093516 | 0.98320852 | 43 | nan | 0.00513576 | nan | nan | 4718.19926075 | nan |
| current | 1 | 1 | 0.14560965 | 0.78978591 | 0.73827110 | 0.69876181 | 0.78749832 | 47 | nan | 0.00743890 | nan | nan | 5214.35993582 | nan |
| current | 1 | 2 | 0.22635591 | 0.44970180 | 0.65505979 | 0.78983382 | 0.44059472 | 27 | nan | 0.01058103 | nan | nan | 2866.05124666 | nan |
| current | 1 | 3 | 0.04459828 | 0.97815400 | 0.82057898 | 0.81997393 | 0.97359665 | 40 | nan | 0.00904610 | nan | nan | 4358.51828299 | nan |
| best_checkpoint | 0 | 1 | 0.05851474 | 0.98180314 | 0.70345480 | 0.69534050 | 0.97897638 | 55 | 40.00000000 | 0.00527564 | 0.12486584 | 0.06635110 | 6204.08952849 | 55.00000000 |
| best_checkpoint | 0 | 2 | 0.03600839 | 0.99327119 | 0.74295686 | 0.77419355 | 0.99042041 | 30 | 15.00000000 | 0.00682675 | 0.15002505 | 0.11401666 | 3292.02753946 | 30.00000000 |
| best_checkpoint | 0 | 3 | 0.03788523 | 0.99222317 | 0.77185136 | 0.78543500 | 0.98975008 | 30 | 15.00000000 | 0.00621717 | 0.03527062 | -0.00261461 | 3346.37903380 | 30.00000000 |
| best_checkpoint | 1 | 1 | 0.05862605 | 0.97240997 | 0.41278218 | 0.84555230 | 0.95785968 | 30 | 15.00000000 | 0.00882275 | 0.09065636 | 0.03203032 | 3365.04075969 | 30.00000000 |
| best_checkpoint | 1 | 2 | 0.08218428 | 0.93486166 | 0.78002306 | 0.81313131 | 0.93086824 | 61 | 46.00000000 | 0.00438682 | 0.11454295 | 0.03235867 | 6739.45921094 | 61.00000000 |
| best_checkpoint | 1 | 3 | 0.10598586 | 0.88572840 | 0.80944245 | 0.81785598 | 0.88162940 | 29 | 14.00000000 | 0.00908791 | 0.14975460 | 0.04376874 | 3325.25333694 | 29.00000000 |
| row_val_best | 0 | 1 | 0.05069123 | 0.98671498 | 0.76414014 | 0.77240143 | 0.98374552 | 49 | 34.00000000 | 0.00309973 | 0.03360947 | -0.01708177 | 5457.53596728 | 49.00000000 |
| row_val_best | 0 | 2 | 0.11566723 | 0.91171197 | 0.74718981 | 0.67530140 | 0.91226024 | 100 | 95.00000000 | 0.00210662 | 0.13699916 | 0.02133193 | 11162.72433091 | 100.00000000 |
| row_val_best | 0 | 3 | 0.06759704 | 0.97440775 | 0.75863543 | 0.71977843 | 0.97209510 | 49 | 34.00000000 | 0.00300172 | 0.04853844 | -0.01905861 | 5503.15076811 | 49.00000000 |
| row_val_best | 1 | 1 | 0.04558360 | 0.97783425 | 0.84726883 | 0.85402411 | 0.97434978 | 98 | 83.00000000 | 0.00238307 | 0.03628046 | -0.00930314 | 11033.37471599 | 98.00000000 |
| row_val_best | 1 | 2 | 0.05911296 | 0.95473990 | 0.82102647 | 0.81687846 | 0.95194067 | 77 | 62.00000000 | 0.00229707 | 0.06432869 | 0.00521573 | 8726.37848605 | 77.00000000 |
| row_val_best | 1 | 3 | 0.03153640 | 0.98990466 | 0.85186054 | 0.83659172 | 0.98619305 | 77 | 62.00000000 | 0.00228131 | 0.04986247 | 0.01832606 | 8728.29297017 | 77.00000000 |
