# Group-Mean Metric Floor Audit

Scope: read-only calculation from `data/ea_ip.csv` and existing seed-42 LOMO NPZs. No models were trained, jobs submitted, or report files changed.

Grouping matches `_phase1_metrics_scratch.md`: group key is `(smiles_A, smiles_B, fracA)`, with groups restricted to at least two `poly_type` values before group means and group-mean R² are calculated. The A-blind null uses actual reconstructed LOMO **training** indices only; validation rows are excluded.

## Analysis A — A-blind null floor

For each test row the null lookup uses train-only `(smiles_B, fracA, poly_type)` mean, then `(smiles_B, poly_type)`, then global train mean. `primary`, `secondary`, and `global` count the rows served by each level.

| target | fold | null_group_mean_r2 | null_overall_r2 | null_mae | null_bias | primary | secondary | global | hpg_hier_group_mean_r2 | hpg_hier_overall_r2 | hpg_hier_mae | hpg_hier_bias | hpg_hier_octamer_group_mean_r2 | hpg_hier_octamer_overall_r2 | hpg_hier_octamer_mae | hpg_hier_octamer_bias | wdmpnn_group_mean_r2 | wdmpnn_overall_r2 | wdmpnn_mae | wdmpnn_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA | 0 | 0.69380 | 0.69158 | 0.26002 | 0.25962 | 4774 | 0 | 0 | 0.92534 | 0.92411 | 0.10681 | -0.09844 | 0.99476 | 0.99138 | 0.03545 | -0.00925 | 0.76013 | 0.75982 | 0.22686 | -0.22674 |
| EA | 1 | 0.48731 | 0.48470 | 0.21665 | -0.21208 | 4774 | 0 | 0 | 0.57514 | 0.56850 | 0.21397 | -0.21316 | 0.98890 | 0.98434 | 0.03141 | 0.00228 | 0.94464 | 0.93036 | 0.08001 | 0.06981 |
| EA | 2 | 0.96114 | 0.95597 | 0.08439 | 0.01491 | 4774 | 0 | 0 | 0.92157 | 0.92365 | 0.11819 | -0.11221 | 0.99464 | 0.99236 | 0.03478 | 0.00108 | 0.98035 | 0.97263 | 0.06299 | 0.03337 |
| EA | 3 | 0.95276 | 0.94747 | 0.07258 | -0.04957 | 4774 | 0 | 0 | 0.97464 | 0.96855 | 0.05469 | 0.04164 | 0.98362 | 0.98052 | 0.04501 | 0.03417 | 0.99032 | 0.98149 | 0.04095 | 0.01418 |
| EA | 4 | 0.88373 | 0.86128 | 0.12697 | 0.08668 | 4774 | 0 | 0 | 0.95060 | 0.93451 | 0.07334 | -0.06048 | 0.89700 | 0.87842 | 0.10833 | -0.10505 | 0.94556 | 0.93114 | 0.07598 | -0.05240 |
| EA | 5 | 0.67571 | 0.67461 | 0.17628 | -0.16868 | 4774 | 0 | 0 | 0.96903 | 0.96482 | 0.05831 | -0.04163 | 0.99138 | 0.98633 | 0.03328 | -0.01636 | 0.96868 | 0.95590 | 0.06451 | -0.04735 |
| EA | 6 | -19.06946 | -19.97913 | 1.05641 | -1.05632 | 4774 | 0 | 0 | 0.91655 | 0.88211 | 0.05805 | -0.03031 | 0.92480 | 0.89776 | 0.05652 | 0.00474 | 0.89363 | 0.86504 | 0.06855 | -0.04368 |
| EA | 7 | 0.09751 | 0.10387 | 0.51370 | 0.51330 | 4774 | 0 | 0 | 0.90187 | 0.90497 | 0.15554 | 0.15218 | 0.98949 | 0.98826 | 0.05566 | -0.03825 | 0.96479 | 0.96299 | 0.10081 | 0.09439 |
| EA | 8 | 0.42772 | 0.42902 | 0.39270 | 0.39212 | 4774 | 0 | 0 | 0.96315 | 0.96150 | 0.09393 | 0.07321 | 0.99396 | 0.99142 | 0.04256 | -0.01746 | 0.99408 | 0.98944 | 0.04414 | -0.02137 |
| IP | 0 | 0.96902 | 0.96650 | 0.04433 | -0.02312 | 4774 | 0 | 0 | 0.92787 | 0.92315 | 0.06460 | 0.06081 | 0.92650 | 0.92311 | 0.06590 | 0.06305 | 0.26951 | 0.26476 | 0.22411 | 0.22406 |
| IP | 1 | 0.50927 | 0.50602 | 0.24599 | -0.24587 | 4774 | 0 | 0 | 0.98116 | 0.97898 | 0.04499 | -0.03605 | 0.96971 | 0.96739 | 0.05792 | 0.05294 | 0.96354 | 0.95362 | 0.06211 | 0.05785 |
| IP | 2 | -1.01943 | -1.00861 | 0.66143 | -0.66143 | 4774 | 0 | 0 | 0.76920 | 0.77432 | 0.19782 | -0.19568 | 0.99387 | 0.99095 | 0.03638 | 0.00368 | 0.97667 | 0.97301 | 0.06592 | 0.03715 |
| IP | 3 | -3.20636 | -3.20740 | 0.42575 | 0.42571 | 4774 | 0 | 0 | 0.97463 | 0.96311 | 0.03347 | -0.00767 | 0.93263 | 0.92418 | 0.05369 | 0.05004 | 0.96794 | 0.95694 | 0.03798 | -0.01517 |
| IP | 4 | -0.25097 | -0.34241 | 0.19383 | 0.18526 | 4774 | 0 | 0 | 0.94379 | 0.93211 | 0.04514 | -0.03523 | 0.91007 | 0.89351 | 0.05746 | 0.05397 | 0.71193 | 0.68960 | 0.09734 | 0.09466 |
| IP | 5 | -7.52773 | -6.93823 | 0.70901 | 0.70901 | 4774 | 0 | 0 | 0.76969 | 0.76798 | 0.11849 | 0.11732 | 0.87641 | 0.87469 | 0.08144 | -0.08044 | 0.94604 | 0.86816 | 0.07590 | 0.03149 |
| IP | 6 | 0.56868 | 0.56208 | 0.23231 | -0.23220 | 4774 | 0 | 0 | 0.97462 | 0.97164 | 0.05101 | 0.03053 | 0.98614 | 0.98415 | 0.03869 | 0.02924 | 0.98380 | 0.96847 | 0.05419 | 0.01927 |
| IP | 7 | 0.40980 | 0.40641 | 0.29355 | -0.29327 | 4774 | 0 | 0 | 0.96862 | 0.96628 | 0.06580 | -0.06212 | 0.98848 | 0.98626 | 0.03965 | 0.03012 | 0.98326 | 0.97779 | 0.04733 | -0.03084 |
| IP | 8 | -0.03401 | -0.06530 | 0.21632 | 0.21607 | 4774 | 0 | 0 | 0.98063 | 0.97622 | 0.03047 | 0.01978 | 0.98371 | 0.97751 | 0.02564 | 0.01796 | 0.97688 | 0.96751 | 0.03397 | 0.02050 |

### Null summary

| target | median_null_group_mean_r2 | mean_null_group_mean_r2 | median_null_overall_r2 | median_null_mae | total_primary | total_secondary | total_global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EA | 0.67571 | -1.54331 | 0.67461 | 0.21665 | 42966 | 0 | 0 |
| IP | -0.03401 | -1.06464 | -0.06530 | 0.24599 | 42966 | 0 | 0 |

Interpretation: EA's A-blind null reaches median group-mean R² `0.67571` using no information about the held-out A. This supports the structural-weakness conclusion: **within-fold EA group-mean R² is not a measure of unseen-monomer chemistry and cannot support a chemistry-extrapolation claim by itself.** IP's A-blind null median is `-0.03401`, so this null does not demonstrate a high IP floor; its table is still reported separately rather than extrapolated from EA.

## Analysis B — Pooled across-fold placement

Pooled group-mean R² concatenates fold-specific group means (fold is retained with each group). Fold-placement uses the raw test-row mean for each equal-sized fold. `placement_r2` is parity R² (`r2_score(true_fold_mean, pred_fold_mean)`), not correlation squared. Bias standard deviation uses population (`ddof=0`) spread across the nine folds.

| model | target | pooled_group_mean_r2 | placement_r2 | placement_slope | placement_intercept | fold_biases_0_to_8 | fold_bias_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | EA | 0.94887 | 0.92306 | 0.91292 | -0.25344 | -0.09844, -0.21316, -0.11221, +0.04164, -0.06048, -0.04163, -0.03031, +0.15218, +0.07321 | 0.10273 |
| hpg_hier | IP | 0.95187 | 0.93921 | 0.80603 | 0.26989 | +0.06081, -0.03605, -0.19568, -0.00767, -0.03523, +0.11732, +0.03053, -0.06212, +0.01978 | 0.08316 |
| hpg_hier_octamer | EA | 0.98907 | 0.98941 | 1.03069 | 0.06198 | -0.00925, +0.00228, +0.00108, +0.03417, -0.10505, -0.01636, +0.00474, -0.03825, -0.01746 | 0.03658 |
| hpg_hier_octamer | IP | 0.98444 | 0.98013 | 1.05093 | -0.04952 | +0.06305, +0.05294, +0.00368, +0.05004, +0.05397, -0.08044, +0.02924, +0.03012, +0.01796 | 0.04131 |
| hpg_hier_junction | EA | 0.97586 | 0.97105 | 0.89190 | -0.27952 | -0.08344, -0.08175, +0.05425, +0.01080, -0.00962, -0.03852, -0.07197, +0.09613, +0.08085 | 0.06585 |
| hpg_hier_junction | IP | 0.95637 | 0.93658 | 0.81603 | 0.24821 | +0.01511, -0.05256, -0.15801, -0.07592, -0.02840, +0.17451, -0.00746, -0.03499, -0.00474 | 0.08365 |
| hpg_hier_junction1 | EA | 0.96543 | 0.94273 | 0.82342 | -0.48399 | +0.03658, -0.04548, -0.13053, -0.00123, -0.06643, -0.01521, -0.21147, +0.03590, +0.08072 | 0.08592 |
| hpg_hier_junction1 | IP | 0.95224 | 0.93139 | 0.77086 | 0.34441 | +0.08084, -0.03298, -0.11786, +0.04601, -0.01976, +0.21143, -0.02142, -0.04996, +0.00615 | 0.08854 |
| wdmpnn | EA | 0.96653 | 0.94550 | 0.99427 | -0.03455 | -0.22674, +0.06981, +0.03337, +0.01418, -0.05240, -0.04735, -0.04368, +0.09439, -0.02137 | 0.08836 |
| wdmpnn | IP | 0.95406 | 0.93604 | 1.00256 | 0.04506 | +0.22406, +0.05785, +0.03715, -0.01517, +0.09466, +0.03149, +0.01927, -0.03084, +0.02050 | 0.07106 |

### Within-fold group-mean compression ratios

| model | target | fold | predicted_group_mean_spread_div_true |
| --- | --- | --- | --- |
| hpg_hier | EA | 0 | 1.11962 |
| hpg_hier_octamer | EA | 0 | 1.01007 |
| hpg_hier_junction | EA | 0 | 1.16839 |
| hpg_hier_junction1 | EA | 0 | 0.97963 |
| wdmpnn | EA | 0 | 1.08080 |
| hpg_hier | EA | 1 | 1.16327 |
| hpg_hier_octamer | EA | 1 | 1.05634 |
| hpg_hier_junction | EA | 1 | 1.07688 |
| hpg_hier_junction1 | EA | 1 | 1.07754 |
| wdmpnn | EA | 1 | 1.04238 |
| hpg_hier | EA | 2 | 1.04806 |
| hpg_hier_octamer | EA | 2 | 1.01971 |
| hpg_hier_junction | EA | 2 | 0.95620 |
| hpg_hier_junction1 | EA | 2 | 1.07119 |
| wdmpnn | EA | 2 | 0.98382 |
| hpg_hier | EA | 3 | 0.91688 |
| hpg_hier_octamer | EA | 3 | 0.93189 |
| hpg_hier_junction | EA | 3 | 0.94248 |
| hpg_hier_junction1 | EA | 3 | 0.91018 |
| wdmpnn | EA | 3 | 0.97068 |
| hpg_hier | EA | 4 | 1.08920 |
| hpg_hier_octamer | EA | 4 | 1.01292 |
| hpg_hier_junction | EA | 4 | 0.95633 |
| hpg_hier_junction1 | EA | 4 | 1.06579 |
| wdmpnn | EA | 4 | 1.02413 |
| hpg_hier | EA | 5 | 1.08754 |
| hpg_hier_octamer | EA | 5 | 1.02007 |
| hpg_hier_junction | EA | 5 | 1.07009 |
| hpg_hier_junction1 | EA | 5 | 1.11705 |
| wdmpnn | EA | 5 | 0.96637 |
| hpg_hier | EA | 6 | 1.08458 |
| hpg_hier_octamer | EA | 6 | 0.88746 |
| hpg_hier_junction | EA | 6 | 1.11158 |
| hpg_hier_junction1 | EA | 6 | 1.02279 |
| wdmpnn | EA | 6 | 1.02865 |
| hpg_hier | EA | 7 | 0.90415 |
| hpg_hier_octamer | EA | 7 | 0.96685 |
| hpg_hier_junction | EA | 7 | 0.99979 |
| hpg_hier_junction1 | EA | 7 | 0.94671 |
| wdmpnn | EA | 7 | 0.97723 |
| hpg_hier | EA | 8 | 0.88987 |
| hpg_hier_octamer | EA | 8 | 1.02606 |
| hpg_hier_junction | EA | 8 | 0.93608 |
| hpg_hier_junction1 | EA | 8 | 0.92565 |
| wdmpnn | EA | 8 | 1.00210 |
| hpg_hier | IP | 0 | 1.06322 |
| hpg_hier_octamer | IP | 0 | 1.08994 |
| hpg_hier_junction | IP | 0 | 0.99481 |
| hpg_hier_junction1 | IP | 0 | 1.19008 |
| wdmpnn | IP | 0 | 1.20323 |
| hpg_hier | IP | 1 | 0.99560 |
| hpg_hier_octamer | IP | 1 | 1.02187 |
| hpg_hier_junction | IP | 1 | 1.00164 |
| hpg_hier_junction1 | IP | 1 | 1.02370 |
| wdmpnn | IP | 1 | 1.05020 |
| hpg_hier | IP | 2 | 0.84890 |
| hpg_hier_octamer | IP | 2 | 0.98340 |
| hpg_hier_junction | IP | 2 | 0.88672 |
| hpg_hier_junction1 | IP | 2 | 0.87414 |
| wdmpnn | IP | 2 | 1.00017 |
| hpg_hier | IP | 3 | 0.96612 |
| hpg_hier_octamer | IP | 3 | 0.91774 |
| hpg_hier_junction | IP | 3 | 1.11577 |
| hpg_hier_junction1 | IP | 3 | 0.93368 |
| wdmpnn | IP | 3 | 1.09510 |
| hpg_hier | IP | 4 | 1.08124 |
| hpg_hier_octamer | IP | 4 | 1.07257 |
| hpg_hier_junction | IP | 4 | 1.12346 |
| hpg_hier_junction1 | IP | 4 | 1.10583 |
| wdmpnn | IP | 4 | 1.21617 |
| hpg_hier | IP | 5 | 1.00542 |
| hpg_hier_octamer | IP | 5 | 0.91603 |
| hpg_hier_junction | IP | 5 | 1.00540 |
| hpg_hier_junction1 | IP | 5 | 1.05052 |
| wdmpnn | IP | 5 | 1.01764 |
| hpg_hier | IP | 6 | 1.07337 |
| hpg_hier_octamer | IP | 6 | 1.01271 |
| hpg_hier_junction | IP | 6 | 1.02193 |
| hpg_hier_junction1 | IP | 6 | 1.01315 |
| wdmpnn | IP | 6 | 0.96064 |
| hpg_hier | IP | 7 | 0.96764 |
| hpg_hier_octamer | IP | 7 | 1.00756 |
| hpg_hier_junction | IP | 7 | 0.92698 |
| hpg_hier_junction1 | IP | 7 | 0.92488 |
| wdmpnn | IP | 7 | 0.98869 |
| hpg_hier | IP | 8 | 0.93057 |
| hpg_hier_octamer | IP | 8 | 1.03480 |
| hpg_hier_junction | IP | 8 | 1.05177 |
| hpg_hier_junction1 | IP | 8 | 1.12885 |
| wdmpnn | IP | 8 | 1.05677 |

## Analysis C — NPZ split/scaling parity

The following values are read from the stored NPZ arrays. `octamer_matches_baseline` compares `n_train`, `n_val`, `n_test`, and `prediction_scale` for the same target/fold.

| model | target | fold | n_train | n_val | n_test | prediction_scale | octamer_matches_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | EA | 0 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 0 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 0 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 0 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 0 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 1 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 1 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 1 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 1 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 1 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 2 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 2 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 2 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 2 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 2 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 3 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 3 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 3 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 3 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 3 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 4 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 4 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 4 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 4 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 4 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 5 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 5 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 5 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 5 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 5 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 6 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 6 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 6 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 6 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 6 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 7 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 7 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 7 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 7 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 7 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | EA | 8 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | EA | 8 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | EA | 8 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | EA | 8 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | EA | 8 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 0 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 0 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 0 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 0 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 0 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 1 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 1 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 1 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 1 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 1 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 2 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 2 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 2 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 2 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 2 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 3 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 3 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 3 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 3 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 3 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 4 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 4 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 4 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 4 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 4 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 5 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 5 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 5 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 5 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 5 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 6 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 6 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 6 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 6 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 6 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 7 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 7 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 7 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 7 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 7 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier | IP | 8 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_octamer | IP | 8 | 33418 | 4774 | 4774 | physical_units | True |
| hpg_hier_junction | IP | 8 | 33418 | 4774 | 4774 | physical_units |  |
| hpg_hier_junction1 | IP | 8 | 33418 | 4774 | 4774 | physical_units |  |
| wdmpnn | IP | 8 | 33418 | 4774 | 4774 | physical_units |  |

## Verdict

- **What within-fold group-mean R² does the A-blind null achieve for EA?** Median `0.67571` across folds (mean `-1.54331`).
- **Under pooled and across-fold placement metrics, does octamer still lead baseline for EA?** Pooled group-mean R² baseline `0.94887` vs octamer `0.98907` (difference `+0.04020`). Placement R² baseline `0.92306` vs octamer `0.98941`; slope baseline `0.91292` vs octamer `1.03069`; fold-bias SD baseline `0.10273` vs octamer `0.03658`.
- **What within-fold group-mean R² does the A-blind null achieve for IP?** Median `-0.03401` across folds (mean `-1.06464`).
- **Under pooled and across-fold placement metrics, does octamer still lead baseline for IP?** Pooled group-mean R² baseline `0.95187` vs octamer `0.98444` (difference `+0.03258`). Placement R² baseline `0.93921` vs octamer `0.98013`; slope baseline `0.80603` vs octamer `1.05093`; fold-bias SD baseline `0.08316` vs octamer `0.04131`.
- **Do n_train / n_val / prediction_scale match between octamer and baseline in all 18 cells?** Yes.
- **Metric interpretation for EA:** The A-blind null median is `0.67571`, so within-fold EA group-mean R² **cannot support a chemistry-extrapolation claim**: a predictor with no held-out-A information attains a high score from observed B/composition/architecture covariates.
- **Metric interpretation for IP:** The A-blind null median is `-0.03401`; this null does **not** establish a high within-fold IP metric floor. The IP conclusion is therefore not inferred from the EA floor.
