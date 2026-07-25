# Seed-42 Diagnostics Figure Index

## Added: LOMO overall breakdown and EA fold-1 deep dive

### New figures

- `13_lomo_overall_r2.png` — Per-fold LOMO Overall R² by held-out monomer for all three models; bars are direct fold values and off-scale values are annotated.
- `14_lomo_overall_mae.png` — Per-fold LOMO Overall MAE (eV) by held-out monomer for all three models; bars are direct fold values and off-scale values are annotated.
- `14b_lomo_overall_rmse.png` — Per-fold LOMO Overall RMSE (eV) by held-out monomer for all three models; bars are direct fold values and off-scale values are annotated.
- `15_ea_lomo_fold1_parity_hpg_vs_wdmpnn.png` — EA LOMO fold-1 overall parity: HPG-hier versus wDMPNN, with R², MAE, bias, and slope.
- `16_ea_lomo_fold1_group_vs_deviation.png` — EA fold-1 group-mean and architecture-deviation MAE with corresponding R² for all models.
- `17_lomo_fold1_target_shift_EA_IP.png` — Fold-1 target distributions: training versus held-out dibenzothiophene sulfone, with mean shift, standard-deviation ratio, and Wasserstein distance.

### EA/IP fold-1 (dibenzothiophene sulfone): overall test-set metrics

| target | model | r2 | mae | rmse | target_sd |
|---|---|---|---|---|---|
| EA | HPG-hier | 0.5685 | 0.2140 | 0.2347 | 0.3573 |
| IP | HPG-hier | 0.9790 | 0.0450 | 0.0543 | 0.3748 |
| EA | wDMPNN | 0.9304 | 0.0800 | 0.0943 | 0.3573 |
| IP | wDMPNN | 0.9536 | 0.0621 | 0.0807 | 0.3748 |
| EA | ChemArch | 0.5153 | 0.2356 | 0.2487 | 0.3573 |
| IP | ChemArch | 0.9205 | 0.0971 | 0.1057 | 0.3748 |

Interpret MAE against `target_sd`: an R² drop can be amplified by a narrow held-out target distribution; MAE and SD are both eV.

### Fold-1 target-distribution shift

| target | train_mean | heldout_mean | mean_shift | train_sd | heldout_sd | std_ratio | wasserstein |
|---|---|---|---|---|---|---|---|
| EA | -2.5618 | -2.3781 | 0.1837 | 0.6208 | 0.3573 | 0.5754 | 0.3000 |
| IP | 1.4169 | 1.7458 | 0.3290 | 0.4812 | 0.3748 | 0.7789 | 0.3470 |

### EA fold-1 decomposition

#### Group means

| model | r2 | mae | rmse | n |
|---|---|---|---|---|
| hpg_hier | 0.5751 | 0.2133 | 0.2330 | 2046 |
| wdmpnn | 0.9446 | 0.0755 | 0.0841 | 2046 |
| chemarch | 0.5227 | 0.2348 | 0.2470 | 2046 |

#### Architecture deviations

| model | r2 | mae | rmse | n |
|---|---|---|---|---|
| hpg_hier | 0.7903 | 0.0187 | 0.0254 | 4774 |
| wdmpnn | 0.4364 | 0.0295 | 0.0416 | 4774 |
| chemarch | 0.8075 | 0.0170 | 0.0243 | 4774 |

### WDMPNN input sanity

- Fold-1 WDMPNN_Input sanity: 4774 unique inputs; fragment counts=[2]; port counts=[4]; RDKit fragment parse failures/port violations=0.

### HPG-hier + wDMPNN arithmetic-mean ensemble

#### Individual models

| target | split | model | r2_median | r2_mean | mae_median | mae_mean |
|---|---|---|---|---|---|---|
| EA | group_disjoint | HPG-hier | 0.9959 | 0.9958 | 0.0283 | 0.0286 |
| EA | monomer_heldout | HPG-hier | 0.9241 | 0.8925 | 0.0939 | 0.1036 |
| EA | pair_disjoint | HPG-hier | 0.9965 | 0.9959 | 0.0250 | 0.0275 |
| IP | group_disjoint | HPG-hier | 0.9962 | 0.9964 | 0.0219 | 0.0216 |
| IP | monomer_heldout | HPG-hier | 0.9631 | 0.9171 | 0.0510 | 0.0724 |
| IP | pair_disjoint | HPG-hier | 0.9956 | 0.9954 | 0.0225 | 0.0234 |
| EA | group_disjoint | wDMPNN | 0.9969 | 0.9970 | 0.0234 | 0.0234 |
| EA | monomer_heldout | wDMPNN | 0.9559 | 0.9276 | 0.0685 | 0.0850 |
| EA | pair_disjoint | wDMPNN | 0.9966 | 0.9965 | 0.0248 | 0.0248 |
| IP | group_disjoint | wDMPNN | 0.9973 | 0.9973 | 0.0175 | 0.0178 |
| IP | monomer_heldout | wDMPNN | 0.9569 | 0.8467 | 0.0621 | 0.0777 |
| IP | pair_disjoint | wDMPNN | 0.9968 | 0.9967 | 0.0197 | 0.0198 |

#### Ensemble

| target | split | r2_median | r2_mean | mae_median | mae_mean |
|---|---|---|---|---|---|
| EA | group_disjoint | 0.9975 | 0.9975 | 0.0208 | 0.0213 |
| EA | monomer_heldout | 0.9445 | 0.9445 | 0.0661 | 0.0781 |
| EA | pair_disjoint | 0.9975 | 0.9974 | 0.0203 | 0.0213 |
| IP | group_disjoint | 0.9979 | 0.9979 | 0.0159 | 0.0160 |
| IP | monomer_heldout | 0.9734 | 0.9251 | 0.0464 | 0.0606 |
| IP | pair_disjoint | 0.9972 | 0.9973 | 0.0181 | 0.0177 |
