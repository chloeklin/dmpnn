# F6 — Demonstration: paired differences

## Prediction source
`predictions/regen_v1/ea_ip_lomo_b_clustered/ea_ip__[target]__[model]__monomer_b_heldout_clustered__fold[4-8]__s[42/43/44].npz`
Seeds 42,43,44 averaged at prediction level.
Models: hpg_hier_octamer (octamer), wdmpnn.
D folds: [4, 5, 6, 7, 8] (cross-scaffold fold group).

## Metric function
`compute_copolymer_metrics` → overall_r2, mae, rmse, group_mean_r2, delta_r2.
Paired difference = octamer − wDMPNN per fold. Win = octamer better:
higher for R² metrics, lower for MAE/RMSE.

## Missing files
None

## Expected vs computed
None — all values match expected.

## Cells: 5 D folds × 2 targets × 5 metrics = 50 values.