# Factor 5 — K=1 vs K=16 octamer reads

## Layout
Two 1×5 rows per figure: one row per target (EA top, IP bottom),
one column per metric in the order:
overall_r2, rmse, mae, group_mean_r2, delta_r2.
Factor 5 overlays three row subsets in each panel; Factor 2 uses `all` rows only.

## Prediction paths and seed handling
Baseline: `predictions/regen_v1/ea_ip_lomo_b_clustered`
Ablated:  `predictions/octamer_k1/ea_ip_lomo_b_clustered`
Model: `hpg_hier_octamer` | split: `monomer_b_heldout_clustered` | seeds averaged at prediction level (42/43/44)
Ablated filename suffix: `__k1`.

## Row subsets used
- `random`: random only (poly_type in ['random'])
- `block_alternating`: block + alternating (poly_type in ['block', 'alternating'])
- `all`: all rows (all rows)

## Split
S folds (same-scaffold interpolation): [0, 1, 2, 3]
D folds (cross-scaffold extrapolation): [4, 5, 6, 7, 8]

## Cells
Expected: 9 folds × 2 targets × 3 subsets × 2 settings × 5 metrics = 540 rows.
Plotted rows: 540.

## Incomplete cells (n_seeds < 3)
None — all plotted cells have 3 seeds.

## Missing prediction files
None

## §5 ablated − baseline range assertions (all-rows subset)
| target | group | metric | computed | expected | result |
| --- | --- | --- | --- | --- | --- |
| EA | S | overall_r2 | -0.01 to 0.001 | -0.01 to 0.001 | PASS |
| EA | S | rmse | -0.002 to 0.011 | -0.002 to 0.011 | PASS |
| EA | S | mae | -0.001 to 0.008 | -0.001 to 0.008 | PASS |
| EA | S | group_mean_r2 | -0.01 to 0.001 | -0.01 to 0.001 | PASS |
| EA | S | delta_r2 | -0.002 to 0.029 | -0.002 to 0.029 | PASS |
| EA | D | overall_r2 | -0.006 to 0.018 | -0.006 to 0.018 | PASS |
| EA | D | rmse | -0.015 to 0.007 | -0.015 to 0.007 | PASS |
| EA | D | mae | -0.01 to 0.002 | -0.01 to 0.002 | PASS |
| EA | D | group_mean_r2 | -0.006 to 0.019 | -0.006 to 0.019 | PASS |
| EA | D | delta_r2 | -0.024 to 0.032 | -0.024 to 0.032 | PASS |
| IP | S | overall_r2 | -0.003 to 0.003 | -0.003 to 0.003 | PASS |
| IP | S | rmse | -0.004 to 0.003 | -0.004 to 0.003 | PASS |
| IP | S | mae | -0.0 to 0.002 | -0.0 to 0.002 | PASS |
| IP | S | group_mean_r2 | -0.004 to 0.003 | -0.004 to 0.003 | PASS |
| IP | S | delta_r2 | -0.019 to 0.005 | -0.019 to 0.005 | PASS |
| IP | D | overall_r2 | -0.006 to 0.01 | -0.006 to 0.01 | PASS |
| IP | D | rmse | -0.006 to 0.008 | -0.006 to 0.008 | PASS |
| IP | D | mae | -0.004 to 0.005 | -0.004 to 0.005 | PASS |
| IP | D | group_mean_r2 | -0.006 to 0.01 | -0.006 to 0.01 | PASS |
| IP | D | delta_r2 | -0.062 to 0.025 | -0.062 to 0.025 | PASS |