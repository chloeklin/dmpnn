# Factor 2 — positional embeddings (off vs on)

## Layout
Two 1×5 rows per figure: one row per target (EA top, IP bottom),
one column per metric in the order:
overall_r2, rmse, mae, group_mean_r2, delta_r2.
Factor 5 overlays three row subsets in each panel; Factor 2 uses `all` rows only.

## Prediction paths and seed handling
Baseline: `predictions/regen_v1/ea_ip_lomo`
Ablated:  `predictions/octamer_posemb/ea_ip_lomo`
Model: `hpg_hier_octamer` | split: `monomer_heldout` | seeds averaged at prediction level (42/43/44)
Ablated filename suffix: `__noposemb`.

## Row subsets used
- `all`: all rows (all rows)

## Split
No S/D boundary (A-split).

## Cells
Expected: 9 folds × 2 targets × 1 subsets × 2 settings × 5 metrics = 180 rows.
Plotted rows: 180.

## Incomplete cells (n_seeds < 3)
None — all plotted cells have 3 seeds.

## Missing prediction files
None

## §5 ablated − baseline range assertions (all-rows subset)
| target | group | metric | computed | expected | result |
| --- | --- | --- | --- | --- | --- |
| EA | all | overall_r2 | -0.008 to 0.035 | -0.008 to 0.035 | PASS |
| EA | all | rmse | -0.06 to 0.012 | -0.06 to 0.012 | PASS |
| EA | all | mae | -0.051 to 0.011 | -0.051 to 0.011 | PASS |
| EA | all | group_mean_r2 | -0.008 to 0.037 | -0.008 to 0.037 | PASS |
| EA | all | delta_r2 | -0.053 to 0.105 | -0.053 to 0.105 | PASS |
| IP | all | overall_r2 | -0.032 to 0.088 | -0.032 to 0.088 | PASS |
| IP | all | rmse | -0.032 to 0.013 | -0.032 to 0.013 | PASS |
| IP | all | mae | -0.027 to 0.013 | -0.027 to 0.013 | PASS |
| IP | all | group_mean_r2 | -0.032 to 0.092 | -0.032 to 0.092 | PASS |
| IP | all | delta_r2 | -0.108 to 0.051 | -0.108 to 0.051 | PASS |