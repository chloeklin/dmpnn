# F5 — Noise floor

## Left panel
Hard-coded from `analysis/model_diagnostics/_noise_floor_results.md`.
6 runs: HPG-octamer, A-split EA, seed 42, 2 folds × 3 repeats, V100 GPU.

## Right panel
`predictions/regen_v1/ea_ip_lomo/ea_ip__[target]__[model]__monomer_heldout__fold[0-8]__s[42/43/44].npz`
Per-seed delta_r2 via `compute_copolymer_metrics`, then sample SD (ddof=1) across 3 seeds.
Cells available: 36 of 36 expected.
Overall median SD: 0.0491