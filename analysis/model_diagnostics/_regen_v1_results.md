# Frozen-protocol regeneration v1

All figures are the mean prediction of three seeds.

## Status

R1 pending: 10/270 run artifacts and sidecars are complete. Analysis is blocked until all 270 R1 runs are present.

## Pre-registered run-quality rule

A single run is flagged as **potentially undertrained** if `best_epoch < 10`. This threshold was chosen before bulk results landed and is applied identically across all models. When results arrive, all headline metric tables will be reported twice: once with every seed, and once with flagged runs excluded. Flag counts will be reported per model.

## Mandatory metric verification

| check | status | reference |
| --- | --- | --- |
| old seed-42 aggregate metrics | PASS | _phase1_metrics_scratch.md |
| old seed-42 per-fold metrics | PASS | _groupmean_metric_floor.md |
| ordering tie convention | PASS | exact y_pred ties receive 0.5 credit |

The canonical metric module reproduces the old seed-42 references to 5 decimal places.

## Ordering tie discrepancy resolved

The committed old inline metric scored exact prediction ties as incorrect because it tested `sign_product > 0`. HPG-hier-octamer has 34 exact tied prediction pairs across EA/IP; every other model has zero. The frozen Phase-1 values instead give exact prediction ties 0.5 credit: this reproduces the octamer ordering medians exactly (EA 0.818263 → 0.81826; IP 0.827061 → 0.82706). The canonical module now documents and uses that convention. No other metric was changed.

## Null-floor comparison

Group-mean R² comparisons use the **fold-specific** A-blind null floor from `_dataset_design_audit.md`, not a median across folds. The median floor is 0.384 for clustered EA but fold-specific floors vary.

## Artifact collection

Task logs must be downloaded alongside NPZs before the final report is generated, so that the frozen-split assertion can be confirmed from logs rather than inferred from output metadata. Use `scripts/shell/download_regen_v1_artifacts.sh` after jobs complete, then grep `logs/regen_v1/r3/tasks/` for `Frozen monomer_b_heldout split assertions passed for all folds`, `B-identity leakage`, `differs from frozen metadata`, or `frozen_protocol`.

## Missing cells

| model | target | fold | seed | available | sidecar |
| --- | --- | --- | --- | --- | --- |
| hpg_hier | EA | 0 | 44 | False | False |
| hpg_hier | EA | 1 | 42 | False | False |
| hpg_hier | EA | 1 | 43 | False | False |
| hpg_hier | EA | 1 | 44 | False | False |
| hpg_hier | EA | 2 | 42 | False | False |
| hpg_hier | EA | 2 | 43 | False | False |
| hpg_hier | EA | 2 | 44 | False | False |
| hpg_hier | EA | 3 | 42 | False | False |
| hpg_hier | EA | 3 | 43 | False | False |
| hpg_hier | EA | 3 | 44 | False | False |
| hpg_hier | EA | 4 | 42 | False | False |
| hpg_hier | EA | 4 | 43 | False | False |
| hpg_hier | EA | 4 | 44 | False | False |
| hpg_hier | EA | 5 | 42 | False | False |
| hpg_hier | EA | 5 | 43 | False | False |
| hpg_hier | EA | 5 | 44 | False | False |
| hpg_hier | EA | 6 | 42 | False | False |
| hpg_hier | EA | 6 | 43 | False | False |
| hpg_hier | EA | 6 | 44 | False | False |
| hpg_hier | EA | 7 | 42 | False | False |
| hpg_hier | EA | 7 | 43 | False | False |
| hpg_hier | EA | 7 | 44 | False | False |
| hpg_hier | EA | 8 | 42 | False | False |
| hpg_hier | EA | 8 | 43 | False | False |
| hpg_hier | EA | 8 | 44 | False | False |
| hpg_hier | IP | 0 | 42 | False | False |
| hpg_hier | IP | 0 | 43 | False | False |
| hpg_hier | IP | 0 | 44 | False | False |
| hpg_hier | IP | 1 | 42 | False | False |
| hpg_hier | IP | 1 | 43 | False | False |
| hpg_hier | IP | 1 | 44 | False | False |
| hpg_hier | IP | 2 | 42 | False | False |
| hpg_hier | IP | 2 | 43 | False | False |
| hpg_hier | IP | 2 | 44 | False | False |
| hpg_hier | IP | 3 | 42 | False | False |
| hpg_hier | IP | 3 | 43 | False | False |
| hpg_hier | IP | 3 | 44 | False | False |
| hpg_hier | IP | 4 | 42 | False | False |
| hpg_hier | IP | 4 | 43 | False | False |
| hpg_hier | IP | 4 | 44 | False | False |
| hpg_hier | IP | 5 | 42 | False | False |
| hpg_hier | IP | 5 | 43 | False | False |
| hpg_hier | IP | 5 | 44 | False | False |
| hpg_hier | IP | 6 | 42 | False | False |
| hpg_hier | IP | 6 | 43 | False | False |
| hpg_hier | IP | 6 | 44 | False | False |
| hpg_hier | IP | 7 | 42 | False | False |
| hpg_hier | IP | 7 | 43 | False | False |
| hpg_hier | IP | 7 | 44 | False | False |
| hpg_hier | IP | 8 | 42 | False | False |
| hpg_hier | IP | 8 | 43 | False | False |
| hpg_hier | IP | 8 | 44 | False | False |
| wdmpnn | EA | 0 | 44 | False | False |
| wdmpnn | EA | 1 | 42 | False | False |
| wdmpnn | EA | 1 | 43 | False | False |
| wdmpnn | EA | 1 | 44 | False | False |
| wdmpnn | EA | 2 | 42 | False | False |
| wdmpnn | EA | 2 | 43 | False | False |
| wdmpnn | EA | 2 | 44 | False | False |
| wdmpnn | EA | 3 | 42 | False | False |
| wdmpnn | EA | 3 | 43 | False | False |
| wdmpnn | EA | 3 | 44 | False | False |
| wdmpnn | EA | 4 | 42 | False | False |
| wdmpnn | EA | 4 | 43 | False | False |
| wdmpnn | EA | 4 | 44 | False | False |
| wdmpnn | EA | 5 | 42 | False | False |
| wdmpnn | EA | 5 | 43 | False | False |
| wdmpnn | EA | 5 | 44 | False | False |
| wdmpnn | EA | 6 | 42 | False | False |
| wdmpnn | EA | 6 | 43 | False | False |
| wdmpnn | EA | 6 | 44 | False | False |
| wdmpnn | EA | 7 | 42 | False | False |
| wdmpnn | EA | 7 | 43 | False | False |
| wdmpnn | EA | 7 | 44 | False | False |
| wdmpnn | EA | 8 | 42 | False | False |
| wdmpnn | EA | 8 | 43 | False | False |
| wdmpnn | EA | 8 | 44 | False | False |
| wdmpnn | IP | 0 | 42 | False | False |
| wdmpnn | IP | 0 | 43 | False | False |
| wdmpnn | IP | 0 | 44 | False | False |
| wdmpnn | IP | 1 | 42 | False | False |
| wdmpnn | IP | 1 | 43 | False | False |
| wdmpnn | IP | 1 | 44 | False | False |
| wdmpnn | IP | 2 | 42 | False | False |
| wdmpnn | IP | 2 | 43 | False | False |
| wdmpnn | IP | 2 | 44 | False | False |
| wdmpnn | IP | 3 | 42 | False | False |
| wdmpnn | IP | 3 | 43 | False | False |
| wdmpnn | IP | 3 | 44 | False | False |
| wdmpnn | IP | 4 | 42 | False | False |
| wdmpnn | IP | 4 | 43 | False | False |
| wdmpnn | IP | 4 | 44 | False | False |
| wdmpnn | IP | 5 | 42 | False | False |
| wdmpnn | IP | 5 | 43 | False | False |
| wdmpnn | IP | 5 | 44 | False | False |
| wdmpnn | IP | 6 | 42 | False | False |
| wdmpnn | IP | 6 | 43 | False | False |
| wdmpnn | IP | 6 | 44 | False | False |
| wdmpnn | IP | 7 | 42 | False | False |
| wdmpnn | IP | 7 | 43 | False | False |
| wdmpnn | IP | 7 | 44 | False | False |
| wdmpnn | IP | 8 | 42 | False | False |
| wdmpnn | IP | 8 | 43 | False | False |
| wdmpnn | IP | 8 | 44 | False | False |
| hpg_hier_octamer | EA | 0 | 44 | False | False |
| hpg_hier_octamer | EA | 1 | 42 | False | False |
| hpg_hier_octamer | EA | 1 | 43 | False | False |
| hpg_hier_octamer | EA | 1 | 44 | False | False |
| hpg_hier_octamer | EA | 2 | 42 | False | False |
| hpg_hier_octamer | EA | 2 | 43 | False | False |
| hpg_hier_octamer | EA | 2 | 44 | False | False |
| hpg_hier_octamer | EA | 3 | 42 | False | False |
| hpg_hier_octamer | EA | 3 | 43 | False | False |
| hpg_hier_octamer | EA | 3 | 44 | False | False |
| hpg_hier_octamer | EA | 4 | 42 | False | False |
| hpg_hier_octamer | EA | 4 | 43 | False | False |
| hpg_hier_octamer | EA | 4 | 44 | False | False |
| hpg_hier_octamer | EA | 5 | 42 | False | False |
| hpg_hier_octamer | EA | 5 | 43 | False | False |
| hpg_hier_octamer | EA | 5 | 44 | False | False |
| hpg_hier_octamer | EA | 6 | 42 | False | False |
| hpg_hier_octamer | EA | 6 | 43 | False | False |
| hpg_hier_octamer | EA | 6 | 44 | False | False |
| hpg_hier_octamer | EA | 7 | 42 | False | False |
| hpg_hier_octamer | EA | 7 | 43 | False | False |
| hpg_hier_octamer | EA | 7 | 44 | False | False |
| hpg_hier_octamer | EA | 8 | 42 | False | False |
| hpg_hier_octamer | EA | 8 | 43 | False | False |
| hpg_hier_octamer | EA | 8 | 44 | False | False |
| hpg_hier_octamer | IP | 0 | 42 | False | False |
| hpg_hier_octamer | IP | 0 | 43 | False | False |
| hpg_hier_octamer | IP | 0 | 44 | False | False |
| hpg_hier_octamer | IP | 1 | 42 | False | False |
| hpg_hier_octamer | IP | 1 | 43 | False | False |
| hpg_hier_octamer | IP | 1 | 44 | False | False |
| hpg_hier_octamer | IP | 2 | 42 | False | False |
| hpg_hier_octamer | IP | 2 | 43 | False | False |
| hpg_hier_octamer | IP | 2 | 44 | False | False |
| hpg_hier_octamer | IP | 3 | 42 | False | False |
| hpg_hier_octamer | IP | 3 | 43 | False | False |
| hpg_hier_octamer | IP | 3 | 44 | False | False |
| hpg_hier_octamer | IP | 4 | 42 | False | False |
| hpg_hier_octamer | IP | 4 | 43 | False | False |
| hpg_hier_octamer | IP | 4 | 44 | False | False |
| hpg_hier_octamer | IP | 5 | 42 | False | False |
| hpg_hier_octamer | IP | 5 | 43 | False | False |
| hpg_hier_octamer | IP | 5 | 44 | False | False |
| hpg_hier_octamer | IP | 6 | 42 | False | False |
| hpg_hier_octamer | IP | 6 | 43 | False | False |
| hpg_hier_octamer | IP | 6 | 44 | False | False |
| hpg_hier_octamer | IP | 7 | 42 | False | False |
| hpg_hier_octamer | IP | 7 | 43 | False | False |
| hpg_hier_octamer | IP | 7 | 44 | False | False |
| hpg_hier_octamer | IP | 8 | 42 | False | False |
| hpg_hier_octamer | IP | 8 | 43 | False | False |
| hpg_hier_octamer | IP | 8 | 44 | False | False |
| hpg_hier_junction | EA | 0 | 44 | False | False |
| hpg_hier_junction | EA | 1 | 42 | False | False |
| hpg_hier_junction | EA | 1 | 43 | False | False |
| hpg_hier_junction | EA | 1 | 44 | False | False |
| hpg_hier_junction | EA | 2 | 42 | False | False |
| hpg_hier_junction | EA | 2 | 43 | False | False |
| hpg_hier_junction | EA | 2 | 44 | False | False |
| hpg_hier_junction | EA | 3 | 42 | False | False |
| hpg_hier_junction | EA | 3 | 43 | False | False |
| hpg_hier_junction | EA | 3 | 44 | False | False |
| hpg_hier_junction | EA | 4 | 42 | False | False |
| hpg_hier_junction | EA | 4 | 43 | False | False |
| hpg_hier_junction | EA | 4 | 44 | False | False |
| hpg_hier_junction | EA | 5 | 42 | False | False |
| hpg_hier_junction | EA | 5 | 43 | False | False |
| hpg_hier_junction | EA | 5 | 44 | False | False |
| hpg_hier_junction | EA | 6 | 42 | False | False |
| hpg_hier_junction | EA | 6 | 43 | False | False |
| hpg_hier_junction | EA | 6 | 44 | False | False |
| hpg_hier_junction | EA | 7 | 42 | False | False |
| hpg_hier_junction | EA | 7 | 43 | False | False |
| hpg_hier_junction | EA | 7 | 44 | False | False |
| hpg_hier_junction | EA | 8 | 42 | False | False |
| hpg_hier_junction | EA | 8 | 43 | False | False |
| hpg_hier_junction | EA | 8 | 44 | False | False |
| hpg_hier_junction | IP | 0 | 42 | False | False |
| hpg_hier_junction | IP | 0 | 43 | False | False |
| hpg_hier_junction | IP | 0 | 44 | False | False |
| hpg_hier_junction | IP | 1 | 42 | False | False |
| hpg_hier_junction | IP | 1 | 43 | False | False |
| hpg_hier_junction | IP | 1 | 44 | False | False |
| hpg_hier_junction | IP | 2 | 42 | False | False |
| hpg_hier_junction | IP | 2 | 43 | False | False |
| hpg_hier_junction | IP | 2 | 44 | False | False |
| hpg_hier_junction | IP | 3 | 42 | False | False |
| hpg_hier_junction | IP | 3 | 43 | False | False |
| hpg_hier_junction | IP | 3 | 44 | False | False |
| hpg_hier_junction | IP | 4 | 42 | False | False |
| hpg_hier_junction | IP | 4 | 43 | False | False |
| hpg_hier_junction | IP | 4 | 44 | False | False |
| hpg_hier_junction | IP | 5 | 42 | False | False |
| hpg_hier_junction | IP | 5 | 43 | False | False |
| hpg_hier_junction | IP | 5 | 44 | False | False |
| hpg_hier_junction | IP | 6 | 42 | False | False |
| hpg_hier_junction | IP | 6 | 43 | False | False |
| hpg_hier_junction | IP | 6 | 44 | False | False |
| hpg_hier_junction | IP | 7 | 42 | False | False |
| hpg_hier_junction | IP | 7 | 43 | False | False |
| hpg_hier_junction | IP | 7 | 44 | False | False |
| hpg_hier_junction | IP | 8 | 42 | False | False |
| hpg_hier_junction | IP | 8 | 43 | False | False |
| hpg_hier_junction | IP | 8 | 44 | False | False |
| hpg_hier_junction1 | EA | 0 | 44 | False | False |
| hpg_hier_junction1 | EA | 1 | 42 | False | False |
| hpg_hier_junction1 | EA | 1 | 43 | False | False |
| hpg_hier_junction1 | EA | 1 | 44 | False | False |
| hpg_hier_junction1 | EA | 2 | 42 | False | False |
| hpg_hier_junction1 | EA | 2 | 43 | False | False |
| hpg_hier_junction1 | EA | 2 | 44 | False | False |
| hpg_hier_junction1 | EA | 3 | 42 | False | False |
| hpg_hier_junction1 | EA | 3 | 43 | False | False |
| hpg_hier_junction1 | EA | 3 | 44 | False | False |
| hpg_hier_junction1 | EA | 4 | 42 | False | False |
| hpg_hier_junction1 | EA | 4 | 43 | False | False |
| hpg_hier_junction1 | EA | 4 | 44 | False | False |
| hpg_hier_junction1 | EA | 5 | 42 | False | False |
| hpg_hier_junction1 | EA | 5 | 43 | False | False |
| hpg_hier_junction1 | EA | 5 | 44 | False | False |
| hpg_hier_junction1 | EA | 6 | 42 | False | False |
| hpg_hier_junction1 | EA | 6 | 43 | False | False |
| hpg_hier_junction1 | EA | 6 | 44 | False | False |
| hpg_hier_junction1 | EA | 7 | 42 | False | False |
| hpg_hier_junction1 | EA | 7 | 43 | False | False |
| hpg_hier_junction1 | EA | 7 | 44 | False | False |
| hpg_hier_junction1 | EA | 8 | 42 | False | False |
| hpg_hier_junction1 | EA | 8 | 43 | False | False |
| hpg_hier_junction1 | EA | 8 | 44 | False | False |
| hpg_hier_junction1 | IP | 0 | 42 | False | False |
| hpg_hier_junction1 | IP | 0 | 43 | False | False |
| hpg_hier_junction1 | IP | 0 | 44 | False | False |
| hpg_hier_junction1 | IP | 1 | 42 | False | False |
| hpg_hier_junction1 | IP | 1 | 43 | False | False |
| hpg_hier_junction1 | IP | 1 | 44 | False | False |
| hpg_hier_junction1 | IP | 2 | 42 | False | False |
| hpg_hier_junction1 | IP | 2 | 43 | False | False |
| hpg_hier_junction1 | IP | 2 | 44 | False | False |
| hpg_hier_junction1 | IP | 3 | 42 | False | False |
| hpg_hier_junction1 | IP | 3 | 43 | False | False |
| hpg_hier_junction1 | IP | 3 | 44 | False | False |
| hpg_hier_junction1 | IP | 4 | 42 | False | False |
| hpg_hier_junction1 | IP | 4 | 43 | False | False |
| hpg_hier_junction1 | IP | 4 | 44 | False | False |
| hpg_hier_junction1 | IP | 5 | 42 | False | False |
| hpg_hier_junction1 | IP | 5 | 43 | False | False |
| hpg_hier_junction1 | IP | 5 | 44 | False | False |
| hpg_hier_junction1 | IP | 6 | 42 | False | False |
| hpg_hier_junction1 | IP | 6 | 43 | False | False |
| hpg_hier_junction1 | IP | 6 | 44 | False | False |
| hpg_hier_junction1 | IP | 7 | 42 | False | False |
| hpg_hier_junction1 | IP | 7 | 43 | False | False |
| hpg_hier_junction1 | IP | 7 | 44 | False | False |
| hpg_hier_junction1 | IP | 8 | 42 | False | False |
| hpg_hier_junction1 | IP | 8 | 43 | False | False |
| hpg_hier_junction1 | IP | 8 | 44 | False | False |
