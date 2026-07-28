# Regen v1 Pilot Verification

Scope: 10 R1 pilot runs (monomer_heldout) and 4 R3 pilot runs (monomer_b_heldout_clustered).

**This report does not authorise submission of the remaining 260 R1 or 212 R3 jobs.**

## A. Did they actually train

| split | model | target | fold | seed | epochs_run | best_epoch | best_val_loss | wall_time_s | wall_time_h | accelerator | device | git_sha | pbs_job_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monomer_heldout | hpg_hier | EA_vs_SHE_eV | 0 | 42 | 55 | 40 | 0.004577 | 6207.070218 | 1.724186 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[0].gadi-pbs |
| monomer_heldout | hpg_hier | EA_vs_SHE_eV | 0 | 43 | 20 | 5 | 0.008415 | 2237.808309 | 0.621613 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[5].gadi-pbs |
| monomer_heldout | hpg_hier_junction1 | EA_vs_SHE_eV | 0 | 42 | 38 | 23 | 0.004321 | 4355.246452 | 1.209791 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[4].gadi-pbs |
| monomer_heldout | hpg_hier_junction1 | EA_vs_SHE_eV | 0 | 43 | 30 | 15 | 0.004306 | 3572.104922 | 0.992251 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[9].gadi-pbs |
| monomer_heldout | hpg_hier_junction | EA_vs_SHE_eV | 0 | 42 | 56 | 41 | 0.005069 | 6581.571612 | 1.828214 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[3].gadi-pbs |
| monomer_heldout | hpg_hier_junction | EA_vs_SHE_eV | 0 | 43 | 42 | 27 | 0.004384 | 4868.142469 | 1.352262 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[8].gadi-pbs |
| monomer_heldout | hpg_hier_octamer | EA_vs_SHE_eV | 0 | 42 | 72 | 57 | 0.003249 | 8199.694746 | 2.277693 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[2].gadi-pbs |
| monomer_heldout | hpg_hier_octamer | EA_vs_SHE_eV | 0 | 43 | 20 | 5 | 0.005642 | 2334.513462 | 0.648476 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[7].gadi-pbs |
| monomer_heldout | wdmpnn | EA_vs_SHE_eV | 0 | 42 | 51 | 36 | 0.012252 | 6905.689296 | 1.918247 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[1].gadi-pbs |
| monomer_heldout | wdmpnn | EA_vs_SHE_eV | 0 | 43 | 52 | 37 | 0.012127 | 6909.388079 | 1.919274 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174849945[6].gadi-pbs |
| monomer_b_heldout_clustered | hpg_hier | EA_vs_SHE_eV | 0 | 42 | 48 | 33 | 0.017444 | 5569.353392 | 1.547043 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174854419[0].gadi-pbs |
| monomer_b_heldout_clustered | hpg_hier | EA_vs_SHE_eV | 1 | 42 | 32 | 17 | 0.010502 | 3716.451681 | 1.032348 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174854419[1].gadi-pbs |
| monomer_b_heldout_clustered | hpg_hier | EA_vs_SHE_eV | 2 | 42 | 29 | 14 | 0.013571 | 3500.742594 | 0.972428 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174854419[2].gadi-pbs |
| monomer_b_heldout_clustered | wdmpnn | EA_vs_SHE_eV | 0 | 42 | 98 | 83 | 0.026002 | 13463.905983 | 3.739974 | cuda | Tesla V100-SXM2-32GB | cec9d5feea303e0f | 174854419[3].gadi-pbs |

No runs flagged for near-zero epochs or implausibly short wall time.

## B. Is the new code path actually being used

| split | model | fold | seed | y_pred_shape | y_pred_final_shape | both_present | hashes_differ | ypred_identical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monomer_heldout | hpg_hier | 0 | 42 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier | 0 | 43 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier_junction1 | 0 | 42 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier_junction1 | 0 | 43 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier_junction | 0 | 42 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier_junction | 0 | 43 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier_octamer | 0 | 42 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | hpg_hier_octamer | 0 | 43 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | wdmpnn | 0 | 42 | (4774,) | (4774,) | True | True | False |
| monomer_heldout | wdmpnn | 0 | 43 | (4774,) | (4774,) | True | True | False |
| monomer_b_heldout_clustered | hpg_hier | 0 | 42 | (4788,) | (4788,) | True | True | False |
| monomer_b_heldout_clustered | hpg_hier | 1 | 42 | (4788,) | (4788,) | True | True | False |
| monomer_b_heldout_clustered | hpg_hier | 2 | 42 | (4788,) | (4788,) | True | True | False |
| monomer_b_heldout_clustered | wdmpnn | 0 | 42 | (4788,) | (4788,) | True | True | False |

### R1 pilot vs pre-regen counterparts
| model | seed | old_file | max_abs_diff | identical |
| --- | --- | --- | --- | --- |
| hpg_hier | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold0__s42.npz | 0.375915 | False |
| hpg_hier | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold0__s42.npz | 0.485784 | False |
| hpg_hier_junction1 | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction1__monomer_heldout__fold0__s42.npz | 0.371763 | False |
| hpg_hier_junction1 | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction1__monomer_heldout__fold0__s42.npz | 0.376781 | False |
| hpg_hier_junction | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction__monomer_heldout__fold0__s42.npz | 0.418842 | False |
| hpg_hier_junction | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction__monomer_heldout__fold0__s42.npz | 0.367303 | False |
| hpg_hier_octamer | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier_octamer__monomer_heldout__fold0__s42.npz | 0.234042 | False |
| hpg_hier_octamer | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier_octamer__monomer_heldout__fold0__s42.npz | 0.575999 | False |
| wdmpnn | 42 | ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_heldout__fold0__s42.npz | 0.189930 | False |
| wdmpnn | 43 | ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_heldout__fold0__s42.npz | 0.179964 | False |

## C. Split integrity

### C.1 Split hash consistency across seeds
| split | model | fold | seeds | unique_hashes | consistent |
| --- | --- | --- | --- | --- | --- |
| monomer_heldout | hpg_hier | 0 | 2 | 1 | True |
| monomer_heldout | hpg_hier_junction1 | 0 | 2 | 1 | True |
| monomer_heldout | hpg_hier_junction | 0 | 2 | 1 | True |
| monomer_heldout | hpg_hier_octamer | 0 | 2 | 1 | True |
| monomer_heldout | wdmpnn | 0 | 2 | 1 | True |
| monomer_b_heldout_clustered | hpg_hier | 0 | 1 | 1 | n/a |
| monomer_b_heldout_clustered | hpg_hier | 1 | 1 | 1 | n/a |
| monomer_b_heldout_clustered | hpg_hier | 2 | 1 | 1 | n/a |
| monomer_b_heldout_clustered | wdmpnn | 0 | 1 | 1 | n/a |

### C.2 R3 held-out B identity and fold sizes
| model | fold | n_train | n_val | n_test | sizes_match_meta | sets_disjoint | test_b_in_held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | 0 | 33390 | 4788 | 4788 | True | True | True |
| hpg_hier | 1 | 33390 | 4788 | 4788 | True | True | True |
| hpg_hier | 2 | 33390 | 4788 | 4788 | True | True | True |
| wdmpnn | 0 | 33390 | 4788 | 4788 | True | True | True |

### C.3 Frozen-split assertion execution
The frozen-split assertion is implemented in `scripts/python/frozen_splits.py`. Its runtime log line is not present in the downloaded artifacts (task logs were not pulled), so direct confirmation is **not available here**. Indirect evidence is that the split indices and held-out B identities reproduce the metadata exactly (see C.2).

## D. Early diagnostic reads

### D.1 Final vs best checkpoint MAE gap (eV)
| split | model | fold | seed | best_mae | final_mae | final_minus_best_mae |
| --- | --- | --- | --- | --- | --- | --- |
| monomer_heldout | hpg_hier | 0 | 42 | 0.086397 | 0.083024 | -0.003373 |
| monomer_heldout | hpg_hier | 0 | 43 | 0.070503 | 0.089173 | 0.018670 |
| monomer_heldout | hpg_hier_junction1 | 0 | 42 | 0.080781 | 0.070670 | -0.010112 |
| monomer_heldout | hpg_hier_junction1 | 0 | 43 | 0.044299 | 0.044342 | 0.000043 |
| monomer_heldout | hpg_hier_junction | 0 | 42 | 0.054050 | 0.198353 | 0.144303 |
| monomer_heldout | hpg_hier_junction | 0 | 43 | 0.087954 | 0.074759 | -0.013195 |
| monomer_heldout | hpg_hier_octamer | 0 | 42 | 0.083275 | 0.055008 | -0.028267 |
| monomer_heldout | hpg_hier_octamer | 0 | 43 | 0.125125 | 0.058451 | -0.066674 |
| monomer_heldout | wdmpnn | 0 | 42 | 0.198623 | 0.216471 | 0.017848 |
| monomer_heldout | wdmpnn | 0 | 43 | 0.192009 | 0.180016 | -0.011993 |
| monomer_b_heldout_clustered | hpg_hier | 0 | 42 | 0.069933 | 0.073809 | 0.003875 |
| monomer_b_heldout_clustered | hpg_hier | 1 | 42 | 0.053236 | 0.051007 | -0.002229 |
| monomer_b_heldout_clustered | hpg_hier | 2 | 42 | 0.046938 | 0.068286 | 0.021348 |
| monomer_b_heldout_clustered | wdmpnn | 0 | 42 | 0.095968 | 0.097340 | 0.001372 |

**Mean final - best MAE across 14 pilots: 0.005115 eV.**

- monomer_b_heldout_clustered: mean gap 0.006092, max 0.021348
- monomer_heldout: mean gap 0.004725, max 0.144303


### D.2 Best epochs
| split | model | fold | seed | best_epoch | patience |
| --- | --- | --- | --- | --- | --- |
| monomer_heldout | hpg_hier | 0 | 42 | 40 | 15 |
| monomer_heldout | hpg_hier | 0 | 43 | 5 | 15 |
| monomer_heldout | hpg_hier_junction1 | 0 | 42 | 23 | 15 |
| monomer_heldout | hpg_hier_junction1 | 0 | 43 | 15 | 15 |
| monomer_heldout | hpg_hier_junction | 0 | 42 | 41 | 15 |
| monomer_heldout | hpg_hier_junction | 0 | 43 | 27 | 15 |
| monomer_heldout | hpg_hier_octamer | 0 | 42 | 57 | 15 |
| monomer_heldout | hpg_hier_octamer | 0 | 43 | 5 | 15 |
| monomer_heldout | wdmpnn | 0 | 42 | 36 | 15 |
| monomer_heldout | wdmpnn | 0 | 43 | 37 | 15 |
| monomer_b_heldout_clustered | hpg_hier | 0 | 42 | 33 | 15 |
| monomer_b_heldout_clustered | hpg_hier | 1 | 42 | 17 | 15 |
| monomer_b_heldout_clustered | hpg_hier | 2 | 42 | 14 | 15 |
| monomer_b_heldout_clustered | wdmpnn | 0 | 42 | 83 | 15 |
Best epochs range 5–83, mean 30.9. Patience is 15; clustering around early epochs is noted but does not block the pilot.

### D.3 R1 pilot vs pre-regen counterparts
| model | seed | old_file | new_group_mean_r2 | old_group_mean_r2 | delta_group_mean_r2 | new_overall_r2 | old_overall_r2 | new_mae | old_mae | new_delta_r2 | old_delta_r2 | new_ordering | old_ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold0__s42.npz | 0.957564 | 0.925337 | 0.032228 | 0.955941 | 0.924113 | 0.086397 | 0.106807 | 0.805296 | 0.733561 | 0.780710 | 0.744542 |
| hpg_hier | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold0__s42.npz | 0.975137 | 0.925337 | 0.049800 | 0.969954 | 0.924113 | 0.070503 | 0.106807 | 0.568077 | 0.733561 | 0.716194 | 0.744542 |
| hpg_hier_junction1 | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction1__monomer_heldout__fold0__s42.npz | 0.964147 | 0.988892 | -0.024745 | 0.962340 | 0.986239 | 0.080781 | 0.046449 | 0.781119 | 0.785473 | 0.790323 | 0.753177 |
| hpg_hier_junction1 | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction1__monomer_heldout__fold0__s42.npz | 0.987664 | 0.988892 | -0.001228 | 0.985018 | 0.986239 | 0.044299 | 0.046449 | 0.776579 | 0.785473 | 0.809058 | 0.753177 |
| hpg_hier_junction | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction__monomer_heldout__fold0__s42.npz | 0.982000 | 0.915314 | 0.066686 | 0.980332 | 0.912206 | 0.054050 | 0.118499 | 0.796493 | 0.771767 | 0.773053 | 0.788042 |
| hpg_hier_junction | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier_junction__monomer_heldout__fold0__s42.npz | 0.953630 | 0.915314 | 0.038316 | 0.951152 | 0.912206 | 0.087954 | 0.118499 | 0.780769 | 0.771767 | 0.810362 | 0.788042 |
| hpg_hier_octamer | 42 | ea_ip__EA_vs_SHE_eV__hpg_hier_octamer__monomer_heldout__fold0__s42.npz | 0.959801 | 0.994756 | -0.034956 | 0.958205 | 0.991382 | 0.083275 | 0.035454 | 0.779416 | 0.708146 | 0.764255 | 0.744053 |
| hpg_hier_octamer | 43 | ea_ip__EA_vs_SHE_eV__hpg_hier_octamer__monomer_heldout__fold0__s42.npz | 0.921815 | 0.994756 | -0.072941 | 0.919723 | 0.991382 | 0.125125 | 0.035454 | 0.698615 | 0.708146 | 0.753666 | 0.744053 |
| wdmpnn | 42 | ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_heldout__fold0__s42.npz | 0.809215 | 0.760135 | 0.049081 | 0.807364 | 0.759817 | 0.198623 | 0.226863 | 0.581887 | 0.683631 | 0.790486 | 0.795536 |
| wdmpnn | 43 | ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_heldout__fold0__s42.npz | 0.825769 | 0.760135 | 0.065634 | 0.822620 | 0.759817 | 0.192009 | 0.226863 | 0.605592 | 0.683631 | 0.792766 | 0.795536 |

_This is a sanity check on direction and magnitude only; single-seed pilot runs are not interpreted as results._


### D.4 R3 group-mean R2 vs per-fold B-blind null floor
| model | fold | seed | group_mean_r2 | overall_r2 | mae | delta_r2 | null_floor | above_floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | 0 | 42 | 0.949808 | 0.948139 | 0.069933 | 0.779445 | 0.541910 | True |
| hpg_hier | 1 | 42 | 0.979869 | 0.978560 | 0.053236 | 0.870380 | 0.383520 | True |
| hpg_hier | 2 | 42 | 0.981602 | 0.980078 | 0.046938 | 0.751778 | 0.747190 | True |
| wdmpnn | 0 | 42 | 0.928941 | 0.926752 | 0.095968 | 0.711030 | 0.541910 | True |

All R3 pilot group-mean R² values are above their per-fold B-blind null floor.
### D.5 PBS task logs

R3 pilot task logs must be downloaded alongside the NPZs to confirm the frozen-split assertion executed. Use `scripts/shell/download_regen_v1_artifacts.sh` after the pilot finishes, then grep the logs for markers such as `Frozen monomer_b_heldout split assertions passed for all folds`, `differs from frozen metadata`, `B-identity leakage`, or `frozen_protocol`.

## E. Budget

- Mean wall time per pilot job: 1.56 h
- Max wall time per pilot job: 3.74 h
- Jobs remaining: R1 = 260, R3 = 212
- Revised GPU-hour estimate (mean walltime basis): R1 405 h, R3 330 h, total 734 h
- Revised GPU-hour estimate (max walltime basis, conservative): R1 972 h, R3 793 h, total 1765 h

## F. Verdict

### R1 remaining 260 jobs: **GO**
- No blocking issues detected in the pilot.

### R3 remaining 212 jobs: **GO**
- No blocking issues detected in the pilot.

### Warnings
- 1 run(s) have final-minus-best MAE gap > 0.05 eV; largest is 0.144 eV
- 2 run(s) reached best epoch ≤ 5 (minimum early-stopping plateau)
- 1 R1 pilot run(s) have new group-mean R² > 0.05 below the pre-regen counterpart
