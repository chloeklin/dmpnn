# F2 — Worked example (selected example)

## Prediction files loaded
`predictions/regen_v1/ea_ip_lomo/ea_ip__[EA|IP]_vs_SHE_eV__[hpg_hier_octamer|wdmpnn]__monomer_heldout__fold[0-8]__s[42/43/44].npz`
Seeds 42,43,44 averaged at the prediction level; metric computed once on averaged predictions.
Both models, both targets (EA, IP), all 9 folds used for pool construction. Figure rendered for EA.

## Selection criterion (selected example)
Candidate pool: fracA==0.5 groups with all 3 poly_type values present in the test fold (682 per fold, asserted).
Eligibility: `|gm_err_octamer - gm_err_wdmpnn| <= 0.01` (chemistry placement tied)
AND `ordering_octamer == 1.0` AND `ordering_wdmpnn == 0.0`.
Among eligible group-folds: selected group with largest true architecture spread (max - min y_true).

## Selected group
- smiles_A: `OB(O)c1cc(F)c(B(O)O)cc1F`
- smiles_B: `Brc1cc(Br)c2cc[nH]c2c1`
- fracA: 0.5
- fold: 2
- true spread: 0.33250 eV  (rank 1 of 61 in EA eligible pool)
- gm_err octamer: 0.05004 eV
- gm_err wDMPNN:  0.04623 eV

## Per-fold eligible-pool counts
| target | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA | 0 | 10 | 12 | 2 | 6 | 12 | 14 | 1 | 4 | 61 |
| IP | 0 | 19 | 5 | 1 | 1 | 0 | 25 | 2 | 1 | 54 |

## Fold-0 marginals (single-model; fold 0 only)
| target · model | ordering==0.0 | mean ordering | gm_err decile thr (eV) | joint w/ decile | expected if independent |
| --- | --- | --- | --- | --- | --- |
| EA · octamer | 69 (10.1%) | 0.814 | 0.01806 | 1 | 6.90 |
| EA · wdmpnn | 43 (6.3%) | 0.858 | 0.09580 | 7 | 4.30 |
| IP · octamer | 27 (4.0%) | 0.886 | 0.04317 | 6 | 2.70 |
| IP · wdmpnn | 42 (6.2%) | 0.822 | 0.10909 | 10 | 4.20 |

**Do not quote the conjunction (joint) alone as the failure rate.** The decile condition
selects 10% of groups by definition (§3 marginals note). The ordering-failure rate is the
relevant quantity describing how often a model fails to rank architecture.

## Median ordering-failure rate across all 9 folds
| model | EA | IP |
| --- | --- | --- |
| octamer | 6.5% | 6.6% |
| wDMPNN  | 8.8% | 8.1% |

EA fold 4 is an outlier: octamer 35.6% failure / mean ordering 0.560,
wDMPNN 39.7% / 0.524 — both near chance. Fold not excluded.

## Missing files
None

## Architecture-spread recovery (fracA=0.5, arch3 groups)

### Selected-group arch_spread_ratio_predavg
`arch_spread_ratio_predavg` = pred_spread / true_spread, computed from the **three-seed
prediction average** (seeds 42/43/44 averaged at the prediction level before scoring).
Distinct from the per-run `arch_spread_ratio_arch3` that appears in
`_regen_v1_results_individual_runs.csv`, which is computed from a single seed's predictions.
Those per-run values are not interchangeable with these and must not be quoted as one number.

| model | true spread (eV) | pred spread (eV) | arch_spread_ratio_predavg |
| --- | --- | --- | --- |
| hpg_hier_octamer | 0.33250 | 0.32904 | 0.9896 |
| wdmpnn | 0.33250 | 0.02400 | 0.0722 |

wDMPNN does not rank the three architectures in the wrong order so much as predict
nearly the same value for all three — it recovers 7% of the true
architecture range on this group, so its ordering is the sign of residual noise.

**Strata note (two-sided):** All spread statistics below are for fracA=0.5 groups with
exactly 3 poly_types present in the test fold. A range over 3 points and a range over
2 points are not the same quantity; the two strata are never pooled.

### Per-fold median arch_spread_ratio (fracA=0.5, arch3, folds 0-8)
| target · model | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA · octamer | 0.649 | 0.747 | 0.929 | 0.689 | 0.811 | 0.833 | 1.559 | 0.713 | 0.768 |
| EA · wdmpnn | 0.775 | 0.247 | 0.378 | 0.526 | 0.909 | 0.582 | 0.610 | 1.390 | 0.616 |
| IP · octamer | 0.570 | 0.947 | 0.927 | 0.904 | 1.162 | 0.738 | 0.967 | 1.118 | 1.034 |
| IP · wdmpnn | 0.467 | 1.484 | 1.074 | 1.219 | 0.968 | 0.265 | 0.350 | 0.229 | 0.551 |

### Per-fold collapse rate (ratio < 0.25, fracA=0.5, arch3, folds 0-8)
| target · model | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA · octamer | 18.5% | 11.7% | 9.8% | 9.5% | 12.6% | 6.3% | 10.9% | 7.2% | 16.7% |
| EA · wdmpnn | 6.5% | 50.0% | 33.4% | 12.5% | 4.1% | 6.2% | 15.4% | 1.2% | 14.2% |
| IP · octamer | 16.0% | 7.8% | 10.1% | 9.1% | 12.6% | 11.9% | 6.7% | 6.6% | 8.5% |
| IP · wdmpnn | 11.6% | 7.8% | 2.8% | 1.6% | 8.8% | 44.3% | 35.3% | 52.3% | 7.5% |

Octamer collapse rates are asserted within 6.3-18.5% on EA and 6.6-16.0% on IP.
wDMPNN spot checks: EA fold 1 = 50.0%,
IP fold 5 = 44.3%,
IP fold 7 = 52.3%.

## Cells: 1 group × 2 models × 3 poly_types × 2 panels = 12 plotted series endpoints.