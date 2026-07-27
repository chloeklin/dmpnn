# HPG-hier training stability

## Decision

The six fixed-seed Gadi V100 runs establish material training variability, but the retained artifacts do **not** prove that short early-stopped runs simply stop before convergence. Epoch count is not monotone with test quality. The stronger diagnosis is unstable model selection under a validation protocol that uses one entire donor monomer, compounded by predicting from the final patience-expired model rather than restoring the best checkpoint.

No B-heldout job or model variant should run until a stability intervention reduces fold-1 MAE SD below 0.02 eV. No GPU job was submitted during this analysis.

## Step A — diagnosis from existing artifacts

### Runtime environment

All six artifacts report CUDA on `Tesla V100-SXM2-32GB`, NVIDIA driver `580.126.20`, torch `2.8.0+cu128`, torch CUDA `12.8`, cuDNN `91002`, deterministic cuDNN requested `True`, global deterministic algorithms enabled `False`, cuDNN deterministic `True`, and cuDNN benchmark `False`.

### Training duration, validation minimum, and test performance

Validation loss is MSE in normalized-target space, not eV². Test metrics are in physical units.

| fold | repeat | epochs run | best validation loss | final test MAE (eV) | final group-mean R² | wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 38 | 0.00681657 | 0.083809 | 0.962021 | 4,283.85 |
| 0 | 2 | 28 | 0.00741245 | 0.054591 | 0.982422 | 3,133.99 |
| 0 | 3 | 43 | 0.00513576 | 0.052108 | 0.985506 | 4,718.20 |
| 1 | 1 | 47 | 0.00743890 | 0.145610 | 0.789786 | 5,214.36 |
| 1 | 2 | 27 | 0.01058103 | 0.226356 | 0.449702 | 2,866.05 |
| 1 | 3 | 40 | 0.00904610 | 0.044598 | 0.978154 | 4,358.52 |

Wall time is almost entirely explained by epoch count: Pearson `r = 0.9969`, Spearman `ρ = 1.0`, approximately 112.75 seconds per additional epoch. This confirms that the wall-time spread reflects different stopping epochs rather than different hardware.

There is **no monotone epoch-performance relationship**:

- Across all six runs, epochs versus MAE has Spearman `ρ = -0.314`; epochs versus group-mean R² has `ρ = +0.257`.
- In fold 0, the 28-epoch run is much better than the 38-epoch run, while the 43-epoch run is best.
- In fold 1, the 27-epoch run is worst and the 40-epoch run is best, but extending to 47 epochs is worse again.
- Within each fold, the three-point Spearman coefficient is only `-0.5` for epochs versus MAE. Three observations do not support inference, but they directly refute a simple “more epochs is always better” account.
- Best validation loss also does not rank test performance reliably. Fold-1 repeat 3 has a worse validation minimum than repeat 1 (`0.00905` versus `0.00744`) but a far better test MAE (`0.0446` versus `0.1456` eV).

Therefore, the short fold-1 run is bad, but short runs are not uniformly bad. The evidence does not establish that patience merely fires before convergence.

### Validation-loss curves and anomalies

The six production runs used `logger=False` and retained only `epochs_actually_run` and the scalar `best_val_loss` in each sidecar. Their checkpoints and PBS stdout/Lightning progress logs are not present on this disk. Consequently:

- Full validation-loss curves cannot be reconstructed.
- Whether loss was still descending when patience fired cannot be tested.
- Epoch-to-epoch validation-loss variance cannot be quantified.
- The exact best epoch is unavailable. With patience 15, the best monitored epoch must precede the stopping point by the final no-improvement window, but it cannot be recovered exactly from the sidecar.
- No run has a non-finite recorded best loss, prediction, or metric, and all six completed normally. There is no positive evidence of NaNs or terminal loss blow-ups.
- Transient loss spikes and gradient explosions cannot be ruled out because neither gradient norms nor full loss histories were retained.

A critical source-level finding is that `ModelCheckpoint(save_top_k=1)` saved the best validation checkpoint, but prediction was called on the in-memory **final model** after patience expired. The best checkpoint was not restored. The reported test score therefore corresponds to a model approximately one patience window beyond its validation optimum, not to the model with `best_val_loss`. This can amplify sensitivity to the exact stopping trajectory.

Future stability runs now retain the complete validation-loss curve, best epoch, selected prediction checkpoint, and existing environment provenance.

### What the validation set is

The A-heldout split has nine fixed folds. In every fold:

- Test is all 4,774 rows for one held-out A monomer.
- Validation is all 4,774 rows for a second A monomer.
- Training contains 33,418 rows from the remaining seven A monomers.
- Train, validation, and test A identities are disjoint.
- The validation identity is chosen deterministically with NumPy RNG seed `42 + fold`; it is fixed across all repeats.

For the measured folds:

| fold | test A monomer | validation A monomer | validation rows | fixed across repeats? |
| --- | --- | --- | ---: | --- |
| 0 | `CC1(C)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21` | `O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21` | 4,774 | yes |
| 1 | `O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21` | `OB(O)c1ccc(-c2ccc(B(O)O)s2)s1` | 4,774 | yes |

The validation loss is not noisy because it is estimated from few rows; 4,774 rows are evaluated every epoch. It is “noisy” in the model-selection sense: one monomer is a high-leverage, monomer-specific proxy for performance on a different unseen monomer, while A identity carries most target variance. The weak relationship between validation minimum and test quality is consistent with that mismatch.

### Primary hypothesis verdict

**Partially supported, not proven.** The validation signal is demonstrably single-monomer and monomer-specific, and it selects trajectories inconsistently with test performance. However, absent curves, the existing artifacts cannot show that validation loss remains descending or quantify epoch noise when patience fires. The lack of monotone epoch-performance behavior argues against the narrower claim that early stopping simply terminates before convergence. The final-versus-best checkpoint mismatch is an additional concrete mechanism.

## Step B — candidate fixes ranked

Every intervention changes training/model selection and therefore invalidates direct comparability with existing seed-42 outputs. If adopted, all baseline and variant cells used in a comparison must be regenerated under the same stabilized protocol.

| rank | intervention | expected variance effect | implementation and GPU cost | scientific/comparability assessment |
| ---: | --- | --- | --- | --- |
| 1 | Keep the current single-monomer validation and restore the best checkpoint | Medium to high. It directly fixes the known final-after-patience prediction bug without changing the split. | Minimal code change and one additional test-set inference pass for diagnosis; training cost remains the current 27–47 epochs. | Preferred if fold-1 MAE SD falls below 0.02 eV because it preserves the current validation design and avoids a new paper-level justification. Existing outputs still require regeneration because their predictions used the final model. |
| 2 | Stratified random row-level validation from the eight non-test A monomers and restore the best checkpoint | High. It fixes checkpoint selection and replaces one high-leverage validation monomer with an IID mixture spanning every available training A. | Moderate code change; roughly current training cost. Validation uses about 10% of each non-test A group. | Use only if rank 1 fails and this arm succeeds. It is defensible as “leave-one-A-out OOD testing with IID validation,” but it changes the validation claim and invalidates direct comparability. |
| 3 | Fixed 60-epoch budget, no early stopping, predict the final epoch | High for stopping-time variance; uncertain for optimization-path variance. | Low implementation cost; about 1.9 V100-hours per run from the observed epoch rate. | Fallback only if neither Step-C arm reaches the target. It changes training/model selection and requires regeneration. |
| 4 | Average several checkpoints | Medium. Averaging may smooth trajectory noise beyond selecting one checkpoint. | Additional checkpoint storage and inference/implementation work. | The averaging window and weighting must be pre-registered; unavailable historical curves do not identify them. |
| 5 | Longer patience, a minimum-epoch floor, or an Adam LR schedule | Medium but uncertain. These may suppress late oscillation but do not isolate the known checkpoint bug. | Low-to-moderate code cost and potentially higher GPU cost. | Defer until the isolated checkpoint test and validation test are resolved. |

The decisive test separates checkpoint restoration from validation redesign. This prevents adopting row-level validation, and its additional scientific justification, if the implementation bug alone explains the instability.

## Step C — decisive 12-run test

The two selected arms are:

1. Arm A, `best_checkpoint`: retain the current single-monomer validation, early stopping, patience 15, and all other settings; predict from the best validation checkpoint.
2. Arm B, `row_val_best`: stratified 10% row-level validation from each of the eight non-test A monomers, early stopping with patience 15, and prediction from the best validation checkpoint.

For each arm: EA folds 0 and 1 × repeats 1–3 × seed 42 on Gadi V100. This is exactly 12 new GPU runs. The current six runs are the unchanged comparator. Fixed 60 epochs is not part of Step C and remains a fallback only if neither arm succeeds. Full validation curves and exact best epochs will be retained.

| configuration | fold 0 MAE SD | fold 0 group-mean R² SD | fold 0 ΔR² SD | fold 1 MAE SD | fold 1 group-mean R² SD | fold 1 ΔR² SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current final model | 0.017630 | 0.012762 | 0.033771 | 0.091067 | 0.267831 | 0.082760 |
| Arm A: current validation + best checkpoint | pending | pending | pending | pending | pending | pending |
| Arm B: row-level validation + best checkpoint | pending | pending | pending | pending | pending | pending |

**Success criterion:** fold-1 MAE SD below 0.02 eV. If Arm A succeeds, retain the existing validation design regardless of Arm B. If Arm A fails and Arm B succeeds, adopt row-level validation with an explicit methods justification. If neither succeeds, test fixed 60 epochs next.

The implementation and analysis are prepared in `scripts/python/run_hpg_generalization.py`, `scripts/shell/generate_hpg_stability_fixes.sh`, and `scripts/python/analyze_hpg_stability_fixes.py`. The generator produces six jobs per arm and asserts a total of 12. It does not submit them. No B-heldout or variant job is included.

### Best-checkpoint bug cost

Every Step-C run performs two test-set inference passes after the same training trajectory: once from the final patience-expired model and once after restoring the best checkpoint. The primary bug-cost statistic is

`final-model test MAE − best-checkpoint test MAE`.

Positive values mean the historical final-model behavior was worse. Report all 12 paired gaps by arm, fold, and repeat, plus mean, median, SD, minimum, and maximum. Arm A is the direct estimate for the existing single-monomer-validation protocol; Arm B is supplementary. This same-run comparison isolates checkpoint choice from CUDA trajectory variation. Historical runs cannot be repaired retrospectively because their checkpoints were not retrieved.

## Step D — implications

### Repeats needed for the nine-fold paired sign test

The headline test is a paired two-sided sign test over nine fold-level model differences, not a test that estimates each cell to ±0.03 eV. Repeats reduce the probability that run noise flips each fold's difference; the nine folds then provide the test replication.

Assume every fold has a true same-sign effect `δ`, each model has independent per-run SD `s`, and each fold/model cell averages `r` runs. The noisy paired difference has SD `sqrt(2) × s / sqrt(r)`, so the probability that one fold has the correct sign is

`p = Φ(δ × sqrt(r) / (sqrt(2) × s))`.

For nine folds, the exact two-sided α = 0.05 sign test rejects when at least 8 of 9 fold differences have one sign. Its power for a positive effect is

`P(K ≥ 8) + P(K ≤ 1)`, where `K ~ Binomial(9, p)`.

For `δ = 0.03 eV`, the exact minimum `r` reaching at least 80% power is:

| post-fix per-model per-run SD `s` (eV) | runs per model/fold cell | exact power | jobs for 144 cells |
| ---: | ---: | ---: | ---: |
| 0.005 | 1 | >0.999999 | 144 |
| 0.010 | 1 | 0.990 | 144 |
| 0.015 | 1 | 0.846 | 144 |
| 0.020 | 2 | 0.883 | 288 |
| 0.030 | 4 | 0.846 | 576 |

The earlier 432/576/1,008 estimates answered the wrong question and are withdrawn. Under these assumptions, one run per cell is adequately powered for a common 0.03 eV effect when stabilized per-run SD is at most 0.015 eV; SD 0.02 eV needs two runs, and SD 0.03 eV needs four.

This calculation is optimistic if effects vary strongly in magnitude or sign across folds, if fold errors are correlated, or if baseline and variant SDs differ. Conversely, paired common random numbers may reduce difference noise. After Step C, use the selected arm's per-run SD for planning, then report the empirical fold signs and exact sign-test result rather than treating this normal-noise calculation as evidence.

### Other pathological folds

EA fold 6 and IP fold 5 must be repeat-tested before any junction-coupling conclusion stands. After selecting one stabilized protocol:

- Run baseline, junction n=1, and junction n=2 on EA fold 6 and IP fold 5.
- Use at least three independent repeats per model/cell with the same fixed data split and seed protocol as Step C.
- Pair comparisons by fold and repeat label, report the distribution and SD of MAE, group-mean R², ΔR², and ordering differences, and retain curves/provenance.
- This is 18 runs for the two folds. IP fold 2 should be the next priority because it also drives a claimed coupling rescue.
- Do not interpret chemistry mechanisms unless the between-model effect exceeds the paired ±2 SD band and has consistent sign across repeats.

### Existing conclusions invalidated at the measured variance scale

The following sections of `variant_results_report.md` are invalid or provisional if this scale persists after stabilization:

- **§1 Headline:** single-run claims that HPG-hier leads OOD architecture recovery require repeated baselines and competitors; fold-1 ΔR² SD is 0.08276.
- **§3 Main three-way comparison:** all seed-42 HPG-hier/wDMPNN/ChemArch rankings lack matched repeat uncertainty.
- **§4 Complementarity/ensemble:** the 0.003 eV EA ensemble margin is negligible relative to measured variability.
- **§5.1 Wedge:** the dropped/retained verdict rests on unmatched single runs and must be repeated if the variant is reconsidered.
- **§5.2 Junction coupling:** depth rankings, EA fold-1 “rescue,” EA fold-6 collapse, and IP fold-5 collapse are not distinguishable from run variance without repeats. The physical rationale built on fold 1 is unsupported.
- **§5.3 Octamer:** the EA chemistry-placement claim is not established. Its largest group-mean R² contribution is fold 1 (`+0.414`), where baseline group-mean R² spans `0.450–0.978`; 8 of 9 recorded EA MAE differences lie inside the current conservative ±2 SD band.
- **§5.4 Fold-1 chemistry weakness:** the “hard monomer” and missing cross-junction-conjugation diagnoses are invalid. The baseline itself can reach group-mean R² `0.978`, comparable to the octamer's `0.989`, without a model change.
- **§6 Variant status:** labels such as dropped, mechanism-only, or real/mis-attributed cannot be final until matched repeat distributions exist.

Dataset-design facts in §2 and artifact-provenance checks remain valid. They establish what was evaluated, not that single-run model differences are stable.

## Stop condition

Do not submit the 144 B-heldout cells and do not start a new model variant. Run only the 12 pre-registered stability jobs. Adopt a fix only if fold-1 MAE SD is below 0.02 eV, then recompute the repeat budget before any B-heldout launch.
