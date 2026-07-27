# Windsurf spec — full regeneration under the frozen protocol

**Why.** Every existing prediction file was produced by a runner that predicted from the final
patience-expired model rather than the best checkpoint. The defect was present in each runner's first
commit, so no artifact predates it. Measured cost for HPG-hier: mean +0.048 eV MAE (median +0.038,
max +0.114). Separately, three identical runs of EA fold 1 gave group-mean R² of 0.450 / 0.790 / 0.978,
so single-run results are not interpretable. Prediction averaging across repeats improved every metric
in every cell tested (6/6 MAE, 24/24 overall).

Everything is regenerated once, under one protocol, and never compared across protocols again.

---

## 1. The frozen protocol — do not vary any of this

| element | setting | rationale |
|---|---|---|
| Validation design | **unchanged** — one held-out A monomer (A-split) / the cyclic next B fold (B-split) | Arm B's row-level validation regressed fold 0 and requires a methodological defence; not adopted |
| Model selection | **best checkpoint**, always, all runners | the bug fix |
| Replicates | **3 runs per cell at seeds 42, 43, 44** | see §1.1 |
| Primary prediction | **row-wise mean of the 3 seeds' `y_pred`** | averaging fixed the instability empirically |
| Diagnostic | `y_pred_final` retained per run | measures what the bug cost each model family |
| Early stopping | patience 15, no min-epoch floor (current settings) | Arm C dropped — premise unsupported, averaging solved the problem |

### 1.1 Use seeds 42/43/44, not three repeats at seed 42

The stability work used repeated runs at a fixed seed. **For regeneration, vary the seed instead.**
Same cost, and it is strictly better:

- it captures initialisation variance as well as nondeterminism,
- it is what the error bars in the paper need,
- "we report the mean of three seeds" is a normal sentence; "we report the mean of three identical
  runs that differed due to GPU nondeterminism" is not.

This also discharges the outstanding seeds-43/44 item. **The split must not depend on the model seed** —
folds must be byte-identical across all three seeds, asserted against the frozen metadata as already
implemented.

### 1.2 Every model gets the same treatment

The 3-seed average is applied to **every** model including wDMPNN, ChemArch, GlobalArch and frac.
Any comparison where one model is averaged and another is not is invalid. State the convention
explicitly wherever results are reported: *"all figures are the mean prediction of three seeds."*

---

## 2. Run matrix, in priority order

| stage | split | models | cells | runs | ~GPU h |
|---|---|---|---|---|---|
| **R1** | A-heldout | hpg_hier, wdmpnn, hpg_hier_octamer, hpg_hier_junction (n=2), hpg_hier_junction1 | 90 | 270 | ~300 |
| **R2** | B-heldout random | hpg_hier, wdmpnn, hpg_hier_octamer, hpg_hier_junction (n=2) | 72 | 216 | ~240 |
| **R3** | B-heldout clustered | same four | 72 | 216 | ~240 |
| **R4** | A-heldout | chemarch, globalarch, frac | 54 | 162 | ~180 |

R1 restores the existing results on trustworthy footing. R2 is the new science. R4 is needed only
because §3 of `variant_results_report.md` cites ChemArch and GlobalArch; if that section is cut, drop R4.

Submit R1 first and **stop for review before R2** — R1 will show whether any of the existing
conclusions survive, which may change what is worth running.

---

## 3. Correctness requirements

1. **Fresh checkpoint and prediction directories for everything.** wDMPNN and the Stage-2D runners skip
   training entirely when they find `TRAINING_COMPLETE` and load the *old* run's checkpoint. Pointing
   regeneration at existing directories would silently return the results being replaced, with no
   visible error. Use a new root, e.g. `predictions/regen_v1/` and `checkpoints/regen_v1/`.
2. **Verify runs actually trained.** After the first ~10 jobs complete, check the sidecars for non-zero
   epochs and plausible wall time. Report the check before letting the rest of the array run.
3. **Split assertions active** on every run — regenerated folds must match the frozen metadata, and
   must be identical across seeds 42/43/44. Fail loudly.
4. **Provenance sidecar per run**: resolved config, git SHA, PBS job ID, seed, epochs run, best epoch,
   best val loss, wall time, accelerator/device/driver/torch/CUDA versions, and both
   `prediction_checkpoint` and `final_prediction_checkpoint` (path + SHA-256).
5. **Both arrays saved**: `y_pred` (best checkpoint, primary) and `y_pred_final` (diagnostic).
6. Idempotent and resumable; skip completed cells; one NPZ per (model, target, fold, seed).

---

## 4. Analysis outputs

Produce `analysis/model_diagnostics/_regen_v1_results.md`. Use the existing metric module — do not
reimplement — and verify it reproduces the old seed-42 numbers before trusting it on new data.

### 4.1 Per cell
- Metrics computed on the **3-seed averaged prediction**: group-mean R², ΔR², pairwise ordering,
  overall R², MAE, mean signed bias. These are the headline numbers.
- Separately, the **mean and SD across the 3 individual seeds** for each metric. The SD is the error
  bar and must appear in every comparison table.

### 4.2 Comparisons
- Paired per-fold comparisons against `hpg_hier`: signed differences, wins/losses, exact two-sided
  sign test across the 9 folds, **Holm-corrected across the comparison family**. State that the
  minimum attainable two-sided p with 9 folds is 0.0039.
- Headline differences as the **median of paired per-fold differences** — never the difference of
  separately ranked medians.
- Flag any difference smaller than the measured per-fold SD as not interpretable.

### 4.3 Required context columns
- **Every group-mean R² reported next to its null floor** from `_dataset_design_audit.md`
  (A-blind for the A-split, B-blind for the B-split). A model that fails to beat its floor is reported
  as failing to beat it.
- Pooled group-mean R², fold-placement R²/slope/intercept, fold-bias SD, within-fold compression ratio.
- **B-split only:** metrics reported twice — full folds, and filtered to exclude held-out B monomers
  with max training Tanimoto ≥ 0.95. Plus the **performance-versus-novelty curve** (bin held-out B
  monomers by max Tanimoto: <0.35, 0.35–0.45, 0.45–0.55, 0.55–0.7, >0.7; report per-bin MAE,
  group-mean R² and ΔR² for every model).

### 4.4 Checkpoint-gap report
Run `analyze_checkpoint_mae_gap.py` over the regenerated tree. Report `final MAE − best MAE` per model
family, mean and SD across all cells, and explicitly compare HPG-hier's mean gap to wDMPNN's. Smoke
runs suggested HPG-hier was damaged far more than wDMPNN and ChemArch (+0.006 vs −0.0001 and +0.0001);
if that holds at full training, the previous comparisons understated HPG-hier and that belongs in the
writeup.

### 4.5 Fold-variance check
Report the 3-seed SD **per fold**. The folds previously treated as chemically interesting — EA 1, EA 6,
IP 5, IP 2 — are under suspicion of simply being high-variance. State for each whether its SD is
elevated relative to the others.

---

## 5. Verification before the results are used

1. The metric module reproduces the old seed-42 numbers in `_phase1_metrics_scratch.md` and
   `_groupmean_metric_floor.md` to 5 dp on the old files. If any number differs, stop and report it
   rather than adjusting the script.
2. Folds are byte-identical across seeds 42/43/44 for at least one model/target, demonstrated.
3. A spot-check that at least one regenerated NPZ differs from its predecessor — if regenerated files
   match the old ones exactly, training was skipped and the run is invalid.

## 6. Out of scope — do not do these

- Do not run Arm C or any further stability arm.
- Do not adopt row-level validation.
- Do not create new model variants.
- Do not edit `variant_results_report.md`; results go in `_regen_v1_results.md` and the report is
  revised separately once R1 is reviewed.
