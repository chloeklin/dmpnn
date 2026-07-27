# Windsurf prompt — diagnose and fix training instability before any further experiments

**Context.** Six Gadi V100 runs of `hpg_hier`, EA, A-heldout, seed 42, current code, varying only the
repeat label, produced:

| fold | group-mean R² across 3 repeats | MAE across 3 repeats | MAE SD |
|---|---|---|---|
| 0 | 0.962 / 0.982 / 0.986 | 0.084 / 0.055 / 0.052 | 0.018 |
| 1 | **0.790 / 0.450 / 0.978** | 0.146 / 0.226 / 0.045 | **0.091** |

Wall times ranged 2,866–5,214 s, so runs are terminating at very different points. A MAE SD of
0.091 eV exceeds every effect this project has reported. **No model comparison is possible until this
is reduced.** Do not run the B-heldout cells or any variant until it is.

Write findings to `analysis/model_diagnostics/_training_stability.md`. Do not edit report files.

---

## Step A — diagnose from artifacts already on disk (no GPU)

From the six provenance sidecars and any Lightning logs retained:

1. **Epochs actually run** and **best validation loss** per run, tabulated against final test MAE and
   group-mean R². Is there a monotone relationship between epochs-run and test performance? If the
   short runs are the bad ones, early stopping is terminating before convergence.
2. **The full validation-loss curve per run**, if logged. Report whether validation loss is still
   descending when patience fires, and how noisy it is epoch-to-epoch.
3. **What the validation set actually is.** Confirm the A-heldout generator's val set is one whole
   held-out A monomer (n = 4,774). State which monomer is used for each fold and whether it is fixed.
4. Whether any run shows loss spikes, NaNs, or gradient blow-ups.

**Primary hypothesis to test:** validation loss is computed on a single held-out monomer, so it is a
noisy, monomer-specific signal; early stopping on it terminates runs at effectively arbitrary points.

## Step B — propose fixes, ranked, with the cost of each

Candidate interventions — evaluate each against the constraint that the **test** monomer must remain
strictly held out:

1. **Change the validation set to a random row-level sample drawn from the training monomers**
   (test monomer still fully excluded). This keeps the OOD test intact while giving a stable stopping
   signal. Note that it changes what "monomer-heldout" means for validation only — argue whether this
   is defensible and how to describe it in a paper.
2. **Longer patience and a minimum-epoch floor**, so runs cannot stop during early noise.
3. **LR schedule** (cosine or plateau) — current source is Adam at 1e-3 with no scheduler.
4. **Checkpoint averaging** or selecting the best of the last k epochs rather than a single early-stop
   point.
5. **Fixed epoch budget with no early stopping**, if the val signal cannot be made reliable.

For each: expected effect on variance, implementation cost, and whether it invalidates comparability
with existing seed-42 results (assume it does, and say so).

## Step C — the test that decides it (12 GPU runs)

Pick the top two candidate fixes from Step B. For each, run **EA folds 0 and 1 × 3 repeats**, seed 42,
Gadi V100 — the same protocol that produced the numbers above, so it is directly comparable.

Report the MAE SD, group-mean R² SD and ΔR² SD per fold for: current config (already measured),
fix 1, fix 2. **Success criterion: fold-1 MAE SD below 0.02 eV**, i.e. comparable to the current
fold-0 value. State plainly whether either fix reaches it.

## Step D — implications to state explicitly

1. **How many runs per cell** are needed for a 0.03 eV effect to be detectable, given the post-fix SD.
   Show the arithmetic. This determines whether the B-heldout experiment is 144, 432, or larger.
2. **Whether folds 6 (EA) and 5 (IP)** — the other "pathological" folds — should also be repeat-tested
   before any conclusion about junction coupling stands. Recommend a protocol.
3. **Which existing conclusions in `variant_results_report.md` are invalidated** if fold-level SD is
   confirmed at this scale. Name the sections. In particular assess: the EA fold-1 "hard monomer"
   diagnosis, the junction-coupling rationale that was built on it, and the octamer's EA chemistry
   result whose largest single contribution is fold 1 (+0.414).

## Constraint

12 GPU runs total for Step C. Do not submit the 144 B-heldout cells. Do not start new model variants.
