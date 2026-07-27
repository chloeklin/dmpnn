# Windsurf spec — Step 0 (design audit + null floors, no GPU) and Step 1 (B-heldout LOMO)

## STATUS — read this first

**Step 0 is complete** (`_dataset_design_audit.md`, both frozen split files). Below is the pre-flight
checklist that must be cleared before any training job is submitted. Items marked *blocking* stop
Step 1; items marked *not blocking* are reporting or budget decisions.

| # | Item | Blocking? | GPU? |
|---|---|---|---|
| 1 | ~~Count near-duplicate held-out B monomers.~~ **DONE and decided: 6 monomers at ≥ 0.95 across 682 — too few to justify rebuilding. Keep the frozen random folds; report every Step-1 metric twice, full and filtered (excluding held-out B with max training Tanimoto ≥ 0.95).** | Resolved | — |
| 2 | **Implement the `monomer_b_heldout` split type** with the seed-independence assertion, the bitwise-reproduction check on the existing A split, and per-run provenance logging (§1.1). | **Blocking** | No |
| 2b | **Demonstrate the *metadata* assertion, not just the seed check.** Failing on `split_seed != 42` only compares a number. Perturb a copy of `monomer_b_heldout.json` (swap two B monomers between folds) and show the run aborts. | **Blocking** | No |
| 2c | **De-duplicate the split loader.** B-split logic now exists separately in `run_hpg_generalization.py` and `run_wdmpnn_generalization.py`. If they drift, HPG and wDMPNN get evaluated on different folds and the comparison is void. Factor into one shared function, or add a test dumping train/val/test indices from both runners for all 9 folds and asserting equality. | **Blocking** | No |
| 3 | ~~Budget decision.~~ **DECIDED: run both assignments, 144 cells.** This makes wiring up the clustered split blocking: register `monomer_b_heldout_clustered` as its own canonical split with its own prediction directory (`predictions/ea_ip_lomo_b_clustered`), reading `metadata/splits/monomer_b_heldout_clustered.json`. Do not hard-code `split_seed=42` in a way that blocks a second frozen split. | **Blocking** | No |
| 4 | Novelty-stratified reporting and the performance-vs-novelty curve (§1.0 D). | Not blocking — required in the Step 1 analysis, not before it | No |
| 5 | The 7-donor training set (§1.0 E). **This is not a bug and must not be "fixed."** Changing the A-split validation design would break comparability with every existing seed-42 result. Document it, do not alter it. | Not blocking | No |

Nothing here needs a GPU. Items 1–3 should take well under a day, and item 1 may change the split
before anything is trained, which is why it comes first.

**Motivation.** The current LOMO split holds out **monomer A only**, and there are just 9 A monomers,
so every "unseen chemistry" result is an extrapolation from 8 training examples. Monomer B has 682
unique values and is always seen. An A-blind null predictor already reaches median group-mean R²
0.676 on EA (and beats the baseline on EA fold 2), so the EA chemistry column of
`variant_results_report.md` may be measuring a near-degenerate metric. A B-heldout split should
invert this: the B-blind null must average over ~606 seen B monomers, which should collapse it toward
the global mean and give the EA chemistry metric real headroom.

Step 0 costs no GPU and determines which Step 1 cells are worth running. **Do Step 0 first and stop
for review before implementing Step 1.**

Do not edit any existing report file. Do not submit jobs.

---

# Step 0 — design audit, split construction, null floors (pandas + RDKit only)

Output: `analysis/model_diagnostics/_dataset_design_audit.md` plus the frozen split file described in 0.3.

## 0.1 Confirm the design (do not assume it)

From `data/ea_ip.csv` report:

- n rows; n unique `smiles_A`; n unique `smiles_B`; whether the A set and B set are **disjoint**.
- Whether the design is a complete factorial: is every (A, B) pair present, and does every pair have
  the same number of rows? Report the exact per-pair row count and the full breakdown of what those
  rows are — the distinct `(fracA, fracB, poly_type)` combinations and their counts.
- Confirm or refute `9 × 682 × 7 = 42,966`. If the design is ragged, report the ragged cells; every
  later step depends on this.
- Rows per A monomer and rows per B monomer (min/median/max).

## 0.2 Which chemical axis carries the signal

For EA and for IP separately, report the share of total variance explained by:

1. A identity alone (one-hot / group means by `smiles_A`)
2. B identity alone
3. A + B identity
4. composition (`fracA`) alone
5. `poly_type` alone, and **within-(A,B,fracA)-group variance** — this should reproduce the
   known "architecture is 1–4% of variance" figure; report whether it does.

Also report mean EA and mean IP by A monomer and the spread across B monomers. **Hypothesis under
test:** A monomers are donors (so IP, HOMO-driven, is A-dominated) and B monomers are acceptors
(so EA, LUMO-driven, is B-dominated). State whether the numbers support it or not.

## 0.3 Construct and freeze the B-heldout split

- Partition the 682 B monomers into **9 folds** of 75–76, giving ~4,788 test rows per fold — matched
  to the existing A-heldout fold size (4,774) so the two splits are directly comparable.
- Test set for fold k = all rows whose `smiles_B` is in fold k. Assert the 9 test sets are disjoint
  and their union is the full dataset.
- Assignment must use a **fixed split seed that is independent of the model seed**, and must be
  written to `metadata/splits/monomer_b_heldout.json` in the same format as
  `metadata/splits/monomer_heldout.json`, storing both the B-monomer lists and the row index arrays.
- **Validation set.** The A-heldout runs use `n_val = 4774`, exactly one A monomer's worth — determine
  from the code what that val set actually is and document it. For the B split, the validation set
  must be **B-disjoint from both train and test** (carve a separate held-out set of B monomers for
  validation), fixed across seeds. State the design you chose and why.
- Produce two variants if cheap: **random** B assignment, and a **scaffold/cluster** assignment
  (Butina or Murcko scaffold clustering) so folds contain chemically related B monomers. Freeze both;
  Step 1 runs the random one first.

## 0.4 Difficulty calibration between the two splits

The two splits are not automatically comparable: holding out 76 of 682 B monomers leaves close
analogues in training, whereas holding out 1 of 9 A monomers may not. Quantify it.

- Morgan fingerprints (r=2, 2048 bits). For each held-out monomer, compute max Tanimoto similarity to
  the training monomers of the same role. Report the per-fold distribution (min/median/max) for:
  the 9 A-heldout folds, the 9 random B folds, and the 9 clustered B folds.
- Reuse whatever is already in `analysis/model_diagnostics/07_monomer_novelty/` rather than writing
  new novelty code; note which script you reused.

## 0.5 Null floors for both splits

Symmetric to the A-blind null already computed in `_groupmean_metric_floor.md`:

- **A-blind null** (existing): predict from train-only `(smiles_B, fracA, poly_type)` mean.
- **B-blind null** (new): predict from train-only `(smiles_A, fracA, poly_type)` mean, evaluated on
  the new B folds.

For each null, split, target and fold report group-mean R², overall R², MAE and mean signed bias,
using the **same grouping rule** as `_phase1_metrics_scratch.md` — group key `(smiles_A, smiles_B,
fracA)` restricted to groups with ≥2 `poly_type` values. Report medians and means across folds.

Also report a **trivial global-mean null** for both splits as an absolute reference.

## 0.6 Verdict section

End `_dataset_design_audit.md` with a table of **(split × target × metric) → headroom**, defined as
the gap between the relevant null floor and 1.0, and a plain statement of which cells are worth
spending GPU on and which are degenerate. Do not soften this; if a cell your current report relies on
turns out degenerate, say so explicitly and name the report section affected.

**Stop here for review.**

---

# Step 1 — B-heldout LOMO runs (after Step 0 is reviewed)

## 1.0 Amendments after reviewing `_dataset_design_audit.md` — read first

**A. Near-duplicate held-out monomers.** Random B folds 0, 1, 2 and 6 contain held-out B monomers
with **max Tanimoto = 1.000** to a training B monomer, i.e. fingerprint-identical at Morgan r=2.
Before any runs: report per fold the count of held-out B monomers with max Tanimoto ≥ 0.99, ≥ 0.95
and ≥ 0.90, and list the ≥ 0.99 pairs with both SMILES so they can be inspected. These are not a
held-out chemistry test. Do not silently drop them — report all Step-1 metrics twice, once on the
full fold and once excluding held-out B monomers with max Tanimoto ≥ 0.95.

**B. The two splits are not difficulty-matched.** Median nearest-neighbour Tanimoto is 0.31–0.47 for
A-heldout, 0.52–0.58 for random B, 0.48–0.50 for clustered B. Any statement comparing absolute scores
across splits must carry this. The **clustered** split is the difficulty-matched comparator to
A-heldout, which raises its priority above "follow-up": if budget allows 144 cells, run random and
clustered together; if not, run random first but do not compare its numbers to the A split without
the caveat.

**C. Each split is the right test for a different target.** From the Step-0 headroom table:

| split | EA headroom | IP headroom |
|---|---|---|
| A-heldout | 0.324 | **1.034** |
| B-heldout random | **0.580** | 0.440 |

So EA chemistry claims belong on the B split and IP chemistry claims belong on the A split. Report
them as a matched pair rather than picking a single "main" split. The architecture axis (ΔR²,
ordering) is not affected by the null floors and should be reported on both.

**D. Novelty-stratified reporting — required, and only possible on this split.** The A split has one
held-out monomer per fold, so its novelty rows are constant (min = median = max). The B split has
~76 per fold, so Step 1 can produce a **performance-versus-novelty curve**: bin held-out B monomers
by max Tanimoto to training (e.g. <0.35, 0.35–0.45, 0.45–0.55, 0.55–0.7, >0.7) and report per-bin
MAE, group-mean R² and ΔR² for every model. This is a stronger result than any single OOD number and
costs nothing once the runs exist. Make it a first-class output, not an appendix.

**E. The A-heldout training set is smaller than assumed.** Step 0 established that the A-heldout
generator excludes the test A *and* a second A for validation, so those models train on **7** donor
monomers, not 8, while A identity carries 0.418 (EA) / 0.500 (IP) of total variance. Note this in any
write-up of A-split chemistry results; it is the likely reason three separate representation changes
failed to move that axis.

## 1.1 Code

- Add `monomer_b_heldout` as a split type in `scripts/python/run_hpg_generalization.py`, loading
  `metadata/splits/monomer_b_heldout.json` with the same "regenerate then assert against the stored
  file" pattern the existing split uses (`run_hpg_generalization.py:104-116`). The run must fail
  loudly if the regenerated indices differ from the frozen file.
- The split must not depend on the model seed. Add the assertion, then demonstrate it fires by
  deliberately passing a wrong split seed.
- Existing `monomer_heldout` behaviour must be unchanged: re-run `hpg_hier` EA fold 0 seed 42 and
  confirm predictions match the existing NPZ bitwise, or explain the difference.
- Persist provenance next to every NPZ, as in `windsurf_spec_seeds_and_readout_ablation.md`:
  resolved config JSON, git SHA, PBS job ID, epochs run, best val loss, wall time. This was missing
  for earlier runs and blocked the training-parity audit.

## 1.2 Runs

`hpg_hier`, `wdmpnn`, `hpg_hier_octamer`, `hpg_hier_junction` (n=2) × EA/IP × folds 0–8 × seed 42
= **72 runs**, random B folds. Print the job count and estimated GPU hours; do not submit.

## 1.3 Analysis

Produce `analysis/model_diagnostics/_b_heldout_results.md` with the **same** metrics and conventions
used for the A split, so the two are directly comparable side by side:

- per-fold group-mean R², ΔR², pairwise ordering, overall R², MAE, mean signed bias
- **each group-mean R² reported next to its B-blind null floor**
- pooled group-mean R², fold-placement R²/slope/intercept, fold-bias SD, compression ratio
- paired per-fold comparisons against `hpg_hier` — signed differences, wins/losses, exact two-sided
  sign test, Holm-corrected across the family; state that the minimum attainable two-sided p with
  9 folds is 0.0039
- headline numbers as **median of paired per-fold differences**, not the difference of separately
  ranked medians

Reuse the metric definitions already used for the A split — factor them into one module rather than
reimplementing — and verify the module reproduces the seed-42 A-split numbers in
`_phase1_metrics_scratch.md` and `_groupmean_metric_floor.md` to 5 dp before trusting it on the B split.

## 1.4 The questions Step 1 must answer

State answers plainly, without hedging toward the existing report:

1. Does HPG-hier still lead architecture recovery (ΔR², ordering) over wDMPNN when the unseen
   monomer is B rather than A?
2. Does the EA chemistry metric have real headroom on this split, and if so does any model actually
   beat the B-blind null by a meaningful margin?
3. Does the octamer's chemistry/placement advantage persist here, or was it specific to
   8-example A extrapolation?
