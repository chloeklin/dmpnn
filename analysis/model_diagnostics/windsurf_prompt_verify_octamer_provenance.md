# Windsurf prompt — verify octamer LOMO provenance before any report claim stands

**Context.** A previous Windsurf run recomputed seed-42 LOMO metrics and rewrote
`analysis/model_diagnostics/variant_results_report.md` to declare `hpg_hier_octamer` the best
single model. The headline evidence is an EA chemistry jump (group-mean R² median 0.925 → 0.989,
MAE 0.094 → 0.043 eV, fold-1 sulfone 0.575 → 0.989) produced by a **Stage-2-only** change with an
**unchanged Stage-1**. Stage-2 cannot add chemistry information about an unseen monomer, so this
result is either (a) an artefact of a different/leaky evaluation set, (b) an artefact of the
K=16 prediction-averaging mini-ensemble, or (c) a genuine readout effect. Determine which.

**Rules for this task.**
- Do **not** train anything, submit any job, or edit any report file.
- Do **not** delete or overwrite `_phase1_metrics_scratch.md`; write findings to a new file
  `_octamer_provenance_check.md`.
- If a check cannot be run because information is not stored in the NPZ or logs, say so
  explicitly and name the file that would contain it. Do not infer.

---

## Check 1 — Evaluation-set identity (highest priority)

For every `(target, fold)` in EA/IP × 0–8, load the seed-42 NPZs for `hpg_hier`, `wdmpnn`,
`hpg_hier_junction`, `hpg_hier_junction1`, `hpg_hier_octamer` and report a table with:

- `n_test` (number of rows)
- `sha1` of `y_true` rounded to 6 dp, as a byte-order-independent hash of the **sorted** values
- `sha1` of the row identifiers (SMILES pair / index column) if present in the NPZ
- whether `n_test` and both hashes are **identical across all five models** for that fold

Flag every fold where the octamer's test set differs from the baseline's in size or content.
If the NPZ has no identifier column, report which arrays it does contain (`list(npz.files)`)
for one octamer file and one baseline file, side by side.

## Check 2 — Row multiplicity / averaging

For the octamer NPZs only:

- Does any row identifier appear more than once? Report the max multiplicity.
- Is `n_test` an integer multiple of the baseline `n_test` (e.g. 16×)? Report the ratio.
- If predictions were averaged over K sampled octamers, confirm the averaging happened
  **before** the NPZ was written, and state the source line in
  `scripts/python/run_hpg_generalization.py` (or wherever the write happens) that proves it.

## Check 3 — Split provenance

- Report file mtimes for all 18 octamer and 18 junction1 LOMO NPZs, and for the 18 baseline ones.
- Locate the job scripts / PBS logs that produced the octamer LOMO NPZs. Quote the exact command
  line, including the split flag, `stage2_mode`, `n_random_samples`, `octamer_len`, seed, and fold.
- Confirm from the log that the split used is `monomer_heldout` with the **same fold definition
  file / same held-out monomer per fold** as the baseline run. Print the held-out monomer SMILES
  per fold for both runs and diff them.
- Confirm the octamer sequence construction / K-sampling happens **after** the train/test split,
  not on the full dataset before splitting. Quote the code path that establishes the ordering.

## Check 4 — Hyperparameter parity

Extract from the octamer and baseline logs/configs and tabulate side by side: hidden sizes,
depth, epochs, early-stopping criterion, LR schedule, batch size, target scaling, loss.
Flag every difference. A model comparison with unmatched training budget is not a model comparison.

## Check 5 — Where the EA gain actually lives

For EA folds 1, 7, 0 (largest apparent octamer chemistry gains):

- Per architecture class (block / alternating / random), report octamer vs baseline MAE and mean
  signed bias. If the gain is concentrated in `random` rows only, it is consistent with the K=16
  averaging confound; if it is uniform across block/alternating (which are deterministic, K=1),
  it is not.
- Report the mean signed bias of the group means. The baseline fold-1 failure was a −0.213 eV
  systematic under-prediction; state whether the octamer removed the bias or merely rescaled.

## Check 6 — Paired per-fold comparison (replaces median-of-columns)

Seeds 43/44 are unavailable, but a paired within-seed comparison is available now and was not done.
For each metric (group-mean R², ΔR², ordering, overall R², MAE) and each target, report octamer vs
baseline as **wins/losses across the 9 folds** plus the per-fold signed difference, and the exact
two-sided sign-test p-value. Do the same for junction n=1 vs baseline and n=2 vs baseline.
State plainly that with 9 folds the minimum attainable two-sided p is 0.0039 and that no result
here is multiple-comparison corrected.

---

## Deliverable

`analysis/model_diagnostics/_octamer_provenance_check.md` containing the tables above and a final
section **"Verdict"** answering exactly three questions:

1. Are the octamer LOMO predictions evaluated on the identical held-out sets as the baseline? (yes / no / cannot determine, with evidence)
2. Is the EA chemistry gain concentrated in `random`-architecture rows? (yes / no / cannot determine)
3. Was training budget matched? (yes / no / cannot determine)

Do not restate any conclusion from `variant_results_report.md`. Do not add new conclusions to it.
