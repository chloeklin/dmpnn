# Windsurf prompt — does within-fold group-mean R² actually test unseen-monomer chemistry?

**Context.** The provenance audit (`_octamer_provenance_check.md`) cleared the octamer run: identical
held-out sets, no row expansion, and the EA gain is present in `block` and `alternating` rows
(deterministic, K=1), so it is not a K=16 averaging artefact. The gain is therefore either real or an
artefact of the **metric**, and the metric has a structural weakness that has never been quantified:

- The LOMO split holds out **monomer A only** (9 folds × 4774 rows = the full 42,966-row dataset).
  Monomer B is always seen in training.
- Within a fold, **A is constant**. All variation across groups comes from B, `fracA/fracB`, and
  `poly_type` — all seen. A model's error on the unseen A enters within-fold group-mean R² almost
  entirely as a **constant offset**, and any change in how the readout weights the A contribution
  changes that offset without the model having learned anything about A.

Two analyses settle this. Both use **existing NPZs and `ea_ip.csv` only** — no training, no jobs.

**Rules.** Do not train, submit jobs, or edit any report file. Write results to a new file
`_groupmean_metric_floor.md`. Report "cannot determine" rather than inferring.

---

## Analysis A — A-blind null floor (no model involved)

For each fold `f` with held-out monomer `A_f`, build a **null predictor that never sees A**:
for each test row, predict the mean true value over all **training** rows sharing the same
`(smiles_B, fracA, poly_type)` — i.e. averaged over every other A monomer. Fall back to the
`(smiles_B, poly_type)` mean, then the global training mean, if a cell is empty; report fallback counts.

Then, per target and fold, report for this null predictor:

- group-mean R² (same grouping code used in `_phase1_metrics_scratch.md`)
- overall R², MAE, mean signed bias

Tabulate null vs `hpg_hier` vs `hpg_hier_octamer` vs `wdmpnn` side by side.

**Interpretation to state explicitly:** if the A-blind null already reaches high within-fold
group-mean R², then within-fold group-mean R² does not measure unseen-monomer chemistry and cannot
support any "chemistry extrapolation" claim in `variant_results_report.md`. Report the numbers and
say which conclusion they support; do not soften either way.

## Analysis B — pooled across-fold placement (the metric that does test unseen A)

Concatenate all 9 folds' test predictions per model per target (each prediction came from a model
that never saw that fold's A). Then report per model per target:

1. **Pooled group-mean R²** over all group means from all folds combined.
2. **Per-fold mean signed bias** (9 numbers) and their standard deviation — the fold-level offset spread.
3. **Across-fold placement R²**: the 9 points (true fold mean, predicted fold mean), R² and fitted slope.
   Slope < 1 indicates shrinkage of unseen-A predictions toward the global mean.
4. **Within-fold predicted group-mean spread ÷ true spread**, per fold — a compression ratio.

Models: `hpg_hier`, `hpg_hier_octamer`, `hpg_hier_junction`, `hpg_hier_junction1`, `wdmpnn`.

State plainly whether the octamer's advantage survives under pooled/across-fold metrics or exists
only within folds.

## Analysis C — finish Check 4 from data already in the files

The NPZs store `n_train`, `n_val`, `n_test`, `prediction_scale`; the previous audit listed these array
names but never printed their values. For all five models × 2 targets × 9 folds, print `n_train`,
`n_val`, and `prediction_scale`, and flag any cell where the octamer differs from the baseline.
This does not recover epoch counts, but it does detect a different training set or target scaling.

---

## Deliverable

`analysis/model_diagnostics/_groupmean_metric_floor.md`, ending with a **"Verdict"** section answering:

1. What within-fold group-mean R² does the A-blind null achieve, per target (median across folds)?
2. Under pooled and across-fold placement metrics, does the octamer still lead the baseline? By how much, and with what per-fold bias spread and slope?
3. Do `n_train` / `n_val` / `prediction_scale` match between octamer and baseline in all 18 cells?
