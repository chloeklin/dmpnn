# Windsurf spec — multi-seed LOMO runs + readout ablation (code + submission + aggregation)

Two deliverables. Implement the code and submission scripts; do **not** submit jobs, and do not edit
any file in `analysis/model_diagnostics/` other than creating the new outputs named below.

---

## Deliverable 1 — `--stage2_readout` flag and the ablation config

Currently the stoichiometry-weighted monomer readout and the octamer's attention pooling are bound to
`stage2_mode`. Separate them.

- Add `--stage2_readout {stoich_weighted,attention}` to `scripts/python/run_hpg_generalization.py`
  and the corresponding argument in `chemprop/models/hpg_hier.py`.
- **Reuse the exact attention-pooling module the octamer already uses.** Factor it out into one class
  and call it from both paths. Do not reimplement it for the transition-graph path — if the two
  differ in any way (temperature, normalisation, parameter count, init) the ablation is meaningless.
- Defaults must preserve current behaviour exactly: `transition_graph` → `stoich_weighted`,
  `octamer_sequence` → `attention`. Prove this: run one fold of `hpg_hier` EA fold 0 seed 42 with
  the new code and confirm the predictions match the existing NPZ bitwise, or explain any difference.
- Record the resolved value of `stage2_readout` in the NPZ metadata.

The four configurations to support:

| run | stage2_mode | stage2_readout | status |
|---|---|---|---|
| A | `transition_graph` | `stoich_weighted` | exists (baseline) |
| B | `octamer_sequence` | `attention` | exists (octamer) |
| **C** | `transition_graph` | `attention` | **new — the decisive ablation** |
| D | `octamer_sequence` | `stoich_weighted` | new — lower priority |

## Deliverable 2 — seed sweep submission

`scripts/shell/submit_hpg_seeds.sh`, a PBS job array over the cross product:

- models: `hpg_hier`, `wdmpnn`, `hpg_hier_octamer`, `hpg_hier_junction` (n=2), and run C
- targets: `EA_vs_SHE_eV`, `IP_vs_SHE_eV`
- folds: 0–8
- seeds: 42, 43, 44 (skip cells whose NPZ already exists — seed 42 mostly exists for A/B/junction)

Requirements, in priority order:

1. **The split must not depend on the model seed.** This is the one bug that would invalidate the
   entire multi-seed analysis. The monomer-heldout fold definition must be byte-identical across
   seeds 42/43/44 and must continue to validate against `metadata/splits/monomer_heldout.json`.
   Use a separate fixed `--split_seed 42` if the current code shares one seed for both. Add an
   assertion that fails the run if the fold's test index array differs from the stored split.
2. **State explicitly what the seed does change**: model init, batch shuffling, dropout, and the
   octamer's K-sample draws. Document whether the validation split is fixed or reseeded, and prefer
   fixed — `n_val = 4774` (one monomer's worth) suggests it is a held-out monomer, and it must be the
   same one across seeds for paired comparison.
3. **Persist provenance next to every NPZ.** The last audit could not verify training parity because
   the executed configs and task logs were absent. Write, per run: resolved config JSON (all CLI args
   and defaults), git commit SHA, PBS job ID, epochs actually run, best val loss, wall time.
   Keep task logs under `logs/hpg_phase1/tasks/`.
4. Idempotent: skip completed cells, resumable after walltime kills, one NPZ per cell.
5. Print the total job count and estimated GPU hours before submission, and do not submit.

**Budget note.** If GPU time forces a cut, drop run D entirely and drop `hpg_hier_junction` to seed 42
only. Seeds 43/44 for A, B, wDMPNN and run C are the minimum that makes any claim in the report
defensible.

## Deliverable 3 — `scripts/python/aggregate_lomo_seeds.py`

One script that produces every number the report needs, so the analysis stops being re-derived
ad hoc each time. It must:

- Load all available NPZs and report an explicit inventory of missing cells (do not silently omit).
- **Average metrics across seeds within a fold first, then take the median across the 9 folds.**
  Never pool seeds and folds into one distribution.
- Report, per model/target: per-fold per-seed metrics; seed spread (SD across seeds within fold);
  median and mean across folds; and the **paired per-fold comparison against a named reference model**
  — signed differences, wins/losses, exact two-sided sign test, and paired Wilcoxon across the 9 folds.
  State that with 9 folds the minimum attainable two-sided p is 0.0039, and report a
  Holm correction across the comparison family.
- Compute the metrics from `_groupmean_metric_floor.md` as first-class outputs: pooled group-mean R²,
  fold-placement R²/slope/intercept, fold-bias SD, within-fold compression ratio, and the
  **A-blind null floor per fold**, so every group-mean R² is reported next to its floor.
- Reuse the existing metric definitions rather than rewriting them; if the definitions live inline in
  a previous scratch script, factor them into one module and note which file they came from.
- Emit a machine-readable `results.parquet`/`.csv` plus a markdown table, to
  `analysis/model_diagnostics/_multiseed_results.md`.

## Verification before this is considered done

1. Run C reproduces run A bitwise when `--stage2_readout stoich_weighted` is passed (Deliverable 1 check).
2. The split assertion fires if you deliberately pass a wrong split seed — demonstrate it.
3. `aggregate_lomo_seeds.py` reproduces the seed-42 numbers in `_phase1_metrics_scratch.md` and the
   pooled/placement numbers in `_groupmean_metric_floor.md` to 5 dp. If any number differs, stop and
   report the discrepancy rather than adjusting the new script to match.
