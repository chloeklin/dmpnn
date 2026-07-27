# Windsurf spec — readout ablation: is the octamer gain sequence encoding or attention pooling?

**Status of the evidence.** Three audits have now cleared the octamer result of the obvious artefacts:
identical held-out sets and row identifiers; no K-expansion; the EA gain present in deterministic
`block`/`alternating` rows; matched `n_train`/`n_val`/`prediction_scale`; and the gain survives pooled
and across-fold placement metrics (EA placement R² 0.923 → 0.989, slope 0.913 → 1.031, fold-bias SD
0.103 → 0.037), so it is not shrinkage toward the global mean.

**The remaining confound is internal to the variant.** `hpg_hier_octamer` changes three things at once
relative to `hpg_hier` Stage-2:

1. explicit 8-instance monomer sequence (instead of a 2-node transition graph),
2. positional embeddings over the 8 positions,
3. **attention pooling over positions, replacing the stoichiometry-weighted monomer readout.**

The measured gain is entirely on **chemistry placement** (unseen-monomer level), while ΔR² and pairwise
ordering — the architecture axis the octamer was designed to improve — are flat on EA (5/4 and 4/5
across folds) and not significant on IP (7/2, p = 0.18). That pattern is what you would expect if the
gain comes from (3) and not from (1). If so, the Q2 result is "a better readout," not "explicit
sequences beat the transition graph," and the report's Q2 framing does not survive.

## The ablation

Add one config and run it on the existing monomer-heldout LOMO protocol, seed 42, both targets,
all 9 folds. Change **nothing** else.

| run | stage2_mode | readout | positional emb. | purpose |
|---|---|---|---|---|
| A (have) | `transition_graph` | stoich-weighted sum | n/a | baseline |
| B (have) | `octamer_sequence` | attention pooling | yes | current octamer |
| **C (new)** | `transition_graph` | **attention pooling** | n/a | isolates the readout |
| **D (new)** | `octamer_sequence` | **stoich-weighted sum over positions** | yes | isolates the sequence |

Runs C and D are the deliverable. Use the same submit template as the octamer LOMO runs
(`scripts/shell/submit_hpg_phase1.sh`), same `--stage1_pool sum --stage2_depth 2 --stage2_edge full
--seed 42`, and **persist the resolved run config and the PBS task log alongside each NPZ** — the last
audit could not verify training parity because those artefacts were absent for the earlier runs.
Fix that for these runs.

## Pre-registered reading of the outcome

State which of these the results match, before looking at anything else:

- **C ≈ B on pooled/placement chemistry metrics** → the gain is the readout. Q2's explicit-sequence
  claim is not supported; the report's Q2 section must be rewritten as a readout finding, and run C
  becomes the new baseline against which the sequence encoder is judged.
- **D ≈ B, C ≈ A** → the gain is the explicit sequence. Q2 stands as written.
- **Both C and D below B** → the two interact; report it as such and do not attribute the gain.

## Metrics to report for C and D

Exactly the metrics used in `_groupmean_metric_floor.md` so the rows are directly comparable:
per-fold group-mean R², ΔR², ordering, overall R², MAE, mean signed bias; pooled group-mean R²;
placement R², slope, intercept; fold-bias SD; and paired per-fold win/loss with exact two-sided sign
tests against run A. Also report per-fold **headroom above the A-blind null** from
`_groupmean_metric_floor.md` for EA, since several EA folds sit close to that floor.

Do not edit `variant_results_report.md`. Write results to `_readout_ablation.md`.

## Priority note

Seeds 43/44 for runs A–D remain the gating requirement for any claim in the paper. If GPU budget
forces a choice, C at three seeds is worth more than C and D at one seed.
