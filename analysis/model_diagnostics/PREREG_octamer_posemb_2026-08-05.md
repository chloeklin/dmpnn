# Pre-registration — octamer positional-embedding ablation

Written **5 August 2026, before any job is submitted.** Nothing below may be edited after the
first result lands. Corrections go in a dated addendum at the bottom.

---

## 1. Question

Does removing the 8 learned position embeddings degrade the octamer? Factor 5 was excluded by
the K=1 arm (outcome C in `PREREG_octamer_k1_2026-07-30.md`).  Factors 1, 2, 3 and 4 remain
(HANDOFF §7).  This arm isolates **factor 2: the 8 learned position vectors** in the octamer
sequence encoder.

## 2. Mechanism, stated precisely

Without position embeddings, every slot holding monomer A receives an identical copy of A's
Stage-1 embedding; every slot holding monomer B receives an identical copy of B's Stage-1
embedding.  Slots remain distinguishable through path structure — end slots (0 and 7) aggregate
one incoming message, interior slots aggregate two — so this is a **reduction of positional
information, not its elimination**.  Do not describe the ablated model as "position-blind".

The change is implemented by the `--octamer_position_embeddings off` flag, which suppresses both
creation of the `position_embeddings` parameter and the `h = h + position_embeddings` addition
in `OctamerEncoder.forward`.  The parameter count therefore differs from the K=16 baseline by
exactly `octamer_len × d_h = 8 × 128 = 1024`.

## 3. Design

| | R1 | R3 |
|---|---|---|
| Split | `monomer_heldout` | `monomer_b_heldout_clustered` |
| Subdirectory | `ea_ip_lomo` | `ea_ip_lomo_b_clustered` |
| Folds | 0–8 | 0–8 |
| Targets | EA, IP | EA, IP |
| Seeds | 42, 43, 44, averaged at the **prediction** level (`--frozen_protocol`) | same |
| Runs | `3 × 9 × 2 = 54` | `3 × 9 × 2 = 54` |
| Changed vs existing `hpg_hier_octamer` | `--octamer_position_embeddings off` only | same |

All other settings match the existing `hpg_hier_octamer` cells: `--n_random_samples 16
--batch_size 64 --epochs 100 --patience 15 --min_epochs 1 --frozen_protocol`.

## 4. Comparators

The existing K=16 `hpg_hier_octamer` cells, 54 complete on each split, same folds and seeds.

## 5. Predictions, written before the result

Primary quantity: **ΔR² on `all` rows**, reported separately for R1, R3-S and R3-D.  Unlike the
K=1 arm, there is no row-subset restriction here, so `delta_r2` is computable directly — **no
metric substitution is needed**.  State this explicitly so the K=1 substitution is not repeated.

Materiality threshold, defined per split before running:

| split | threshold | source | cells |
|---|---|---|---|
| R1 | **0.051** | median per-cell across-seed SD of `delta_r2` for `hpg_hier_octamer` on `all` rows (`_regen_v1_results_individual_runs.csv`) | 18 |
| R3 | **0.024** | median per-cell across-seed SD of `delta_r2` for `hpg_hier_octamer` on `all` rows (`_regen_v1_r3_results_individual_runs.csv`) | 17 |

A single shared threshold would make the A split look artificially decisive because the R1
distribution is heavily right-tailed (mean SD 0.123 against median 0.051).  Per-split thresholds
avoid that.

| outcome | reading |
|---|---|
| **Large drop, both splits** | Position embeddings are a principal source of the octamer's advantage. Factor 2 is a leading explanation and factors 1, 3 and 4 shrink in importance. |
| **Large drop, one split only** | The mechanism is split-dependent; report both and do not generalise. |
| **No material change on either** | Factor 2 is excluded, like factor 5. Remaining candidates are factor 1 (8-slot topology) and factor 4 (discarded 16-d port-pair edge features). **No ablation at the current noise floor can separate those two** — state this as a limit of the dataset, not of effort. |
| **Performance improves without them** | Not anticipated. Treat as a possible bug until the §6 controls pass. |

"Large drop" means the ΔR² difference (position-embedding-off minus K=16 baseline) falls below
`−threshold` for the relevant split and fold group.  "No material change" means it stays within
`[−threshold, +threshold]`.

## 6. Negative controls

1. **Sidecar check.** Every sidecar must show:
   - `resolved_config.octamer_position_embeddings == "off"`
   - `resolved_config.n_random_samples == 16`
   - `resolved_variant.stage2_mode == "octamer_sequence"`
   - `resolved_variant.stage2_readout == "attention"`
   - `resolved_config.batch_size == 64`
   - `resolved_config.frozen_protocol == true`
2. **Parameter-count check.** `n_octamer_params` must differ from the K=16 baseline by exactly 1024.
3. **Path-collision check.** Output paths use the token `__noposemb` under
   `predictions/octamer_posemb` and `checkpoints/octamer_posemb`; no file may overwrite an
   existing K=16 prediction.
4. **Cap check.** No run may have `best_epoch` at the 100-epoch cap.

## 7. Statistical caution carried from the K=1 arm

With three seeds per cell, an across-seed SD estimate carries roughly 40% relative error.  Any
claim about a change in stability therefore requires a **paired per-cell sign test across cells**,
not a comparison of two medians.  Pre-commit to that test now.

## 8. What this arm does not resolve

Factors 1 (8-slot topology) and 4 (the discarded 16-d port-pair edge features) stay confounded
whatever the result.  Do not claim this arm distinguishes them.

---

## Addenda

### 2026-08-08 — Refinement of the fourth pre-registered outcome

The fourth row of §5 reads too narrowly: a measured *improvement* without position embeddings is
not automatically a bug signal.  With `use_position_embeddings = False`, every slot holding the
same monomer receives the same Stage-1 embedding.  The path message passing is bidirectional and
the attention readout is permutation-invariant over slots, so the ablated model treats a sequence
and its reverse as identical (for example AABB and BBAA map to the same representation).  For a
linear polymer chain read from either end, reversal is the same molecule, so this invariance is
physically correct.  The position-embedding-on model is permitted to distinguish a sequence from its
reverse, which may not be chemically meaningful and may spend capacity on a spurious asymmetry.

Therefore, outcome 4 should be split into two competing readings, separated only by the §6
controls, not by assumption:

1. **Bug in the ablation.** Ruled out if all §6 controls pass (correct flag in the sidecar,
   `n_octamer_params` reduced by exactly 1024, no path collisions, no cap runs).  If any control
   fails, the result is *not* reportable until the failure is resolved.
2. **Genuine finding.** If all controls pass, the improvement is reportable as evidence that the
   position embeddings were fitting a chemically non-physical orientation asymmetry, and that the
   octamer gain was coming from some other aspect of the sequence representation.

The controls-passing gate is what distinguishes these two readings.  State that gate explicitly
when reporting outcome 4.
