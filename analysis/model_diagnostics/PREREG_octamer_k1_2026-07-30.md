# Pre-registration — octamer replica ablation (K=1)

Written **30 July 2026, before the arm was submitted.** Nothing below may be edited after the
first result lands. Corrections go in a dated addendum at the bottom.

---

## 1. Question

The octamer (`hpg_hier_octamer`) is the most stable model in the campaign and shows the largest
ΔR² on the B-split cross-scaffold folds. Five factors differ between it and `hpg_hier`
(HANDOFF §7). This arm isolates **factor 5: the 16-replica in-loss ensemble.**

## 2. What the octamer does today, and what changes

`_build_octamer_sequences` enumerates the `C(8, n_A)` arrangements of the 8 slots.

- **Non-uniform transition matrix** (`poly_type` = `block`, `alternating`) → argmax, **one** sequence.
- **Uniform transition matrix** (`poly_type` = `random`) → sample `n_random_samples` = **16** with
  replacement.

On the 16-replica rows, `forward` returns `pred_sum / replica_counts` and `_loss` takes MSE
against that mean. So the octamer is *trained* as a 16-member ensemble: predictions are averaged
**and** the gradient is variance-reduced.

`--n_random_samples 1` removes both at once. It changes the **training objective**, not just
test-time sampling. It is not "fewer test-time samples" and must not be described that way.

## 3. Affected subset — counted, not assumed

From `data/ea_ip.csv`, 42,966 rows:

| `poly_type` | rows | share | sequences built | affected by K=1 |
|---|---|---|---|---|
| `random` | 18,414 | 42.86% | 16 sampled | **yes** |
| `block` | 18,414 | 42.86% | 1 (argmax) | no |
| `alternating` | 6,138 | 14.29% | 1 (argmax) | no |

**All metrics are reported split by `poly_type`, random rows separately from the rest.** A
pooled number would dilute the effect by ~57% and is not reportable for this arm.

## 4. Design

| | |
|---|---|
| Split | `monomer_b_heldout_clustered` (R3) only. Claim 8 lives here. R1 is not run. |
| Fold grouping | S (folds 0–3) and D (folds 4–8) reported separately. Never pooled — HANDOFF §5. |
| Folds | 0–8 |
| Targets | EA, IP |
| Seeds | 42, 43, 44, averaged at the **prediction** level (frozen protocol, HANDOFF §4) |
| Runs | 3 × 9 × 2 = 54 |
| Changed vs the existing octamer arm | `--n_random_samples 1` **only** |
| Cost | ~2.2 kSU at 41.1 SU/run |

Comparator is the existing `hpg_hier_octamer` K=16 cells, same folds, same seeds, same protocol.

## 5. Predictions, written before the result

Primary quantity: **ΔR² on random rows, D folds**, K=1 versus K=16.

| # | outcome | reading |
|---|---|---|
| A | seed SD rises **and** ΔR² falls on random rows | The in-loss ensemble was carrying the octamer. Claim 8's "explicit sequence" story weakens — the gain was variance reduction, not sequence representation. Rewrite Claim 8. |
| B | seed SD rises, ΔR² **holds** | The *stability* came from ensembling; the architecture effect survives. This is HANDOFF §7's standing prediction. |
| C | neither moves materially | The 16 replicas were not contributing. Remaining candidates are factor 2 (positional embeddings) and factor 4 (missing edge features). Arms C/D do not address either — say so. |
| D | ΔR² **rises** at K=1 | Not anticipated. Treat as a signal of a bug in the arm, not a finding, until the §6 controls pass. |

**"Materially" is defined now, not later:** a change is material only if it exceeds the measured
per-cell across-seed SD of ΔR² for the octamer on R3, **0.024** (median, computed from
`_regen_v1_r3_results_individual_runs.csv`). Changes below 0.024 are reported as "within
run-to-run variation" regardless of sign consistency.

## 6. Negative controls — must pass before any result is interpreted

1. **`block` and `alternating` rows must not move materially.** They take the argmax path, which
   `--n_random_samples 1` does not touch. If they shift by more than 0.024 in ΔR², something other
   than the replica count changed and the arm is not clean. Given that
   `octamer_sequence + stoich_weighted` silently runs the plain baseline today (HANDOFF §7), a
   silent config failure in this codebase is a live risk, not a hypothetical.
2. **Sidecar check.** Every sidecar must show `resolved_config.n_random_samples == 1`,
   `stage2_mode == "octamer_sequence"`, `stage2_readout == "attention"`, `batch_size == 64`,
   `epochs == 100`, `patience == 15`, `frozen_protocol == true`.
3. **Output paths must not collide with the K=16 runs.** The existing filename template
   (`generate_regen_v1_r3.sh`) does not encode `n_random_samples`, and the PBS resume guard skips
   any cell whose `.npz` already exists. Without a distinct token every K=1 job would silently
   skip. Confirm 54 new files exist and none overwrote a K=16 file.
4. **Cap check.** No run may have `best_epoch` at the 100 cap. One K=16 octamer cell already hit
   97 (HANDOFF §8); if K=1 trains slower this could bite more often.

## 7. What this arm does not resolve

Factors 2 (positional embeddings) and 4 (the missing 16-d port-pair edge features) stay
confounded inside the 8-chain arm whatever K=1 shows. State as a limitation.

---

## Addenda

### 2026-08-05 — Correction on the `block`/`alternating` negative control

The pre-registered statement that "`block` and `alternating` rows must not move
materially" is too strong.  Those rows do take the argmax path, so
`--n_random_samples 1` does not change their forward pass *for a fixed trained
model*.  However, training with K=1 changes the shared model weights relative
to the K=16 run, and therefore changes predictions on *all* rows, including
block and alternating rows.  A material shift on block/alternating is not, by
itself, evidence of a bug or a configuration error; it only rules out the
hypothesis that the two settings produced identical trained models.  It does
not rule out weight-driven spillover from the random rows.  The interpretation
of the negative control should be updated accordingly.

### 2026-08-05 — Metric substitution for the primary quantity

The pre-registration defined the primary quantity as ΔR² (the difference in the
named `delta_r2` metric between K=1 and K=16) on **random rows, D folds**.
`delta_r2` cannot be computed on a random-only subset because
`compute_copolymer_metrics` retains only copolymer groups containing at least
two distinct `poly_type` values, and a random-only subset leaves one row per
group.  The same issue blocks `group_mean_r2` and `ordering` on the random-only
subset.

The primary quantity was therefore replaced post-hoc with the K=1-minus-K=16
difference in **overall R²** on random rows, D folds.  This is a different
metric from the pre-registered ΔR², but the operational question — does the
K=1 arm differ from the K=16 arm on the random-held-out rows? — is the same.

The conclusion is unchanged under three alternative readings that are
computable from the saved predictions:

1. **overall R² on random rows, D folds** (the reported substitution): the
   median paired difference is well inside the 0.024 material threshold.
2. **`delta_r2` computed from group means taken over all rows and then
   evaluated on the random subset** (materialized in `_octamer_k1_r3_results.md`
   as the row set `random_via_all_groups`): for the median paired K=1-minus-K=16
   difference, D folds = −0.002 (EA +0.002, IP −0.006) and S folds = +0.002
   (EA +0.009, IP −0.006). Both are inside the 0.024 material threshold.
3. **`delta_r2` on the full fold (`all` rows), D folds**: the median paired
   difference is inside the 0.024 material threshold.

Because every computable reading shows no material shift, the supported outcome
remains C.  This substitution is disclosed as a post-hoc change to the primary
outcome metric.
