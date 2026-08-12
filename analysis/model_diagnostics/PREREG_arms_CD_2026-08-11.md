# Pre-registration — octamer arms C and D (topology vs. readout)

Written **11 August 2026, before any job for arms C or D is submitted.**
Nothing below may be edited after the first result lands.
Corrections go in a dated addendum at the bottom.

---

## 1. Question

Do factors **1** (8-slot chain topology) and **3** (attention readout)
separate, and which of them carries the octamer's advantage over HPG-hier?

After the K=1 and positional-embedding arms, three factors remain open:

| # | Factor | Status |
| --- | --- | --- |
| 1 | 8-slot chain vs 2-node transition graph | open |
| 3 | Attention readout vs stoichiometry-weighted | open |
| 4 | Discarded 16-d port-pair edge features | open |

Factors 1 and 3 are confounded. The octamer changed both at once, so the
2x2 grid is the only informative comparison:

|  | Stoichiometry-weighted / mean readout | Attention readout |
| --- | --- | --- |
| 2-node graph | HPG-hier — have | **arm C** |
| 8-slot chain | **arm D** | octamer — have |

Running one without the other leaves attribution ambiguous. They are treated
as a single arm with two branches.

## 2. Mechanism, stated precisely

- **Arm C** is a 2-node transition graph with the attention readout (model
  `hpg_hier_attention`). It holds the topology of HPG-hier constant and swaps
  the readout from stoichiometry-weighted sum to learned attention.

- **Arm D** is the 8-slot chain with mean pooling (model `hpg_hier_octamer`
  with `--stage2_readout stoich_weighted`). It holds the attention-based
  readout out of the octamer and replaces it with mean pooling over slots.

On an 8-slot chain the number of slots holding monomer A is
`n_A = round(8 · fracA)`, so the slot counts already encode composition.
Mean pooling over positions is the exact analogue of the stoichiometry-weighted
sum `f_A · h_A + f_B · h_B` used on the 2-node graph. Pooling happens after
message passing, so arrangement information still reaches the readout; only
the learned weighting of positions is removed. This matches HANDOFF §7.

No other pooling variant is introduced. Max, sum, or gated pooling would turn
the arm into a search rather than an attribution test.

## 3. Design

The pilot is restricted to **R1 (monomer_heldout)**, **EA**, folds **0 and 4**,
three seeds (42, 43, 44), both arms. That is **12 runs**.

|  | R1 pilot |
| --- | --- |
| Split | `monomer_heldout` |
| Subdirectory | `ea_ip_lomo` |
| Folds | 0, 4 |
| Targets | EA only |
| Seeds | 42, 43, 44, averaged at the **prediction** level (`--frozen_protocol`) |
| Runs | 2 arms × 2 folds × 1 target × 3 seeds = **12** |

The full R1 arm would be **108 runs** (2 arms × 9 folds × 2 targets × 3 seeds),
roughly 3.9 kSU at the measured octamer per-run cost. It is **not generated or
submitted** from this pilot.

R3 (`monomer_b_heldout_clustered`) is a separately gated generalisation arm and
is not part of the pilot.

All other settings match the existing `hpg_hier_octamer` frozen protocol:
`--n_random_samples 16 --batch_size 64 --epochs 100 --patience 15
--min_epochs 1 --octamer_len 8 --frozen_protocol`.

Arm-specific readout:

| arm | model token | additional CLI flag | `stage2_mode` | `stage2_readout` resolved |
| --- | --- | --- | --- | --- |
| C | `hpg_hier_attention` | none | `transition_graph` | `attention` |
| D | `hpg_hier_octamer` | `--stage2_readout stoich_weighted` | `octamer_sequence` | `stoich_weighted` (OctamerEncoder `readout="mean"`) |

## 4. Comparators

Existing frozen cells from `predictions/regen_v1/ea_ip_lomo`:

- HPG-hier: 2-node transition graph, stoichiometry-weighted (`hpg_hier`).
- Octamer: 8-slot chain, attention (`hpg_hier_octamer`).

## 5. Predictions, written before the result

Primary quantity: **ΔR² on `all` rows**, reported separately for R1, R3-S, and
R3-D when those arms are run. The pilot reports R1 only.

The pre-registered reading, lifted from HANDOFF §7:

| outcome | reading |
| --- | --- |
| **D ≈ octamer and C ≈ HPG-hier** | the sequence did it — **factor 1** |
| **C ≈ octamer and D ≈ HPG-hier** | the readout did it — **factor 3** |
| **Both midway** | they interact; do not attribute |
| **Both ≈ HPG-hier** | it was **factor 4** — the discarded edge features |

### Materiality threshold

The threshold is the median per-cell across-seed SD of `delta_r2` for
`hpg_hier_octamer` on the `all` row set, taken from the existing
`_regen_v1_results_individual_runs.csv` (R1) and
`_regen_v1_r3_results_individual_runs.csv` (R3).

Re-deriving today:

| split | re-derived value | cells | source |
| --- | --- | --- | --- |
| R1 | **0.0446** | 18 | `_regen_v1_results_individual_runs.csv` |
| R3 | **0.0205** | 18 | `_regen_v1_r3_results_individual_runs.csv` |

The posemb pre-registration recorded **0.051** for R1. The 2026-08-11 addendum
to `PREREG_octamer_posemb_2026-08-05.md` shows that 0.051 was the 10th sorted
value (rank 10, one-indexed), while the true median of 18 values is the average
of ranks 9 and 10, i.e. `(0.038223 + 0.050963) / 2 = 0.044593`. The data are
unchanged; only the median convention slipped.

**This arm freezes the re-derived R1 value 0.0446.** This is a new
pre-registration written before any result, so it can adopt the corrected
median. The posemb value 0.051 remains frozen for the posemb arm because a
pre-registered number may not be altered after that arm began; the discrepancy
is documented rather than propagated.

A result is outside the noise floor only if the arm's ΔR² (arm minus its
comparator) falls below `−threshold` or above `+threshold` for the relevant
split and fold group.

## 6. Negative controls

1. **Sidecar check.** Every sidecar must show:
   - `resolved_config.batch_size == 64`
   - `resolved_config.epochs == 100`
   - `resolved_config.patience == 15`
   - `resolved_config.frozen_protocol == true`
   - `resolved_config.n_random_samples == 16`
   - `resolved_variant.stage2_mode` matches the arm
   - `resolved_variant.stage2_readout` matches the arm

2. **Parameter-count check.** Reconstruct total parameter counts from the model
   configs (`d_h = 128`, `stage2_depth = 2`, `octamer_len = 8`):

| configuration | total params |
| --- | --- |
| HPG-hier (baseline) | 256 897 |
| arm C (transition graph + attention) | 257 026 |
| octamer (baseline) | 390 402 |
| arm D (8-slot chain + mean) | 390 273 |

   Arm C adds exactly 129 parameters to HPG-hier and arm D removes exactly 129
   parameters from the octamer. That is `d_h + 1` for the single attention
   readout (`Linear(d_h, 1)` with weight `(128, 1)` and bias `(1,)`).

3. **Path-collision check.** Output paths use the tokens `__armC` and `__armD`
   under a new prediction directory (`predictions/octamer_cd`) and a new
   checkpoint directory (`checkpoints/octamer_cd/hpg`). No file may overwrite an
   existing `regen_v1`, `octamer_k1`, or `octamer_posemb` prediction.

4. **Cap check.** No run may have `best_epoch` at the 100-epoch cap.

## 7. Statistical caution

Three seeds per cell gives an across-seed SD estimate with roughly 40% relative
error. Any claim about stability or a change in stability requires a paired
per-cell sign test across cells, not a comparison of two medians.

## 8. What this arm does not resolve

Factor 4 — the discarded 16-d port-pair edge features — stays open whatever the
result. Say so.

---

## Addenda

*None yet.*
