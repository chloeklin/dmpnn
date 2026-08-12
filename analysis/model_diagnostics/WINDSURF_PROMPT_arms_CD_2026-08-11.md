# Windsurf task — complete one missing run, then implement and pilot arms C and D

Two independent parts. **Part A is small and can start immediately. Part B needs a
pre-registration written before any job is submitted.**

---

# Part A — one missing run, and one run that must NOT be re-run

## A1. Complete the K=1 arm: IP fold 7, seed 44

`predictions/octamer_k1/ea_ip_lomo_b_clustered/` contains 53 files where 54 are expected.
The cell `IP_vs_SHE_eV × fold 7` has seeds 42 and 43 only — **seed 44 is missing**.

This is a genuine protocol gap: the frozen protocol requires exactly three seeds averaged at
the prediction level, and that cell is currently reported as a 2-seed average.

Generate and submit the single job:

```
target = IP vs SHE (eV)
split  = monomer_b_heldout_clustered
fold   = 7
seed   = 44
setting = K=1  (single fixed sequence, --n_random_samples 1)
output = predictions/octamer_k1/ea_ip_lomo_b_clustered/
         ea_ip__IP_vs_SHE_eV__hpg_hier_octamer__monomer_b_heldout_clustered__fold7__s44__k1.npz
```

Match the configuration of the existing `fold7__s42__k1` run exactly — read its
`.config.json` and reproduce every field except the seed. Confirm before submitting that the
resolved config matches on `n_random_samples`, `batch_size`, `epochs`, `patience`,
`stage2_mode`, `stage2_readout` and `frozen_protocol`.

After it lands, re-run the K=1 analysis and report whether any conclusion changes. It should
not — the arm's outcome C rests on medians across 5 and 4 folds — but say so explicitly rather
than assuming.

## A2. Do NOT re-run Factor 2, EA fold 0

The position-embedding arm shows EA fold 0 moving in opposite directions on the two axes:
overall R² **+0.035** while ΔR² is **−0.053**.

**That cell is complete and on-protocol.** All three seeds exist, best-checkpoint predictions,
three seeds averaged at the prediction level. Nothing is broken.

Re-running a completed cell because its number is inconvenient is precisely what the
pre-registration exists to prevent. Adding seeds to that one cell would also break protocol
symmetry with the other 17 cells. **Leave it alone and report it as observed.**

If anyone wants a better variance estimate on that cell in future, the correct route is a
separately pre-registered repeat study across *all* cells, not a top-up on one.

---

# Part B — arms C and D: separating factor 1 from factor 3

## B1. Why these two, and why together

After the K=1 and position-embedding arms, three factors remain open:

| # | Factor | Status |
|---|---|---|
| 1 | 8-slot chain vs 2-node transition graph | open |
| 3 | Attention readout vs stoichiometry-weighted | open |
| 4 | Discarded 16-d port-pair edge features | open |

**Factors 1 and 3 are confounded with each other.** The octamer changed both at once. Only the
2×2 separates them, and only the pair is informative:

| | Stoichiometry-weighted / mean readout | Attention readout |
|---|---|---|
| **2-node graph** | HPG-hier — *have* | **arm C** |
| **8-slot chain** | **arm D** | octamer — *have* |

Running one without the other leaves the attribution ambiguous. Treat them as a single arm.

## B2. Arm C needs no code change — verify this first

`chemprop/models/hpg_hier.py` already constructs an attention readout for the transition-graph
path:

```python
self.stage2_attention_readout = (
    AttentionReadout(d_h)
    if stage2_mode == "transition_graph" and stage2_readout == "attention" else None
)
```

**Before generating any jobs, verify that this object is actually used in `forward`**, not
merely constructed. A constructed-but-unused readout would silently train the stoichiometric
baseline — exactly the failure mode the arm-D guard was added to prevent. Trace the
transition-graph path through `forward` and confirm `stage2_attention_readout` is applied.
If it is not wired up, say so and stop; that is a bug to fix before the arm means anything.

## B3. Arm D needs a small patch

The constructor currently raises for `octamer_sequence` with any readout other than attention,
with a message referencing HANDOFF §7 arm D. That guard is correct and should stay for
unimplemented combinations — but arm D now needs implementing.

`OctamerEncoder.forward` ends:

```python
h = h_flat.reshape(n_reps, L, self.d_h)
return self.attention_readout(h)
```

Add a `readout` argument to `OctamerEncoder.__init__` taking `"attention"` or `"mean"`, and in
`forward` return `h.mean(dim=1)` when `readout == "mean"`. Then allow the constructor to build
`OctamerEncoder(..., readout="mean")` when
`stage2_mode == "octamer_sequence" and stage2_readout == "stoich_weighted"`.

**Why mean pooling is the correct analogue, not a substitute.** On an 8-slot chain the number
of slots holding monomer A is `n_A = round(8 · fracA)`, so the slot counts already encode
composition. Mean pooling over positions is therefore the exact analogue of the
stoichiometry-weighted sum `f_A·h_A + f_B·h_B` used on the 2-node graph. Pooling happens after
message passing, so arrangement information still reaches the readout; only the *learned
weighting of positions* is removed. This is stated in HANDOFF §7 — do not substitute max, sum
or gated pooling, and do not add more than one readout variant. Adding variants turns
attribution into a search.

**Keep the guard** for genuinely unimplemented combinations, and update its message so it no
longer implies arm D is unimplemented.

## B4. Write the pre-registration BEFORE submitting anything

Create `analysis/model_diagnostics/PREREG_arms_CD_2026-08-11.md`. Follow the structure of
`PREREG_octamer_posemb_2026-08-05.md`. It must contain, written before any result:

**The question.** Do factors 1 (8-slot topology) and 3 (attention readout) separate, and which
carries the octamer's advantage?

**The pre-registered reading**, lifted from HANDOFF §7 — this was written down in advance and
must not be altered now:

| Outcome | Reading |
|---|---|
| D ≈ octamer and C ≈ HPG-hier | **the sequence did it** — factor 1 |
| C ≈ octamer and D ≈ HPG-hier | **the readout did it** — factor 3 |
| Both midway | they interact; **do not attribute** |
| Both ≈ HPG-hier | it was factor 4 — the discarded edge features |

**The primary quantity: ΔR² on `all` rows**, reported separately for R1, R3-S and R3-D.

**The materiality threshold.** Derive it the same way the posemb pre-registration did — the
median per-cell across-seed SD of ΔR² for `hpg_hier_octamer` on `all` rows, from
`_regen_v1_results_individual_runs.csv`. Note in the document that re-deriving that quantity
today gives **0.0446** for R1, that the posemb pre-registration recorded **0.051**, and that
the discrepancy is documented in the 2026-08-11 addendum to that file. **State which value
this arm freezes and why**, before running.

**Negative controls**, by analogy with §6 of the posemb pre-registration:

- every sidecar shows the intended `stage2_mode` and `stage2_readout`
- arm D's parameter count differs from the octamer's by exactly the attention-readout
  parameters, and arm C's differs from HPG-hier's by the same amount — verify by
  reconstruction, since older sidecars do not record parameter counts
- output paths carry distinct tokens (`__armC`, `__armD`) under a new prediction directory; no
  file may overwrite an existing octamer or HPG-hier prediction
- no run at the epoch cap

**What this arm does not resolve.** Factor 4 stays open whatever the result. Say so.

## B5. Pilot first, then decide

Do **not** generate the full arm. Follow the pattern that worked for the position-embedding
arm:

**Pilot:** folds 0 and 4, EA only, three seeds, both arms — **12 runs**.

Fold 0 and fold 4 are chosen because both already have complete octamer and HPG-hier
comparators, and fold 4 is one of the higher-variance cells, so it tests the arms under
unfavourable conditions.

Report the pilot with **all five metrics** — overall R², RMSE, MAE, group-mean R² and ΔR² —
not ΔR² alone. Both previous ablations reported every metric and all of them agreed; that
agreement is load-bearing evidence that ΔR² is not simply a metric that finds differences
wherever it is pointed. Keep the practice.

The full R1 arm would be 2 arms × 9 folds × 2 targets × 3 seeds = **108 runs**, roughly 3.9
kSU at the octamer's measured per-run cost. **Do not submit it without approval.**

## B6. Protocol constraints

- Three seeds (42/43/44), averaged at the **prediction** level, metric computed once.
- `y_pred` (best checkpoint), never `y_pred_final`.
- All other settings identical to the existing octamer runs: batch 64, 100 epochs, patience 15,
  16 sampled sequences, chain length 8, `--frozen_protocol`.
- If any file is missing, list it and stop — do not substitute.
- Do not modify `evaluation/metrics.py`, the figures, or any dated document.

## B7. Report back

1. The result of the B2 verification — is `stage2_attention_readout` actually used in `forward`?
2. The arm-D patch, with the parameter-count reconstruction from B4.
3. The pre-registration file, with the threshold decision and its justification.
4. The pilot generator, job count, and confirmation that nothing was submitted beyond the pilot.
5. `py_compile` clean on every modified file; every file written, listed.
