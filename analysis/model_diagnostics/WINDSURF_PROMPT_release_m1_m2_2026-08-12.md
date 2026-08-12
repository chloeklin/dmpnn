# Windsurf task — land the provenance fix, then emit the M1 and M2 pilot jobs

Two pilots are written but no PBS files exist yet. One blocker must clear first.

**Generate only. Do not train, do not submit, do not log in to Gadi.**

---

## 1. First: the `resolved_variant` fix has not landed

`scripts/python/run_hpg_generalization.py` still reads:

```python
line 555:  resolved_variant = _VARIANT_FLAGS.get(model_token, {})   # raw preset
line 571:  "resolved_variant": resolved_variant,                    # written unmodified
line 572:  "resolved_stage2_readout": args.stage2_readout or resolved_variant.get("stage2_readout"),
```

This is the same code as before, shifted by six lines. It was reported as fixed; it is not.

**Why it blocks M2 specifically.** M2 is a new variant whose readout differs from its preset —
exactly the arm D situation, where the sidecar claimed `attention` while the model trained with
mean pooling, and only a reconstructed parameter count could tell them apart. Releasing 27 M2
runs now bakes that ambiguity into a pre-registered ablation.

Apply the fix from `WINDSURF_PROMPT_fix_resolved_variant_2026-08-12.md` in full:

- build the resolved variant dict **once**, near where `resolved_stage2_readout` is already
  computed, and use that same object to construct the model and to write the sidecar
- audit **all five** `_VARIANT_FLAGS` keys — `stage2_edge_weight`, `stage2_mode`,
  `stage2_readout`, `junction_coupling`, `n_coupling_steps` — and report which could disagree
  with the built model, not just the readout
- also record `octamer_position_embeddings` and `octamer_len`, both of which change the
  architecture and neither of which is in the preset table
- keep `resolved_stage2_readout`, and add an assertion that the two agree
- backfill the 12 arm C/D sidecars, recovering values only from each file's own `cli_args`, and
  add a `provenance_corrected` key so corrected files are distinguishable

**Report the audit before moving on.** If a second field can drift, that changes what the arm
C/D pilot means and I want to know before more jobs are written.

---

## 2. Comparator status — one gap remains

| Comparator | Status |
|---|---|
| `hpg_hier`, all 108 cells | **complete** — EA fold 2 seed 43 landed, CUDA, batch 64 / 100 ep / patience 15 |
| `hpg_hier_octamer`, all 108 | complete |
| `wdmpnn` (regen_v1), all 108 | complete |
| **`hpg_hier_junction`** | **107/108 — A split, EA, fold 2, seed 42 still missing** |

The junction gap does **not** block M1 or M2 — neither uses it as a comparator. It blocks the
separate junction analysis only. Do not let it hold up job generation, but do report it if
either generator's pre-flight touches it.

**Keep the hard-fail behaviour in both generators.** Blocking on a missing comparator was the
correct call.

---

## 3. Emit the M1 pilot jobs

Re-run `scripts/shell/generate_m1_pilot.sh`.

Expected: **27 jobs** — EA only, folds 0–8, seeds 42/43/44, our configuration (batch 64,
100 epochs, patience 15, `--frozen_protocol`), written to `logs/m1/pilot/pbs/`, token `__m1`,
prediction directory `predictions/m1/`.

Plus the separate **27-job config-bridge manifest** at the published configuration (batch 50,
30 epochs, patience 30), token `__m1pub` — in its own manifest so it can be held back until
M1(ours) looks sane.

If the count is not 27 per manifest, stop and report rather than adjusting the generator.

---

## 4. Emit the M2 pilot jobs

Only after §1 has landed. Re-run `scripts/shell/generate_m2_pilot.sh`.

Expected: **27 jobs** — EA only, folds 0–8, seeds 42/43/44, our configuration, written to
`logs/m2/pilot/pbs/`, token `__m2`, prediction directory `predictions/m2/`.

**Report which folds lack an arm D comparator.** Arm D currently exists for folds 0 and 4 only,
so the M2 ↔ arm D contrast — the one that separates topology from edge features — will be
incomplete on 7 of 9 folds. State this plainly in the report and in the M2 pre-registration; do
not let it be discovered at analysis time.

---

## 5. Verify before reporting done

For **one** job file from each pilot, confirm the resolved arguments contain:

- the correct `--split_types monomer_heldout`, fold, seed
- `--batch_size 64 --epochs 100 --patience 15 --frozen_protocol`
- for M2, the variant token and whatever flag selects the edge-feature path
- for M1, the flag selecting the monomer-level aggregation
- the correct output path and token
- a PBS header matching `generate_octamer_posemb_r1.sh` — `gpuvolta`, `ncpus=12`, `ngpus=1`,
  `mem=100GB`, `jobfs=100GB`, `walltime=06:00:00`, project `hm62`, storage
  `scratch/um09+gdata/dk92`

Confirm no `qsub`, `ssh`, `scp` or `rsync` appears in either generator or any emitted job.

---

## 6. Report back

1. The §1 audit — which variant fields could drift, which now cannot, and the 12 backfilled
   sidecars with before/after.
2. Job counts: M1 pilot, M1 bridge, M2 pilot.
3. Which folds lack an arm D comparator.
4. The §5 spot-check for one job from each pilot.
5. Confirmation that nothing was submitted and nothing trained locally.
6. `py_compile` clean; every file written, listed.

Estimated cost once submitted: M1 pilot ≈ 0.2 kSU, M1 bridge ≈ 0.2 kSU, M2 pilot ≈ 1.0 kSU.
