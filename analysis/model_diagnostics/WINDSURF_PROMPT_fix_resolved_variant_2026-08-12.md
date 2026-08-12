# Windsurf task — `resolved_variant` records the preset, not what actually ran

A provenance bug in the sidecar. **No training is affected — the model is built correctly.**
But the field that a reader would trust to say what ran is wrong, and 108 more runs are about
to write it.

**Do not train, do not submit, do not log in to Gadi.**

---

## 1. The bug

`scripts/python/run_hpg_generalization.py`:

```python
line 339:  resolved_stage2_readout = args.stage2_readout or variant["stage2_readout"]
line 352:  ... stage2_readout=resolved_stage2_readout, ...        # model built CORRECTLY

line 549:  resolved_variant = _VARIANT_FLAGS.get(model_token, {})  # <- raw preset, unmodified
line 565:  "resolved_variant": resolved_variant,                   # <- written to sidecar as-is
line 566:  "resolved_stage2_readout": args.stage2_readout or resolved_variant.get("stage2_readout"),
```

So `resolved_variant` is the **preset table entry**, ignoring every CLI override. Only the
separate top-level `resolved_stage2_readout` carries the truth, and the two disagree.

### Evidence from the arms C/D pilot

All six arm D runs were launched with `--stage2_readout stoich_weighted`. Their sidecars say:

| field | value |
|---|---|
| `resolved_variant.stage2_readout` | **`attention`** ← wrong |
| `resolved_stage2_readout` | `stoich_weighted` ← correct |

The model did train with mean pooling — confirmed independently by parameter count:
arm D records `n_octamer_params = 133376`, and the octamer is 133505 (132481 from the
position-embedding arm, plus 8 × 128 = 1024 for the embeddings). The difference is exactly
**129**, a `Linear(128, 1)` attention readout that arm D does not have.

**So the only thing distinguishing arm D from the octamer in its own provenance record is a
parameter count that has to be reconstructed from a third arm.** That is not acceptable
provenance for a pre-registered ablation.

---

## 2. The fix

Make `resolved_variant` describe **what was actually built**, not the preset.

The clean approach: build the resolved variant dict once, near line 339 where
`resolved_stage2_readout` is already computed, and use that same object both to construct the
model and to write the sidecar. Do not compute it twice in two places — that is what allowed
them to drift.

**Check every field, not just the readout.** `_VARIANT_FLAGS` entries carry five keys:

```
stage2_edge_weight, stage2_mode, stage2_readout, junction_coupling, n_coupling_steps
```

and the CLI exposes overrides that can affect them, including `--stage2_readout`,
`--stage2_edge`, `--n_coupling_steps`, and `--octamer_position_embeddings`. **Audit each one**:
for every key in the variant dict, determine whether a CLI argument can change what the model
actually does, and make the recorded value reflect the override. Report any field where the
preset and the built model can currently disagree — the readout may not be the only one.

Also record `octamer_position_embeddings` and `octamer_len` in the sidecar if they are not
already there, since both change the architecture and neither is in `_VARIANT_FLAGS`.

### Keep `resolved_stage2_readout`

Do not delete the top-level field. Other code and several analysis documents reference it, and
removing it would break the one thing that was correct. After the fix the two must agree; add
an assertion that they do, so any future drift fails loudly rather than silently.

---

## 3. Backfill the existing sidecars — carefully

Twelve arm C/D sidecars carry the wrong `resolved_variant.stage2_readout`. Correct them, but:

- **Only correct fields you can prove from evidence already in the file** — the CLI args are
  recorded in `cli_args`, so the resolved value is recoverable without guessing.
- **Do not touch any other field**, and do not rewrite sidecars for arms where preset and
  override agree.
- Add a `provenance_corrected` key recording the date and what changed, so a corrected file is
  distinguishable from one that was written correctly in the first place.
- Report exactly which files were modified and what changed in each.

If you cannot recover the resolved value from a file's own contents, leave it alone and report
it. Do not infer from filenames.

---

## 4. Tests

- A run with `--stage2_readout stoich_weighted` on `hpg_hier_octamer` writes
  `resolved_variant.stage2_readout == "stoich_weighted"`.
- A run with no override writes the preset value unchanged.
- The assertion in §2 fires when the two fields are made to disagree.
- One test per additional field found in the §2 audit.

Tests must not train — construct the resolved dict directly.

---

## 5. Report back

1. The audit from §2 — which fields could disagree, and which now cannot.
2. The single place `resolved_variant` is now built.
3. Which of the 12 sidecars were corrected, and the before/after for each.
4. Test results, `py_compile` clean, every file written.

**This must land before the full 108-run arm is submitted**, or those runs will write the same
misleading field.
