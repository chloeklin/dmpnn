# Windsurf task — analyse the completed octamer position-embedding R1 arm

All 54 `__noposemb` predictions have landed. The arm is complete. Analyse it **strictly against
`analysis/model_diagnostics/PREREG_octamer_posemb_2026-08-05.md`**, which is immutable — any
post-hoc change goes in a dated addendum at the bottom of that file, never as an edit.

**Scope:** `scripts/python/analyze_octamer_posemb.py` and its report output. Do not modify the
pre-registration body, the figures under `analysis/paper1_figures/`, or `evaluation/metrics.py`.

---

## 1. Run the analysis without `--partial`

```bash
.venv/bin/python scripts/python/analyze_octamer_posemb.py
```

Drop `--partial` — all 18 cells now exist. **This is the first time the 3-seed assertion at
`analyze_octamer_posemb.py:584` will actually execute**; it was previously gated off and was only
ever tested against a synthetic frame. If it raises, that is the guard working. Report the message
verbatim rather than adding `--partial` to get past it.

---

## 2. Numbers to reproduce as assertions

Computed independently from the raw `.npz` files, three seeds averaged at the **prediction** level,
metric computed once on the averaged predictions. Pre-registered R1 threshold: **±0.051**.

`delta_r2`, all rows, noposemb minus baseline:

| fold | EA base | EA abl | EA diff | IP base | IP abl | IP diff |
|---|---|---|---|---|---|---|
| 0 | 0.8098 | 0.7572 | **−0.0527** | 0.8425 | 0.7346 | **−0.1080** |
| 1 | 0.8744 | 0.8645 | −0.0100 | 0.8856 | 0.8686 | −0.0170 |
| 2 | 0.8898 | 0.8772 | −0.0126 | 0.7515 | 0.6644 | **−0.0871** |
| 3 | 0.8652 | 0.8575 | −0.0077 | 0.9441 | 0.9164 | −0.0277 |
| 4 | 0.4151 | 0.3998 | −0.0153 | 0.5694 | 0.6203 | +0.0509 |
| 5 | 0.8851 | 0.8866 | +0.0015 | 0.9132 | 0.9297 | +0.0165 |
| 6 | −0.1409 | −0.0361 | **+0.1048** | 0.9287 | 0.9335 | +0.0048 |
| 7 | 0.8126 | 0.8176 | +0.0050 | 0.8605 | 0.8509 | −0.0096 |
| 8 | 0.8489 | 0.8345 | −0.0144 | 0.8892 | 0.9075 | +0.0183 |

Summary statistics to assert:

| | EA | IP |
|---|---|---|
| median diff | **−0.0100** | **−0.0096** |
| folds outside ±0.051 | 2 of 9 | 2 of 9 |
| negative folds | 6 of 9 | 5 of 9 |
| paired sign test, two-sided | **p = 0.5078** | **p = 1.0000** |

Minimum attainable p at n = 9 is 0.0039. Neither target approaches it.

**If your run disagrees with any figure above, stop and report the discrepancy. Do not adjust the
aggregation to make them match.**

---

## 3. The pre-registered outcome

Both medians sit well inside `[−0.051, +0.051]`, and neither sign test is close to significance.
By §5 of the pre-registration this is **outcome 3 — "no material change"** on R1.

Report it in exactly those terms, and carry the consequence the pre-registration already committed
to in writing:

> Factor 2 is excluded, like factor 5. Remaining candidates are factor 1 (8-slot topology) and
> factor 4 (discarded 16-d port-pair edge features). **No ablation at the current noise floor can
> separate those two** — state this as a limit of the dataset, not of effort.

Constraints on how this is written:

- The pre-registration's outcome 3 is phrased across both splits. **R3 has not been run.** Report
  the outcome as **R1-only** and say so explicitly; do not imply the arm is closed on both splits.
- Do not describe the ablated model as "position-blind". §2 of the pre-registration is explicit
  that this is a *reduction* of positional information, not its elimination — end slots and
  interior slots remain distinguishable through path structure.
- Do not read the two out-of-band folds per target as a trend. See §4.

---

## 4. Cells that must be flagged as unstable

Three cells have baseline across-seed `delta_r2` SD large enough that their diffs carry almost no
information. Report the SD beside every diff, and name these three explicitly:

| cell | baseline ΔR² | baseline seed SD |
|---|---|---|
| EA fold 6 | **−0.1409** | **0.8125** |
| EA fold 4 | 0.4151 | 0.4817 |
| IP fold 4 | 0.5694 | 0.3426 |

EA fold 6's baseline is *negative* with an SD of 0.81 — larger than the entire plausible range of
the metric. Its "+0.1048 improvement" is not evidence of anything and must not be reported as the
arm's largest positive effect without that caveat attached in the same sentence.

Add a **sensitivity check**: recompute the medians and sign tests with these three cells excluded,
report both versions side by side, and state whether the outcome-3 conclusion changes. Present this
as a robustness check, not as a replacement for the pre-registered analysis — the pre-registered
numbers remain the headline.

---

## 5. Negative controls (§6 of the pre-registration)

I have already checked four of these across all 54 ablated sidecars. Reproduce them and confirm:

| control | result |
|---|---|
| `octamer_position_embeddings == "off"` | **passes**, 0 violations of 54 |
| `frozen_protocol == true` | **passes**, 0 of 54 |
| `batch_size == 64` | **passes**, 0 of 54 |
| `stage2_mode == "octamer_sequence"`, readout `attention` | **passes**, 0 of 54 |
| no run at the 100-epoch cap | **passes** — max `best_epoch` is 16 |
| `__noposemb` path token, no collision with K=16 | **passes**, 54 sidecars, all tokened |

### 5.1 Control 2 cannot be verified from artefacts — resolve it

The parameter-count check ("`n_octamer_params` must differ from the K=16 baseline by exactly
1024") **cannot currently be performed.** Ablated sidecars record
`resolved_config.n_octamer_params = 132481`. The baseline `regen_v1` sidecars **do not contain the
field at all** — it postdates those runs. There is no parameter count anywhere in them.

Resolve it by construction rather than declaring the control passed:

1. Instantiate the octamer model twice from the recorded baseline config — once with position
   embeddings on, once off — and count parameters directly.
2. Confirm the on-minus-off difference is exactly `octamer_len × d_h = 8 × 128 = 1024`, and that
   the "off" count equals the 132481 recorded in the ablated sidecars.
3. Record in the report that this control was verified **by reconstruction, not from the baseline
   artefacts**, and state why.

### 5.2 The two arms ran on different commits — check it

Baseline sidecars: `git_commit = cec9d5feea303e0f655c22c94e76034ca7bd45cb`.
Ablated sidecars: `git_commit = a1e85cec074b37aeafc9d8bb4ac8cd6ffc128695`.

The flag had to be added, so a difference is expected. What is **not** established is that nothing
else on the octamer code path changed between those two commits. Given that an unrelated 18 June
change silently voided an entire family of earlier results, this must be checked, not assumed.

Run `git diff cec9d5f..a1e85ce` restricted to the octamer path — at minimum
`chemprop/models/hpg_hier.py`, `chemprop/featurizers/molgraph/hpg_hier.py`,
`scripts/python/run_hpg_generalization.py` — and report every hunk that is not the
position-embedding flag. If any behavioural change is found, **the comparison is confounded and
the arm is not reportable until it is resolved.** Say so plainly rather than noting it in passing.

---

## 6. Deliverables

1. The regenerated report, with per-fold diffs, per-cell seed SDs, both sign tests, the outcome-3
   statement scoped to R1, and the §4 sensitivity check.
2. Confirmation of every §2 assertion, individually.
3. The §5.1 parameter-count reconstruction with the numbers produced.
4. The §5.2 commit diff, with every non-flag hunk on the octamer path listed, or an explicit
   statement that there are none.
5. Every file written, listed. `py_compile` clean on anything modified.

Do **not** generate or submit the R3 arm. That gate is opened separately once this analysis is
reviewed.
