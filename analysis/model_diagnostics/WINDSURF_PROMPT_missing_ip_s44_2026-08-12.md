# Windsurf task — diagnose 4 missing wD-MPNN B-split runs

Four cells of the published-config wD-MPNN B-split arm did not land. **50 of 54 predictions
exist.** This task is forensics plus a readiness check. **Do not submit anything, do not train
anything, do not log in to Gadi.**

---

## 1. What is already established — do not re-derive

I have done the local forensics. Take these as given and start from §2.

**The 4 missing cells:**

| Target | Split | Fold | Seed |
|---|---|---|---|
| IP vs SHE (eV) | `monomer_b_heldout_clustered` | 5 | 44 |
| IP vs SHE (eV) | `monomer_b_heldout_clustered` | 6 | 44 |
| IP vs SHE (eV) | `monomer_b_heldout_clustered` | 7 | 44 |
| IP vs SHE (eV) | `monomer_b_heldout_clustered` | 8 | 44 |

**What is on disk:**

- All 4 rows **are present** in `logs/wdmpnn_original/r1_r3/manifests/r1_r3_after_review.manifest`
  at 0-indexed rows **102, 103, 104, 105** — the last four rows of a 106-row manifest.
- The matching job files **exist**: `logs/wdmpnn_original/r1_r3/pbs/wdmpnn_orig_r_{102,103,104,105}.pbs`.
- **No partial artefacts locally** — no `.npz`, no `.config.json`, for any of the four.
- Every landed cell has the correct configuration: `batch_size=50, epochs=30, patience=30`,
  accelerator `cuda`.
- Three git commits appear across the arm (`7e936002`, `90e129b6`, `ef37695b`) but the wD-MPNN
  code path is **byte-identical** across all three — verified by
  `git diff --stat` over `chemprop/nn/message_passing/`, `chemprop/featurizers/` and
  `scripts/python/run_wdmpnn_generalization.py`. Not a cause.
- `logs/wdmpnn_original/` retains only **3 task logs** — the rest were pruned in a cleanup, so
  local logs cannot diagnose this.

**The pattern is the tail of the batch.** Rows 102–105 are the final four of 106, all seed 44,
all IP, consecutive folds. That is much more consistent with a truncated submission or a batch
tail failure than with four independent per-fold problems.

---

## 2. What to do locally

### 2.1 Verify the four job files are correct and ready to resubmit unchanged

For each of `wdmpnn_orig_r_{102,103,104,105}.pbs`, confirm:

- the `sed -n "<N>p"` line index it reads resolves to the intended manifest row (note
  `wdmpnn_orig_r_N.pbs` reads line `N+1`)
- the resolved arguments contain `--split_types monomer_b_heldout_clustered`, the right
  `--folds`, `--seed 44`, and `--batch_size 50 --epochs 30 --patience 30
  --frozen_protocol --protocol_variant original_paper`
- the PBS header matches the other 102 jobs — queue, project, storage, walltime, modules, venv
- the output path is the `__orig`-tokened B-split path

Report any discrepancy. If all four are correct, say so plainly: **they can be resubmitted
as-is, no regeneration needed.**

### 2.2 Check the resume guard will not block them

These jobs contain a guard that exits 1 if a partial output exists:

```
if [[ -e "$OUTPUT" || -e "${OUTPUT%.npz}.config.json" || -e "$BASE_OUTPUT" || -e "$BASE_CONFIG" ]]; then
    printf 'Partial output exists; refusing ambiguous resume: %s\n' "$OUTPUT" >&2
    exit 1
fi
```

Note it checks **both** the `__orig` path and the un-tokened `BASE_OUTPUT` the runner writes
before the rename. Nothing exists locally — but the jobs run against `/scratch` on Gadi, where
a half-written base file from a previous attempt would silently block all four.

**Write this into the Gadi checklist in §3 rather than assuming.**

### 2.3 Look for a submission-side cause

Search the repository for whatever was used to submit this arm — a loop over the PBS directory,
a submit script, a recorded job-id list (`submitted_job_ids.txt` exists for other arms).
Determine whether the submission covered all 106 jobs or stopped early. If a submit script
exists and has a limit, an off-by-one, or a `head`/`seq` bound that would cut the tail, that is
the answer and it should be reported and fixed.

If there is no record of what was submitted, say so — that is itself worth knowing, and worth
fixing for future arms by writing a job-id list at submission time.

---

## 3. Produce a Gadi diagnostic checklist for Chloe to run

**You must not run these.** Produce them as a copy-pasteable block with a one-line explanation
of what each answers.

Cover at least:

1. Are the jobs still queued or running? (`qstat -u $USER`, and history if available)
2. Do PBS `.e` / `.o` files exist for those four job names, and what do they say?
3. Do partial outputs exist under
   `/scratch/um09/hl4138/dmpnn/predictions/wdmpnn_original/ea_ip_lomo_b_clustered/`
   for `IP … fold{5,6,7,8}__s44` — including the **un-tokened base filename**, which is what
   would trip the resume guard?
4. Are there orphaned checkpoint directories under
   `/scratch/um09/hl4138/dmpnn/checkpoints/wdmpnn_original/wdmpnn/` for those cells?
5. Is the `hm62` allocation still live and does it have headroom? (`nci_account -P hm62`)
6. Did the four jobs hit the 6-hour walltime? (from the `.e` file or `qstat -fx`)

For each, state what the answer would imply and what to do next.

---

## 4. Do not

- Do not submit jobs, log in to Gadi, or transfer files.
- Do not regenerate the manifest — it is correct and other jobs read it by line index, so
  renumbering would break the mapping for every other job.
- Do not train locally. The `--frozen_protocol` CUDA guard should prevent this; confirm it is
  in place and would fire.
- Do not modify any landed prediction or sidecar.

---

## 5. Report back

1. §2.1 verdict — are the four job files resubmittable as-is?
2. §2.3 — was there a submission-side cause, and is there a record of what was submitted?
3. The §3 checklist, ready to paste.
4. A one-line recommendation: resubmit as-is, or fix something first.
