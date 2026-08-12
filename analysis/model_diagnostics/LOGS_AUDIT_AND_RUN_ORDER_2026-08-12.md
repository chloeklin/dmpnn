# Run order, and a logs/ audit

Written 12 August 2026. Two parts: what to run and in what order (§1), and what to do with the
14 directories under `logs/` (§2–§4).

---

# 1. Run order

From `ARCHITECTURE_LADDER_2026-08-12.md`. Every rung is chemprop 2.2.0, three seeds averaged at
the prediction level, frozen protocol. `wdmpnn_original` is the only baseline and its
configuration is never changed.

## Do these first — they unblock a paper or cost nothing

| | Job | Runs | ~SU | Code | What it gives you |
|---|---|---|---|---|---|
| **1** | **Published-config wD-MPNN on the B split** | 54 | 0.35 k | **none — PBS files already exist** | The last hole in Paper 1. Nothing else blocks submission |
| **2** | **Analyse the junction on / 1-step / off runs** | **0** | **0** | none | Already run. Answers "does atom-level cross-monomer messaging help inside the hierarchy?" |

Job 1 is `logs/wdmpnn_original/r1_r3/pbs/` — the generator covered both splits from the start,
only the A half was ever submitted. Remove the one stray B-split EA fold-0 prediction first or
its job will hit the partial-output guard.

## Then the ladder, cheapest and most informative first

| | Job | Runs | ~SU | Code | Isolates |
|---|---|---|---|---|---|
| **3** | Arms C + D **pilot** | 12 | 0.45 k | done | **D4, readout** — which §2 of the ladder doc shows carries the accuracy gain |
| **4** | **M1 pilot**, both configs | 24 | 0.2 k | small patch | **D1**, monomer-level representation — plus the config bridge |
| **5** | M1 full, both configs | 108 | 0.7 k | — | as above, if the pilot is clean |
| **6** | M2 pilot → full | 12 → 54 | 0.45 → 2.0 k | moderate patch | **Topology vs edge features** — the confound arms C/D cannot break |
| **7** | Full arms C + D | 108 | 3.9 k | done | Only if step 3 warrants it |

**Steps 1–5 total ≈ 1.7 kSU** and cover both the accuracy story and the ΔR² story. That is the
package to take to a supervisor.

**Why arms C/D pilot before M1:** the code is already written and the pre-registration is
already filed. M1 needs a patch first. Run 3 while 4 is being implemented.

## Not in this ladder

The protocol-matching confound. Every rung trains against labels computed on 8-unit chains.
Only the chain-length sweep in the Paper 2 dataset addresses it.

---

# 2. logs/ audit — the headline

**Total 4.2 GB across 14 directories. About 95% of that is progress-bar rewrites, not content.**

A single 13.4 MB task log contains 103,148 carriage returns. Stripping each line back to its
final state leaves **613 KB — 4.5% of the original** — with every real line intact.

So the first action is not deletion. It is:

```bash
# for each logs/**/tasks/*.log, keep only the final state of each CR-rewritten line
python - <<'PY'
import glob, os
for p in glob.glob('logs/**/tasks/*.log', recursive=True):
    raw = open(p,'rb').read()
    if raw.count(b'\r') < 1000:      # nothing to gain
        continue
    clean = b'\n'.join(l.split(b'\r')[-1] for l in raw.split(b'\n'))
    open(p,'wb').write(clean)
PY
```

**Expected: 4.2 GB → roughly 200 MB, with no loss of information.**

**Do this before deleting anything**, because the size problem largely disappears and the
delete/keep decision becomes much less pressing.

### Why not just delete the logs

`scripts/python/analyze_regen_v1.py:322` requires the frozen-split assertion to be **confirmed
from the task logs, not inferred from output metadata** — you grep
`logs/regen_v1/r3/tasks/` for `Frozen monomer_b_heldout split assertions passed for all folds`,
`B-identity leakage` and `differs from frozen metadata`. Those logs are provenance evidence for
the current campaign. Strip them; never delete them.

---

# 3. Directory-by-directory

Dates are the first submitted job in each directory. **The dual-checkpoint runner landed
between 27 and 28 July** — `regen_v1` predictions carry `y_pred_final` and
`split_indices_sha256`; anything earlier does not, and is therefore affected by the
model-selection bug.

| Directory | Size | Submitted | Verdict | Action |
|---|---|---|---|---|
| `regen_v1` | 2.1 G | 28 Jul | **Current frozen campaign.** Logs are cited evidence | **Strip logs. Never delete** |
| `wdmpnn_original` | 5.3 M | 2 Aug | Current — published-config baseline | Keep |
| `backfill_regen_v1_missing_5` | 14 M | 2 Aug | Five missing regen_v1 cells | Strip, then **fold into `regen_v1`** — it is not a separate campaign |
| `octamer_k1` | 292 K | 3 Aug | Current — factor 5 ablation | Keep |
| `octamer_posemb` | 616 K | 10 Aug | Current — factor 2 ablation | Keep |
| `octamer_cd` | 56 K | 12 Aug | Current — arms C/D. **Contains MPS-trained runs** | Keep; add the quarantine note |
| `b_heldout_step1` | 292 K | — | Built the B-split scaffold metadata | Keep — it is split provenance |
| `hpg_stability_fixes` | 60 K | — | Feeds `predictions/stability_fixes/` | **Ask yourself:** is anything in the current results traced to it? If not, superseded |
| `hpg_noise_floor` | 24 K | 27 Jul | **Pre-checkpoint-fix.** The repeat study we removed from the deck | Mark superseded |
| `hpg_phase1` | **1.2 G** | 26 Jul | **Pre-fix → predictions void** | Strip logs; keep `pbs/` + manifest |
| `hpg_phase1_gates` | 95 M | 26 Jul | Pre-fix | Strip; keep manifest |
| `seed42_hpg_hier_vs_baselines` | 653 M | 20 Jul | **Pre-fix and single-seed → void** | Strip; keep manifest |
| `hpg_gates` | 8 K | 19 Jul | Manifests only, `group_disjoint` split we no longer use | Archive |
| `seeded_diagnostics` | 20 K | 21 Jul | Two stray PBS output files | **Safe to delete** |

**Only one directory I would delete outright** is `seeded_diagnostics` — two orphaned `.e`/`.o`
files with no manifest.

**Everything else: strip, then keep.** For void campaigns the `pbs/` and `manifest` files are
worth keeping even though the results are not — they document what was run, which is what a
thesis appendix and a reviewer will ask for. They are kilobytes once stripped.

---

# 4. Naming — fix the structure, not just the labels

Renaming alone will not fix "I'm lost". The problem is that fourteen sibling directories give
no signal about which are live. **Sort them into three buckets first**, then rename:

```
logs/
├── current/          # feeds live results — treat as evidence
│   ├── frozen3seed-all-models_2026-07/        <- regen_v1  (+ backfill folded in)
│   ├── baseline-published-config_2026-08/     <- wdmpnn_original
│   ├── ablation-sampling-K1_2026-08/          <- octamer_k1
│   ├── ablation-position-embeddings_2026-08/  <- octamer_posemb
│   ├── ablation-readout-armsCD_2026-08/       <- octamer_cd
│   └── split-construction-b-scaffold_2026-08/ <- b_heldout_step1
├── superseded/       # ran correctly, replaced by something better
│   ├── repeat-study-prefix_2026-07/           <- hpg_noise_floor
│   └── stability-fixes_2026-07/               <- hpg_stability_fixes
└── void/             # known-invalid: pre-checkpoint-fix or single-seed
    ├── phase1_2026-07/                        <- hpg_phase1
    ├── phase1-gates_2026-07/                  <- hpg_phase1_gates
    ├── singleseed-hpg-vs-baselines_2026-07/   <- seed42_hpg_hier_vs_baselines
    └── gates-group-disjoint_2026-07/          <- hpg_gates
```

### The naming rule

```
<what-varied>_<YYYY-MM>
```

*What varied*, not what it was called at the time. `regen_v1` tells you nothing;
`frozen3seed-all-models` tells you it is the three-seed frozen-protocol campaign across all
models. A name should answer "why does this directory exist?" without opening it.

### Add a README

One `logs/README.md`, ten lines, stating: what each bucket means, that `void/` is retained for
provenance and its results must never be quoted, and the date the dual-checkpoint runner landed
(27–28 July) — which is the single fact that determines which bucket a directory belongs in.

### Do the same to `predictions/`

`predictions/` has the same problem — 24 directories including `_checkpoint_smoke_hpg`,
`_dual_checkpoint_smoke_stage2d`, `wDMPNN_Pilot_Lambda`, `HPG2Stage_Ablation`. Same three
buckets, same rule. Worth doing in the same pass while the mapping is fresh.

---

## Suggested order for the cleanup

1. **Strip the progress bars.** Non-destructive, recovers ~4 GB, makes everything else easier.
2. **Delete `seeded_diagnostics`.** The only clear-cut deletion.
3. **Create the three buckets and move directories in.** Keep a mapping table in the README so
   old paths in old documents can still be traced.
4. **Update the two references** in `analyze_regen_v1.py` that hard-code `logs/regen_v1/r3/tasks/`.
5. Repeat for `predictions/` once `logs/` is settled.

Step 4 is the only one that can break something. Do it in the same commit as the move.
