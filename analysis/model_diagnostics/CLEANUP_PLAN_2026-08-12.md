# logs/ and predictions/ cleanup — revised after your pass

Written 12 August 2026, after re-inspecting. **logs/ is down from 4.2 GB to 65 MB.** Three
directories are gone entirely and the big ones are stripped.

One thing was lost that needs replacing, and it is fixable with something better than what it
replaced. That is §1. Then predictions/ (§2), the naming scheme (§3), and the code fix (§4).

---

## 1. First: the regen_v1 task logs are gone, and the analyzer still cites them

`logs/regen_v1/` now contains **6 manifest files and nothing else** — `r1/tasks/`, `r3/tasks/`
and both `pbs/` directories are deleted.

That matters because `analyze_regen_v1.py:322` and `:438` state, in the generated report:

> *"Task logs must be downloaded alongside NPZs before the final report is generated, so that
> the frozen-split assertion can be confirmed from logs rather than inferred from output
> metadata … grep `logs/regen_v1/r3/tasks/` for `Frozen monomer_b_heldout split assertions
> passed for all folds` …"*

That instruction can no longer be followed. The assertion string now survives only in
`logs/backfill_regen_v1_missing_5/tasks/`, which covers 5 backfilled cells out of 483.

**Do not re-download the logs.** There is a better source of the same guarantee, and it was
there all along.

### The replacement: verify the split hash, not a log line

Every `regen_v1` prediction file carries `split_indices_sha256` — e.g. the B-split octamer
fold-0 EA cell records `031443b6…f7b2df4e`. And the frozen split definitions are on disk at
`metadata/splits/monomer_b_heldout_clustered.json` and `monomer_heldout.json`.

So the check becomes: **recompute the hash from the frozen metadata and compare it to the hash
recorded in each prediction file.**

This is strictly stronger than grepping a log:

- a log line says "the runner asserted the split matched"; the hash says "the indices actually
  used are byte-identical to the frozen definition"
- it covers **every cell**, not just those whose logs were retained
- it is reproducible now and in two years, from artefacts that must exist anyway
- it cannot silently pass because a log was regenerated, truncated or lost

**This is an upgrade, not a workaround.** Say so in the report rather than apologising for the
missing logs.

---

## 2. predictions/ — 216 MB, 27 directories

Same problem as logs/ had: no signal about which are live.

| Directory | Size | npz | Verdict |
|---|---|---|---|
| `regen_v1` | 40 M | 483 | **Current** — the frozen three-seed campaign |
| `wdmpnn_original` | 5.1 M | 55 | **Current** — published-config baseline |
| `octamer_k1` | 4.0 M | 53 | **Current** — factor 5 |
| `octamer_posemb` | 4.5 M | 54 | **Current** — factor 2 |
| `octamer_cd` | 0 | **0** | **Current but empty** — MPS runs correctly quarantined |
| `_quarantine_local_mps` | 580 K | 7 | **Quarantine** — keep, never analyse |
| `DMPNN` | 41 M | 1014 | Pre-fix baselines |
| `GAT` / `GIN` / `AttentiveFP` | 23 / 23 / 21 M | 339 / 333 / 480 | Pre-fix baselines |
| `ea_ip_lomo` / `ea_ip_group` / `ea_ip_pair` | 16 M each | 263 / 154 / 150 | Pre-fix, split-named — the naming is by *split*, not by campaign |
| `IdentityBaseline` | 5.2 M | 110 | Pre-fix |
| `HPG2Stage_Ablation` | 3.2 M | 54 | Pre-fix |
| `stability_fixes` | 992 K | 12 | Superseded |
| `noise_floor` | 384 K | 6 | **Superseded — pre-checkpoint-fix, 27 July** |
| `wDMPNN_Gen` / `wDMPNN_Pilot_Lambda` | 244 K / 496 K | 2 / 4 | Pilots |
| `_checkpoint_smoke_*` × 5, `_dual_checkpoint_smoke*` × 3 | ~200 K total | 1 each | **Smoke tests — safe to delete** |

**Delete outright:** the eight `_checkpoint_smoke_*` / `_dual_checkpoint_smoke*` directories.
One npz each, they were one-off verification runs, and the thing they verified is now covered
by the `y_pred_final` field being present in every real prediction.

**Keep everything else**, including the pre-fix baselines. 216 MB is not a problem, and those
predictions are the raw material for the "what the February–May phase produced" section of the
thesis. Their *results* are void; their existence is evidence.

**Note `ea_ip_lomo`, `ea_ip_group`, `ea_ip_pair` are named by split, while everything else is
named by campaign.** That inconsistency is a large part of why the directory is confusing.

---

## 3. Naming — the scheme, applied to what is actually there now

Three buckets, then `<what-varied>_<YYYY-MM>`. The name answers *"why does this exist?"*
without opening it.

### logs/

```
logs/
├── current/
│   ├── frozen3seed-all-models_2026-07/       <- regen_v1  (+ backfill_regen_v1_missing_5 folded in)
│   ├── baseline-published-config_2026-08/    <- wdmpnn_original
│   ├── ablation-sampling-K1_2026-08/         <- octamer_k1
│   ├── ablation-position-embeddings_2026-08/ <- octamer_posemb
│   ├── ablation-readout-armsCD_2026-08/      <- octamer_cd
│   └── split-construction-bscaffold_2026-08/ <- b_heldout_step1
├── superseded/
│   └── stability-fixes_2026-07/              <- hpg_stability_fixes
└── void/                                      # pre-checkpoint-fix (before 28 July)
    ├── phase1_2026-07/                        <- hpg_phase1
    ├── phase1-gates_2026-07/                  <- hpg_phase1_gates
    └── singleseed-hpg-vs-baselines_2026-07/   <- seed42_hpg_hier_vs_baselines
```

### predictions/

```
predictions/
├── current/
│   ├── frozen3seed-all-models_2026-07/       <- regen_v1
│   ├── baseline-published-config_2026-08/    <- wdmpnn_original
│   ├── ablation-sampling-K1_2026-08/         <- octamer_k1
│   ├── ablation-position-embeddings_2026-08/ <- octamer_posemb
│   └── ablation-readout-armsCD_2026-08/      <- octamer_cd
├── quarantine/
│   └── local-mps-wrong-hardware_2026-08/     <- _quarantine_local_mps
├── superseded/
│   ├── repeat-study-prefix_2026-07/          <- noise_floor
│   └── stability-fixes_2026-07/              <- stability_fixes
└── void/                                      # pre-checkpoint-fix
    ├── baselines-dmpnn_2026-0X/              <- DMPNN
    ├── baselines-gat_2026-0X/                <- GAT
    ├── baselines-gin_2026-0X/                <- GIN
    ├── baselines-attentivefp_2026-0X/        <- AttentiveFP
    ├── baselines-identity_2026-0X/           <- IdentityBaseline
    ├── hpg2stage-ablation_2026-0X/           <- HPG2Stage_Ablation
    ├── bysplit-lomo_2026-0X/                 <- ea_ip_lomo
    ├── bysplit-group_2026-0X/                <- ea_ip_group
    ├── bysplit-pair_2026-0X/                 <- ea_ip_pair
    └── wdmpnn-pilots_2026-0X/                <- wDMPNN_Gen, wDMPNN_Pilot_Lambda
```

**Two rules worth writing down:**

- **`void/` is retained for provenance. Nothing in it may be quoted as a result.** Put that
  sentence in the README so a future you, or a co-author, cannot mistake it.
- **The boundary is 28 July 2026** — the date the dual-checkpoint runner landed. A directory
  belongs in `void/` if its predictions lack `y_pred_final`. That is a mechanical test, not a
  judgement call:

```bash
python - <<'PY'
import numpy as np, glob, os
for d in sorted(glob.glob('predictions/*/')):
    fs = glob.glob(os.path.join(d,'**','*.npz'), recursive=True)
    if not fs: continue
    keys = np.load(fs[0], allow_pickle=True).files
    print(f"{'POST-FIX' if 'y_pred_final' in keys else 'VOID    '}  {d}")
PY
```

Run that and it classifies every directory for you.

### Add a README to each

`logs/README.md` and `predictions/README.md`, ten lines each: what the buckets mean, the
28 July boundary, the "never quote void/" rule, and **a mapping table from old names to new**
so paths in older documents can still be traced.

---

## 4. The code fix

Two sites in `scripts/python/analyze_regen_v1.py` hard-code `logs/regen_v1/r3/tasks/`:

- **line 322** — a note in the verification-gates section
- **line 438** — the provenance paragraph in the generated report

Both must change, because the path is gone *and* because the hash check is better evidence.

### What to do

**Replace the log-grep provenance with a split-hash verification.** Concretely:

1. Add a function that, for each prediction file, reads `split_indices_sha256`, recomputes the
   hash from the corresponding `metadata/splits/<split_type>.json` fold definition, and
   compares.
2. Make it a **hard gate**: raise if any cell's hash does not match, naming the cell. This
   replaces a manual grep with an automatic check, which is the right direction.
3. Rewrite both text blocks to state what is actually verified:

   > *Split integrity is verified per cell by comparing `split_indices_sha256` recorded in each
   > prediction file against the hash recomputed from `metadata/splits/<split>.json`. This is
   > checked for every cell at analysis time and gates report generation. It supersedes the
   > earlier practice of grepping task logs for a runner assertion, which covered only cells
   > whose logs were retained.*

4. Add a short note to `HANDOFF_2026-08-05.md` §10 recording that the provenance mechanism
   changed and why — the logs were pruned, and the replacement is stronger.

### Sequencing

Do the code fix **before** the directory moves, not in the same commit. If the hash gate is in
place first, you can run the analyzer immediately after the move and it will tell you whether
anything broke. Doing both at once means a failure could be either the move or the rewrite.

Suggested order:

1. Implement and test the hash gate; confirm all 483 regen_v1 cells pass. **Commit.**
2. Delete the eight `_*_smoke*` prediction directories. **Commit.**
3. Create the buckets, move directories, write both READMEs with the mapping tables. **Commit.**
4. Grep the repo for any remaining hard-coded old paths (`regen_v1`, `ea_ip_lomo`,
   `wdmpnn_original`) in `scripts/`, `analysis/` and `evaluation/`, and update. **Commit.**

Step 4 is the one that will surface surprises — figure builders and analyzers reference
prediction paths in a lot of places.
