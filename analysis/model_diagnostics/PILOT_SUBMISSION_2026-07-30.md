# Pilot submission — wDMPNN original-paper (Arm A) + octamer K=1 (Arm B)

Billing project: **`hm62`** (both Arm A wDMPNN original and Arm B octamer
K=1). Copy/paste in order. Nothing in this document has been run for you —
every `qsub` is your call.

**Two environments are used below, labeled on every code block:**

- **[LOCAL]** — your machine, repo root (`/Users/u6788552/Desktop/experiments/dmpnn`).
  Manifest/PBS generation and all sidecar/result checks run here, since
  results are always downloaded locally after jobs finish.
- **[GADI]** — an SSH session on the Gadi login node. Only `qsub`/`qstat`
  and live log tailing (before you've synced logs back) happen here.

On Gadi, set:

```bash
# [GADI]
export PROJECT_DIR=/scratch/um09/hl4138/dmpnn
```

Whenever a step below generates manifests/PBS files locally, you must
**sync `logs/` to Gadi before `qsub`** (the generated `.pbs` files embed
absolute `$PROJECT_DIR/logs/...` manifest paths), and **sync
`predictions/` back to local before running any sidecar check**:

```bash
# [LOCAL] push generated manifests/PBS to Gadi (replace `gadi` with your SSH alias/host)
rsync -avz logs/ gadi:/scratch/um09/hl4138/dmpnn/logs/

# [LOCAL] pull results back after jobs finish
rsync -avz gadi:/scratch/um09/hl4138/dmpnn/predictions/ predictions/
```

---

## 1. Grant check

Check the `hm62` compute balance before spending anything:

```bash
# [GADI]
nci_account -P hm62
```

---

## 2. Arm A — wDMPNN original-paper pilot (2 jobs)

Ready now, no prerequisites.

### 2.1 Generate the manifests + per-task PBS files

```bash
# [LOCAL]
bash scripts/shell/generate_wdmpnn_original_r1_r3.sh
```

Confirm the printed summary says `Pilot jobs: 2` and `Post-review jobs: 106`.
Then sync `logs/` to Gadi (see the rsync command above).

### 2.2 Submit the first pilot job (R1 / monomer_heldout)

```bash
# [GADI]
qsub $PROJECT_DIR/logs/wdmpnn_original/r1_r3/pbs/wdmpnn_orig_p_0.pbs
```

### 2.3 Submit the second pilot job (R3 / monomer_b_heldout_clustered)

```bash
# [GADI]
qsub $PROJECT_DIR/logs/wdmpnn_original/r1_r3/pbs/wdmpnn_orig_p_1.pbs
```

### 2.4 Check job status

```bash
# [GADI]
qstat -u $USER
```

---

## 3. Arm B — octamer K=1 pilot (2 jobs)

Prerequisite: all 54 K=16 comparators must exist under
`predictions/regen_v1/ea_ip_lomo_b_clustered/`, and the fold 7 / seed 44 /
IP / hpg_hier_octamer backfill cell must have finished.

### 3.1 Count the 54 K=16 comparators

```bash
# [LOCAL]
find "predictions/regen_v1/ea_ip_lomo_b_clustered" -maxdepth 1 -name 'ea_ip__*__hpg_hier_octamer__monomer_b_heldout_clustered__fold*__s*.npz' | wc -l
```

Expect `54`. If it is anything else, stop — do not run the generator.

### 3.2 Check the specific backfill cell (hpg_hier_octamer, IP, fold 7, seed 44)

```bash
# [LOCAL]
test -f "predictions/regen_v1/ea_ip_lomo_b_clustered/ea_ip__IP_vs_SHE_eV__hpg_hier_octamer__monomer_b_heldout_clustered__fold7__s44.npz" && echo "EXISTS" || echo "MISSING - backfill still running, do not proceed"
```

`EXISTS` must print. If `MISSING` prints (or the count in 3.1 was not 54),
stop here — do not run 3.3 or submit anything.

### 3.3 Only if both 3.1 and 3.2 pass — generate the manifests + per-task PBS files

```bash
# [LOCAL]
bash scripts/shell/generate_octamer_k1_r3.sh
```

This script re-checks both conditions itself (against your local
`predictions/` copy) and exits 1 if either fails, so if 3.1/3.2 passed but
something changed underneath you, the generator will refuse and print the
offending path(s). Confirm the printed summary says `Pilot jobs: 2` and
`Post-pilot jobs: 52`. Then sync `logs/` to Gadi (see the rsync command
above) — the generated PBS files embed Gadi-absolute manifest paths and
won't run until the manifests exist there too.

### 3.4 Submit the first pilot job (fold 0)

```bash
# [GADI]
qsub $PROJECT_DIR/logs/octamer_k1/r3/pbs/oct_k1_p_0.pbs
```

### 3.5 Submit the second pilot job (fold 4)

```bash
# [GADI]
qsub $PROJECT_DIR/logs/octamer_k1/r3/pbs/oct_k1_p_1.pbs
```

Check status the same way as Arm A:

```bash
# [GADI]
qstat -u $USER
```

---

## 4. When the pilots finish

### 4.1 Arm A — tail the job logs

While the job is still running (before syncing back), tail it directly on
Gadi:

```bash
# [GADI]
tail -n 100 $PROJECT_DIR/logs/wdmpnn_original/r1_r3/tasks/wdmpnn_orig_p_0_*.log
tail -n 100 $PROJECT_DIR/logs/wdmpnn_original/r1_r3/tasks/wdmpnn_orig_p_1_*.log
```

Once finished, sync `logs/` and `predictions/` back (see rsync commands at
the top) and you can tail/inspect locally instead.

### 4.2 Arm A — sidecar check

```bash
# [LOCAL] — run after syncing predictions/ back from Gadi
python3 <<PY
import json

paths = [
    "predictions/wdmpnn_original/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_heldout__fold0__s42__orig.config.json",
    "predictions/wdmpnn_original/ea_ip_lomo_b_clustered/ea_ip__EA_vs_SHE_eV__wdmpnn__monomer_b_heldout_clustered__fold0__s42__orig.config.json",
]
for p in paths:
    d = json.load(open(p))
    epochs = d["epochs_actually_run"]
    wall = d["wall_time_seconds"]
    print(p)
    print("  resolved_config:", d["resolved_config"])
    print("  epochs_actually_run:", epochs)
    print("  wall_time_seconds:", wall)
    print("  seconds_per_epoch:", wall / epochs)
    print()
PY
```

Expected values in `resolved_config`: `batch_size=50`, `epochs=30`,
`patience=30`, `protocol_variant='original_paper'`, `frozen_protocol=True`.
Expected `epochs_actually_run`: `30`.

### 4.3 Arm B — tail the job logs

```bash
# [GADI]
tail -n 100 $PROJECT_DIR/logs/octamer_k1/r3/tasks/oct_k1_p_0_*.log
tail -n 100 $PROJECT_DIR/logs/octamer_k1/r3/tasks/oct_k1_p_1_*.log
```

### 4.4 Arm B — sidecar check

```bash
# [LOCAL] — run after syncing predictions/ back from Gadi
python3 <<PY
import json

paths = [
    "predictions/octamer_k1/ea_ip_lomo_b_clustered/ea_ip__EA_vs_SHE_eV__hpg_hier_octamer__monomer_b_heldout_clustered__fold0__s42__k1.config.json",
    "predictions/octamer_k1/ea_ip_lomo_b_clustered/ea_ip__EA_vs_SHE_eV__hpg_hier_octamer__monomer_b_heldout_clustered__fold4__s42__k1.config.json",
]
for p in paths:
    d = json.load(open(p))
    epochs = d["epochs_actually_run"]
    wall = d["wall_time_seconds"]
    print(p)
    print("  resolved_config:", d["resolved_config"])
    print("  epochs_actually_run:", epochs)
    print("  best_epoch:", d.get("best_epoch"))
    print("  wall_time_seconds:", wall)
    print("  seconds_per_epoch:", wall / epochs)
    print()
PY
```

Expected values in `resolved_config`: `n_random_samples=1`, `batch_size=64`,
`epochs=100`, `patience=15`, `frozen_protocol=True`. Expected `best_epoch`
below `100` (early stopping should have fired, unlike Arm A).

---

## 5. Stop conditions — do not skip these

1. **wDMPNN per-epoch time.** If `seconds_per_epoch` from 4.2 is above
   `700` for either pilot job, **do not submit the remaining 106 jobs**.
   Walltime is `06:00:00` = 21,600 s across 30 epochs, so anything above
   `720` s/epoch will be killed mid-run. If this happens, raise `WALLTIME`
   in `scripts/shell/generate_wdmpnn_original_r1_r3.sh` and re-run the
   generator (step 2.1) before submitting anything else.

2. **wDMPNN epoch count.** If `epochs_actually_run` from 4.2 is anything
   other than `30` for either pilot job, **stop**. It means early stopping
   fired and the NoamLR schedule was truncated — the exact problem this
   re-run exists to fix. Do not submit the remaining 106 jobs; go back and
   diagnose why `patience=30` against `epochs=30` didn't prevent it (per-task
   PBS already asserts this and would have failed the job, so if the job
   "succeeded" with `epochs_actually_run != 30`, treat that as a bug in the
   guard, not a green light).

---

## 6. Submit the remainders — only after the checks in §5 pass

wDMPNN, 106 jobs. The remainder manifest is ordered split-major, so task
indices `0`-`52` are R1 `monomer_heldout` (53 jobs) and `53`-`105` are R3
`monomer_b_heldout_clustered` (53 jobs) — submit either subset independently
if you want to stage them:

```bash
# [GADI] — R1 monomer_heldout only (53 jobs)
for i in $(seq 0 52); do qsub $PROJECT_DIR/logs/wdmpnn_original/r1_r3/pbs/wdmpnn_orig_r_${i}.pbs; done
```

```bash
# [GADI] — R3 monomer_b_heldout_clustered only (53 jobs)
for i in $(seq 53 105); do qsub $PROJECT_DIR/logs/wdmpnn_original/r1_r3/pbs/wdmpnn_orig_r_${i}.pbs; done
```

```bash
# [GADI] — or all 106 at once
for f in $PROJECT_DIR/logs/wdmpnn_original/r1_r3/pbs/wdmpnn_orig_r_*.pbs; do qsub "$f"; done
```

Octamer K=1, 52 jobs:

```bash
# [GADI]
for f in $PROJECT_DIR/logs/octamer_k1/r3/pbs/oct_k1_r_*.pbs; do qsub "$f"; done
```
