# Windsurf — STOP the local pilot. PBS scripts only.

**Read this before doing anything else. Stop the background runner first.**

---

## 1. Stop, now

Kill the background runner (`90d2ae`) and do not restart it. Do not launch any further
training on this machine.

---

## 2. What went wrong

Training was executed **locally on Apple Silicon (MPS)**. Every existing result in this
project was produced on Gadi, on NVIDIA V100 GPUs under CUDA. The two are not comparable:

| | Runs you just produced | Every existing baseline |
|---|---|---|
| accelerator | **mps** | **cuda** |
| device | Apple Silicon | Tesla V100-SXM2-32GB |
| torch | **2.7.1** | **2.8.0+cu128** |
| `cudnn_deterministic` | False | True |
| git commit | 90e129b6 | cec9d5fe / 52cf8712 |

Arms C and D exist to be compared against `hpg_hier` and `hpg_hier_octamer`. If the arms run
on MPS with torch 2.7.1 and the baselines ran on CUDA with torch 2.8.0, **any difference
between them confounds the thing being tested with the hardware and library stack.** The arm
cannot answer its question. This is the same class of error that previously voided a whole
family of results in this project through an unnoticed commit difference.

### 2.1 The K=1 seed-44 run made a clean cell worse

`IP fold 7` of the K=1 arm previously had two seeds, both CUDA/V100/torch 2.8.0, commit
52cf8712. It was incomplete but internally consistent, and it was flagged as such.

It now has three seeds, of which one ran on MPS/torch 2.7.1/commit 90e129b6. **A flagged
2-seed cell is honest; a 3-seed cell that silently mixes two hardware stacks is not.** This is
a regression, not a fix.

---

## 3. Quarantine, do not delete

Move — do not delete — every locally-trained artefact to `predictions/_quarantine_local_mps/`,
preserving relative paths, and write a `README.md` in that directory stating what they are, why
they are quarantined, and the date.

Files to move:

1. Everything under `predictions/octamer_cd/` produced by the local runner (all `__armC` /
   `__armD` `.npz` and `.config.json`).
2. `predictions/octamer_k1/ea_ip_lomo_b_clustered/ea_ip__IP_vs_SHE_eV__hpg_hier_octamer__monomer_b_heldout_clustered__fold7__s44__k1.npz`
   and its `.config.json`.

Then **re-run `analyze_octamer_k1.py`** so IP fold 7 returns to a flagged 2-seed cell, and
confirm the outcome is still C.

Before moving anything, verify by reading each sidecar's
`runtime_environment.accelerator` — quarantine exactly those with `accelerator != "cuda"`, and
report the list. Do not quarantine on filename alone.

---

## 4. What to produce instead: PBS scripts, generated locally, submitted by hand

**Never execute training on this machine. Never attempt to log in to Gadi, submit jobs, or
transfer files.** Your job ends at writing PBS scripts and a manifest to disk. Chloe submits
them.

Follow the existing pattern exactly — `scripts/shell/generate_octamer_posemb_r1.sh` is the
template. Note its header: *"run locally (off Gadi) to generate PBS files. It does not
submit."* Match that behaviour and say so in your own script's header.

Write `scripts/shell/generate_octamer_cd_pilot.sh` (replacing the current version) so that it:

- writes PBS job files to `logs/octamer_cd/pilot/pbs/` and a manifest to
  `logs/octamer_cd/pilot/manifests/`
- **submits nothing and trains nothing**
- carries the same PBS header fields as the posemb generator: queue `gpuvolta`, `ncpus=12`,
  `ngpus=1`, `mem=100GB`, `jobfs=100GB`, walltime `06:00:00`, the same module loads, the same
  venv activation path, and `PROJECT_DIR=/scratch/um09/hl4138/dmpnn`
- uses the same storage directive, and states the charge project explicitly at the top so it
  can be checked before submission
- runs the same pre-flight checks: refuse to emit a job whose output already exists, and verify
  each comparator prediction is present

### The pilot, unchanged in scope

12 runs: arms C and D × folds 0 and 4 × EA only × seeds 42, 43, 44.

Do **not** generate the 108-run full R1 arm.

---

## 5. Keep the work that is still valid

None of the following is affected by the hardware problem, and none of it should be redone:

- `PREREG_arms_CD_2026-08-11.md` — written before any job ran. Keep as is.
- The arm-D mean-pooling patch to `OctamerEncoder`.
- The verification that `stage2_attention_readout` is genuinely used in `forward` (non-zero
  gradient on `pool_score.weight`).
- The parameter-count check: arm C = HPG-hier + 129, arm D = octamer − 129, matching a
  `Linear(128, 1)` attention readout at `d_h = 128`. This is correct.
- `hpg_hier_attention` added to `evaluation/naming.py`.
- The guard test.

---

## 6. Add a hardware guard so this cannot recur silently

Add a check in `scripts/python/run_hpg_generalization.py`: when `--frozen_protocol` is set,
**raise unless the accelerator is CUDA**, with a message naming the frozen-protocol
requirement and pointing at this incident. Allow an explicit `--allow_non_cuda` escape hatch
for deliberate local smoke tests, which must also force a `_localsmoke` token into the output
filename so such runs can never be mistaken for protocol runs.

Add a unit test covering both branches.

---

## 7. Report back

1. Confirmation the runner is stopped.
2. The list of files quarantined, with each one's recorded `accelerator`.
3. Confirmation that the K=1 arm re-analysis returns IP fold 7 to a 2-seed cell and still gives
   outcome C.
4. The new generator, the PBS files it wrote, and the job count — with explicit confirmation
   that nothing was submitted and nothing was trained locally.
5. The hardware guard and its test.
