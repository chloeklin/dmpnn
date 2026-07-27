# Windsurf prompt — is the baseline reproduction failure code drift or training noise?

**Context.** Rerunning `hpg_hier`, EA, fold 0, seed 42 against the frozen A-heldout split reproduced
the split, test indices and targets bitwise, but **not** the predictions: max prediction difference
0.364 eV, R² 0.92411 → 0.95877, MAE 0.10681 → 0.08451. The rerun is materially *better*, not merely
different.

The existing baseline and wDMPNN A-split NPZs are dated **2026-07-20**; the junction n=1 and octamer
NPZs are dated **2026-07-26**. If the model code changed in between, then every comparison in
`variant_results_report.md` is an old baseline against new variants, and part of the octamer's
apparent advantage may be six days of unrelated code changes rather than the variant itself.

Two candidate causes:

- **Code drift** — `hpg_hier` in its baseline configuration behaves differently now than on 20 July.
- **Training nondeterminism** — identical code and seed simply do not reproduce on GPU.

They imply different remediation, so distinguish them before deciding what to run.
**Steps 1 and 2 first; stop for review before Step 3.**

Do not edit any report file. Do not submit PBS jobs. Write findings to
`analysis/model_diagnostics/_code_drift_investigation.md`.

---

## Step 1 — git history (free, minutes)

Window: NPZ mtimes record when a run **finished**, and jobs may have queued for hours or days before
starting. Do not use a tight bracket. Search **2026-07-10 → 2026-07-27**, and state that assumption.

1. List every commit in that window touching any of:
   - `chemprop/models/hpg_hier.py`
   - `chemprop/featurizers/molgraph/hpg_hier.py`
   - `chemprop/data/hpg_hier.py`
   - `scripts/python/run_hpg_generalization.py`
   - `scripts/python/run_wdmpnn_generalization.py`
   - anything else in `chemprop/` reachable from the `hpg_hier` forward pass, the loss, the target
     scaler, the optimizer, or the training loop

   For each: SHA, author date, files touched, one-line summary.

2. **For each commit, classify its effect on the *baseline* configuration**
   (`stage2_mode=transition_graph`, `stage2_readout=stoich_weighted`, `junction_coupling=off`):
   `affects baseline numerics: yes / no / unclear`, with the reason. Many changes may be gated behind
   `stage2_mode == octamer_sequence` or the junction path and therefore cannot touch the baseline —
   say so explicitly where true. This classification is the whole point of Step 1; do not skip it.

3. Report whether **default hyperparameters** changed in that window: `d_h`, stage-1 depth, stage-2
   depth, max epochs, early-stopping patience, LR, batch size, target scaling, loss.

4. Report any change to environment pinning (`requirements*.txt`, `pyproject.toml`, `environment.yml`,
   lockfiles) in the window.

5. Note explicitly that uncommitted working-tree changes at run time would **not** appear in
   `git log`, so a clean history is suggestive but not proof. Check `git reflog` and any branches
   merged in the window in case the runs were executed from a branch.

## Step 2 — the decisive test (1 GPU run)

Git history alone cannot settle it. Run this:

- Check out the last commit **before 2026-07-20T07:00 UTC** into a separate worktree
  (`git worktree add`) so the current tree is untouched.
- With that old code, rerun `hpg_hier`, `EA_vs_SHE_eV`, A-heldout fold 0, seed 42, writing to a
  scratch `--prediction_dir`. Use the same environment you used for the failed reproduction.
- Compare predictions bitwise to the canonical `2026-07-20` NPZ, and report max absolute difference,
  correlation, R² and MAE alongside the two numbers already on record.

Reading of the outcome — state which applies:

- **Old code reproduces the canonical NPZ (or comes far closer than current code did)** → code drift
  confirmed. The 20 July baselines are stale and must be regenerated under current code before any
  variant comparison stands.
- **Old code reproduces no better than current code** → nondeterminism dominates. The historical
  numbers are one draw from a distribution whose width is unknown, and the priority becomes measuring
  that width.

Also report, from the provenance sidecars now being written: **actual wall time per run**. The
1,728 GPU-hour figure for 72 cells is a ceiling from requesting 24 h per job, not observed usage, and
Chloe needs the real number to plan the queue.

**Stop here for review.**

---

## Step 3 — noise floor (6 GPU runs, after review)

Required regardless of how Steps 1–2 resolve, because no one currently knows the run-to-run spread.

**Must run on Gadi CUDA in the production environment — not locally on MPS.** Steps 1–2 were executed
on Apple MPS while the canonical NPZs came from Gadi GPUs, so hardware is confounded with code version
and with nondeterminism in those results. A noise floor measured on MPS does not describe the
production runs and cannot be used to judge them. Report the accelerator, driver/CUDA version, torch
version and whether deterministic kernels were requested, alongside every number.

- `hpg_hier`, current code, A-heldout folds 0 and 1, EA, seed 42, **three independent repeats each**
  (vary only whatever nondeterminism exists — do not change the seed).
- Report per fold: mean and SD across the three repeats for group-mean R², ΔR², ordering, overall R²
  and MAE, plus max pairwise difference.
- Compare that SD against the octamer-vs-baseline per-fold differences already recorded
  (EA MAE: +0.071, +0.183, +0.083, +0.010, −0.035, +0.025, +0.002, +0.100, +0.051). State plainly how
  many of those differences fall **inside** the measured noise band.

This SD becomes a required column in every future comparison table.

Also report **actual V100 wall time per run** from the provenance sidecars. The ~59-minute figure on
record is an MPS measurement and does not transfer.

## Step 4 — decide the replicate strategy before submitting the B split

The Step-3 SD determines how many runs per cell the B-heldout experiment needs. Report both options
with real GPU-hour estimates and let Chloe choose; do not submit either.

- **If the SD is small relative to the effects being tested** (per-fold MAE SD well under ~0.02 eV):
  144 cells at seed 42 is an adequate screen, as originally specified.
- **If the SD is comparable to those effects:** a single run per cell cannot support any conclusion.
  Run **three seeds (42/43/44) per cell** and treat the seed as the replicate unit — this absorbs both
  initialisation variance and nondeterminism, and is what the error bars needed anyway. That is 432
  cells for random + clustered, so the likely trade is to run random-only at three seeds (216 cells)
  and defer clustered.

State which case the measurement puts you in, with the numbers.

## Note on scope

The B-heldout cells are **not** contaminated by the July-20 issue — they will all run on current code
and are internally consistent. Only the existing A-split tables are affected. But how many runs per
cell they need is decided by Step 3, so Step 3 comes first.
