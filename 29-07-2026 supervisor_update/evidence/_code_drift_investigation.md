# Code-drift investigation

## Scope and timing assumption

The history window is 2026-07-10 00:00 UTC through 2026-07-27 23:59 UTC. NPZ modification times record completion, not checkout or queue-entry time, so this deliberately broad window avoids treating the artifact timestamp as a tight code-version bracket.

The canonical fold-0 HPG NPZ was created and modified at 2026-07-20 17:39:00 +10:00 (07:39 UTC). The old-code test used `f3ca8af9a715583631fe3492790cb0d81cd8b453`, the last commit reachable before 2026-07-20 07:00 UTC; it was authored at 2026-07-20 16:19:05 +10:00 (06:19 UTC).

## Step 1: history

### Relevant commits

| SHA | Author date | Relevant files touched | Summary | Affects baseline numerics? | Reason |
| --- | --- | --- | --- | --- | --- |
| `e150a59005d823086d58b34464817f0cca4526a1` | 2026-07-14 00:21:28 +10:00 | `chemprop/data/dataloader.py`, `chemprop/data/copolymer.py`, `chemprop/data/samplers.py`, `chemprop/models/copolymer.py`, `chemprop/nn/within_group_loss.py`, `scripts/python/run_wdmpnn_generalization.py`, `scripts/python/utils.py` | Added within-group-loss and group-aware wDMPNN machinery. | No for HPG; potentially yes for nonzero-lambda wDMPNN only. | HPG-hier uses its own PyTorch `DataLoader`, model, MSE, and training function. The wDMPNN baseline keeps `lambda_within=0`, for which the new path is gated off. |
| `8b63f5db4a2c3c1250d80920bb830ed67d7e46cf` | 2026-07-14 13:35:00 +10:00 | `scripts/python/run_wdmpnn_generalization.py` | Changed wDMPNN prediction naming/tagging. | No. | Saving happens after prediction and does not alter HPG or wDMPNN baseline training numerics. |
| `e70ffc3eefef098d82bfd2cf1825f4184d246607` | 2026-07-18 18:59:22 +10:00 | `chemprop/data/hpg_hier.py`, `chemprop/featurizers/molgraph/hpg_hier.py`, `chemprop/models/hpg_hier.py`, `scripts/python/run_hpg_generalization.py`; also non-hierarchical HPG files | Introduced the HPG-hier model, featurizer, collation, runner path, and its baseline defaults. | Yes, foundational. | This commit defines the baseline itself. It precedes the canonical artifact. The four HPG-hier files are byte-identical between `e70ffc3` and the selected pre-run commit `f3ca8af`, so there was no committed HPG-hier drift between introduction and the tested pre-run revision. |
| `6b0c2262b8cb4a270d13929c95ef6b166984f901` | 2026-07-25 13:48:39 +10:00 | `chemprop/data/hpg_hier.py`, `chemprop/featurizers/molgraph/hpg_hier.py`, `chemprop/models/hpg_hier.py`, `scripts/python/run_hpg_generalization.py` | Added edge-weight, octamer-sequence, and junction-coupling variants. | No for the stated baseline configuration. | The baseline selects `stage2_edge_weight=feature`, `stage2_mode=transition_graph`, and `junction_coupling=off`. In `feature` mode the Stage-2 message expression and dimensions are the original expression. Octamer construction/forward is reached only for `octamer_sequence`; junction graph construction/forward is reached only when coupling is `on`. Added batch fields are `None` on the baseline path. |
| `fea3a8f85e6c2ccbd80155ce20e1adf87d1e5cf9` | 2026-07-25 21:01:00 +10:00 | `chemprop/featurizers/molgraph/hpg_hier.py`, `chemprop/models/hpg_hier.py` | Added octamer positional embeddings and attention pooling and changed octamer candidate handling. | No. | All numerical changes are under octamer sequence construction or the octamer encoder; `transition_graph` never executes them. |
| `7524e67cfd442e38d5467c90f226bf6199dcbed1` | 2026-07-25 21:04:49 +10:00 | `chemprop/data/hpg_hier.py`, `chemprop/featurizers/molgraph/hpg_hier.py`, `chemprop/models/hpg_hier.py` | Allowed variable octamer replicate counts and changed deterministic-vs-random octamer sampling. | No. | The changed tensors and replica averaging exist only when octamer sequences are present. They are absent for `transition_graph`. |
| `5e515041b2beb7ac1ecdc5de372a5a8299f77631` | 2026-07-26 10:10:43 +10:00 | `scripts/python/run_hpg_generalization.py` | Added the one-step junction token and resolved coupling-step counts per variant. | No. | Baseline coupling remains `off`, now with an explicitly unused count of zero. The junction layer is not instantiated or called. |

No other commit in the window touched `MABBondMessagePassing`, the target transform, HPG-hier MSE, its Adam optimizer, or its Lightning training path. The only other reachable shared-file edits were additions for non-hierarchical HPG or gated copolymer/wDMPNN functionality.

### Default hyperparameters

No committed change in the window altered the HPG-hier baseline defaults after its introduction:

| Setting | Value |
| --- | --- |
| Hidden dimension `d_h` | 128 |
| Stage-1 depth | 4 |
| Stage-2 depth | 2 |
| Maximum epochs | 100 |
| Early-stopping patience | 15 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Target scaling | Fit a standard scaler on training targets; normalize train and validation; unscale predictions |
| Loss | Mean squared error in normalized target space |
| Optimizer | Adam |
| Stage-1 pooling | Sum |
| Stage-2 readout | Stoichiometry-weighted sum |

The current working tree has an uncommitted Stage-2 readout option. For the baseline, its default resolves to `stoich_weighted`; the attention module is not instantiated and the original weighted-sum expression executes. This uncommitted code therefore appears gated away from baseline numerics, but its existence is important: git log alone cannot reconstruct arbitrary working-tree state used by historical jobs.

### Environment pinning

No commit in the window changed `requirements*.txt`, `pyproject.toml`, `environment.yml`, lockfiles, or equivalent environment-pinning files. This means git provides no evidence that the July 20 and July 26 environments were identical; it means only that no pinned environment change was committed.

### Branches, merges, and reflog

Only local `main` and `origin/main` are currently visible. The reflog shows normal commits, pushes, and fast-forward pulls. One merge, `1baedd015237c1df96c2600060a1604e842bea7f`, occurred on 2026-07-20 10:59:52 +10:00; its merged side introduced no change to the relevant HPG, HPG-hier, runner, loss, scaler, optimizer, or environment files. No surviving side branch contains an alternative relevant implementation.

Reflog retention and branch visibility are not archival guarantees. Deleted branches, expired reflog entries, cluster-side clones, and uncommitted working-tree changes at run time may be absent. A clean committed history is suggestive, not proof of the code actually executed.

## Step 2: old-code run

A detached worktree at `/tmp/dmpnn_pre20_worktree` was created from `f3ca8af9a715583631fe3492790cb0d81cd8b453`; the current worktree was untouched. The old runner lacked `--prediction_dir`, so the detached worktree received only a sink-only patch adding that argument and replacing the prediction output root. The model, featurizer, split, optimizer, scaler, loss, and training code were not changed. The same existing virtual environment was used. Output went to `/tmp/dmpnn_pre20_reproduction`.

The invocation used HPG-hier, EA, A-heldout fold 0, seed 42, Stage-1 sum pooling, Stage-2 depth 2, full Stage-2 edges, 100 maximum epochs, patience 15, and batch size 64.

### Prediction comparison

| Artifact | Code | Overall R² | MAE (eV) | Difference from canonical: max abs (eV) | Difference from canonical: correlation | Difference from canonical: RMSE (eV) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Canonical July 20 | Historical state not fully recorded | 0.9241134900 | 0.1068073920 | 0 | 1 | 0 |
| Current-code reproduction | Current working tree | 0.9587659282 | 0.0845148754 | 0.3642779589 | 0.9947367761 | 0.0711799460 |
| Pre-July-20-code reproduction | `f3ca8af` plus output-sink patch | 0.9667159231 | 0.0668704021 | 0.3134304285 | 0.9945889457 | 0.0854562276 |

The old-code prediction is not bitwise equal to the canonical prediction. Test indices and true targets are bitwise equal in all three artifacts. Old code is not systematically closer: its maximum absolute difference is somewhat smaller, but its prediction-difference RMSE is larger, its correlation is lower, and its R²/MAE are even farther from the canonical metrics. Current-code versus old-code predictions also differ (maximum 0.2443060875 eV, correlation 0.9956304464, prediction-difference RMSE 0.0578683473 eV).

### Decisive reading

**The second outcome applies: old code reproduces no better than current code, so training nondeterminism dominates this test.** The committed post-July-20 changes are gated away from the baseline path, and running the pre-July-20 implementation does not recover the historical prediction vector or historical metrics. This test does not confirm baseline numerical code drift.

The historical canonical result must therefore be treated as one draw from a distribution whose width is unknown. It does not establish that code drift is exactly zero: the old run used the currently installed environment, historical uncommitted state is unavailable, and a single old/current pair cannot quantify stochastic spread. Those limitations are why the proposed repeat experiment remains necessary.

### Observed wall time

The pre-July-20-code run took **3565.91 seconds = 59.43 minutes = 0.9905 accelerator-device hours**. It used the local MPS accelerator, reached a best checkpoint at epoch 52, and stopped after the patience window. This is observed process wall time from `/usr/bin/time`, not a scheduler-reported CUDA GPU-hour measurement and not the 24-hour PBS request ceiling.

The current-code reproduction's NPZ was produced, but its provenance sidecar write failed on a non-serializable `Path` after training. Consequently no trustworthy sidecar wall-time field exists for that run. The serialization defect has been fixed for future runs. The old commit predates sidecars, so its wall time was measured externally. As a first wall-time proxy for queue planning, one observed fold is approximately one accelerator-hour; extrapolating linearly gives roughly 72 device-hours for 72 cells or 144 device-hours for 144 cells, before accelerator differences, model-dependent variation, and scheduler overhead. The earlier 1,728 GPU-hour value is only `72 × 24 h` requested-walltime capacity and is not observed usage.

## Scope and stopping point

No PBS jobs were submitted. No existing report file was edited. Step 3 was not started.

The B-heldout cells remain internally comparable because they will all use current code. Existing A-split variant tables remain subject to an unmeasured run-to-run noise floor even though this investigation finds no evidence that committed, baseline-active code drift explains their differences.
