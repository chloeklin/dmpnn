# Octamer Provenance Check

Scope: read-only audit of existing seed-42 LOMO NPZ files and locally available scripts/logs. No model training or report files were modified.

## Check 1 — Evaluation-set identity

Model order in hash columns: `hpg_hier`, `wdmpnn`, `hpg_hier_junction`, `hpg_hier_junction1`, `hpg_hier_octamer`. `y_true` SHA1 is computed over sorted values rounded to 6 dp as little-endian float64 bytes. Row-ID SHA1 is computed over sorted `test_indices|smiles_A|smiles_B` strings.

| target | fold | n_test_all_models | y_true_sha1_all_models | row_id_sha1_all_models | all_equal |
| --- | --- | --- | --- | --- | --- |
| EA | 0 | 4774,4774,4774,4774,4774 | 748c7e7e40cd,96f6302aeabf,748c7e7e40cd,748c7e7e40cd,748c7e7e40cd | 8d73e19c7959,8d73e19c7959,8d73e19c7959,8d73e19c7959,8d73e19c7959 | False |
| EA | 1 | 4774,4774,4774,4774,4774 | a8e7c5bc66c0,da890d5d7d50,a8e7c5bc66c0,a8e7c5bc66c0,a8e7c5bc66c0 | 37216336fe23,37216336fe23,37216336fe23,37216336fe23,37216336fe23 | False |
| EA | 2 | 4774,4774,4774,4774,4774 | 8ecc81066722,f225c799c747,8ecc81066722,8ecc81066722,8ecc81066722 | 7d9a43fc90fd,7d9a43fc90fd,7d9a43fc90fd,7d9a43fc90fd,7d9a43fc90fd | False |
| EA | 3 | 4774,4774,4774,4774,4774 | d8c551d2e286,ce517a8287c5,d8c551d2e286,d8c551d2e286,d8c551d2e286 | 35a2d742a657,35a2d742a657,35a2d742a657,35a2d742a657,35a2d742a657 | False |
| EA | 4 | 4774,4774,4774,4774,4774 | 41ccd5c2430d,ff5dcbf75c6f,41ccd5c2430d,41ccd5c2430d,41ccd5c2430d | c785dfd303de,c785dfd303de,c785dfd303de,c785dfd303de,c785dfd303de | False |
| EA | 5 | 4774,4774,4774,4774,4774 | 9cf135da4135,b3b50c10a116,9cf135da4135,9cf135da4135,9cf135da4135 | d84139b992b2,d84139b992b2,d84139b992b2,d84139b992b2,d84139b992b2 | False |
| EA | 6 | 4774,4774,4774,4774,4774 | 0eded281d94d,555a20837dc1,0eded281d94d,0eded281d94d,0eded281d94d | 28bbc8d8f002,28bbc8d8f002,28bbc8d8f002,28bbc8d8f002,28bbc8d8f002 | False |
| EA | 7 | 4774,4774,4774,4774,4774 | 93be41a7b9d2,3e16a405948a,93be41a7b9d2,93be41a7b9d2,93be41a7b9d2 | 437296d298fd,437296d298fd,437296d298fd,437296d298fd,437296d298fd | False |
| EA | 8 | 4774,4774,4774,4774,4774 | a4c3b4fe77dd,6d78ced143a1,a4c3b4fe77dd,a4c3b4fe77dd,a4c3b4fe77dd | 69f5bc1d6c97,69f5bc1d6c97,69f5bc1d6c97,69f5bc1d6c97,69f5bc1d6c97 | False |
| IP | 0 | 4774,4774,4774,4774,4774 | 6f9d7ed32ede,c469b983ce27,6f9d7ed32ede,6f9d7ed32ede,6f9d7ed32ede | 8d73e19c7959,8d73e19c7959,8d73e19c7959,8d73e19c7959,8d73e19c7959 | False |
| IP | 1 | 4774,4774,4774,4774,4774 | e8012fb9951d,a11fadbecd8b,e8012fb9951d,e8012fb9951d,e8012fb9951d | 37216336fe23,37216336fe23,37216336fe23,37216336fe23,37216336fe23 | False |
| IP | 2 | 4774,4774,4774,4774,4774 | a6c8b27301df,68189f0b9dae,a6c8b27301df,a6c8b27301df,a6c8b27301df | 7d9a43fc90fd,7d9a43fc90fd,7d9a43fc90fd,7d9a43fc90fd,7d9a43fc90fd | False |
| IP | 3 | 4774,4774,4774,4774,4774 | e240c9c2e12f,2b8d3558c700,e240c9c2e12f,e240c9c2e12f,e240c9c2e12f | 35a2d742a657,35a2d742a657,35a2d742a657,35a2d742a657,35a2d742a657 | False |
| IP | 4 | 4774,4774,4774,4774,4774 | e71ed2f7169f,8683a62fe27f,e71ed2f7169f,e71ed2f7169f,e71ed2f7169f | c785dfd303de,c785dfd303de,c785dfd303de,c785dfd303de,c785dfd303de | False |
| IP | 5 | 4774,4774,4774,4774,4774 | 7d5b8cd0f8ce,ddb59658c377,7d5b8cd0f8ce,7d5b8cd0f8ce,7d5b8cd0f8ce | d84139b992b2,d84139b992b2,d84139b992b2,d84139b992b2,d84139b992b2 | False |
| IP | 6 | 4774,4774,4774,4774,4774 | a37f7b15f842,a47214a06542,a37f7b15f842,a37f7b15f842,a37f7b15f842 | 28bbc8d8f002,28bbc8d8f002,28bbc8d8f002,28bbc8d8f002,28bbc8d8f002 | False |
| IP | 7 | 4774,4774,4774,4774,4774 | 227354a4a301,5178ad32f138,227354a4a301,227354a4a301,227354a4a301 | 437296d298fd,437296d298fd,437296d298fd,437296d298fd,437296d298fd | False |
| IP | 8 | 4774,4774,4774,4774,4774 | c46ef5c380d4,1b01fdadedfb,c46ef5c380d4,c46ef5c380d4,c46ef5c380d4 | 69f5bc1d6c97,69f5bc1d6c97,69f5bc1d6c97,69f5bc1d6c97,69f5bc1d6c97 | False |

EA fold-0 NPZ arrays — baseline: `y_true, y_pred, test_indices, split_type, model, target, fold, seed, n_train, n_val, n_test, prediction_scale, smiles_A, smiles_B, fracA, fracB, poly_type`

EA fold-0 NPZ arrays — octamer: `y_true, y_pred, test_indices, split_type, model, target, fold, seed, n_train, n_val, n_test, prediction_scale, smiles_A, smiles_B, fracA, fracB, poly_type`

Result: no octamer fold differs from baseline in test-row count, rounded target-value multiset, or stored row identifiers. The all-five `all_equal` check is `False` solely because the wDMPNN `y_true` arrays differ from the other four models at the sixth decimal place despite identical stored test indices/SMILES pairs; e.g. EA fold 0 has maximum absolute baseline–wDMPNN y_true difference `1.19e-07`. Baseline, both junction variants, and octamer have identical rounded-y_true hashes in every cell.

## Check 2 — Row multiplicity / averaging

| target | fold | max_identifier_multiplicity | octamer_to_baseline_n_ratio |
| --- | --- | --- | --- |
| EA | 0 | 1 | 1.000 |
| EA | 1 | 1 | 1.000 |
| EA | 2 | 1 | 1.000 |
| EA | 3 | 1 | 1.000 |
| EA | 4 | 1 | 1.000 |
| EA | 5 | 1 | 1.000 |
| EA | 6 | 1 | 1.000 |
| EA | 7 | 1 | 1.000 |
| EA | 8 | 1 | 1.000 |
| IP | 0 | 1 | 1.000 |
| IP | 1 | 1 | 1.000 |
| IP | 2 | 1 | 1.000 |
| IP | 3 | 1 | 1.000 |
| IP | 4 | 1 | 1.000 |
| IP | 5 | 1 | 1.000 |
| IP | 6 | 1 | 1.000 |
| IP | 7 | 1 | 1.000 |
| IP | 8 | 1 | 1.000 |

No stored octamer test identifier repeats; every octamer NPZ has the same row count as baseline (ratio 1.0), not a 16x expanded test set. Source establishes prediction averaging before writing: `chemprop/models/hpg_hier.py:215-222` computes `pred_sum / replica_counts`; `scripts/python/run_hpg_generalization.py:205-206` concatenates the resulting per-row predictions; `scripts/python/run_hpg_generalization.py:302-312` asserts prediction/target shape equality then writes one prediction per test index.

## Check 3 — Split provenance

### NPZ mtimes

| model | target | fold | mtime_utc |
| --- | --- | --- | --- |
| hpg_hier | EA | 0 | 2026-07-20T07:39:00+00:00 |
| hpg_hier | EA | 1 | 2026-07-20T07:35:14+00:00 |
| hpg_hier | EA | 2 | 2026-07-20T07:48:14+00:00 |
| hpg_hier | EA | 3 | 2026-07-20T07:20:56+00:00 |
| hpg_hier | EA | 4 | 2026-07-20T08:29:35+00:00 |
| hpg_hier | EA | 5 | 2026-07-20T07:22:30+00:00 |
| hpg_hier | EA | 6 | 2026-07-20T07:45:17+00:00 |
| hpg_hier | EA | 7 | 2026-07-20T08:18:13+00:00 |
| hpg_hier | EA | 8 | 2026-07-20T08:28:33+00:00 |
| hpg_hier | IP | 0 | 2026-07-20T07:14:49+00:00 |
| hpg_hier | IP | 1 | 2026-07-20T07:01:06+00:00 |
| hpg_hier | IP | 2 | 2026-07-20T07:39:11+00:00 |
| hpg_hier | IP | 3 | 2026-07-20T07:14:46+00:00 |
| hpg_hier | IP | 4 | 2026-07-20T07:48:36+00:00 |
| hpg_hier | IP | 5 | 2026-07-20T07:38:30+00:00 |
| hpg_hier | IP | 6 | 2026-07-20T08:29:46+00:00 |
| hpg_hier | IP | 7 | 2026-07-20T08:15:36+00:00 |
| hpg_hier | IP | 8 | 2026-07-20T10:29:40+00:00 |
| hpg_hier_junction1 | EA | 0 | 2026-07-26T01:36:49+00:00 |
| hpg_hier_junction1 | EA | 1 | 2026-07-26T01:18:07+00:00 |
| hpg_hier_junction1 | EA | 2 | 2026-07-26T01:12:56+00:00 |
| hpg_hier_junction1 | EA | 3 | 2026-07-26T01:38:39+00:00 |
| hpg_hier_junction1 | EA | 4 | 2026-07-26T01:32:53+00:00 |
| hpg_hier_junction1 | EA | 5 | 2026-07-26T01:12:26+00:00 |
| hpg_hier_junction1 | EA | 6 | 2026-07-26T02:03:18+00:00 |
| hpg_hier_junction1 | EA | 7 | 2026-07-26T02:55:12+00:00 |
| hpg_hier_junction1 | EA | 8 | 2026-07-26T02:04:09+00:00 |
| hpg_hier_junction1 | IP | 0 | 2026-07-26T01:47:42+00:00 |
| hpg_hier_junction1 | IP | 1 | 2026-07-26T01:00:07+00:00 |
| hpg_hier_junction1 | IP | 2 | 2026-07-26T01:16:42+00:00 |
| hpg_hier_junction1 | IP | 3 | 2026-07-26T01:06:10+00:00 |
| hpg_hier_junction1 | IP | 4 | 2026-07-26T01:07:13+00:00 |
| hpg_hier_junction1 | IP | 5 | 2026-07-26T01:41:59+00:00 |
| hpg_hier_junction1 | IP | 6 | 2026-07-26T02:55:04+00:00 |
| hpg_hier_junction1 | IP | 7 | 2026-07-26T02:28:25+00:00 |
| hpg_hier_junction1 | IP | 8 | 2026-07-26T03:55:37+00:00 |
| hpg_hier_octamer | EA | 0 | 2026-07-26T04:30:48+00:00 |
| hpg_hier_octamer | EA | 1 | 2026-07-26T04:16:47+00:00 |
| hpg_hier_octamer | EA | 2 | 2026-07-26T04:08:57+00:00 |
| hpg_hier_octamer | EA | 3 | 2026-07-26T05:18:05+00:00 |
| hpg_hier_octamer | EA | 4 | 2026-07-26T04:00:25+00:00 |
| hpg_hier_octamer | EA | 5 | 2026-07-26T04:17:10+00:00 |
| hpg_hier_octamer | EA | 6 | 2026-07-26T05:32:36+00:00 |
| hpg_hier_octamer | EA | 7 | 2026-07-26T04:37:37+00:00 |
| hpg_hier_octamer | EA | 8 | 2026-07-26T04:48:49+00:00 |
| hpg_hier_octamer | IP | 0 | 2026-07-26T04:37:24+00:00 |
| hpg_hier_octamer | IP | 1 | 2026-07-26T03:46:47+00:00 |
| hpg_hier_octamer | IP | 2 | 2026-07-26T04:57:37+00:00 |
| hpg_hier_octamer | IP | 3 | 2026-07-26T04:29:17+00:00 |
| hpg_hier_octamer | IP | 4 | 2026-07-26T03:56:38+00:00 |
| hpg_hier_octamer | IP | 5 | 2026-07-26T04:17:48+00:00 |
| hpg_hier_octamer | IP | 6 | 2026-07-26T04:43:43+00:00 |
| hpg_hier_octamer | IP | 7 | 2026-07-26T04:43:33+00:00 |
| hpg_hier_octamer | IP | 8 | 2026-07-26T05:05:58+00:00 |

### Held-out monomer identity from stored test indices

| target | fold | baseline_heldout_smiles_A | octamer_heldout_smiles_A | same |
| --- | --- | --- | --- | --- |
| EA | 0 | CC1(C)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | CC1(C)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | True |
| EA | 1 | O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | True |
| EA | 2 | OB(O)c1cc(F)c(B(O)O)cc1F | OB(O)c1cc(F)c(B(O)O)cc1F | True |
| EA | 3 | OB(O)c1cc2cc3sc(B(O)O)cc3cc2s1 | OB(O)c1cc2cc3sc(B(O)O)cc3cc2s1 | True |
| EA | 4 | OB(O)c1cc2ccc3cc(B(O)O)cc4ccc(c1)c2c34 | OB(O)c1cc2ccc3cc(B(O)O)cc4ccc(c1)c2c34 | True |
| EA | 5 | OB(O)c1ccc(-c2ccc(B(O)O)s2)s1 | OB(O)c1ccc(-c2ccc(B(O)O)s2)s1 | True |
| EA | 6 | OB(O)c1ccc(B(O)O)c2nsnc12 | OB(O)c1ccc(B(O)O)c2nsnc12 | True |
| EA | 7 | OB(O)c1ccc(B(O)O)cc1 | OB(O)c1ccc(B(O)O)cc1 | True |
| EA | 8 | OB(O)c1ccc2c(c1)\[nH\]c1cc(B(O)O)ccc12 | OB(O)c1ccc2c(c1)\[nH\]c1cc(B(O)O)ccc12 | True |
| IP | 0 | CC1(C)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | CC1(C)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | True |
| IP | 1 | O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | True |
| IP | 2 | OB(O)c1cc(F)c(B(O)O)cc1F | OB(O)c1cc(F)c(B(O)O)cc1F | True |
| IP | 3 | OB(O)c1cc2cc3sc(B(O)O)cc3cc2s1 | OB(O)c1cc2cc3sc(B(O)O)cc3cc2s1 | True |
| IP | 4 | OB(O)c1cc2ccc3cc(B(O)O)cc4ccc(c1)c2c34 | OB(O)c1cc2ccc3cc(B(O)O)cc4ccc(c1)c2c34 | True |
| IP | 5 | OB(O)c1ccc(-c2ccc(B(O)O)s2)s1 | OB(O)c1ccc(-c2ccc(B(O)O)s2)s1 | True |
| IP | 6 | OB(O)c1ccc(B(O)O)c2nsnc12 | OB(O)c1ccc(B(O)O)c2nsnc12 | True |
| IP | 7 | OB(O)c1ccc(B(O)O)cc1 | OB(O)c1ccc(B(O)O)cc1 | True |
| IP | 8 | OB(O)c1ccc2c(c1)\[nH\]c1cc(B(O)O)ccc12 | OB(O)c1ccc2c(c1)\[nH\]c1cc(B(O)O)ccc12 | True |

All baseline/octamer held-out monomer-SMILES lists are identical. The runner source additionally regenerates the monomer-heldout split with seed 42 then checks each fold index array against `metadata/splits/monomer_heldout.json`: `scripts/python/run_hpg_generalization.py:104-116`.

No executed octamer LOMO PBS scripts or task logs are present in this workspace. The expected task-log location is `logs/hpg_phase1/tasks/task_<task-index>_<PBS_JOBID>.log`; an executed generated PBS file would be under `logs/hpg_phase1/pbs/phase1_*_hpg_hier_octamer_monomer_heldout_*.pbs`. Therefore the exact executed command line and actual runtime provenance cannot be confirmed from logs. The current submit-template command is not treated as evidence of what ran; it would have supplied `--split_types monomer_heldout --folds <fold> --models hpg_hier_octamer --stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --n_coupling_steps 2 --seed 42` at `scripts/shell/submit_hpg_phase1.sh:49-73`.

Sequence construction does **not** occur after splitting: the runner builds `hier_graphs_by_token` for all dataframe rows at `scripts/python/run_hpg_generalization.py:261-273`, then builds splits at line 274. The featurizer consumes only each row's `WDMPNN_Input` and no target values (`scripts/python/run_hpg_generalization.py:159-170`; `chemprop/featurizers/molgraph/hpg_hier.py:232-253`). This establishes construction-before-split, but does not by itself establish leakage.

## Check 4 — Hyperparameter parity

| Setting | Baseline source configuration | Octamer source configuration | Difference / run-proof status |
| --- | --- | --- | --- |
| Model hidden size | `d_h=128` | `d_h=128` | matched in runner source |
| Stage-1 depth / pool | default 4 / CLI `sum` | default 4 / CLI `sum` | matched in current submit template; executed logs absent |
| Stage-2 depth / edge | CLI 2 / CLI `full` | CLI 2 / CLI `full` | matched in current submit template; executed logs absent |
| Stage-2 mode | `transition_graph` | `octamer_sequence` | intentional architecture difference |
| Octamer length / samples | not used | CLI 8 / 16 | intentional octamer-only difference |
| Max epochs | 100 | 100 | runner default; executed logs absent |
| Early stopping | `val_loss`, mode `min`, patience 15 | same | source matched |
| Optimizer / LR schedule | Adam / 1e-3 / no scheduler in source | same | source matched |
| Batch size | 64 | 64 | runner default; executed logs absent |
| Target scaling | training-set standard scaler | same | source matched |
| Loss | mean squared error | same | source matched |

The source defaults and current submit template imply matched training settings, but the missing executed PBS/task logs and persisted run configs mean the historical training budget cannot be confirmed.

## Check 5 — EA gain location

| fold | poly_type | hpg_hier_mae | hpg_hier_bias | hpg_hier_octamer_mae | hpg_hier_octamer_bias |
| --- | --- | --- | --- | --- | --- |
| 0 | alternating | 0.076092 | -0.058467 | 0.053342 | 0.022048 |
| 0 | block | 0.111945 | -0.106147 | 0.035017 | -0.022593 |
| 0 | random | 0.111908 | -0.104061 | 0.029928 | -0.006345 |
| 0 | ALL_GROUP_MEANS | 0.105366 | -0.099416 | 0.029077 | -0.010464 |
| 1 | alternating | 0.204823 | -0.204488 | 0.052791 | -0.002169 |
| 1 | block | 0.225172 | -0.224379 | 0.026290 | -0.000527 |
| 1 | random | 0.205826 | -0.204821 | 0.029399 | 0.006571 |
| 1 | ALL_GROUP_MEANS | 0.213325 | -0.212620 | 0.027414 | 0.002400 |
| 7 | alternating | 0.121276 | 0.116619 | 0.060002 | -0.015358 |
| 7 | block | 0.167219 | 0.163293 | 0.058715 | -0.048432 |
| 7 | random | 0.155271 | 0.152922 | 0.051159 | -0.035703 |
| 7 | ALL_GROUP_MEANS | 0.157173 | 0.154999 | 0.053536 | -0.039250 |

Bias is prediction minus truth. `ALL_GROUP_MEANS` is the mean signed bias after one mean per matched chemistry group. The stored NPZ does not contain per-row octamer sequence counts or sampled sequences, so the exact K for each row cannot be reconstructed from outputs. The source makes K conditional on transition weights: non-uniform transitions return one deterministic candidate, while exactly uniform transitions sample `n_random_samples` candidates (`chemprop/featurizers/molgraph/hpg_hier.py:175-200`).

## Check 6 — Paired per-fold comparison

Signed differences are variant minus baseline for R²/ordering; for MAE they are baseline minus variant, so positive always means variant better. Wins/losses exclude exact ties. Exact two-sided sign tests are across the nine folds; minimum attainable two-sided p with nine non-tied folds is 0.0039. No result is multiple-comparison corrected.

| comparison | target | metric | wins | losses | ties | signed_difference_by_fold_0_to_8 | sign_test_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| octamer | EA | group_mean_r2 | 8 | 1 | 0 | +0.0694, +0.4138, +0.0731, +0.0090, -0.0536, +0.0224, +0.0083, +0.0876, +0.0308 | 0.039062 |
| octamer | EA | delta_r2 | 5 | 4 | 0 | -0.0254, +0.0434, +0.0127, +0.0756, -0.2082, -0.0352, +0.0374, +0.1393, -0.0569 | 1.000000 |
| octamer | EA | ordering | 4 | 5 | 0 | -0.0005, +0.0503, +0.0322, +0.0799, -0.0059, -0.0123, +0.0214, -0.0071, -0.0358 | 1.000000 |
| octamer | EA | overall_r2 | 8 | 1 | 0 | +0.0673, +0.4158, +0.0687, +0.0120, -0.0561, +0.0215, +0.0156, +0.0833, +0.0299 | 0.039062 |
| octamer | EA | overall_mae | 8 | 1 | 0 | +0.0714, +0.1826, +0.0834, +0.0097, -0.0350, +0.0250, +0.0015, +0.0999, +0.0514 | 0.039062 |
| octamer | IP | group_mean_r2 | 5 | 4 | 0 | -0.0014, -0.0115, +0.2247, -0.0420, -0.0337, +0.1067, +0.0115, +0.0199, +0.0031 | 1.000000 |
| octamer | IP | delta_r2 | 7 | 2 | 0 | +0.0927, +0.0253, +0.3053, +0.0651, -0.0620, +0.0452, +0.0319, +0.0865, -0.0119 | 0.179688 |
| octamer | IP | ordering | 5 | 4 | 0 | +0.0414, -0.0309, -0.0996, +0.0534, +0.0454, +0.0053, -0.0191, +0.0263, -0.0001 | 1.000000 |
| octamer | IP | overall_r2 | 5 | 4 | 0 | -0.0000, -0.0116, +0.2166, -0.0389, -0.0386, +0.1067, +0.0125, +0.0200, +0.0013 | 1.000000 |
| octamer | IP | overall_mae | 5 | 4 | 0 | -0.0013, -0.0129, +0.1614, -0.0202, -0.0123, +0.0371, +0.0123, +0.0262, +0.0048 | 1.000000 |
| junction_n1 | EA | group_mean_r2 | 6 | 3 | 0 | +0.0636, +0.3867, -0.0179, +0.0102, +0.0046, +0.0069, -0.6850, +0.0848, -0.0004 | 0.507812 |
| junction_n1 | EA | delta_r2 | 6 | 3 | 0 | +0.0519, +0.0232, -0.1731, +0.0745, +0.2361, -0.0133, +0.0482, +0.1338, -0.0554 | 0.507812 |
| junction_n1 | EA | ordering | 4 | 5 | 0 | +0.0086, -0.0065, -0.0544, +0.0678, +0.0204, -0.0381, -0.0562, +0.0078, -0.0373 | 1.000000 |
| junction_n1 | EA | overall_r2 | 6 | 3 | 0 | +0.0621, +0.3882, -0.0211, +0.0133, +0.0107, +0.0071, -0.7255, +0.0802, -0.0003 | 0.507812 |
| junction_n1 | EA | overall_mae | 5 | 4 | 0 | +0.0604, +0.1561, -0.0190, +0.0136, -0.0010, +0.0138, -0.1535, +0.0969, -0.0018 | 1.000000 |
| junction_n1 | IP | group_mean_r2 | 5 | 4 | 0 | -0.0557, +0.0001, +0.1456, -0.0610, +0.0083, -0.5302, +0.0177, +0.0052, -0.0123 | 1.000000 |
| junction_n1 | IP | delta_r2 | 4 | 5 | 0 | +0.0944, +0.0488, +0.3531, +0.0342, -0.0205, -0.3310, -0.0212, -0.0104, -0.2782 | 1.000000 |
| junction_n1 | IP | ordering | 3 | 6 | 0 | +0.0362, -0.0156, +0.0182, +0.0021, -0.0262, -0.0735, -0.0375, -0.0266, -0.0106 | 0.507812 |
| junction_n1 | IP | overall_r2 | 5 | 4 | 0 | -0.0550, +0.0007, +0.1412, -0.0576, +0.0101, -0.5328, +0.0168, +0.0039, -0.0251 | 1.000000 |
| junction_n1 | IP | overall_mae | 5 | 4 | 0 | -0.0237, +0.0009, +0.0750, -0.0262, +0.0038, -0.0938, +0.0211, +0.0080, -0.0059 | 1.000000 |
| junction_n2 | EA | group_mean_r2 | 7 | 2 | 0 | -0.0100, +0.3495, +0.0569, +0.0114, +0.0295, +0.0055, -0.0728, +0.0631, +0.0078 | 0.179688 |
| junction_n2 | EA | delta_r2 | 4 | 5 | 0 | +0.0382, -0.0257, -0.0144, -0.0337, +0.2142, +0.0017, -0.9308, +0.1383, -0.0370 | 1.000000 |
| junction_n2 | EA | ordering | 3 | 6 | 0 | +0.0435, -0.0886, -0.0303, +0.0697, -0.0425, -0.0060, -0.1235, +0.0251, -0.0331 | 0.507812 |
| junction_n2 | EA | overall_r2 | 7 | 2 | 0 | -0.0119, +0.3495, +0.0528, +0.0115, +0.0388, +0.0069, -0.1182, +0.0598, +0.0074 | 0.179688 |
| junction_n2 | EA | overall_mae | 7 | 2 | 0 | -0.0117, +0.1275, +0.0527, +0.0141, +0.0211, +0.0089, -0.0309, +0.0558, +0.0042 | 0.179688 |
| junction_n2 | IP | group_mean_r2 | 5 | 4 | 0 | +0.0534, -0.0194, +0.1016, -0.1293, -0.0165, -0.2756, +0.0184, +0.0115, +0.0095 | 1.000000 |
| junction_n2 | IP | delta_r2 | 5 | 4 | 0 | +0.0484, +0.0376, +0.4606, -0.0864, +0.0152, -0.1294, -0.0779, +0.0582, -0.0410 | 1.000000 |
| junction_n2 | IP | ordering | 4 | 5 | 0 | +0.0037, -0.0081, +0.0081, -0.0125, +0.0371, -0.0365, -0.0409, +0.0179, -0.0275 | 1.000000 |
| junction_n2 | IP | overall_r2 | 5 | 4 | 0 | +0.0533, -0.0184, +0.0990, -0.1302, -0.0123, -0.2690, +0.0162, +0.0117, +0.0075 | 1.000000 |
| junction_n2 | IP | overall_mae | 5 | 4 | 0 | +0.0318, -0.0178, +0.0386, -0.0477, -0.0044, -0.0571, +0.0195, +0.0181, +0.0071 | 1.000000 |

## Verdict

- **Are the octamer LOMO predictions evaluated on the identical held-out sets as the baseline? Yes.** Baseline and octamer match in all 18 target/fold cells for n_test, order-independent rounded-y_true SHA1, stored index/SMILES row-ID SHA1, and held-out-monomer SMILES. Across all five models, n_test and row identifiers match, but wDMPNN y_true has sixth-decimal float-precision differences, so the all-five rounded-y_true hash criterion is not met.
- **Is the EA chemistry gain concentrated in random-architecture rows? No.** In EA folds 0, 1, and 7, octamer reduces MAE and shifts the negative bias toward zero for block and alternating rows as well as random rows. The NPZs do not store per-row K/sample sequences, so this rules out a random-only concentration but cannot quantify the sampling contribution per row.
- **Was training budget matched? Cannot determine.** The current source/template specifies matched optimization, scaling, loss, batch size, epoch cap, and early stopping, but executed octamer/baseline PBS task logs and persisted run configs are absent, so the actual historical commands and training budgets cannot be verified.
