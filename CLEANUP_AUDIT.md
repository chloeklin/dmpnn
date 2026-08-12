# Repository cleanup audit

**Repository:** `dmpnn` (Chemprop v2 fork + polymer property prediction research)
**Audited:** 11 August 2026 — read-only inspection. **Nothing has been deleted, moved, or renamed.**

---

## 0. Summary of what was inspected

| Fact | Value |
|---|---|
| Git repository | Yes, branch `main` |
| Tracked files | 2,131 |
| Untracked (not gitignored) files | 90 |
| Modified but uncommitted | 9 |
| Tracked but deleted on disk | 2 (`RESEARCH_REVIEW/`) |
| Top-level entries | 41 |
| Python virtual environments present | 2 (`venv/`, `.venv/`) |

**Method.** For every deletion candidate I checked: `git ls-files` (tracked?), `.gitignore` (ignored?), `git grep` across all tracked source, shell, YAML and Markdown (referenced?), and content hashing (`git ls-files -s`, `md5sum`) for duplicates. Where a reference exists I have named it.

**One structural finding drives most of the recommendations:** this repository is doing three jobs at once — (a) a packaged Python library (`chemprop/`, governed by `pyproject.toml`/`MANIFEST.in`), (b) a research pipeline (`scripts/`, `experiments/`, `analysis/`), and (c) a document/deliverable store (dated folders, decks, handoffs, paper drafts). The clutter is almost entirely in (c) leaking into the root. The reorganisation below separates them without touching (a).

---

## 1. Safe to delete

Everything in this section is either gitignored, untracked, or a byte-identical duplicate, **and** has no reference anywhere in tracked source, config, or documentation.

### 1.1 Editor / OS metadata — tracked in git (should not be)

| Path | What it is | Why deletable | Referenced by |
|---|---|---|---|
| `results/.DS_Store` | macOS Finder metadata | OS artifact, no content value | Nothing |
| `results/DMPNN/.DS_Store` | ditto | ditto | Nothing |
| `results/DMPNN_DiffPool/.DS_Store` | ditto | ditto | Nothing |
| `results/GAT/.DS_Store` | ditto | ditto | Nothing |
| `results/GIN/.DS_Store` | ditto | ditto | Nothing |

These four under model subfolders are byte-identical to each other. They are tracked because `.gitignore` line `!results/**/*` re-includes everything under `results/`, overriding the global `*.DS_Store` rule. Deleting them requires `git rm --cached` **and** a `.gitignore` fix (see §7 step 1), otherwise they return.

### 1.2 Editor / OS metadata — on disk, already gitignored

13 further `.DS_Store` files, safe to delete but cosmetic only (git already ignores them):

```
./.DS_Store                                  ./experiments/hpg2stage/.DS_Store
./analysis/.DS_Store                         ./experiments/hpg2stage/output/.DS_Store
./analysis/model_diagnostics/.DS_Store       ./experiments/diagnostics/.DS_Store
./experiments/.DS_Store                      ./29-07-2026 supervisor_update/.DS_Store
./experiments/tabular/figures_ea_ip/.DS_Store ./22-07-2026 report_figures/.DS_Store
./scripts/.DS_Store                          ./hpg_hier_design/.DS_Store
./scripts/shell/.DS_Store
```

### 1.3 Jupyter checkpoint files — tracked in git (29 files)

All of `results/*/.ipynb_checkpoints/*-checkpoint.csv` — 29 tracked files across `AttentiveFP`, `DMPNN`, `DMPNN_DiffPool`, `GIN`, `IdentityBaseline`, `PPG`, `tabular`.

- **What they are:** Jupyter's autosave copies of result CSVs.
- **Why deletable:** four of them are byte-identical to the live file next to them (e.g. `results/IdentityBaseline/.ipynb_checkpoints/block__copoly_mix_results-checkpoint.csv` == `results/IdentityBaseline/block__copoly_mix_results.csv`). The rest are stale intermediate saves. `.ipynb_checkpoints` is standard disposable Jupyter state.
- **Referenced by:** nothing. `git grep ipynb_checkpoints` across all source returns no hits.
- **Same caveat as §1.1:** tracked only because of the `!results/**/*` rule.

Also delete the three empty `predictions/*/.ipynb_checkpoints/` directories (`GAT`, `DMPNN`, `GIN`).

### 1.4 Python bytecode caches (23 directories, untracked)

```
analysis/diagnostics/__pycache__          chemprop/__pycache__ (+ 9 nested)
analysis/model_diagnostics/__pycache__    polymer_input/__pycache__ (+ 2 nested)
analysis/paper1_figures/__pycache__       scripts/__pycache__
evaluation/__pycache__                    scripts/python/__pycache__
experiments/hpg2stage/scripts/__pycache__ tests/__pycache__
29-07-2026 supervisor_update/__pycache__
```

Gitignored, regenerated automatically. Worth noting for a different reason: `scripts/python/__pycache__` contains `.pyc` files for **modules that no longer exist** (`plot_colors`, `diagnose_ordering_ties`, `run_wdmpnn_generalization`, `regeneration`, `frozen_splits`… some do exist, several don't). That is a symptom, not a problem — it just confirms the cache is stale.

### 1.5 `.pytest_cache/` (root, untracked, gitignored)

Pytest's run cache. Regenerated on next test run.

### 1.6 `pred.zip` (3.7 MB, root, untracked, gitignored)

- **What it is:** a transfer archive of 104 prediction artifacts under `predictions/octamer_posemb/ea_ip_lomo/` — evidently an HPC download bundle.
- **Why deletable:** I extracted the file list and checked each path. **All 104 files already exist, unzipped, at their target locations.** The archive is pure duplication.
- **Referenced by:** nothing.

### 1.7 Byte-identical duplicated evidence documents (9 files)

`29-07-2026 supervisor_update/` contains exact copies of files that live canonically in `analysis/model_diagnostics/`. Verified identical by git blob hash:

| Duplicate copy (delete) | Canonical original (keep) |
|---|---|
| `29-07-2026 supervisor_update/evidence/_a_heldout_bitwise_reproduction.md` | `analysis/model_diagnostics/_a_heldout_bitwise_reproduction.md` |
| `.../evidence/_code_drift_investigation.md` | `analysis/model_diagnostics/_code_drift_investigation.md` |
| `.../evidence/_groupmean_metric_floor.md` | `analysis/model_diagnostics/_groupmean_metric_floor.md` |
| `.../evidence/_noise_floor_results.md` | `analysis/model_diagnostics/_noise_floor_results.md` |
| `.../evidence/_octamer_provenance_check.md` | `analysis/model_diagnostics/_octamer_provenance_check.md` |
| `.../evidence/_training_stability_stepc_results.md` | `analysis/model_diagnostics/_training_stability_stepc_results.md` |
| `.../evidence/variant_results_report.md` | `analysis/model_diagnostics/variant_results_report.md` |
| `.../specs/windsurf_spec_b_heldout_split.md` | `analysis/model_diagnostics/windsurf_spec_b_heldout_split.md` |
| `.../specs/windsurf_spec_regeneration.md` | `analysis/model_diagnostics/windsurf_spec_regeneration.md` |

**Referenced by:** nothing reads the `evidence/` or `specs/` copies. The scripts in that folder (`figures_deck.py`, `figures_results.py`) read from `analysis/model_diagnostics/`, i.e. the originals.

**Note:** `29-07-2026 supervisor_update/evidence/_dataset_design_audit.md` is *not* on this list — it differs from the `analysis/` version. See §4.

### 1.8 Duplicated slide renders in `analysis/model_diagnostics/` (untracked, 54 JPGs)

These are per-slide preview renders of the PowerPoint decks, produced during iterative deck building. Verified by MD5:

- `slide-01.jpg` … `slide-17.jpg` are **byte-identical** to `s-01.jpg` … `s-17.jpg`. One of the two sets is pure duplication (17 files).
- `e-03.jpg` == `f-03.jpg` == `g-03.jpg`; `e-07.jpg` == `f-07.jpg` == `g-07.jpg`; `e-08.jpg` == `f-08.jpg`; `f-09.jpg` == `g-09.jpg`; `h-12.jpg` == `k-12.jpg`.
- Total: 54 JPGs, of which 26 are exact duplicates of another JPG in the same folder.

**Why the whole set is deletable, not just the duplicates:** they are re-renderable at any time from the `.pptx` files that sit beside them. They are untracked, unreferenced, and their names (`e-`, `f-`, `g-`, `h-`, `k-`, `t-`, `s-`, `slide-`) encode nothing but the order in which draft renders were made.

### 1.9 Empty directories (14)

```
analysis/model_diagnostics/09_per_fold_case_studies/fold_01 … fold_05, fold_07, fold_08   (7)
hpg_hier_design/seed_42_diagnostic/09_per_fold_case_studies/fold_01 … fold_05, fold_07, fold_08 (7)
hpg_hier_design/seed_42_diagnostic/10_summary
checkpoints/AttentiveFP/insulator____rep0
tests/_diag_ckpt_tmp/  (contains only an empty logs/)
logs/octamer_k1/r3/tasks, logs/octamer_posemb/r1/tasks, logs/octamer_posemb/r3/tasks,
logs/hpg_stability_fixes/tasks
```

Git does not track empty directories, so removing them is a filesystem-only change with zero git impact. **Caveat on the four empty `logs/*/tasks/` directories:** `scripts/python/analyze_regen_v1.py` (lines 322, 438) instructs the reader to `grep logs/regen_v1/r3/tasks/` for frozen-split assertions — so this *class* of directory is meaningful. The four listed above are empty because those task logs were never downloaded. Deleting the empty shells is harmless; the download script will recreate them.

### 1.10 Explicit backup file

| Path | Why |
|---|---|
| `experiments/hpg2stage/scripts/generate_stage2d_paper_outputs.py.bak` | 70 KB `.bak` of what became `generate_paper_outputs.py` (35 KB). Superseded — the current file's docstring is a trimmed rewrite of the `.bak`'s. Referenced by nothing. Its history is recoverable from git if it was ever committed. |

### 1.11 macOS "copy" artifact

| Path | Why |
|---|---|
| `22-07-2026 report_figures/all/README copy.md` | Finder-duplicate filename. **Note:** it is *not* byte-identical to `README.md` beside it — it is an earlier, shorter version describing only the "LOMO overall breakdown" additions, whose content is superseded by the fuller `README.md`. Delete only after a quick eyeball (30 seconds) that nothing in it is unique. Referenced by nothing. |

---

## 2. Probably safe to delete — but review first

### 2.1 One of the two virtual environments

| Path | Evidence |
|---|---|
| `.venv/` | Python 3.13.2. `pyvenv.cfg` records it was created at **`/Users/u6788552/Desktop/experiments/models/dmpnn/.venv`** — a *different* directory from this repository. It was copied here. Its `bin/` shebangs will point at the old path. Referenced in `analysis/paper1_figures/build_all_figures.py` docstring (`.venv/bin/python …`) and `.devin/config.local.json`. |
| `venv/` | Python 3.12.2. `pyvenv.cfg` records creation at **`/Users/u6788552/Desktop/experiments/dmpnn/venv`** — this repository. Newer library versions (rdkit 2025.9.1 vs 2025.3.3, lightning 2.5.5 vs 2.5.2). Also referenced in `.devin/config.local.json`. |

**Risk:** both have `chemprop` installed as an editable package. Deleting the wrong one breaks your workflow immediately (recoverable, but annoying — you'd reinstall). **Do not delete either until you run `which python` in your normal working shell and confirm which one you actually activate.** Both are gitignored, so this is a local-disk decision only.

### 2.2 `paper_figures/` (16 PNGs, untracked, explicitly gitignored)

- **What it is:** figure panels for an *older* paper — styrene / PS-b-PMMA adjacency matrices, ECFP4 fingerprints, embeddings, and `row_*.png` / `col_header.png` composition strips. Last modified 23 June.
- **Why probably deletable:** listed in `.gitignore` (so it was deliberately excluded from version control), no script in the repository generates or reads it, and no document references it. The current paper pipeline is `analysis/paper1_figures/`.
- **Uncertainty:** because it is untracked *and* gitignored, **there is no git copy**. If these panels are not reproducible from a script, deletion is irreversible. Confirm they are superseded before removing; otherwise archive them (§3 route).

### 2.3 `22-07-2026 report_figures/` (35 tracked files)

- **What it is:** a July 22 figure drop — `fig1_scorecard.png`, `fig2_chemistry.png`, `fig3_architecture.png`, `fig_architecture_comparison.svg`, plus an `all/` subfolder of 17 numbered diagnostic figures with a descriptive `README.md`.
- **Why probably deletable / archivable:** superseded by `29-07-2026 supervisor_update/figures/` (one week later) and by `analysis/paper1_figures/` (August). Referenced by **nothing** in the repository — `git grep "22-07-2026"` returns zero hits.
- **Uncertainty:** it is tracked, so it is recoverable from git history after deletion. My recommendation is **archive rather than delete** (§3), because the `all/README.md` contains figure-by-figure prose and a metrics table that is a genuine provenance record.

### 2.4 Root-level one-off scripts

| Path | What it is | Uncertainty |
|---|---|---|
| `analyze_embeddings.py` | Ad-hoc script to find complete embedding sets. **Hardcodes the absolute path `/Users/u6788552/Desktop/experiments/dmpnn/results/embeddings`**, which is gitignored (`results/embeddings/`) and not present. Referenced by nothing. | It is broken as written (absolute path, missing input). Low risk, but confirm you don't still run it manually. |
| `test_variant_filter.py` | Docstring: "Test variant filtering logic for `visualize_combined_results.py`". A hand-run assertion script, not a pytest test (it lives at root, not in `tests/`). Referenced by nothing. | Pytest's `addopts = "--cov chemprop"` means a bare `pytest` from root *would* try to collect it (name matches `test_*.py`). Move to `tests/` or delete — leaving it at root is the worst option. |

### 2.5 Root-level orphaned data snippets

| Path | What it is | Uncertainty |
|---|---|---|
| `insulator_excluded_problematic_smiles.txt` (130 B) | List of SMILES excluded from the insulator dataset | Referenced by nothing in code. But this is **provenance for a data-cleaning decision** — the kind of thing that is unreferenced yet must not be lost. Recommend moving into `data/` rather than deleting. |
| `insulator_skipped_indices.txt` (405 B) | Row indices skipped | Same reasoning. |
| `graph_vs_tabular_improvement.png` (190 KB) | A figure | **Is referenced:** `experiments/tabular/compare_tabular_vs_graph.py` writes it. It is a generated output sitting at the repository root. Move, don't delete (§4 rename table). |

### 2.6 Smoke-test checkpoints and predictions

`checkpoints/_checkpoint_smoke_hpg`, `_checkpoint_smoke_hpg_final`, `_checkpoint_smoke_stage2d`, `_checkpoint_smoke_wdmpnn`, `_checkpoint_smoke_wdmpnn_fast`, `_checkpoint_smoke_wdmpnn_final`, `_dual_checkpoint_smoke`, `_dual_checkpoint_smoke_stage2d`, `_dual_checkpoint_smoke_wdmpnn` — and the matching eight directories under `predictions/`.

- **Why probably deletable:** the leading `_` marks them as scratch. `git grep "checkpoint_smoke"` returns **zero references** in any script, config, or document. `checkpoints/` is entirely gitignored; the `predictions/` copies are small (8–136 KB each).
- **Uncertainty:** these were the artifacts of a checkpoint-selection investigation (`analyze_checkpoint_mae_gap.py` exists and writes `_regen_v1_checkpoint_gap.csv`). If that investigation's conclusion is still being cited in the paper, the smoke runs are its raw evidence. Check whether `HANDOFF_2026-08-05.md` or `CONTEXT_PACK_2026-08-09.md` still leans on them before deleting.

### 2.7 Superseded Windsurf prompt files (13 files)

| Group | Files |
|---|---|
| `analysis/model_diagnostics/` (tracked) | `windsurf_prompt_analyze_phase1_results.md`, `windsurf_prompt_code_drift_vs_noise.md`, `windsurf_prompt_groupmean_metric_floor.md`, `windsurf_prompt_readout_ablation_spec.md`, `windsurf_prompt_training_stability.md`, `windsurf_prompt_verify_octamer_provenance.md`, `windsurf_spec_seeds_and_readout_ablation.md` |
| `analysis/model_diagnostics/` (untracked) | `WINDSURF_PROMPT_posemb_r1_analysis.md` |
| `analysis/paper1_figures/` (untracked) | `WINDSURF_PROMPT_F2_rebuild.md`, `WINDSURF_PROMPT_arch_spread_fixes.md`, `WINDSURF_PROMPT_arch_spread_recovery.md` |
| kept as duplicates elsewhere | `windsurf_spec_b_heldout_split.md`, `windsurf_spec_regeneration.md` (see §1.7) |

- **What they are:** instructions written *to* an AI agent to produce a specific analysis. Each has a corresponding output document (e.g. `windsurf_prompt_groupmean_metric_floor.md` → `_groupmean_metric_floor.md`).
- **Why probably deletable:** the output supersedes the prompt. Nothing in code references them.
- **Uncertainty — and this is a real one:** for a methods paper, the prompt *is* part of the method's audit trail. `windsurf_spec_b_heldout_split.md` in particular specifies a split design that ended up in the paper. **My recommendation is archive, not delete** — move the whole set to `docs/archive/agent-prompts/`. Deleting loses the record of what was actually asked for.

### 2.8 `.devin/config.local.json`

Agent tool-permission config (`Exec(python)`, `Exec(ls)`, …) for the Devin/Windsurf agent. Untracked. Machine-local and personal; safe to delete if you no longer use that agent, harmless to keep. **Not a secret**, but it does leak your local absolute paths.

---

## 3. Keep

Items that look like clutter but should stay.

| Path | Why it must stay |
|---|---|
| `logs/` (923 tracked files, 893 `.pbs`) | **Misleadingly named — these are not logs.** They are the generated PBS job scripts and manifests for every HPC campaign (`regen_v1`, `octamer_k1`, `octamer_posemb`, `wdmpnn_original`, …). They are the reproducibility record of exactly what was submitted. `scripts/python/verify_regen_v1_pilot.py` reads `logs/regen_v1/r1/manifests/r1_pilot.manifest` directly, and `scripts/python/analyze_regen_v1.py` documents grepping `logs/regen_v1/r3/tasks/`. **Rename, never delete** (§5). |
| `metadata/splits/*.json` (5 files) | Frozen cross-validation split definitions. `.gitignore` ignores `*.json` globally but has an explicit `!metadata/splits/*.json` exception — that exception exists precisely because these are load-bearing. `analysis/diagnostics/config.py` sets `META_DIR = PROJECT_ROOT/'metadata'/'splits'`. Deleting these makes every past result unreproducible. |
| `configs/*.yaml` (4) + `configs/README_visualization_config.md` | `.gitignore` ignores all YAML except three named exceptions; these survive by explicit `!train_config.yaml` / `!scripts/shell/*.yaml` rules. `wdmpnn_a_held_out.yaml`, `wdmpnn_group_disjoint.yaml`, `wdmpnn_pair_disjoint.yaml` are experiment definitions. |
| `pyproject.toml`, `MANIFEST.in`, `LICENSE.txt` | Packaging and licence. `pyproject.toml` defines the `chemprop` console script, pytest config (`addopts = "--cov chemprop"`), black/isort settings. `MANIFEST.in` controls the sdist. `LICENSE.txt` is referenced by `pyproject.toml`'s author field. |
| `.gitignore` | Needs *editing* (§7 step 1), not deleting. |
| `chemprop/scripts/__init__.py`, `evaluation/__init__.py` | Zero-byte, but they are package markers. Deleting them breaks `from evaluation.metrics import …`, which appears in 8+ scripts. |
| `evaluation/` (3 files) | `metrics.py` and `naming.py` are imported by `scripts/evaluate_ea_ip_predictions.py`, `scripts/generate_split_metadata.py`, `scripts/migrate_prediction_filenames.py`, `scripts/python/analyze_octamer_k1.py`, `analysis/paper1_figures/build_all_figures.py`, `experiments/hpg2stage/scripts/generate_paper_outputs.py`, `chemprop/cli/train.py`, and more. Small folder, high fan-in. |
| `polymer_input/` (17 files) | Self-contained featurizer/parsing package with its own README and tests. Internally cohesive. |
| `analysis/model_diagnostics/` **directory name** | Hardcoded as an output root in ~15 scripts: `analysis/diagnostics/config.py` (`OUT_ROOT`), `followup_summary.py`, `run_all_diagnostics.py`, `scripts/python/aggregate_lomo_seeds.py`, `aggregate_seeded_diagnostics.py`, `analyze_checkpoint_mae_gap.py`, `analyze_hpg_noise_floor.py`, `analyze_hpg_stability_fixes.py`, `analyze_octamer_k1.py`, `analyze_octamer_posemb.py`, `analyze_regen_v1.py`, and `29-07-2026 supervisor_update/figures_deck.py` + `figures_results.py`. **Do not rename this directory** without a coordinated find-and-replace. |
| `29-07-2026 supervisor_update/figstyle.py` | **Load-bearing despite the folder name.** `analysis/paper1_figures/build_all_figures.py` loads it via `importlib.util.spec_from_file_location("figstyle", ROOT / "29-07-2026 supervisor_update" / "figstyle.py")` — a hardcoded path including the space and the date. Renaming the folder breaks the paper figure build. |
| `tests/*.py` (18 test modules) | Active test suite including `test_octamer_position_embeddings.py` (Aug 7) and `test_arch_spread_metrics.py` (Aug 11, untracked — commit it). |
| `data/` | Entirely gitignored but contains the 15 source datasets (`ea_ip.csv`, `htpmd.csv`, `opv_camb3lyp.csv`, …). `analysis/diagnostics/config.py` sets `DATA_PATH = PROJECT_ROOT/'data'/'ea_ip.csv'`. **Never delete.** |
| `preprocessing/` (803 entries) | Gitignored. Fitted scalers, correlation masks, feature-removal indices per split. Required for reproducible inference on saved checkpoints — documented in `README.md`. Bulky but load-bearing. |
| `writing/paper1_draft.md`, `writing/paper2_outline.md` | The actual papers. `paper1_draft.md` is untracked (Aug 11) — **commit it before doing anything else.** |
| `chemprop/` entire tree | The packaged library. `pyproject.toml` `[tool.setuptools.packages.find] include = ["chemprop"]`. Do not restructure. |

---

## 4. Needs investigation

I cannot classify these confidently. Each needs a specific check you can do in a minute or two.

| Path | What to check |
|---|---|
| `RESEARCH_REVIEW/01_stage1_inventory.md`, `RESEARCH_REVIEW/02_stage2_research_questions.md` | **Tracked in git but deleted from disk** — an uncommitted deletion sitting in your working tree. Did you delete them deliberately (then: `git rm` and commit) or by accident (then: `git checkout -- RESEARCH_REVIEW/`)? Nothing else references them. Resolve this before any other git operation, or it will get tangled in the cleanup commits. |
| `hpg_hier_design/seed_42_diagnostic/` (subdirs `01_validation` … `10_summary`) | An **older snapshot of the same diagnostics pipeline** that now writes to `analysis/model_diagnostics/`. I diffed them: same filenames, **different contents** — so it is a genuinely different (earlier) run, not a copy. Referenced by nothing (`git grep seed_42_diagnostic` → 0 hits). Question: is this a superseded run to archive, or the seed-42 arm of a multi-seed comparison whose numbers are still cited? `analysis/diagnostics/config.py` has `set_active_seed()` writing to `analysis/model_diagnostics/seed_{seed}/`, which suggests the newer convention replaced this folder. |
| `29-07-2026 supervisor_update/evidence/_dataset_design_audit.md` | The only file in `evidence/` that is **not** byte-identical to its `analysis/model_diagnostics/` twin. One is a modified copy. Diff them and decide which is canonical before deleting either. |
| `results/DMPNN/block_results.csv` == `results/GIN/block_results.csv` | **Byte-identical results files for two different architectures.** Either (a) a copy/paste error that mislabelled one model's results, or (b) a genuine coincidence on a degenerate task. This is a correctness question, not a tidiness one — worth checking before either file is cited. |
| `experiments/wdmpnn_diagnostics/` (8 PNGs) | No script in the repository generates or reads these (`git grep wdmpnn_diagnostics` → 0 hits). Orphaned figure output. Is there a generator that was deleted, or were these produced ad hoc? Determines archive vs delete. |
| `analysis/model_diagnostics/14_lambda_pilot_comparison/` (1 file) and `13_chemarch_residual_ablation/` | `13_` is produced by `analysis/diagnostics/chemarch_residual_ablation.py`. `14_` contains a single `pareto_plots.png` with no matching module in `analysis/diagnostics/`. Confirm `14_` is still current. |
| `experiments/README.md` | Documents an `experiments/eda/` directory with 7 named scripts (`plot_target_distributions.py`, `feature_space_analysis.py`, …). **That directory does not exist.** Either the folder was deleted and the README is stale (fix the README), or it was moved and should be restored. |
| `analysis/model_diagnostics/junction_n2_failures/` | Produced by `scripts/python/diagnose_junction_failures.py` and cited in `hpg_hier_design/paper1_objective_design_draft.md`. Keep — but it is the only non-numbered analysis subfolder, so it sits oddly. Confirm whether it belongs under one of the numbered steps. |
| `predictions/migration_log.json` | Written by `scripts/migrate_prediction_filenames.py`. Is the migration complete (→ archive the log and possibly the script), or is it still in use? |
| `analysis/model_diagnostics/_check_octamer_posemb_pilot.py` + its 4 output files | Untracked ad-hoc verification script living in an *output* directory. Its results appear to be folded into `_octamer_posemb_r1_results.md`. If so, archive; if it is still re-run, it belongs in `scripts/python/`. |
| `results/embeddings/` | Gitignored and absent from disk, but `analyze_embeddings.py` and `scripts/python/run_embeddings_only.py` expect it. Was it deleted to save space, or never generated on this machine? |

---

## 5. Proposed renames

**Rule applied:** I only propose renaming things that are *not* hardcoded in source. Where a name is bad *and* hardcoded, I say so and list the references to update.

### 5.1 Renames with no reference impact (safe)

| | |
|---|---|
| **Current** | `22-07-2026 report_figures/` |
| **Proposed** | `docs/archive/2026-07-22-report-figures/` |
| **Reason** | Leading `DD-MM-YYYY` sorts wrongly (22-07 sorts before 29-07 only by luck; 01-08 would sort first). The space breaks shell globbing and required quoting in every command I ran. Zero references — `git grep "22-07-2026"` → 0 hits. |

| | |
|---|---|
| **Current** | `logs/` |
| **Proposed** | `jobs/` |
| **Reason** | It contains 893 generated `.pbs` **job scripts** and manifests, not logs. Anyone opening `logs/` expects stdout/stderr and will not find it. **References to update:** `scripts/python/verify_regen_v1_pilot.py:9` (`MANIFEST = ROOT/"logs"/...`), the prose instructions in `scripts/python/analyze_regen_v1.py:322,438`, and `LOG_DIR="$LOCAL_PROJECT/logs/..."` in the `scripts/shell/*.sh` generators (at least `backfill_regen_v1_missing_5.sh:43`). **Do not** confuse these with the unrelated `logs/` used for Lightning checkpoints inside `checkpoints/*/logs/` (`scripts/python/utils.py:1171,1456,3509`) — those are a different thing and must not be touched. |

| | |
|---|---|
| **Current** | `graph_vs_tabular_improvement.png` (repository root) |
| **Proposed** | `experiments/tabular/figures/graph_vs_tabular_improvement.png` |
| **Reason** | A generated figure at the repository root. **Reference to update:** the output path in `experiments/tabular/compare_tabular_vs_graph.py`. |

| | |
|---|---|
| **Current** | `insulator_excluded_problematic_smiles.txt`, `insulator_skipped_indices.txt` (root) |
| **Proposed** | `data/insulator/excluded_problematic_smiles.txt`, `data/insulator/skipped_indices.txt` |
| **Reason** | Dataset provenance belongs with the dataset. No code references either, so the move is free. (Note `data/` is gitignored — see §7 step 1 if you want these versioned.) |

| | |
|---|---|
| **Current** | `test_variant_filter.py` (root) |
| **Proposed** | `tests/test_variant_filter.py` (if kept — see §2.4) |
| **Reason** | Matches pytest's `test_*.py` collection pattern but sits outside `tests/`, so `pytest` from root collects it unexpectedly. |

| | |
|---|---|
| **Current** | `analyze_embeddings.py` (root) |
| **Proposed** | `scripts/python/analyze_embeddings.py` |
| **Reason** | It is a script; every other script lives under `scripts/python/`. Also replace its hardcoded `/Users/u6788552/...` path with a `PROJECT_ROOT`-relative one. |

| | |
|---|---|
| **Current** | `analysis/model_diagnostics/{s,slide,e,f,g,h,k,t}-NN.jpg` (54 files) |
| **Proposed** | *(delete — see §1.8)*; if any are kept, `docs/archive/deck-renders/2026-08-10-research-update/slide-NN.jpg` |
| **Reason** | Single-letter prefixes carry no meaning. Rendered deck previews do not belong in a pipeline output directory. |

| | |
|---|---|
| **Current** | `analysis/model_diagnostics/13_final_followup_summary.md` |
| **Proposed** | `analysis/model_diagnostics/13_final_followup_summary.md` *(keep, but see note)* |
| **Reason** | It collides conceptually with the directory `13_chemarch_residual_ablation/` — two different things both numbered 13. Written by `analysis/diagnostics/followup_summary.py` (docstring line 1), so **renaming requires editing that module**. Low priority; flagging the collision. |

### 5.2 Renames I recommend **against**

| Current | Why not to rename |
|---|---|
| `analysis/model_diagnostics/` | Hardcoded in ~15 modules as an output root (full list in §3). The rename is mechanical but touches the code that generates your paper's numbers. Not worth the risk for a name that is already descriptive. |
| `29-07-2026 supervisor_update/` | **`figstyle.py` inside it is imported by absolute path** from `analysis/paper1_figures/build_all_figures.py` via `importlib`. Because it is loaded by path rather than by import statement, a rename fails *silently at runtime*, not at lint time. If you do rename it, fix that line first and re-run `build_all_figures.py` to confirm. Recommended target if you proceed: `docs/archive/2026-07-29-supervisor-update/`, **with `figstyle.py` first promoted to `analysis/figstyle.py`** (see §6.3). |
| `chemprop/` and everything under it | Package name in `pyproject.toml`, `MANIFEST.in`, the `chemprop = "chemprop.cli.main:main"` console script, and every import in the repository. |
| `metadata/splits/` | `analysis/diagnostics/config.py:9`. |
| `predictions/`, `checkpoints/`, `preprocessing/`, `data/`, `results/` | All hardcoded in `analysis/diagnostics/config.py` and dozens of scripts, and documented in `README.md`. These are conventional names doing their job. |

---

## 6. Consolidation opportunities

### 6.1 Documentation is scattered across 8 locations

Current README-type files: `README.md` (root, "Project Structure"), `COPOLYMER_USAGE.md`, `PREDICTIONS_README.md`, `chemprop/README.md`, `chemprop/MODEL_GUIDE.md`, `chemprop/IMPLEMENTING_CUSTOM_MPNN.md`, `scripts/README.md`, `experiments/README.md`, `experiments/hpg2stage/README.md`, `configs/README_visualization_config.md`, `polymer_input/README.md`, `22-07-2026 report_figures/all/README.md`.

**Recommendation:** keep exactly one `README.md` at root as an *entry point and index*, and move the rest into `docs/`:

```
docs/
├── copolymer-training.md        ← COPOLYMER_USAGE.md
├── predictions-format.md        ← PREDICTIONS_README.md
├── visualization-config.md      ← configs/README_visualization_config.md
└── ...
```

Leave `chemprop/README.md`, `chemprop/MODEL_GUIDE.md`, `chemprop/IMPLEMENTING_CUSTOM_MPNN.md`, `scripts/README.md`, `experiments/README.md` and `polymer_input/README.md` **where they are** — a README next to the code it describes is the right convention, and `MANIFEST.in` ships the chemprop ones.

**Also:** the root `README.md` is declared as the package readme in `pyproject.toml` (`readme = "README.md"`) but its content is the research project's directory guide, not a chemprop description. If you ever publish this package, that mismatch surfaces on PyPI. Low priority, worth knowing.

### 6.2 Handoff / context documents overlap heavily

In `analysis/model_diagnostics/`:

| File | Date | Size |
|---|---|---|
| `PROJECT_STATE_HANDOFF.md` | 26 Jul | 10 KB |
| `HANDOFF_2026-07-29.md` | 29 Jul | 24 KB |
| `HANDOFF_2026-08-05.md` | 5 Aug | 15 KB |
| `CONTEXT_PACK_2026-08-09.md` | 9 Aug | 9 KB |
| `PANEL_BRIEFING_2026-08-10.md` | 10 Aug | 11 KB |
| `MODELS_AND_COMPUTE_brief_2026-07-30.md` | 30 Jul | 34 KB |

`HANDOFF_2026-08-05.md` opens: *"Extends `HANDOFF_2026-07-29.md`. That document is **not edited**; where this one disagrees, this one wins."* — so this is an intentional append-only chain, not accidental duplication. **Do not merge them.** But do:

1. Add a `docs/HANDOFF_INDEX.md` (or a header block in the newest file) stating the chain order and which is current.
2. Move superseded ones to `docs/archive/handoffs/` keeping the newest (`CONTEXT_PACK_2026-08-09.md` + `PANEL_BRIEFING_2026-08-10.md`) at `docs/`.

`PROJECT_STATE_HANDOFF.md` is the one genuinely ambiguous name here — it is dated 26 July, i.e. the *oldest*, but its name implies "current state". Rename to `HANDOFF_2026-07-26.md` for consistency with the chain.

### 6.3 Deck-building scripts: three near-identical generations

| Script | Output |
|---|---|
| `analysis/model_diagnostics/build_eval_framework_deck.py` | `Eval_framework_paper1.pptx` |
| `analysis/model_diagnostics/build_eval_framework_deck_v2.py` | `Eval_framework_paper1_v2.pptx` |
| `analysis/model_diagnostics/build_update_deck.py` | `Research_update_2026-08-10.pptx` |
| `29-07-2026 supervisor_update/build_deck.py` | `week_review_monomerB_split.pptx` |

Plus plan documents `DECK_PLAN_eval_framework.md` and `DECK_PLAN_eval_framework_v2.md` (16 KB / 15 KB, both untracked, both Aug 11).

**Recommendation:** the `_v2` pairs are strict supersessions — keep `_v2`, archive `v1`. Consolidate all four builders into `scripts/decks/` and factor the shared styling into the single `figstyle.py` (which currently lives in the dated folder and is imported by absolute path — promoting it to `analysis/figstyle.py` or `scripts/python/figstyle.py` fixes that fragility at the same time).

### 6.4 Figure directories: five serving the same purpose

`paper_figures/` · `analysis/paper1_figures/` · `22-07-2026 report_figures/` · `29-07-2026 supervisor_update/figures/` · `experiments/tabular/figures_ea_ip/` · `experiments/wdmpnn_diagnostics/` · `experiments/diagnostics/feature_conditioned_transfer/`

**Recommendation:** `analysis/paper1_figures/` is the live one (has per-figure `fN_manifest.md` provenance files — good practice, keep it). Archive the two dated ones. Leave the `experiments/*` ones alongside their generating scripts, which is correct. Consider deleting `paper_figures/` (§2.2).

### 6.5 Two diagnostics trees

`analysis/model_diagnostics/01_…14_` and `hpg_hier_design/seed_42_diagnostic/01_…10_` are the same pipeline's output at different times. Once §4 resolves which is superseded, collapse to one — either under `analysis/model_diagnostics/seed_42/` (matching the `set_active_seed()` convention already in `config.py`) or archived.

### 6.6 Small: analysis result files split by convention

`analysis/model_diagnostics/` mixes ~44 `.md` files, ~40 `_*.csv` files, numbered output subdirectories, deck builders, PPTX files, and 54 JPGs in one flat directory of 150+ entries. The leading-underscore convention (`_regen_v1_results.md`, `_multiseed_summary.csv`) does distinguish generated reports from hand-written ones, which is genuinely useful — but it is undocumented. **Add a short `analysis/model_diagnostics/README.md` explaining the `_` prefix and the `NN_` numbering.** That single file would do more for navigability than any reorganisation.

---

## 7. Proposed final structure

Conservative: it preserves every convention that code depends on, and moves only documents and archives.

```
dmpnn/
├── README.md                      # entry point + index (rewritten to link the rest)
├── LICENSE.txt
├── pyproject.toml
├── MANIFEST.in
├── .gitignore                     # fixed: exclude .DS_Store / .ipynb_checkpoints under results/
│
├── chemprop/                      # ← UNCHANGED. The packaged library.
│   ├── README.md, MODEL_GUIDE.md, IMPLEMENTING_CUSTOM_MPNN.md
│   ├── cli/  data/  featurizers/  models/  nn/  uncertainty/  utils/
│   └── ...
│
├── polymer_input/                 # ← UNCHANGED. Featurizer/parsing package.
├── evaluation/                    # ← UNCHANGED. metrics.py, naming.py (high fan-in).
│
├── scripts/                       # ← MOSTLY UNCHANGED
│   ├── README.md
│   ├── python/                    # + analyze_embeddings.py moved in from root
│   ├── shell/
│   ├── decks/                     # NEW: the 4 deck builders consolidated here
│   └── evaluate_ea_ip_predictions.py, generate_split_metadata.py, migrate_prediction_filenames.py
│
├── configs/                       # ← UNCHANGED. Experiment + visualization YAML.
├── metadata/splits/               # ← UNCHANGED. Frozen split definitions. Load-bearing.
│
├── tests/                         # ← UNCHANGED (+ test_variant_filter.py moved in)
│
├── experiments/
│   ├── README.md                  # FIX: remove the eda/ section, or restore eda/
│   ├── tabular/                   # + figures/graph_vs_tabular_improvement.png from root
│   ├── hpg/
│   ├── diagnostics/
│   ├── hpg2stage/
│   └── wdmpnn_diagnostics/        # pending §4
│
├── analysis/
│   ├── figstyle.py                # MOVED here from "29-07-2026 supervisor_update/"
│   ├── diagnostics/               # ← UNCHANGED. The pipeline modules.
│   ├── model_diagnostics/         # ← NAME UNCHANGED (hardcoded). Contents thinned:
│   │   ├── README.md              # NEW: explains the _ prefix and NN_ numbering
│   │   ├── 01_validation/ … 14_lambda_pilot_comparison/
│   │   ├── _*.md, _*.csv          # generated reports (kept)
│   │   └── (JPGs, decks, deck plans, windsurf prompts → moved out)
│   └── paper1_figures/            # ← UNCHANGED. Has fN_manifest.md provenance. Good.
│
├── writing/
│   ├── paper1_draft.md            # COMMIT THIS — currently untracked
│   └── paper2_outline.md
│
├── docs/                          # NEW — was gitignored; un-ignore it
│   ├── copolymer-training.md      ← COPOLYMER_USAGE.md
│   ├── predictions-format.md      ← PREDICTIONS_README.md
│   ├── visualization-config.md    ← configs/README_visualization_config.md
│   ├── HANDOFF_INDEX.md           # NEW: which handoff is current, and the chain order
│   ├── CONTEXT_PACK_2026-08-09.md
│   ├── PANEL_BRIEFING_2026-08-10.md
│   └── archive/
│       ├── handoffs/              ← HANDOFF_2026-07-26/-07-29/-08-05, MODELS_AND_COMPUTE_brief
│       ├── agent-prompts/         ← the 13 windsurf_prompt_* / WINDSURF_PROMPT_* files
│       ├── deck-plans/            ← DECK_PLAN_eval_framework{,_v2}.md
│       ├── 2026-07-22-report-figures/    ← "22-07-2026 report_figures/"
│       └── 2026-07-29-supervisor-update/ ← "29-07-2026 supervisor_update/" minus figstyle.py
│
├── decks/                         # NEW: the .pptx deliverables, out of the analysis tree
│   ├── Eval_framework_paper1_v2.pptx
│   ├── Research_update_2026-08-10.pptx
│   └── week_review_monomerB_split.pptx
│
│   # ── gitignored data & artifacts (unchanged, listed for completeness) ──
├── data/                          # + insulator/ provenance txt files from root
├── preprocessing/
├── checkpoints/
├── predictions/
├── results/
├── jobs/                          # RENAMED from logs/ — generated PBS scripts + manifests
└── venv/  or  .venv/              # keep exactly one
```

**Removed from root by this plan:** `pred.zip`, `.pytest_cache/`, `analyze_embeddings.py`, `test_variant_filter.py`, `graph_vs_tabular_improvement.png`, `insulator_*.txt`, `COPOLYMER_USAGE.md`, `PREDICTIONS_README.md`, `22-07-2026 report_figures/`, `29-07-2026 supervisor_update/`, `paper_figures/`, `hpg_hier_design/`, one of the two venvs. Root goes from 41 entries to about 22, of which the top 12 are self-explanatory.

**What I deliberately did *not* do:** invent a `src/` layout. `pyproject.toml` uses `[tool.setuptools.packages.find] include = ["chemprop"]` with a flat layout, and every script imports accordingly. Changing it would be churn for no navigability gain.

---

## 8. Action plan

Ordered so that each step is independently revertible and nothing depends on a later step.

**Step 0 — Prerequisites (do these first, they are cheap and they protect everything after).**

- Resolve the `RESEARCH_REVIEW/` deletion (§4): either `git rm` + commit, or `git checkout --`.
- **Commit the 9 modified files and the untracked work that matters** — especially `writing/paper1_draft.md` and `tests/test_arch_spread_metrics.py`. Cleanup on a dirty tree is how work gets lost.
- Create a branch: `git checkout -b cleanup/repo-organisation`.
- Tag the current state: `git tag pre-cleanup-2026-08-11`.

**Step 1 — Fix `.gitignore` (before deleting anything, so nothing returns).**

Add under the existing `!results/*` / `!results/**/*` block:
```
results/**/.DS_Store
results/**/.ipynb_checkpoints/
```
Also decide whether `docs/` should stay ignored — the current `docs/` rule will silently swallow the new documentation folder. Change it to `docs/_build/` only.
*References affected:* none. Commit this alone.

**Step 2 — Delete confirmed generated/temporary files (§1).**

```
git rm --cached results/**/.DS_Store results/**/.ipynb_checkpoints/*   # then delete on disk
```
Then on disk only: the 13 untracked `.DS_Store`, 23 `__pycache__/`, `.pytest_cache/`, `pred.zip`, the 54 JPGs, `generate_stage2d_paper_outputs.py.bak`, `README copy.md`.
*References affected:* none — all verified unreferenced.
*Recoverable?* Tracked items yes (git). `pred.zip` yes (contents already on disk). JPGs yes (re-render from `.pptx`).

**Step 3 — Delete the 9 duplicated evidence documents (§1.7).**

`git rm "29-07-2026 supervisor_update/evidence/"{_a_heldout_bitwise_reproduction,_code_drift_investigation,_groupmean_metric_floor,_noise_floor_results,_octamer_provenance_check,_training_stability_stepc_results,variant_results_report}.md` and the two `specs/windsurf_spec_*.md`.
*Hold back* `evidence/_dataset_design_audit.md` until §4 is resolved.
*References affected:* none — `figures_deck.py` and `figures_results.py` read from `analysis/model_diagnostics/`, which is untouched.

**Step 4 — Remove empty directories (§1.9).**

Filesystem only, zero git impact.

**Step 5 — Create the new organisational folders.**

`mkdir -p docs/archive/{handoffs,agent-prompts,deck-plans} decks scripts/decks`
Nothing moves yet.

**Step 6 — Move documents (use `git mv` throughout to preserve history).**

6a. `git mv COPOLYMER_USAGE.md docs/copolymer-training.md` — *update:* any link in `README.md`.
6b. `git mv PREDICTIONS_README.md docs/predictions-format.md` — *update:* links in `README.md`; note it references `evaluation.metrics`/`evaluation.naming`, which do not move.
6c. `git mv configs/README_visualization_config.md docs/visualization-config.md` — *update:* links in `configs/` and `scripts/README.md`.
6d. Move the 13 windsurf prompt files → `docs/archive/agent-prompts/`.
6e. Move superseded handoffs → `docs/archive/handoffs/`.
6f. Move `DECK_PLAN_*` → `docs/archive/deck-plans/`; the 3 `.pptx` → `decks/`.
*References affected:* documentation links only. No code reads any of these — verified by `git grep`.

**Step 7 — Promote `figstyle.py`, then archive the dated folders.**

7a. `git mv "29-07-2026 supervisor_update/figstyle.py" analysis/figstyle.py`
7b. **Update `analysis/paper1_figures/build_all_figures.py` lines 29–32** — the `importlib.util.spec_from_file_location("figstyle", ROOT / "29-07-2026 supervisor_update" / "figstyle.py")` call becomes a plain `from analysis.figstyle import ...` or a path pointing at the new location. The module is then used at lines 70, 74, 107, 477, 711, 827, 907.
7c. Two scripts in that folder import it as a *sibling* (`from figstyle import ...`), which works only when run from inside the folder: `figures_deck.py:31` and `figures_results.py:42`. Update both. `figstyle.py:13` documents this usage in its own docstring.
7d. **Run `python analysis/paper1_figures/build_all_figures.py` and confirm all six figures regenerate byte-identically.** This is the single highest-risk change in the plan — do not skip the check.
7e. Only then: `git mv "29-07-2026 supervisor_update" docs/archive/2026-07-29-supervisor-update` and `git mv "22-07-2026 report_figures" docs/archive/2026-07-22-report-figures`.
*References affected:* `analysis/paper1_figures/build_all_figures.py` (hardcoded absolute path), and any of the five scripts in that folder that import `figstyle`.

**Step 8 — Move stray root files.**

- `git mv analyze_embeddings.py scripts/python/` — *and* fix its hardcoded `/Users/u6788552/...` path to `PROJECT_ROOT`-relative.
- `git mv test_variant_filter.py tests/` (or delete per §2.4).
- `git mv graph_vs_tabular_improvement.png experiments/tabular/figures/` — *update:* the output path in `experiments/tabular/compare_tabular_vs_graph.py`.
- Move `insulator_*.txt` into `data/insulator/` (note: `data/` is gitignored, so these leave version control — if you want them tracked, put them in `metadata/` instead).

**Step 9 — Rename `logs/` → `jobs/`.**

`git mv logs jobs`, then update:
- `scripts/python/verify_regen_v1_pilot.py:9`
- `scripts/python/analyze_regen_v1.py:322,438` (prose instructions)
- `LOG_DIR=`/`PBS_DIR=`/`TASK_LOG_DIR=` in `scripts/shell/backfill_regen_v1_missing_5.sh:43-47` and every sibling `generate_*.sh` / `submit_*.sh` that writes there
- any mention in `README.md`

**Do not** touch `checkpoints/*/logs/` — that is Lightning's own directory, referenced at `scripts/python/utils.py:1171,1456,1677,3509,3518`, `analysis/diagnostics/chemarch_residual_ablation.py:62`, and `experiments/hpg2stage/scripts/stage2d_postrerun_analysis.py:123`.

*This step is optional and the most invasive rename in the plan. If you want a lower-risk version: leave the directory named `logs/` and add a `logs/README.md` saying "these are generated PBS job scripts, not runtime logs."*

**Step 10 — Fix stale documentation.**

- `experiments/README.md`: remove or restore the `eda/` section (§4).
- Rewrite root `README.md` as an index reflecting the new layout.
- Add `analysis/model_diagnostics/README.md` explaining the `_` and `NN_` conventions (§6.6).
- Add `docs/HANDOFF_INDEX.md` (§6.2).

**Step 11 — Verify.**

- `python -c "import chemprop"` and `python -c "from evaluation.metrics import compute_copolymer_metrics"`
- `pytest tests/ -x -q --no-cov` (the `--cov chemprop` default in `pyproject.toml` will slow this down)
- `python analysis/paper1_figures/build_all_figures.py` and diff the six outputs against the pre-cleanup copies
- `grep -rn "29-07-2026\|22-07-2026\|logs/" --include=*.py --include=*.sh --include=*.md .` to catch any reference the plan missed
- `git status` should be clean; `git log --stat` should show `git mv` as renames, not delete+add

**Step 12 — Handle the §2 and §4 items separately, one at a time,** after the mechanical cleanup is committed and verified. In particular: decide the venv (§2.1), diff `_dataset_design_audit.md` (§4), and check the `block_results.csv` collision (§4) — that last one is a possible data-correctness issue, not a tidiness one, and deserves attention regardless of whether you do the rest of this cleanup.

---

## 9. Risk summary

| Risk | Where | Mitigation |
|---|---|---|
| **Highest** — silent runtime break | `figstyle.py` loaded by absolute path from `analysis/paper1_figures/build_all_figures.py`. `importlib` path loads fail at run time, not import time. | Step 7d: regenerate all six paper figures and diff. |
| High | Renaming `logs/` misses a `.sh` reference; job generation writes to a path that no longer exists. | Step 9 grep + Step 11 grep. Or skip Step 9 entirely. |
| Medium | Deleting `paper_figures/` — untracked *and* gitignored, so **no git copy exists**. | Archive instead of delete until confirmed reproducible. |
| Medium | Deleting the wrong venv. | Step 12; check `which python` first. |
| Low | Everything in §1 — all verified unreferenced, and tracked items are recoverable from git. | The `pre-cleanup-2026-08-11` tag. |
| Not addressed by cleanup | `results/DMPNN/block_results.csv` == `results/GIN/block_results.csv`. | Investigate independently — this may be a mislabelled result. |
