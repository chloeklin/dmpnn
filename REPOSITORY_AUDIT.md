# Repository Audit Report

Generated: 2025-06-23

---

## Part 1: Unused / Stale Files — Candidates for Removal

### Root-level files (definitely removable)

| File | Reason |
|------|--------|
| `analyze_embeddings.py` | One-off exploration script, not referenced anywhere |
| `test_variant_filter.py` | One-off test script, not referenced anywhere |
| `insulator_excluded_problematic_smiles.txt` | Not referenced by any script |
| `insulator_skipped_indices.txt` | Not referenced by any script |
| `graph_vs_tabular_improvement.png` | Output image; only referenced by `experiments/tabular/compare_tabular_vs_graph.py` (which generates it). Belongs in experiments/tabular output, not root |

### Root-level directories (empty or stale)

| Directory | Reason |
|-----------|--------|
| `out/` | Empty directory |
| `styrene/` | Contains 9 diagnostic PNG images; not referenced by any script. Early exploration artifact |
| `paper_figures/` | 7 PNG images (row_adj, row_ecfp, etc.); not referenced by any script |
| `logs/` | Contains only empty subdirs (`generalization/`, `learning_curve/`). No actual logs |
| `docs/` | Sphinx skeleton (`Makefile`, `source/`) from upstream chemprop. Not used for this project |

### `scripts/python/` — stale scripts

| File | Reason |
|------|--------|
| `_plot_comparison_core4.py` | Tiny throwaway script (73 lines); hardcoded paths, not imported anywhere |
| `debug_gat_viz.py` | Debug script ("check why GAT results aren't showing"); not referenced |
| `lookup_electrolyte_smiles.py` | Electrolyte dataset utility; not referenced, references non-existent `data/electrolytes/` |
| `analyze_rdkit_impact.py` | One-off analysis; not referenced by any other file |
| `plot_htpmd_fusion_variants.py` | HTPMD-specific; not referenced |
| `plot_htpmd_tabular_desc_vs_graph_plain.py` | HTPMD-specific; not referenced |
| `plot_block_identity_vs_structure.py` | Not referenced by any file |
| `plot_graph_vs_tabular.py` | Not referenced by any file |
| `polyinfo_paper_prepro/` | Two preprocessing scripts for polyimide data; not referenced anywhere. References `data/polyimide.csv` which likely doesn't exist |

### `scripts/shell/` — stale scripts

| File | Reason |
|------|--------|
| `analyze_missing_preprocessing.sh` | Not referenced anywhere |
| `check_embedding_generate_evaluation.sh` | Not referenced anywhere |
| `generate_opv_evaluation_scripts.sh` | Only mentioned once in `scripts/README.md`; likely obsolete |
| `attentive.sh` | Tiny (698B); still used by train_attentivefp pipeline? Marginal |

### `experiments/hpg2stage/output/` — intermediate analysis artifacts

These are *intermediate debugging reports* from development. The final paper outputs are in `output/paper/`. Consider removing:

| File | Reason |
|------|--------|
| `arch_metric_definition_audit.md` | Internal audit during development; superseded by final paper outputs |
| `arch_metric_discrepancy_explanation.md` | Internal debugging; explains LC vs final discrepancy (now resolved) |
| `recompute_archdev_report.md` | Intermediate debugging report |
| `stage2d_learning_curve_final_plan.md` | Planning document; execution is complete |
| `full_archdev_comparison.csv` | Intermediate data |
| `lc_vs_stage2d_archdev_comparison.csv` | Intermediate comparison |

---

## Part 2: Markdown Files — Assessment

### Keep (active documentation)

| File | Status |
|------|--------|
| `README.md` | Project root README — keep |
| `COPOLYMER_USAGE.md` | Still describes valid workflow for train_graph.py copolymer mode |
| `experiments/README.md` | Experiment directory guide |
| `experiments/hpg2stage/README.md` | HPG2Stage experiment guide |
| `scripts/README.md` | Scripts directory guide |
| `polymer_input/README.md` | Module documentation |
| `chemprop/README.md` | Library documentation |
| `chemprop/MODEL_GUIDE.md` | Model reference |
| `chemprop/IMPLEMENTING_CUSTOM_MPNN.md` | Developer guide |
| `scripts/python/MODEL_ARCHITECTURES_EA_IP.md` | Architecture reference for ea_ip models |
| `experiments/hpg2stage/output/paper/*.md` | Final paper outputs (just generated) |
| `experiments/hpg2stage/output/postrerun/*.md` | Final analysis results — keep |

### Consider removing (stale/obsolete docs)

| File | Reason |
|------|--------|
| `GRAPHORMER_QUICKSTART.md` | Graphormer was experimental; scripts exist but no results used in paper |
| `COLOR_SCHEME_STANDARD.md` | Documents color scheme policy. Useful but not actively used — the actual colors in paper figures use a different scheme (`COLORS` dict in generate_stage2d_paper_outputs.py) |
| `configs/README_visualization_config.md` | Documents visualization_config.yaml usage; could be consolidated into scripts/README.md |
| `scripts/shell/README_LEARNING_CURVES.md` | Documents LC script generation; still valid if you use that pipeline |
| `scripts/shell/README_dataset_specific_descriptors.md` | Documents descriptor config; still valid for that pipeline |
| `experiments/tabular/README_feature_importance.md` | Tabular feature importance guide; keep if tabular experiments still relevant |
| `experiments/tabular/figures_ea_ip/reports/*.md` | `further_experiments.md`, `executive_summary.md`, `supervisor_summary.md` — older reports; superseded by stage2d paper outputs? |
| `experiments/hpg2stage/output/diagnostics/*.md` | Intermediate diagnostic reports (5 files); superseded by final paper summary |
| `experiments/hpg2stage/output/bottleneck/*.md` | `architecture_learning_curve.md`, `architecture_signal_analysis.md` — intermediate; superseded |

---

## Part 3: `scripts/python/` Docstring Audit

### Missing module-level docstrings

| File | Issue |
|------|-------|
| **`train_graph.py`** | **No module docstring at all.** This is the core training script. Needs a docstring describing purpose, usage, key arguments, and pipeline stages |
| **`train_tabular.py`** | **No module docstring.** Needs description of tabular training pipeline |
| **`evaluate_model.py`** | **No module docstring.** Needs description of evaluation workflow |
| **`evaluate_utils.py`** | **No module docstring.** Needs description of evaluation helper functions |
| **`train_attentivefp.py`** | **No module docstring.** Needs description |
| **`attentivefp_utils.py`** | Not checked but likely missing based on pattern |

### Outdated/incomplete docstrings

| File | Issue |
|------|-------|
| **`make_wdmpnn_input.py`** | Docstring says filename is `bigsmiles_to_wdmpnn.py` but actual file is `make_wdmpnn_input.py` |
| **`train_tabular.py` → `train()`** | Docstring missing `smiles_column` and `poly_type_array` parameters (added after docstring was written) |
| **`utils.py`** | Module docstring is generic ("Utility functions for model training and data processing"). Doesn't mention key capabilities: `split_type`, `a_held_out`, `stage2d` modes, `save_predictions`, `build_copolymer_model_and_trainer`, etc. |

### Correct/adequate docstrings

| File | Status |
|------|--------|
| `utils.py` | Has module docstring (generic but present) |
| `tabular_utils.py` | Has module docstring |
| `train_graphormer.py` | Has good module docstring with usage |
| `train_identity_baseline.py` | Has excellent detailed docstring |
| `run_stage2d_generalization.py` | Has good module docstring |
| `run_wdmpnn_generalization.py` | Has good module docstring |
| `run_all_evaluations.py` | Has module docstring |
| `run_embeddings_only.py` | Has module docstring with usage |
| `psmiles_to_wdmpnn.py` | Has module docstring with usage |
| `plot_colors.py` | Has module docstring |
| `plot_phase1d.py` | Has module docstring |
| `plot_ea_ip_random_vs_monomer.py` | Has module docstring |
| `train_pae_tg.py` | Has module docstring |

---

## Part 4: Summary of Recommended Actions

### High priority (cleanup)
1. **Delete** root-level stale files: `analyze_embeddings.py`, `test_variant_filter.py`, `insulator_*.txt`, `graph_vs_tabular_improvement.png`
2. **Delete** empty/stale dirs: `out/`, `styrene/`, `paper_figures/`, `logs/`
3. **Delete** `scripts/python/` throwaway scripts: `_plot_comparison_core4.py`, `debug_gat_viz.py`, `lookup_electrolyte_smiles.py`
4. **Fix** `make_wdmpnn_input.py` docstring (wrong filename reference)
5. **Fix** `train_tabular.py` `train()` docstring (missing params)

### Medium priority (docstrings)
6. **Add** module docstring to `train_graph.py`
7. **Add** module docstring to `train_tabular.py`
8. **Add** module docstring to `evaluate_model.py`
9. **Add** module docstring to `evaluate_utils.py`
10. **Add** module docstring to `train_attentivefp.py`
11. **Update** `utils.py` module docstring to reflect current capabilities

### Low priority (optional cleanup)
12. **Delete or archive** `GRAPHORMER_QUICKSTART.md` if Graphormer not used in final paper
13. **Delete or archive** `docs/` (unused Sphinx skeleton)
14. **Delete** `scripts/python/polyinfo_paper_prepro/` if polyimide work is abandoned
15. **Delete** intermediate debug .md files from `experiments/hpg2stage/output/`
16. **Consider** moving `COLOR_SCHEME_STANDARD.md` content into `scripts/python/plot_colors.py` docstring
