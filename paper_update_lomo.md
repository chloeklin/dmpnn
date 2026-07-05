# Paper Update: A-Held-Out → LOMO (Leave-One-Monomer-Out)

## Overview

Updated `experiments/hpg2stage/scripts/generate_stage2d_paper_outputs.py` to replace the
primary evaluation split from the original **A-held-out** (`HPG2Stage/`) to
**LOMO** (`HPG2Stage_LOMAO/`). No models were retrained. Only the paper-generation
script was modified.

---

## Detected LOMO Configuration

| Property | Value |
|----------|-------|
| **Directory** | `predictions/HPG2Stage_LOMAO/` |
| **Filename split token** | `a_held_out` (embedded in the filename; the directory is the LOMO marker) |
| **Number of folds** | 9 (split0–split8, one per unique monomer A identity) |
| **Auto-detection** | `_detect_lomo_dir()` inspects `HPG2Stage_LOMAO/` for `*frac*split*.npz` files and detects the token via regex |

---

## Sections Updated

### New Loaders
- **`load_lomao_predictions(model_suffix, target_key)`** — loads stage2d predictions from `HPG2Stage_LOMAO/` (9 folds, denormalized using LOMO-derived norm params, index-remapped via value-map)
- **`load_wdmpnn_lomao_predictions(target_key)`** — loads wDMPNN LOMO predictions (`ea_ip__{target}__split{fold}.npz` pattern in `HPG2Stage_LOMAO/`)
- Both include clear `[WARN]` messages if files are missing, including the expected filename pattern and any discovered files for debugging

### Normalization Parameters
- **`estimate_normalization_params()`** now derives (intercept, slope) from the 9-fold LOMO frac predictions in `HPG2Stage_LOMAO/`
- **`estimate_normalization_params_orig()`** (new) preserves the original 5-fold a_held_out norm params for backward compatibility with `load_hpg2stage_predictions()` (used only by wDMPNN/gen loaders that still reference the original `HPG2Stage/` directory)
- Both are initialised in `main()` as `_NORM_PARAMS` (LOMO) and `_NORM_PARAMS_ORIG` (original)

### Task 1 — Inventory
- Section heading: `"Final Stage 2D (a_held_out)"` → `"Final Stage 2D (LOMO — Leave-One-Monomer-Out)"`
- Status column now checks against `N_FOLDS_LOMO` (9) instead of 5
- wDMPNN section now reads from `HPG2Stage_LOMAO/` (`load_wdmpnn_lomao_predictions`)
- Header block states detected split token, directory, and fold count
- Removed `a_held_out` row from wDMPNN generalization section (only group/pair-disjoint listed there)

### Task 2 — Tables
| Table | Change |
|-------|--------|
| **Table 1** (overall R² / MAE) | All 4 model loaders replaced with LOMO loaders; markdown title updated |
| **Table 2** (arch-deviation R² / MAE) | Reuses LOMO loaders from Table 1; markdown title updated |
| **Table 3** (generalization comparison) | `a_held_out` row removed; LOMO added as the third entry; splits now ordered by increasing extrapolation difficulty: `group_disjoint` < `pair_disjoint` < `LOMO (Leave-One-Monomer-Out)` |

### Task 3 — Figures
| Figure | Change |
|--------|--------|
| **Figure D** | `load_hpg2stage_predictions` / `load_wdmpnn_predictions` → `load_lomao_predictions` / `load_wdmpnn_lomao_predictions`; suptitle updated from "A-Held-Out:" to "LOMO (Leave-One-Monomer-Out):" |
| **Figure F** | x-axis: `['group_disjoint', 'pair_disjoint', 'a_held_out']` → `['group_disjoint', 'pair_disjoint', 'lomo']`; x-tick labels: `'A-held-out'` → `'LOMO'`; LOMO data loaded from `HPG2Stage_LOMAO/` |
| **Figures A, B, G** | **Unchanged** — variance decomposition, architecture transfer diagnostics, and learning curves do not depend on the primary split |

### Task 4 — Manifest
- Figure D caption: `"A-held-out comparison"` → `"LOMO comparison"`; data source updated to `HPG2Stage_LOMAO`
- Figure F caption: `"a_held_out, group-disjoint, pair-disjoint splits"` → `"group-disjoint, pair-disjoint, LOMO splits (ordered by difficulty)"`
- Table 1 caption: `"(a_held_out)"` → `"(LOMO)"`
- Table 2 caption: added `"(LOMO)"`

### Task 5 — Summary
- Top-level blockquote added: identifies LOMO as the primary benchmark with directory, token, and fold count
- Section 2 heading: `"Architecture Residual Contribution"` → `"Architecture Residual Contribution (LOMO)"`
- Section 5 (`Generalization Findings`): now ordered by difficulty (group-disjoint → pair-disjoint → LOMO); LOMO numbers read live from `load_lomao_predictions`; added paragraph affirming LOMO as primary benchmark
- Section 7 (`wDMPNN Comparison`): `"R²(Δ, a_held_out)"` → `"R²(Δ)"` with `(LOMO)` qualifier; narrative reframed around LOMO stringency vs group/pair-disjoint

---

## Files Changed

| File | Type of change |
|------|----------------|
| `experiments/hpg2stage/scripts/generate_stage2d_paper_outputs.py` | Primary script — all LOMO changes |
| `paper_update_lomo.md` (this file) | New — change report |

---

## A-Held-Out References Remaining

The following `a_held_out` references remain in the script but are **intentional** — they refer to the original 5-fold `HPG2Stage/` data which is still used for backward-compatible loaders:

| Location | Reason retained |
|----------|-----------------|
| `_detect_lomo_dir()` docstring / fallback token | Documents the actual filename token embedded in LOMAO files |
| `estimate_normalization_params_orig()` | Derives norm params from original `HPG2Stage/` frac files (5-fold) for `load_hpg2stage_predictions` |
| `load_hpg2stage_predictions()` docstring / default arg | Kept for wDMPNN/gen loader backward compatibility; not called from tables or figures |
| `load_wdmpnn_predictions()` docstring | Not called from tables/figures; kept for backward compatibility |
| `load_lc_predictions()` filename pattern | Learning curve files are in `HPG2Stage_LC_Final/` and use `a_held_out` — those are separate from the primary split and are **unchanged** |
| `N_FOLDS = 5` comment | Clarifying annotation |
| `_NORM_PARAMS_ORIG` | Internal global for original norm params |

---

## Confirmation

- **LOMO is now the primary benchmark** for Tables 1, 2, 3 and Figures D, F
- **No models were retrained**
- **Diagnostics (Figures A, B), learning curves (Figure G)** are unchanged
- **Robust error handling**: missing LOMO files emit `[WARN]` with the expected filename pattern
- **Automatic fold detection**: `_detect_lomo_dir()` reads the actual `HPG2Stage_LOMAO/` directory to determine token and fold count — no hardcoding required
