# wD-MPNN blast radius

Pre-fix / post-fix is determined against the vectorisation commit `4d3eb86` (2026-07-30 23:04:41 +1000, +10:00 offset). A prediction file is classed as **pre-fix** if its mtime is strictly before that timestamp and **post-fix** if it is on or after it.

## Pre-fix wDMPNN prediction files (mtime < 2026-07-30 23:04:41 +1000)

| parent directory | files | mtime range (AEST) | splits / targets | consumers | recorded claims at risk |
|---|---|---|---|---|---|
| `predictions/ea_ip_group/` | 30 | 2026-07-07 13:22 — 2026-07-21 00:06 | group_disjoint, EA & IP | `analysis/architecture_diagnostic/run_architecture_diagnostic.py` (pre/post comparison, then excluded); `analysis/gate/gate_v3_analysis.py` / `GATE_RESULT_v3.md` (summary metrics include a `wdmpnn` row) | `GATE_RESULT_v3` wdmpnn numbers; the architecture diagnostic could not report wdmpnn results |
| `predictions/ea_ip_pair/` | 30 | 2026-07-07 11:40 — 2026-07-21 01:09 | pair_disjoint, EA & IP | same as above | same as above |
| `predictions/ea_ip_lomo/` | 47 | 2026-07-07 23:12 — 2026-07-21 03:20 | LOMAO (monomer_heldout), EA & IP | same as above | same as above |
| `predictions/regen_v1/ea_ip_lomo/` | 54 | 2026-07-28 01:57 — 2026-07-29 03:22 | LOMAO, EA & IP, seeds 42–44 | `analysis/paper1_figures/build_all_figures.py`, `f2_manifest.md`, `f3_manifest.md`; `analysis/model_diagnostics/_regen_v1_results.md`, `_regen_v1_r3_results.md`, `_wdmpnn_original_results.md` (right-hand/regen_v1 side), `_pilot_verification.md` | f2/f3 wDMPNN statistics, architecture-spread and ordering numbers; regen_v1 comparison metrics; `_wdmpnn_original_results.md` paired-difference table (regen_v1 minus original) |
| `predictions/regen_v1/ea_ip_lomo_b_clustered/` | 54 | 2026-07-28 05:43 — 2026-07-29 08:08 | monomer_b_heldout_clustered, EA & IP, seeds 42–44 | `analysis/paper1_figures/build_all_figures.py`, `f6_manifest.md`; `analysis/model_diagnostics/_regen_v1_r3_results.md` | f6 paired wDMPNN vs octamer numbers; R3 regen_v1 metrics |
| `predictions/wDMPNN_Gen/` | 2 | 2026-07-07 11:40 — 13:22 | pair & group, EA | none found in `analysis/`, `handoffs/`, or `planning/` | no recorded claim |
| `predictions/wDMPNN_Pilot_Lambda/ea_ip_group/` | 4 | 2026-07-14 16:02 — 2026-07-15 02:29 | group, EA, lambda ablations | none found | no recorded claim (exploratory lambda sweep) |
| `predictions/_checkpoint_smoke_wdmpnn_*` | 3 | 2026-07-27 20:57 — 21:31 | smoke/checkpoint | none found | no recorded claim |

**Total pre-fix wDMPNN prediction files: 224** out of 359 wDMPNN `.npz` files.

## Post-fix wDMPNN prediction files (mtime ≥ 2026-07-30 23:04:41 +1000)

| parent directory | files | mtime range (AEST) | consumers | recorded claims at risk |
|---|---|---|---|---|
| `predictions/wdmpnn_original/ea_ip_lomo/` | 54 | 2026-07-31 03:01 — 2026-08-04 00:26 | LOMAO, EA & IP, seeds 42–44, `__orig` suffix | `analysis/model_diagnostics/_wdmpnn_original_results.md` (left-hand / original-paper side) | wDMPNN "original-paper" per-fold metrics and the paired comparison against regen_v1 |
| `predictions/wdmpnn_original/ea_ip_lomo_b_clustered/` | 54 | 2026-08-04 00:07 — 2026-08-12 22:49 | monomer_b_heldout_clustered, EA & IP, seeds 42–44 | same | same for the B-clustered split |
| `predictions/m1/ea_ip_lomo/` | 27 | 2026-08-13 12:16 — 13:12 | LOMAO, EA & IP | none found | no recorded claim |

**Total post-fix wDMPNN prediction files: 135**.

## Key recorded claims that depend on these files

- `analysis/gate/GATE_RESULT_v3.md` reports `wdmpnn` metrics in the LOMAO, group, and pair-disjoint tables; these come from the pre-fix `ea_ip_*` directories. Its G1–G4 summary verdicts include the `wdmpnn` row.
- `analysis/paper1_figures/f2_manifest.md`, `f3_manifest.md`, `f6_manifest.md` and `build_all_figures.py` consume `predictions/regen_v1/` (pre-fix) for the wDMPNN vs HPG-octamer comparisons. The figures and captions quote wDMPNN group-mean error, architecture spread, collapse rate, and ordering.
- `analysis/model_diagnostics/_wdmpnn_original_results.md` is a protocol-parity test: left-hand side is `wdmpnn_original/` (post-fix), right-hand side is `regen_v1/` (pre-fix). Its per-fold metrics and paired-difference table depend on both sides.
- `analysis/verification/abstract_numbers_report.md` does **not** contain any wDMPNN-dependent numerical claims.
- `handoffs/MODELS_AND_COMPUTE_brief_2026-07-30.md` and `handoffs/HANDOFF_2026-08-05.md` discuss wDMPNN architecture, but those are source-code descriptions, not numbers derived from the prediction files.

## Caveat

Pre-fix / post-fix here is defined only by file mtime against the vectorisation commit. The exact code state used to train each checkpoint is not recorded in the `.npz`; mtime gives an upper bound on the commit that could have produced the file. This is sufficient for blast-radius scoping, not for asserting the code version inside each checkpoint.
