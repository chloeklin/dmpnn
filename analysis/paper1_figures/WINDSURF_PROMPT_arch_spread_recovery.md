# Windsurf task — architecture-spread recovery metric, and F2 caption fix

Three deliverables, in this order. **Task 1 must not change any existing number anywhere in the
repository** — that constraint is the whole risk in this task, see §1.4.

---

## 1. New metric — `arch_spread_recovery`

### 1.1 Why

`delta_r2` measures whether a model recovers the *ordering* of architectures within a chemistry
group. Nothing currently measures whether it recovers the *magnitude*. A model that predicts one
value for all three architectures can still score a non-trivial `delta_r2` from the sign of its
residual noise, and `ordering` cannot distinguish that from real signal.

Note `compression_ratio` (`evaluation/metrics.py:88`) is **not** this quantity — it is computed on
group *means*, i.e. the chemistry axis. Do not modify or reuse it.

### 1.2 Definition

Operate on the same matched set `delta_r2` uses: groups keyed
`smiles_A||smiles_B||fracA` retaining only groups with ≥ 2 distinct `poly_type`.

Per group:

```
true_spread = max(y_true) - min(y_true)
pred_spread = max(y_pred) - min(y_pred)
ratio       = pred_spread / true_spread      # NaN if true_spread < 1e-9
```

**The ratio is two-sided.** 1.0 is perfect; 0.0 is total collapse; values above 1 are
over-prediction of architecture spread and are also wrong. Do not summarise it with a plain mean
or median as though larger were better — that error is the reason this task exists.

Report these keys:

| key | definition |
|---|---|
| `arch_spread_recovery_median` | median `ratio` — descriptive only, **not** a quality score |
| `arch_spread_recovery_logerr` | median `abs(log2(ratio))` over groups with `ratio > 0`; **this is the quality score, lower is better** |
| `arch_spread_collapsed_frac` | fraction of groups with `ratio < 0.25` |
| `arch_spread_within_2x_frac` | fraction of groups with `0.5 <= ratio <= 2.0` |
| `arch_spread_n_groups` | number of groups contributing |

### 1.3 Stratification — do not pool

The benchmark is an exact factorial: `block` and `random` at fracA 0.25/0.50/0.75, `alternating`
at 0.50 only. So groups at fracA 0.5 have **3** architectures and groups at 0.25/0.75 have **2**.
A range over 3 points and a range over 2 points are not the same quantity.

Compute and report the five keys **separately** for the 3-architecture and 2-architecture strata,
suffixed `_arch3` and `_arch2`. **Never pool them into a single figure.** This mirrors the
existing rule that S and D fold groups on the B split are never pooled.

### 1.4 Backward-compatibility — the hard constraint

Adding keys to the dict returned by `compute_copolymer_metrics` must not alter a single existing
reported number. Six analyzers consume it, each with its own hardcoded `METRICS` tuple:

`scripts/python/analyze_regen_v1.py`, `analyze_regen_v1_r3.py`, `analyze_octamer_posemb.py`,
`analyze_wdmpnn_original.py`, `analysis/model_diagnostics/_check_octamer_posemb_pilot.py`,
`analysis/paper1_figures/build_all_figures.py`.

Requirements:

- **Add keys only.** Do not rename, reorder, remove or recompute any existing key.
- **Do not add the new keys to any existing `METRICS` tuple or list.** Leave all six untouched.
- Verify no consumer iterates the returned dict wholesale (e.g. `for k, v in metrics.items()`) in
  a way that would widen a committed CSV. If one does, report it and stop rather than changing it.
- **Proof of no-op:** regenerate the output of `scripts/python/analyze_regen_v1.py` before and
  after your change and diff them. The diff must be empty. Paste the diff command and its result
  in your report. If it is not empty, stop and report.

### 1.5 Tests

Add unit tests covering: a perfectly-recovered group (ratio 1.0), a fully-collapsed group
(pred_spread 0 → ratio 0.0, counted in `collapsed_frac`, excluded from `logerr`), an
over-predicting group (ratio 2.0 → `abs(log2)` = 1.0), and a degenerate group with
`true_spread == 0` (NaN, excluded from all counts).

---

## 2. Numbers to reproduce as assertions

Computed independently on `predictions/regen_v1/ea_ip_lomo`, three seeds averaged at the
prediction level, fracA=0.5 3-architecture groups only, split `monomer_heldout`.
**If your run disagrees, stop and report — do not adjust the definition to match.**

Per-fold **median ratio**, folds 0–8:

| target · model | medians |
|---|---|
| EA · octamer | 0.649, 0.747, 0.929, 0.689, 0.811, 0.833, 1.559, 0.713, 0.768 |
| EA · wdmpnn | 0.775, 0.247, 0.378, 0.526, 0.909, 0.582, 0.610, 1.390, 0.616 |
| IP · octamer | 0.570, 0.947, 0.927, 0.904, 1.162, 0.738, 0.967, 1.118, 1.034 |
| IP · wdmpnn | 0.467, 1.484, 1.074, 1.219, 0.968, 0.265, 0.350, 0.229, 0.551 |

Collapse rate (`ratio < 0.25`) spot checks: wDMPNN EA fold 1 = **50.0%**, IP fold 5 = **44.3%**,
IP fold 7 = **52.3%**. Octamer stays within **6.3–18.5%** on EA and **6.6–16.0%** on IP across all
nine folds.

The F2 selected group (fold 2, EA, `OB(O)c1cc(F)c(B(O)O)cc1F` / `Brc1cc(Br)c2cc[nH]c2c1`):

| | spread (eV) | ratio |
|---|---|---|
| true | 0.33250 | — |
| octamer | 0.32904 | **0.9896** |
| wDMPNN | 0.02400 | **0.0722** |

---

## 3. F2 figure and manifest

Do **not** change the selection criterion or the selected group. F2 has been verified and the
group is correct. Only the reporting changes.

1. Add `arch_spread_recovery` for both models to the right-hand panel annotation, beside the
   existing `ordering` values: `octamer 0.99 / wDMPNN 0.07`.
2. Add a short manifest section stating, in words: *wDMPNN does not rank the three architectures
   in the wrong order so much as predict nearly the same value for all three — it recovers 7% of
   the true architecture range on this group, so its ordering is the sign of residual noise.*
3. Add the per-fold median-ratio and collapse-rate tables from §2 to the manifest, with the
   two-sided caveat from §1.2 stated explicitly.
4. Add `arch_spread_recovery` as a column in `f2_worked_example.csv`, per model.

**Cosmetic:** both annotations currently collide with the axes — the left panel renders as
"m err:" with the leading `g` clipped, and the right panel's text overlaps the `alternating`
marker. Move both inside the axes with padding, or place them below the panel titles.

---

## 4. Protocol constraints

- Three seeds (42/43/44) averaged at the **prediction** level, then metric computed once. Never
  average per-seed metrics.
- `y_pred` (best checkpoint), never `y_pred_final`.
- If any `.npz` is missing, list it and stop — do not substitute.
- Do not modify F1, F3, F4, F5, F6, or any file under `analysis/model_diagnostics/`.

---

## 5. Report back

State: the empty diff from §1.4 with the exact command used; each §2 assertion and whether it
passed; the test results from §1.5; and every file you actually wrote. Confirm `py_compile`
passes on each modified Python file.
