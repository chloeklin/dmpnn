# Windsurf task — rebuild F2 as a paired two-model panel

**Scope: `build_f2` in `analysis/paper1_figures/build_all_figures.py` only.** Do not modify
F1, F3, F4, F5 or F6, and do not modify any file outside `analysis/paper1_figures/`.

---

## 1. Why this is being changed

F2 currently illustrates "two metrics, opposite conclusions" using a single group drawn from
`hpg_hier_octamer` alone. That is the wrong vehicle for the claim. The octamer is the model that
is *good* at architecture ordering — on EA fold 0 it ranks all three architecture pairs correctly
in 72.0% of groups, with a mean pairwise-concordance of 0.814 against a chance baseline of 0.500.
Selecting its single most discordant group therefore illustrates a counterexample to that model's
own behaviour, not the phenomenon the paper is arguing for.

The claim the figure must support is: **two models produce near-identical chemistry placement on
the same group, and disagree completely on architecture ordering.** That requires both models in
the figure.

---

## 2. What to build

A single figure, one selected group, **four series**: true values, octamer predictions, wDMPNN
predictions — shown in the existing two-panel layout (left = raw values with group means; right =
deviations from group mean).

Keep the existing visual conventions: colour by `poly_type`, circles for true, triangles for
predicted, dashed group-mean lines, `figstyle` palette. Distinguish the two models by marker or
line style, not by colour — colour is already carrying architecture.

---

## 3. Group selection — deterministic, no discretion

Candidate pool, computed identically for both models:

- Split `monomer_heldout` (R1), all 9 folds, both targets. Build the figure for **EA**; compute
  the statistics for both targets.
- Restrict to `fracA == 0.5` groups where all 3 `poly_type` values are present in the test fold.
  This gives **exactly 682 groups per fold** — assert this.
- Per group: `gm_err = |mean(y_true) − mean(y_pred)|`, and `ordering` = mean pairwise
  concordance over the 3 architecture pairs, using the existing tie handling (0.5 on predicted
  ties, skip pairs where `y_true` values are equal).

Eligibility for selection:

```
abs(gm_err_octamer - gm_err_wdmpnn) <= 0.01      # chemistry placement tied
and ordering_octamer == 1.0                       # octamer ranks all 3 pairs correctly
and ordering_wdmpnn  == 0.0                       # wDMPNN reverses all 3 pairs
```

Among eligible group-folds, select the one with the **largest true architecture spread**
(`max(y_true) − min(y_true)` within the group), so the disagreement is visible at plotting scale.
Report its rank and the pool size. Do not introduce any further filter.

---

## 4. Numbers to reproduce as assertions

These were computed independently. If your run disagrees with any of them, **stop and report the
discrepancy — do not adjust the criterion to make them match.**

Eligible-pool sizes (tied placement, opposite ordering), per fold 0–8:

| target | per-fold counts | total |
|---|---|---|
| EA | `[0, 10, 12, 2, 6, 12, 14, 1, 4]` | **61** of 6138 (1.0%) |
| IP | `[0, 19, 5, 1, 1, 0, 25, 2, 1]` | **54** of 6138 (0.9%) |

Single-model marginals on **fold 0**, for the manifest:

| target · model | ordering==0.0 | mean ordering | gm_err decile thr (eV) | joint w/ decile |
|---|---|---|---|---|
| EA · octamer | 69 (10.1%) | 0.814 | 0.01806 | 1 |
| EA · wdmpnn | 43 (6.3%) | 0.858 | 0.09580 | 7 |
| IP · octamer | 27 (4.0%) | 0.886 | 0.04317 | 6 |
| IP · wdmpnn | 42 (6.2%) | 0.822 | 0.10909 | 10 |

Median ordering-failure rate across all 9 folds: octamer EA 6.5%, IP 6.6%; wDMPNN EA 8.8%,
IP 8.1%. **EA fold 4 is an outlier** — octamer 35.6% failure / mean ordering 0.560, wDMPNN 39.7%
/ 0.524, i.e. both models at roughly chance. Record this in the manifest; do not exclude the fold.

---

## 5. Manifest requirements

Write `f2_manifest.md` containing:

1. Prediction file paths actually loaded, with seeds.
2. The selection criterion verbatim, and the words **"selected example"**.
3. Pool size and the selected group's rank within it.
4. The full marginals table from §4 — **all three of** ordering-failure rate, decile count, and
   the conjunction, plus the expected-under-independence value for the conjunction.
5. An explicit line: *the conjunction alone must not be quoted as the failure rate, because the
   decile condition selects 10% of groups by construction.*
6. Per-fold eligible-pool counts for both targets.
7. A `## Missing files` section — `None` if complete, otherwise list them.
8. Cell count.

---

## 6. Protocol constraints — these have caused errors before

- **Average the three seeds (42/43/44) at the prediction level, then compute the metric.** Never
  compute per-seed metrics and average those.
- Use `y_pred` (best checkpoint), not `y_pred_final`.
- Do not rebuild any null predictor. F2 does not use one.
- If any `.npz` is missing, write the manifest with the missing files listed and skip the render —
  do not silently substitute a different fold, seed, or model.
- Leave the §3 marginals code added on 10 August in place; extend it, don't replace it.

---

## 7. Deliverables

1. Modified `build_f2` in `build_all_figures.py`.
2. Re-rendered `f2_worked_example.png` **and** `f2_worked_example.csv` (the CSV must now carry
   both models' predictions, with a `model` column — do not leave it as NaN).
3. Rewritten `f2_manifest.md`.
4. A short report stating: the selected group's SMILES pair and fold, its spread rank and pool
   size, and confirmation that each assertion in §4 passed.

Confirm `python -m py_compile` passes and list every file you actually wrote.
