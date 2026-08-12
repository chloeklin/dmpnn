# Windsurf task — per-fold line plots for the factor 2 and factor 5 ablations

Two figures, one per ablation. Purpose: show at a glance that ablated and baseline track each
other across folds on **every** metric, rather than asking the reader to trust a summary
statistic.

**Scope:** a new script under `analysis/paper1_figures/`. Do not modify `build_all_figures.py`,
`evaluation/metrics.py`, or any dated document.

---

## 1. What to plot

For each ablation, one figure with **five panels — one per metric**, in this order:

`overall_r2`, `rmse`, `mae`, `group_mean_r2`, `delta_r2`

In every panel:

- **x axis** = fold index
- **y axis** = the metric value (absolute, not the difference)
- **two lines**: baseline and ablated, one marker per fold, clearly distinguished in the legend
- separate panels or separate figures for EA and IP — **never pooled**

So each figure is a 5 × 2 grid (5 metrics × 2 targets), or two 1 × 5 rows. Pick whichever reads
better and say which you chose.

**Plot absolute values, not differences.** The difference plots already exist in the tables; the
point of these figures is to let the reader see that the two curves lie on top of each other,
and to see the fold-to-fold shape of the metric itself.

---

## 2. The two arms, and an asymmetry to respect

| | Factor 5 (K=1 vs K=16) | Factor 2 (position embeddings off vs on) |
|---|---|---|
| Split | `monomer_b_heldout_clustered` (**B split**) | `monomer_heldout` (**A split**) |
| Folds | 0–8 | 0–8 |
| Ablated predictions | `predictions/octamer_k1/ea_ip_lomo_b_clustered/…__k1.npz` | `predictions/octamer_posemb/ea_ip_lomo/…__noposemb.npz` |
| Baseline predictions | `predictions/regen_v1/ea_ip_lomo_b_clustered/` | `predictions/regen_v1/ea_ip_lomo/` |

**These two arms are on different splits. Do not put them on the same axes and do not imply
they are comparable.** Two separate figures, each labelled with its split in the title.

**On the B split, mark the S/D boundary.** Folds 0–3 are same-scaffold (interpolation), folds
4–8 are cross-scaffold (extrapolation). Add a vertical divider between fold 3 and fold 4 and
label the two regions. They are never pooled, and the figure should make that visible.

### 2.1 Factor 5 must be plotted by row subset — this is essential

**The K=1 ablation does not touch every polymer.** `_build_octamer_sequences` behaves
differently depending on the transition matrix:

- **`random` polymers** — uniform transition matrix → the featurizer *samples*
  `n_random_samples` sequences. Setting it to 1 changes these directly. **This is where the
  ablation acts.**
- **`block` and `alternating`** — non-uniform transition matrix → the featurizer takes an
  argmax and produces **one** sequence regardless of `n_random_samples`. Their featurisation is
  identical under K=1 and K=16.

So plotting factor 5 on all rows pooled **dilutes the effect being tested** with rows the
ablation cannot act on directly. The pre-registration
(`PREREG_octamer_k1_2026-07-30.md` §5) names **random rows, D folds** as the primary quantity
for exactly this reason.

Produce the factor-5 figure with **three row subsets**, clearly labelled:

| Subset | Role |
|---|---|
| `random` only | **where the ablation acts** — the primary series |
| `block` + `alternating` | **negative control** |
| all rows | context, reported but not the headline |

**The control is not a no-op, and do not describe it as one.** Changing the featurisation of
random rows changes the training data, which changes the learned weights, which changes
predictions on *every* row. So block+alternating measures **spillover**: it should stay flat,
and if it moves as much as the random rows do, the arm is not isolating what it claims to.

**One caveat on ΔR² for the random-only subset:** ΔR² needs groups containing ≥ 2 distinct
`poly_type` values. Restricted to random rows alone every group has exactly one, so **ΔR² is
undefined there** — this is the metric substitution disclosed in the K=1 pre-registration
addendum. Plot ΔR² for the all-rows and block+alternating subsets only, and put an explicit
"not computable on this subset" note in the random-only ΔR² panel rather than leaving it blank
or silently dropping it.

Factor 2 (position embeddings) affects every row, so it does **not** need the row-subset
breakdown. All rows is correct there.

---

## 3. Known data gaps — handle explicitly, do not silently paper over

- **Factor 5, IP fold 7 currently has only 2 seeds** (seed 44 missing; a job to complete it may
  already be running). If only two seeds are present, plot the point and **mark it distinctly**
  — hollow marker, or an annotation — and state it in the manifest. Do not drop it and do not
  present it as a 3-seed cell.
- If any other cell is incomplete, list it in the manifest and mark it the same way.

---

## 4. Protocol

- Three seeds (42/43/44) averaged **at the prediction level**, then each metric computed
  **once** on the averaged predictions. Never average per-seed metric values.
- `y_pred` (best checkpoint), never `y_pred_final`.
- Use the existing metric functions in `evaluation/metrics.py` — do not reimplement them.

---

## 5. Numbers to reproduce as assertions

These were computed independently. **If your run disagrees, stop and report — do not adjust
anything to make them match.**

**Factor 2 (A split, 9 folds) — ablated minus baseline, full range across folds:**

| Metric | EA range | IP range |
|---|---|---|
| Overall R² | −0.008, +0.035 | −0.032, +0.088 |
| RMSE | −0.060, +0.012 | −0.032, +0.013 |
| MAE | −0.051, +0.011 | −0.027, +0.013 |
| Group-mean R² | −0.008, +0.037 | −0.032, +0.092 |
| ΔR² | −0.053, +0.105 | −0.108, +0.051 |

**Factor 5 (B split) — ablated minus baseline, full range across folds:**

| Metric | EA S folds | EA D folds | IP S folds | IP D folds |
|---|---|---|---|---|
| Overall R² | −0.010, +0.001 | −0.006, +0.018 | −0.003, +0.003 | −0.006, +0.010 |
| RMSE | −0.002, +0.011 | −0.015, +0.007 | −0.004, +0.003 | −0.006, +0.008 |
| MAE | −0.001, +0.008 | −0.010, +0.002 | −0.000, +0.002 | −0.004, +0.005 |
| Group-mean R² | −0.010, +0.001 | −0.006, +0.019 | −0.004, +0.003 | −0.006, +0.010 |
| ΔR² | −0.002, +0.029 | −0.024, +0.032 | −0.019, +0.005 | −0.062, +0.025 |

Assert the min and max of `ablated − baseline` per metric per target (and per fold group for
factor 5) against these, to 3 decimal places.

---

## 6. Plotting requirements — learned from earlier figures in this repo

- **Any text annotation or legend must sit above the data**: `zorder=20` plus
  `bbox=dict(facecolor='white', alpha=0.85, edgecolor='none')`. Markers are drawn at `zorder=5`
  and will paint over text otherwise.
- **Do not use a `twinx` axis.** A twin axes is drawn entirely above its parent, so artist
  zorder cannot lift a label above a twin's line. Give each metric its own panel instead.
- **Watch the y-limits.** Some folds have large negative R² values. If a point falls outside the
  chosen limits, add an off-scale marker at the axis edge **labelled with the actual value** —
  do not let a line silently vanish and reappear.
- Place annotations in genuinely empty regions; check the rendered PNG, do not assume.

---

## 7. Deliverables

1. The script, under `analysis/paper1_figures/`.
2. `factor2_posemb_per_fold.png` and `factor5_k1_per_fold.png` (plus PDF).
3. A CSV per figure with every plotted value: fold, target, metric, setting, value, n_seeds.
4. A manifest per figure recording prediction paths, seed handling, cell counts, any incomplete
   cells, and confirmation of each §5 assertion.
5. `py_compile` clean; every file written, listed.

**Report the rendered figures back for review before they go in a slide.**
