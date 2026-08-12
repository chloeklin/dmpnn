# Deck plan v2 — responses to your three points

Written 11 August 2026. Supersedes `DECK_PLAN_eval_framework.md` on components 2, 5 and 6.
Components 1, 3 and 4 are unchanged from v1.

Summary of what changed:

| Your point | Outcome |
|---|---|
| 1. Component 2 should cover all folds and both properties | **Done.** New table below. It is a better slide and it changes the claim slightly. |
| 2. Drop component 5 until we know why it is unstable | **Agreed — and there is a stronger reason than the one you gave.** Those runs are off-protocol. Replacement proposed. |
| 3. Component 6 should show both splits, and means as well as medians | **Done.** This turned out to be the most valuable of the three. It reveals something the D-folds-only version was hiding. |

---

## Point 1 — Component 2 across all folds and both properties

You were right that one group from one fold does not cover the story. Here is the whole thing.

### The table

Every fold, both properties. Each row summarises **682 groups** — every set of polymers
sharing monomer A, monomer B and 50/50 composition, differing only in arrangement.

**Placement error** — median distance between the predicted group average and the true group
average, in eV. Lower is better.

**Spread ratio** — median of (predicted spread ÷ true spread) within a group.
**1.00 is perfect. Below 1 means the model flattens the differences between arrangements;
above 1 means it exaggerates them.** Both directions are wrong.

#### EA

| Fold | Placement error, octamer | Placement error, wDMPNN | Spread ratio, octamer | Spread ratio, wDMPNN |
|---|---|---|---|---|
| 0 | 0.082 | 0.160 | 0.65 | 0.78 |
| 1 | 0.027 | 0.026 | 0.75 | 0.25 |
| 2 | 0.047 | 0.041 | 0.93 | 0.38 |
| 3 | 0.020 | 0.048 | 0.69 | 0.53 |
| 4 | 0.054 | 0.071 | 0.81 | 0.91 |
| 5 | 0.018 | 0.039 | 0.83 | 0.58 |
| 6 | 0.036 | 0.052 | 1.56 | 0.61 |
| 7 | 0.063 | 0.202 | 0.71 | 1.39 |
| 8 | 0.051 | 0.025 | 0.77 | 0.62 |
| **Median** | **0.047** | **0.048** | **0.77** | **0.61** |
| **Mean** | **0.044** | **0.074** | **0.86** | **0.67** |

#### IP

| Fold | Placement error, octamer | Placement error, wDMPNN | Spread ratio, octamer | Spread ratio, wDMPNN |
|---|---|---|---|---|
| 0 | 0.129 | 0.192 | 0.57 | 0.47 |
| 1 | 0.041 | 0.044 | 0.95 | 1.48 |
| 2 | 0.022 | 0.051 | 0.93 | 1.07 |
| 3 | 0.029 | 0.022 | 0.90 | 1.22 |
| 4 | 0.037 | 0.045 | 1.16 | 0.97 |
| 5 | 0.024 | 0.099 | 0.74 | 0.27 |
| 6 | 0.017 | 0.029 | 0.97 | 0.35 |
| 7 | 0.036 | 0.265 | 1.12 | 0.23 |
| 8 | 0.012 | 0.017 | 1.03 | 0.55 |
| **Median** | **0.029** | **0.045** | **0.95** | **0.55** |
| **Mean** | **0.038** | **0.085** | **0.93** | **0.73** |

### One summary number, because the ratio is two-sided

Because a ratio of 1.4 is as wrong as 0.7, you cannot just say "higher is better". The clean
summary is **how far the ratio sits from 1, in either direction** (technically the median
absolute log-ratio; 0 means perfect):

| | Octamer | wDMPNN |
|---|---|---|
| EA | **0.42** | 0.71 |
| IP | **0.15** | 0.86 |

Lower is better. The octamer is closer to correct on both properties, and much closer on IP.

### How the numbers are computed

1. Take one fold. Average the three seeds' **predictions** first, then work with one
   prediction per polymer.
2. Find every group of polymers sharing monomer A, monomer B and 50/50 composition, where
   all three arrangements are present in the test set. There are 682 such groups per fold.
3. For each group: **placement error** = |average of predictions − average of true values|.
   **Spread ratio** = (highest prediction − lowest) ÷ (highest true value − lowest true).
4. Take the median across the 682 groups. That is one row.

### How to read it — and how the claim changes

The single-group example suggested "the models tie on placement and differ enormously on
spread". Across all folds that is **half right, and the honest version is better**:

**On placement they are close.** On EA the medians are 0.047 against 0.048 — a genuine tie.
On IP the octamer is somewhat better (0.029 against 0.045).

**On spread the octamer is consistently closer to correct**, on both properties, by the
two-sided measure. But it is not the dramatic 0.99-versus-0.07 of the selected example —
that group was extreme by design.

**And wDMPNN errs in both directions.** It flattens badly on some folds (EA fold 1: 0.25;
IP fold 7: 0.23) and exaggerates on others (EA fold 7: 1.39; IP fold 1: 1.48). The octamer's
worst case is EA fold 6 at 1.56. So neither model is uniformly well behaved — this is a
difference of degree, and the table shows that honestly.

Note also the means differ from the medians for wDMPNN's placement error (0.048 median but
0.074 mean on EA) — a couple of bad folds pull it up. The octamer's are close together,
which is itself a stability point.

### Recommendation

**Lead the slide with this table.** Keep the single-group worked example as an optional
second slide or a backup, clearly labelled as an illustration of what the numbers mean
rather than as evidence. That way nobody can accuse you of choosing a favourable group,
because the favourable group is no longer carrying the argument.

---

## Point 2 — Drop the noise floor. Agreed, but for a stronger reason

You said to exclude it until we know why it is unstable. I looked into the cause, and found
two things that matter more than the instability itself.

### Finding A — those runs are off-protocol and should not be shown at all

The repeat-study runs are dated **27 July**. The checkpoint fix landed **29 July**.

Their prediction files contain no `y_pred_final` field and no `split_indices_sha256` — they
predate the dual-checkpoint runner entirely. That means their `y_pred` is the **final,
patience-expired model, not the best checkpoint**. That is precisely the model-selection bug
that voided the February–May results.

So the 0.268 figure is not a clean measurement of run-to-run noise. It very likely
**overstates** it, because it mixes genuine variation with the checkpoint bug.

### Finding B — the figure is labelled with the wrong model

The figure title says "HPG-octamer". The underlying runs are `hpg_hier` — the configuration
records `stage2_mode: transition_graph` and `stage2_readout: stoich_weighted`, whereas the
octamer uses `octamer_sequence` and `attention`. **The noise floor was measured on a
different model from the one the deck presents as the lead.** This would have been an
awkward question to get in a viva.

### What is actually causing the instability — we now largely know

Two mechanisms, and neither is mysterious:

1. **Full determinism was requested but not enabled.** The run environment records
   `deterministic_kernels_requested: true` and `cudnn_deterministic: true`, but
   `deterministic_algorithms_enabled: false`. So some GPU operations still return slightly
   different results run to run. Identical seeds therefore drift apart.

2. **Early stopping then amplifies that drift.** The six repeats stopped after 38, 28, 43,
   47, 27 and 40 epochs. On the unstable fold, the run that stopped earliest (27 epochs)
   also had the worst validation loss (0.0106) and scored worst on test (0.450). The runs
   are not the same model measured three times — they are three different models.

So the honest description is not "unexplained noise" but "**early stopping lands on
materially different models when the training trajectory is not reproducible**". That is a
real finding, and a fixable one. But it is not the finding the current slide claims.

### What to do instead

**Do not just delete the component — replace its evidence.** The question it answers ("how
big does a difference have to be before it means anything?") is essential, because the
materiality thresholds used everywhere else in this work depend on the answer.

There is already a clean, on-protocol replacement: the **across-seed spread of ΔR² from the
frozen-protocol runs**. Those cover all 18 cells, use the correct models, use best-checkpoint
predictions, and are the source of the pre-registered threshold. Median across-seed SD for
the octamer is 0.045; the pre-registration froze the threshold at 0.051.

That is a defensible noise floor computed the right way. The right panel of the current
figure already shows it — it is only the left panel that is contaminated.

**Recommended action:** keep component 5, drop the repeat study, present the across-seed
spread instead. If you would rather cut the component entirely, that also works, but then
the 0.051 threshold appears in the deck with no visible justification.

**Separately, and regardless of the deck:** the determinism gap is worth fixing in the
training script. Enabling deterministic algorithms would make repeat runs reproducible and
turn this from a caveat into a solved problem.

---

## Point 3 — Both splits, means as well as medians

This was the most valuable of your three points. Adding the A split changes what the
component means.

### The table

Octamer minus wDMPNN, per fold, three seeds averaged at the prediction level. "Wins" counts
folds where the octamer is better. p is a two-sided paired sign test.

#### A split — held-out monomer A, 9 folds

| Metric | EA median | EA mean | EA wins | p | IP median | IP mean | IP wins | p |
|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.024 | +0.041 | 9/9 | **0.004** | +0.030 | +0.130 | 8/9 | 0.039 |
| MAE | −0.019 | −0.037 | 8/9 | 0.039 | −0.029 | −0.051 | 8/9 | 0.039 |
| RMSE | −0.028 | −0.042 | 9/9 | **0.004** | −0.041 | −0.060 | 8/9 | 0.039 |
| Group-mean R² | +0.016 | +0.038 | 8/9 | 0.039 | +0.021 | +0.118 | 8/9 | 0.039 |
| **ΔR² (architecture)** | **+0.257** | **+0.263** | 8/9 | 0.039 | **+0.246** | **+0.400** | 9/9 | **0.004** |

#### B split, S folds — held-out monomer has close relatives in training, 4 folds

| Metric | EA median | EA mean | EA wins | p | IP median | IP mean | IP wins | p |
|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.014 | +0.012 | 4/4 | 0.125 | +0.005 | +0.004 | 3/4 | 0.625 |
| MAE | −0.022 | −0.019 | 4/4 | 0.125 | −0.013 | −0.013 | 4/4 | 0.125 |
| RMSE | −0.023 | −0.018 | 4/4 | 0.125 | −0.010 | −0.010 | 3/4 | 0.625 |
| Group-mean R² | +0.012 | +0.010 | 3/4 | 0.625 | +0.002 | +0.002 | 3/4 | 0.625 |
| **ΔR² (architecture)** | **+0.186** | **+0.167** | 4/4 | 0.125 | **+0.122** | **+0.137** | 4/4 | 0.125 |

#### B split, D folds — held-out monomer has no close relative, 5 folds

| Metric | EA median | EA mean | EA wins | p | IP median | IP mean | IP wins | p |
|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.003 | **−0.006** | 3/5 | 1.000 | +0.010 | **−0.014** | 3/5 | 1.000 |
| MAE | −0.000 | **+0.001** | 3/5 | 1.000 | −0.010 | −0.003 | 3/5 | 1.000 |
| RMSE | −0.004 | **+0.005** | 3/5 | 1.000 | −0.012 | **+0.005** | 3/5 | 1.000 |
| Group-mean R² | +0.002 | **−0.008** | 3/5 | 1.000 | +0.007 | **−0.016** | 3/5 | 1.000 |
| **ΔR² (architecture)** | **+0.260** | **+0.225** | 5/5 | 0.063 | **+0.152** | **+0.198** | 5/5 | 0.063 |

Bold in the D-fold table marks where the mean and the median **disagree in sign**.

### How the numbers are computed

1. For each fold and each model, average the three seeds' predictions, then compute each
   metric once on those averaged predictions.
2. Subtract wDMPNN's value from the octamer's, fold by fold. That gives one difference per
   fold per metric per target.
3. Report the median and the mean of those differences, and count how many folds favour the
   octamer. The p-value is a sign test on that count.
4. For MAE and RMSE lower is better, so a negative difference is a win. For the three R²
   measures higher is better.

### How to read it — this is the part that changed

**On the A split, all five metrics agree.** The octamer wins on everything, 8 or 9 folds out
of 9, with p between 0.004 and 0.039. There is no disagreement between metrics here at all.

**On the B split's easy folds, everything points the same way too**, just without enough
folds to reach significance — with 4 folds the best attainable p is 0.125, so nothing here
can be significant regardless of how large the effect is.

**Only on the hard cross-scaffold folds do the metrics come apart.** Four accuracy measures
split 3 folds to 2 — a coin toss. ΔR² goes 5 out of 5 on both properties, with a gap roughly
fifty times larger.

This is a better argument than the one the D-folds-only slide was making. The claim becomes:

> *The choice of metric does not matter while the task is easy. It matters exactly when you
> start extrapolating to unfamiliar chemistry — which is the case everyone actually cares
> about.*

That is more useful than "these two metrics disagree", and it pre-empts the obvious
objection that you picked the folds where they disagree.

### Why including the mean was worth doing

On the cross-scaffold folds, **the mean and the median disagree in sign for four of the four
accuracy metrics on EA, and three of four on IP.** The median says the octamer is very
slightly ahead; the mean says it is very slightly behind. Neither is wrong — they disagree
because one bad fold drags the mean around.

ΔR² does not do this. Its mean and median agree on both properties (+0.260 / +0.225 on EA,
+0.152 / +0.198 on IP).

So the accuracy metrics are not merely insensitive here — they are **unstable enough that
the answer depends on which summary statistic you choose.** That is an additional argument
for the two-axis metric that we did not previously have, and it only appears once you print
the mean beside the median.

### One caution to keep on the slide

With 9 folds the smallest achievable p is 0.004; with 5 it is 0.063; with 4 it is 0.125. So
the D-fold and S-fold results **cannot** reach the conventional 0.05 threshold, however large
the effect. Say this rather than letting someone else point it out. The A-split row is where
the significance claims live.

---

## Revised deck outline

| Slide | Content | Format |
|---|---|---|
| 1 | Title | — |
| 2 | The framework at a glance | cards |
| 3 | Architecture is ~1% of the target | table |
| 4 | Two axes: placement and spread, all folds | table |
| 5 | *(optional)* One group, worked through | table |
| 6 | Null floor — what a lookup table scores | table |
| 7 | Splits are not interchangeable | table + bar chart |
| 8 | How big must a difference be? *(revised source)* | table |
| 9 | Metrics agree until you extrapolate | table + bar chart |
| 10 | What the framework establishes | cards |
| 11 | Paper strategy — four papers | cards |
| 12 | Why Paper 1 goes first | cards |
| 13 | Provenance | table |

---

## Please confirm before I build

1. **Component 5** — keep it with the across-seed evidence, or cut it entirely and lose the
   visible justification for the 0.051 threshold?
2. **The worked example (slide 5)** — keep as an optional illustration, or drop it now that
   the all-folds table carries the argument?
3. **The reframed component 6 claim** — "metric choice matters only when extrapolating" —
   is that a claim you are comfortable defending? It is stronger than the old one but it
   commits you to the A-split result showing agreement.
