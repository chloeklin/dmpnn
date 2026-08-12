# Evaluation-framework deck — content plan

Written 11 August 2026, for review **before** the deck is rebuilt.

Purpose of this document: show exactly what will go on each slide, where every number
comes from, and how to say what it means in plain language. Nothing here is new work —
every value already exists in `analysis/paper1_figures/` and has been checked twice.

---

## Decision: what stays a figure, what becomes a table

The current figures were built to be technically correct, not to be read quickly. Three of
the six do not work as pictures, for reasons that are not fixable by adjusting them:

| # | Component | Now | Proposed | Why |
|---|---|---|---|---|
| 1 | Variance decomposition | stacked bar | **table** | The thing we want you to notice is 1% of the bar. It is invisible by construction. |
| 2 | Worked example | paired dot plot | **table** | Three polymers, three numbers each. A table shows it instantly; the plot needs a paragraph of explanation. |
| 3 | Null floor | line plot, 9 folds | **table** | One value is −19.07. Any axis that shows it flattens everything else; any axis that doesn't hides it. |
| 4 | Split design | histogram + bar chart | **table + keep the bar chart** | The histogram is unreadable (long tail). The fold-overlap bar chart is simple and works — keep it. |
| 5 | Noise floor | dual-axis line plot | **table** | Six numbers total. A two-y-axis chart is harder to read than the six numbers. |
| 6 | The demonstration | strip plot | **simple bar chart + small table** | This one genuinely works as a picture. A plain grouped bar chart is clearer still. |

Net result: four tables, two charts, both charts of the most ordinary kind (grouped bars).

---

## 1. Architecture is about 1% of what we are predicting

**The claim.** The number everyone reports is dominated by which monomers were used. The
property we actually study — how the monomers are arranged along the chain — is a tiny
slice. So the standard metric cannot see it.

### The table

| What varies | Share of total variation, EA | Share of total variation, IP |
|---|---|---|
| Which monomer A | 41.8% | 50.0% |
| Which monomer B (given A) | 51.1% | 39.6% |
| How much of each (composition) | 6.1% | 8.9% |
| **How they are arranged (architecture)** | **1.0%** | **1.5%** |
| Unexplained | 0.0% | 0.0% |

Second, smaller table on the same slide:

| | EA | IP |
|---|---|---|
| Architecture as a share of **what is left after the monomers are known** | **13.9%** | **14.1%** |

### How the numbers are computed

1. Take all 42,966 polymers in the benchmark.
2. Group them by monomer A. Replace every polymer's value with its group average. Ask: how
   much of the original variation does that reproduce? Answer: 41.8% for EA. That is monomer
   A's share.
3. Now group by monomer A **and** monomer B and do the same. That reproduces 92.9%. The
   *extra* 51.1% is monomer B's contribution once A is known.
4. Repeat adding composition (92.9% → 99.0%, so composition adds 6.1%), then adding
   architecture (99.0% → 100%, so architecture adds 1.0%).
5. For the second table: after monomers A and B are known, 7.1% of the variation is still
   unexplained. Architecture accounts for 0.98 of those 7.1 points — that is 13.9%.

No model is involved. This is a property of the dataset alone.

### How to read it

Say: *"If I know which two monomers went in, I already know 93% of the answer. Everything
about how they are arranged along the chain is one percent. A score that averages over all
of it cannot tell you whether the model understands arrangement."*

The second table is the counterweight, and it is the more encouraging number: **once you
remove the part that is just monomer identity, architecture is a seventh of what remains.**
The signal is small in absolute terms but not negligible in the space where it lives. That
is the argument for measuring the residual separately, which is component 2.

### What it does not show

It does not show that any model is good or bad. It is a statement about the dataset.

---

## 2. Two axes: did it place the chemistry, and did it recover the arrangement?

**The claim.** One number cannot answer both questions. Split the score into two.

### The table

One group of three polymers — **same monomer A, same monomer B, 50/50 composition** — that
differ only in how the units are arranged. Held-out monomer, fold 2, EA.

| Arrangement | True value (eV) | Octamer predicts | wDMPNN predicts |
|---|---|---|---|
| alternating | −3.381 | −3.346 | −3.174 |
| random | −3.259 | −3.177 | −3.179 |
| block | −3.049 | −3.017 | −3.198 |
| **Average of the three** | **−3.230** | **−3.180** | **−3.184** |
| **Spread (highest − lowest)** | **0.333** | **0.329** | **0.024** |

### How the numbers are computed

1. Pick a group: all polymers sharing monomer A, monomer B and composition, differing only
   in arrangement. Here that is three polymers.
2. For each model, average the predictions from the three random seeds **first**, then read
   off one prediction per polymer. (Averaging predictions, not averaging scores — this
   matters and is the standing protocol.)
3. **Average of the three** = the group's mean. Comparing model mean to true mean gives
   *chemistry placement*: octamer is out by 0.050 eV, wDMPNN by 0.046 eV. Effectively tied.
4. **Spread** = highest minus lowest within the group. True spread is 0.333 eV. The octamer
   reproduces 0.329 of that — 99%. wDMPNN reproduces 0.024 — 7%.

### How to read it

Say: *"Both models put this group of polymers in almost exactly the right place — they are
within 0.05 eV of the true average, and they are equally good at that. But look down the
columns. The true values spread over a third of an electron volt. The octamer's predictions
spread over the same range. wDMPNN gives essentially the same answer three times."*

The key point, and the one to say out loud: **wDMPNN is not ranking the arrangements in the
wrong order — it is barely distinguishing them at all.** Whatever ordering it appears to
produce is the leftover noise. That is a stronger and safer claim than "it gets the order
backwards".

### What it does not show — say this before anyone asks

This is a **selected example**, chosen deliberately to make the contrast visible. Out of
6,138 comparable group-folds on EA, 61 (1.0%) show this pattern this clearly. Across all
groups the typical picture is milder: the octamer recovers a median of about 77% of the
true spread on EA and 95% on IP; wDMPNN about 61% and 55%.

One caution on those medians: the ratio can exceed 1 (a model can *over*-state the spread),
so **closer to 1 is better, not larger**. Do not present 0.77 versus 0.61 as a ranking
without that sentence attached.

---

## 3. The null floor: how well can you score knowing nothing?

**The claim.** Before believing a score, check what a model with no knowledge would get.

### The table

Group-mean R² per fold. "Lookup table" is a predictor that never sees the held-out monomer.

| Fold | EA lookup | EA octamer | EA wDMPNN | | IP lookup | IP octamer | IP wDMPNN |
|---|---|---|---|---|---|---|---|
| 0 | 0.694 | 0.958 | 0.882 | | **0.969** | 0.748 | 0.496 |
| 1 | 0.487 | 0.991 | 0.982 | | 0.509 | 0.970 | 0.982 |
| 2 | **0.961** | 0.986 | 0.981 | | −1.019 | 0.995 | 0.976 |
| 3 | **0.953** | 0.995 | 0.979 | | −3.206 | 0.982 | 0.962 |
| 4 | 0.884 | 0.962 | 0.947 | | −0.251 | 0.960 | 0.918 |
| 5 | 0.676 | 0.994 | 0.970 | | −7.528 | 0.977 | 0.755 |
| 6 | −19.069 | 0.937 | 0.878 | | 0.569 | 0.996 | 0.986 |
| 7 | 0.098 | 0.982 | 0.846 | | 0.410 | 0.988 | 0.482 |
| 8 | 0.428 | 0.989 | 0.992 | | −0.034 | 0.995 | 0.987 |
| **Median** | **0.676** | | | | **−0.034** | | |

Bold marks the cases that carry the argument.

### How the numbers are computed

1. The test set holds out one monomer A entirely. The lookup predictor is not trained. For
   each test polymer it simply reports the average value of training polymers sharing the
   same monomer B and composition. If it has never seen that combination it falls back to
   the overall training average.
2. Score it exactly like a real model, on the same folds, with the same metric.
3. The trained-model columns are the octamer and wDMPNN under the frozen protocol, three
   seeds averaged at the prediction level.

### How to read it

Two things, in this order.

**On EA the lookup table is often excellent.** On folds 2 and 3 it scores 0.961 and 0.953 —
knowing nothing about the held-out monomer. So a paper reporting 0.95 on those folds has
reported a number a lookup table matches. Median across folds is 0.676.

**On IP it is mostly useless** — median −0.034, and several folds far below zero. Same
benchmark, same split design, opposite conclusion. *You cannot know which situation you are
in without computing the floor.*

**And on IP fold 0 the lookup table scores 0.969 and beats both trained models.**

### Note on the extreme values

Negative R² means worse than predicting a constant, and it is unbounded below — that is why
−19.07 and −7.53 appear. These are real, not errors. This is precisely why the table
replaces the line chart: no axis can show −19.07 and the difference between 0.95 and 0.96
at the same time.

---

## 4. The splits are not interchangeable

**The claim.** The benchmark's B-monomer space is dominated by two chemical families, so a
"held-out monomer" test means two completely different things depending on the fold.

### Table 1 — the scaffold families

| | Count | Share of 682 B monomers |
|---|---|---|
| Largest family | 317 | 46.5% |
| Second largest | 109 | 16.0% |
| **Those two together** | **426** | **62.5%** |
| All 110 other families | 256 | 37.5% |
| (of which, families with a single member) | 64 | — |

### Chart — keep this one

A simple bar chart, nine bars, one per fold, height 0 to 1: *what fraction of the held-out
B monomers have a close chemical relative in the training set?* Coloured in two groups.

| Fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Fraction with a relative in training | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 | 0.43 | 0.00 | 0.00 |
| Group | S | S | S | S | D | D | D | D | D |

### How the numbers are computed

1. Reduce each B monomer to its core ring skeleton (a standard, automatic procedure —
   strip the decorations, keep the frame). Monomers sharing a skeleton form a family.
2. Count family sizes across all 682 B monomers. Two families cover 62.5% of them.
3. For each fold, take the held-out B monomers and ask what fraction have at least one
   family member in the training set. Folds 0–3 give 1.00; folds 5, 7 and 8 give 0.00.

### How to read it

Say: *"In folds 0 to 3 every held-out monomer has a close relative in training — the model
is interpolating. In folds 5, 7 and 8 none do — the model is extrapolating to unfamiliar
chemistry. Those are two different experiments. Averaging them into one number reports
something that describes neither."*

Hence: S folds and D folds are reported separately throughout, never pooled. The
demonstration in component 6 uses the D folds only, because that is the honest hard case.

---

## 5. How big does a difference have to be before it is real?

**The claim.** Run the identical experiment three times and you get materially different
scores. Most published improvements are smaller than that.

### The table

Identical configuration, identical seed, three repeats.

| | Repeat 1 | Repeat 2 | Repeat 3 | Spread (SD) |
|---|---|---|---|---|
| **Fold 0** — group-mean R² | 0.962 | 0.982 | 0.986 | 0.013 |
| **Fold 0** — error (MAE, eV) | 0.084 | 0.055 | 0.052 | 0.018 |
| **Fold 1** — group-mean R² | **0.790** | **0.450** | **0.978** | **0.268** |
| **Fold 1** — error (MAE, eV) | 0.146 | 0.226 | 0.045 | 0.091 |

### How the numbers are computed

1. Fix everything — model, data, fold, random seed — and train three separate times.
2. Score each run the same way. SD is the ordinary sample standard deviation of the three.

Remaining run-to-run variation comes from non-deterministic GPU operations, not from any
setting we chose.

### How to read it

Say: *"On fold 1, three identical runs scored 0.79, 0.45 and 0.98. If I had run this once
and reported it, I could have honestly claimed almost any result. The spread is 0.27. Almost
every improvement reported in this literature is smaller than that."*

Two further points worth making:

**Stability depends on the metric.** MAE varies by 0.018 eV on fold 0 and 0.091 on fold 1 —
much better behaved than R². Choosing a metric is also choosing how noisy your answer is.

**Stability depends on the fold.** Fold 0 is well behaved, fold 1 is not. So a single fold
cannot tell you your noise level either.

This is why every comparison in this work averages three seeds at the prediction level, and
why we use paired per-fold sign tests rather than comparing two averages.

### What it does not show

We do not fully know *why* fold 1 is unstable and fold 0 is not. It is a property of that
fold's group structure. Reporting it honestly is the contribution; explaining it is not yet
possible.

---

## 6. The demonstration: same predictions, opposite conclusions

**The claim.** Take one fixed set of predictions. Score it two defensible ways. The two
answers disagree. This is the whole paper in one slide.

### The chart — a plain grouped bar chart

Five metrics along the bottom. For each, one bar for EA and one for IP, showing the median
difference between the two models. Four bars sit at almost zero; the fifth is large.

### The table beside it

B split, cross-scaffold (D) folds only, 5 folds.

| Metric | Median difference, EA | Folds octamer wins | Median difference, IP | Folds octamer wins |
|---|---|---|---|---|
| Overall R² | +0.003 | 3 of 5 | +0.010 | 3 of 5 |
| MAE | −0.000 | 3 of 5 | −0.010 | 3 of 5 |
| RMSE | −0.004 | 3 of 5 | −0.012 | 3 of 5 |
| Group-mean R² | +0.002 | 3 of 5 | +0.007 | 3 of 5 |
| **ΔR² (architecture recovery)** | **+0.260** | **5 of 5** | **+0.152** | **5 of 5** |

### How the numbers are computed

1. For each of the 5 cross-scaffold folds and each model, average the three seeds'
   predictions, then compute every metric once on the averaged predictions.
2. Subtract wDMPNN's value from the octamer's, fold by fold. That gives 5 paired
   differences per metric per target.
3. Report the median of those 5, and count how many of the 5 favour the octamer.

The last row, ΔR², is architecture recovery: remove each group's average, then ask how well
the model reproduces what is left — the part that is purely about arrangement.

### How to read it

Say: *"These are the same predictions in every row. On the four standard measures the two
models split three folds to two — that is what a coin does. On architecture recovery the
octamer wins five out of five on both targets, and the gap is about a hundred times larger
than the accuracy differences. One set of predictions, two reasonable metrics, opposite
answers. That is the problem the paper is about."*

### The honest caveat, stated on the slide

With 5 folds, the strongest possible result from this test is 5 out of 5, which corresponds
to p = 0.0625. So this **does not** clear the conventional 0.05 threshold — it cannot,
by design. Say that before someone else does. The argument here rests on the *size and
consistency* of the gap, not on a significance claim.

---

## What the deck will look like

12 slides, same overall shape as before:

1. Title
2. The six components at a glance
3. Component 1 — table
4. Component 2 — table
5. Component 3 — table
6. Component 4 — table + bar chart
7. Component 5 — table
8. Component 6 — bar chart + table
9. What the framework establishes (four claims, each tied to a component)
10. Paper strategy — four papers
11. Why Paper 1 goes first
12. Provenance — where every number comes from

Two charts total, both plain grouped bar charts.

---

## Please check before I build

1. **Is the table-first approach right for a supervisor audience**, or would you rather keep
   more visuals even if they need explaining?
2. **Component 3's table is the biggest** — 9 rows by 7 columns. Would you prefer it cut to
   just the lookup column plus a "beats the models?" flag?
3. **Component 2 is a selected example.** I have written the base rate into the slide. Are
   you comfortable presenting it that way, or would you rather lead with the typical case?
4. **Component 6's p = 0.0625 caveat** — I have put it on the slide. Confirm you want it
   there rather than held for questions.
