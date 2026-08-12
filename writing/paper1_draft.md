# Averaged accuracy is not evidence: evaluating whether polymer representations encode chain architecture

**Draft v1 — 11 August 2026.** Sections 1–3 are drafted in full; Sections 4–6 are drafted at
outline-plus-prose level. Citations marked `[CITE]` need filling. All numerical results trace
to `analysis/paper1_figures/` manifests or to the verification recorded on 11 August 2026.

---

## Abstract

*(to be written last — placeholder)*

The standard copolymer benchmark established that a graph representation encoding chain
architecture improves property prediction, on the basis of an ablation under random
cross-validation splits. We show that this evidence, though sound, does not transfer to the
regime in which the benchmark is now used to assess generalisation. Under random splits every
monomer appears in training, so monomer identity is effectively free and architecture
accounts for roughly 14% of the remaining signal, where an averaged metric resolves it
easily. Under held-out-monomer splits monomer identity becomes the dominant unknown and
architecture falls to 0.98% (electron affinity) and 1.46% (ionisation potential) of target
variance, at which point averaged accuracy cannot distinguish a model that recovers
architecture from one that does not. The architecture ablation has never been run in that
regime. We propose an evaluation that restores sensitivity by removing the chemistry
placement term and scoring the residual, reported against a parameter-free null floor, per
fold, with both mean and median. It requires no retraining and is computable from any
existing set of predictions. Applied to five models on 42,966 copolymers, it shows that
averaged accuracy and architecture recovery agree completely under interpolation and disagree
completely under chemistry extrapolation, where four accuracy metrics split three folds to
two while architecture recovery splits five to zero. On one fold a parameter-free lookup
table that never sees the held-out monomer outperforms every trained model tested.

---

## 1. Introduction

### 1.1 What the wD-MPNN paper claims, and what it established

Polymers are not single molecules but ensembles of related chains. A copolymer of monomers A
and B with a fixed composition can be realised as many distinct sequences, and materials with
identical chemistry and identical composition but different chain architecture — blocky,
alternating, or random — can have different properties. Representing that distinction is a
central objective of machine learning for polymers [CITE].

Aldeghi and Coley introduced the weighted directed message passing neural network (wD-MPNN)
for exactly this purpose, together with the benchmark of 42,966 computed copolymers that is
now standard in the field [Aldeghi & Coley, *Chem. Sci.* 2022, **13**, 10486-10498]. Their
paper contains a section headed *"The wD-MPNN captures how polymer properties depend on chain
architecture and monomer stoichiometry"*, supported by an ablation in which representational
components are withheld and the resulting accuracy compared.

**The evidence is good, and we want to state it accurately before saying what it does not
cover.** Their Table 2 and Table S1 report a four-way ablation on the original EA/IP dataset:

| Representation | EA R² | EA RMSE (eV) | IP R² | IP RMSE (eV) |
|---|---|---|---|---|
| Monomers only | 0.917 | 0.173 | 0.883 | 0.165 |
| + chain architecture | 0.929 | 0.159 | 0.898 | 0.154 |
| + stoichiometry | 0.987 | 0.069 | 0.982 | 0.065 |
| **+ both (the full wD-MPNN)** | **0.997** | **0.035** | **0.997** | **0.027** |

Adding chain architecture on top of stoichiometry reduces RMSE by **49% on EA and 58% on IP**.
This is not a marginal effect. It appears small in R² only because R² saturates near 1: the
same step moves R² by 0.010 and 0.015 while removing essentially all remaining residual
variance.

The same ablation on a second, independent dataset — the diblock copolymer phase data of
Arora *et al.* [CITE] — points the same way, with chain architecture alone raising the area
under the precision-recall curve from 0.47 to 0.49 against a stoichiometry contribution of
0.67.

We therefore accept the claim as stated: **the wD-MPNN's representation does carry chain
architecture information, and that information improves prediction.** Our argument is not
that this evidence is weak.

### 1.2 The gap: the evidence covers a different regime from the one the benchmark now tests

The ablation above is run, in the authors' own words, under *"10-fold cross validation based
on random splits"*. That detail governs what it can establish.

The benchmark contains only **9 distinct A monomers** and 682 B monomers, in an exact
factorial of 6,138 pairs across seven architecture-composition cells. Under a random split,
every one of those 9 A monomers appears in training. Monomer identity is therefore effectively
memorised and contributes almost nothing to test error. What remains to be explained is
composition and architecture — and in that reduced space architecture is a substantial share,
which is why the ablation detects it so clearly.

The regime the field uses the benchmark for is not this one. Generalisation is assessed by
holding out a monomer entirely; the same paper reports wD-MPNN RMSEs of 0.10 ± 0.01 and
0.09 ± 0.02 eV under a 9-fold monomer-A-held-out split, roughly three times the random-split
error. In that setting monomer identity is no longer free: it is the dominant unknown, and it
dominates the aggregate error.

**The architecture ablation is never run under the held-out-monomer split.** Overall wD-MPNN
and baseline performance is reported there; the ablation series is not.

This matters because the two regimes place architecture in completely different positions.
Our variance decomposition (Section 3.1) quantifies both with a single calculation:

| | EA | IP |
|---|---|---|
| Architecture as a share of **total** variance — the held-out-monomer regime | **0.98%** | **1.46%** |
| Architecture as a share of variance remaining **once monomers are known** — the random-split regime | **13.9%** | **14.1%** |

Under random splits a model is working in the second row, where architecture is roughly a
seventh of the available signal and an aggregate metric can resolve it. Under held-out-monomer
splits it is working in the first row, where architecture is about one percent of the target
and a model that captured it perfectly would differ from one that ignored it entirely by
around two points of aggregate R².

So the position is not that the original evidence was too small. It is that **the evidence was
obtained where aggregate accuracy can see architecture, and is routinely cited in support of
work evaluated where aggregate accuracy cannot.** No amount of additional seeds or folds
repairs this, because the limitation is in the choice of reported quantity, not its precision.

There is a further sign, in the same paper, that aggregate scores on these tasks need
external calibration. On the diblock dataset the authors report that a random forest using
**mole fractions alone** achieves a PRC of 0.69 ± 0.01, and one using volume fractions alone
0.71 ± 0.01, against the full wD-MPNN's 0.68 ± 0.01. A baseline carrying no architectural
information whatever outperforms the architecture-aware model on that task. The observation is
made once, in passing. We argue it should be systematic.

### 1.3 A better evaluation, and why it works

Our proposal follows directly from Section 1.2. If the difficulty under held-out-monomer
splits is that chemistry placement dominates the aggregate error and buries architecture, then
the fix is to **remove the chemistry-placement term and score what is left**.

Define a group as the set of test polymers sharing monomer A, monomer B and composition, so
that within a group only architecture differs. Every prediction then splits into the group
mean and a within-group deviation. This yields two scores from one unchanged set of
predictions:

- **Group-mean R²** — did the model place each chemistry correctly? This is the term that
  dominates the aggregate metric under held-out-monomer splits.
- **ΔR²** — within a fixed chemistry and composition, did the model reproduce what
  architecture does? Because the group mean is subtracted from both truth and prediction,
  **this quantity cannot be improved by getting the chemistry right.**

ΔR² restores, under held-out-monomer splits, the property that made the original ablation
informative under random splits: it measures the model in the residual space where
architecture is a seventh of the signal rather than a hundredth. It requires no new data and
no retraining, so it is computable on any existing set of predictions for this benchmark,
including those already published.

We add three further requirements, each motivated by a specific failure we document:

- **Report against a null floor.** A parameter-free lookup table that never sees the held-out
  monomer achieves a median group-mean R² of 0.676 on EA, and on one fold beats every trained
  model we tested (Section 3.2). This generalises the mole-fraction baseline observation above
  from a single remark into a per-fold requirement.
- **Report per fold.** The folds of the standard scaffold-based split are not exchangeable:
  four of them test interpolation within a known scaffold family and five test extrapolation
  beyond it (Section 3.3).
- **Report mean and median.** On the extrapolation folds these disagree in sign for four of
  the four accuracy metrics we examine (Section 3.5).

### 1.4 Contributions

1. A variance decomposition of the standard copolymer benchmark that quantifies why an
   architecture ablation is informative under random splits and uninformative under
   held-out-monomer splits (Section 3.1).
2. A two-axis decomposition of predictive performance into chemistry placement and
   architecture recovery, computed from unchanged predictions (Section 2.4).
3. A parameter-free null floor for both axes, and evidence that it is competitive with
   trained models on a substantial minority of folds (Section 3.2).
4. An audit of the benchmark's scaffold structure showing that its cross-validation folds
   test two qualitatively different tasks (Section 3.3).
5. A demonstration, on five models and 42,966 polymers, that averaged accuracy and
   architecture recovery agree under interpolation and disagree under extrapolation
   (Section 3.5).

We deliberately do **not** claim that any model evaluated here is the best available. One
model (HPG-octamer) performs strongly throughout, but establishing that claim requires
external validity testing that is outside this paper's scope; it appears here as one of five
systems used to demonstrate a measurement problem.

---

## 2. Methods

### 2.1 Dataset

We use the copolymer dataset of Aldeghi and Coley [CITE]: 42,966 copolymers, each defined by
two monomers (A and B), a mole fraction, and a chain architecture. Targets are electron
affinity (EA) and ionisation potential (IP) versus the standard hydrogen electrode, in eV,
computed by density-functional-parameterised tight binding.

The dataset is an **exact factorial**, which matters for the analysis. Architectures are
`block`, `random` and `alternating`; mole fractions of monomer A are 0.25, 0.50 and 0.75.
`block` and `random` occur at all three fractions; `alternating` occurs only at 0.50. Every
occupied cell contains exactly 6,138 rows. Consequently:

- at fracA = 0.50, architecture is a **three-way** discrimination;
- at fracA = 0.25 and 0.75, it is a **two-way** discrimination;
- across all 42,966 rows there are only **three distinct transition matrices**, and they do
  not vary with composition, so architecture and composition are independent axes.

Labels were computed on 8-unit chains, over up to 32 sampled sequences per polymer and 8
conformers per sequence, then averaged. We return to the significance of this in
Section 5.

### 2.2 Models compared

Five trained models plus one untrained reference. All trained models predict a single scalar
per polymer and are trained with the same loss, optimiser and data pipeline.

**wD-MPNN (baseline).** The published architecture [CITE], evaluated in two configurations:

- *Published configuration* (`protocol_variant = original_paper`) — batch size 50, 30 epochs,
  initial learning rate 1e-4, patience 30. Patience cannot fire within 30 epochs, which
  reproduces the absence of early stopping in the released implementation. These settings
  follow the original authors' code rather than our own defaults, so that the baseline cannot
  be said to have been disadvantaged by our choices. **Currently run on the A split only.**
- *Harmonised configuration* — batch size 512, 300 epochs, patience 15, matching the training
  budget given to our own models. Run on both splits.

**Which configuration is used where.** Section 3.5's A-split comparison uses the *published*
configuration. The B-split comparisons use the *harmonised* configuration, because the
published-configuration arm has not yet been run on the B split. This is a gap, and it
affects the paper's headline result: see Appendix B. On the A split, where both are
available, the published configuration is the **stronger** baseline of the two (median
overall R² 0.967 / 0.971 against 0.958 / 0.949 for EA / IP), so using it is the conservative
choice there.

The wD-MPNN represents a copolymer as a single graph in which monomers are joined by
stochastic edges carrying transition probabilities, and atom representations are weighted by
monomer mole fraction. Architecture enters through the stochastic edge weights;
stoichiometry through the atom weighting.

**HPG-hier.** A two-stage hierarchical model of our own. Stage 1 encodes each monomer
independently with a standard directed MPNN. Stage 2 operates on a small graph whose nodes
are the two monomers and whose edges carry the transition probabilities of the architecture
(`stage2_mode = transition_graph`), with the graph read out by stoichiometry-weighted pooling
(`stage2_readout = stoich_weighted`). Architecture enters through the stage-2 edge features.

**HPG-hier-junction (2 steps) and HPG-hier-junction1 (1 step).** As HPG-hier, with an
additional junction-coupling stage that exchanges information between monomer representations
across the bond that would join them, for two and one coupling steps respectively. Included
to separate the effect of junction chemistry from the effect of the architecture encoding.

**HPG-octamer.** As HPG-hier, but stage 2 operates on an explicit **8-slot linear chain**
(`stage2_mode = octamer_sequence`) rather than a two-node transition graph. For each polymer,
16 sequences of 8 monomer units are sampled consistent with the requested composition and
architecture; each slot receives its monomer's stage-1 embedding plus a learned position
embedding; message passing runs along the chain; and the chain is read out by attention
(`stage2_readout = attention`). The 16 sampled sequences are encoded independently and their
**predictions** averaged. Architecture enters through which monomer occupies which slot.

**Null predictor (untrained).** A parameter-free group-mean lookup. For a test polymer it
returns the mean training value of polymers sharing the same monomer B, mole fraction and
architecture; failing that, the same monomer B and architecture; failing that, the global
training mean. Because the held-out monomer A is by construction absent from training, this
predictor is *blind to monomer A* and cannot use any information about it. It is scored
exactly like a trained model on the same folds with the same metrics.

Common settings for all HPG variants: batch size 64, maximum 100 epochs, patience 15, 16
sampled sequences, chain length 8.

### 2.3 Splits

Two cross-validation designs, each 9 folds, never pooled with one another.

**A split (`monomer_heldout`).** One monomer A is held out entirely per fold, together with a
validation monomer drawn from the remainder; the model trains on the other seven. Tests
generalisation to unseen A chemistry.

**B split (`monomer_b_heldout_clustered`).** Monomer B is held out, with folds constructed by
clustering B monomers on their Bemis–Murcko scaffolds. As we show in Section 3.3, the
resulting folds fall into two qualitatively different groups, which we label and always
report separately:

- **S folds (0–3)** — every held-out B monomer has a scaffold relative in the training set.
  The task is interpolation within known chemistry.
- **D folds (4–8)** — held-out B monomers largely have no scaffold relative in training. The
  task is extrapolation to unfamiliar chemistry.

**S and D folds are never averaged together.** Doing so produces a number describing neither
regime.

### 2.4 Metrics

This section is the paper's methodological core. All metrics are computed from a single set
of predictions; none requires retraining.

#### 2.4.1 The matched-group construction

Define a **group** as the set of test polymers sharing monomer A, monomer B and mole
fraction. Within a group, the only thing that differs is chain architecture. We retain groups
containing at least two distinct architectures; groups with one are discarded, since they
carry no architectural contrast.

For each group *g* let $\bar{y}_g$ and $\bar{\hat{y}}_g$ be the mean true and mean predicted
value. Each polymer's value then decomposes as

$$y_i = \bar{y}_{g(i)} + \delta_i, \qquad \delta_i = y_i - \bar{y}_{g(i)}$$

and identically for predictions. The first term is *where the chemistry sits*; the second is
*what architecture does within that chemistry*. The two axes score these separately.

#### 2.4.2 Axis 1 — chemistry placement (group-mean R²)

Compute R² between $\{\bar{y}_g\}$ and $\{\bar{\hat{y}}_g\}$ across groups. This asks whether
the model places each chemistry-and-composition combination at the right value, and is
insensitive to whether it resolves architecture at all.

#### 2.4.3 Axis 2 — architecture recovery (ΔR²)

Compute R² between $\{\delta_i\}$ and $\{\hat{\delta}_i\}$ over all retained polymers. Because
the group mean has been subtracted from both, **this quantity is by construction unable to
reward getting the chemistry right.** It measures only whether within-group deviations —
which are caused by architecture alone — are reproduced.

ΔR² is our primary quantity. It is not a rescaling of overall R²: two models can be
indistinguishable on overall R² and differ by 0.26 on ΔR² (Section 3.5).

#### 2.4.4 Supporting quantities

**Architecture-spread ratio.** For each group, (highest prediction − lowest prediction)
÷ (highest true value − lowest true value). A value of 1 means the model reproduces the full
range of variation across architectures; 0 means it predicts the same value for every
architecture; values above 1 mean it exaggerates.

This quantity is **two-sided**: 1.4 is as wrong as 0.71. It must not be summarised with a
plain mean or median as though larger were better. Where a single summary is needed we report
the median of $|\log_2(\text{ratio})|$, for which 0 is perfect and larger is worse.

We report the ratio separately for three-architecture groups (fracA = 0.50) and
two-architecture groups (fracA = 0.25, 0.75), never pooled, because a range over three points
and a range over two are not the same statistic.

**Ordering.** The fraction of within-group pairs ranked in the correct order, with ties
scored 0.5. Reported as an interpretable companion to ΔR², not as an independent test — it is
a different function of the same predictions.

**Conventional metrics.** Overall R², MAE and RMSE on all test rows, reported for continuity
with existing literature.

#### 2.4.5 The null floor

Every group-mean R² is reported beside the score achieved on the same fold by the null
predictor of Section 2.2. We define the skill score

$$\text{skill} = \frac{R^2_{\text{model}} - R^2_{\text{null}}}{1 - R^2_{\text{null}}}$$

which is zero when a model merely matches the lookup table and one when it is perfect. A
model can post a high absolute R² and a skill score near zero; Section 3.2 shows this occurs.

### 2.5 Training and evaluation protocol

The protocol below was frozen before the comparisons reported here and is enforced in code by
a `--frozen_protocol` flag.

**Three seeds, averaged at the prediction level.** Every cell (model × target × fold) is
trained with seeds 42, 43 and 44. The three prediction vectors are **averaged first**, and
each metric is then computed **once** on the averaged predictions.

This ordering is not incidental. Averaging per-seed metric values instead produces different
numbers, and is unstable when a single run fails: in one cell, per-seed ΔR² values of +0.426,
−0.215 and +0.729 average to 0.313 as metrics but give 0.415 when the predictions are averaged
first. We report only the latter.

We use three seeds for stability of the reported quantity. We do not claim three seeds is
sufficient to characterise run-to-run variability, and we make no claims about variance
itself, since an SD estimated from three samples carries roughly 40% relative error.

**Best-checkpoint predictions.** Predictions come from the checkpoint with the lowest
validation loss, not from the final model at patience expiry. This distinction is recorded
explicitly in every prediction file.

**No test-set information** enters training, validation-fold selection, or early stopping.

### 2.6 Statistical treatment

Model comparisons are made **fold by fold, paired**, on the same data with the same seeds,
and assessed with a two-sided sign test on the number of folds favouring one model. We
prefer the sign test because per-fold differences on these splits are neither independent nor
normally distributed, and because it makes no assumption about the size of an effect.

We report the **median and the mean** of the paired differences, always both. Section 3.5
shows they can disagree in sign, which is itself a result.

**The minimum attainable p-value is a property of the design and we state it wherever a test
is reported.** With 9 folds it is 0.0039; with 5 folds, 0.0625; with 4 folds, 0.125.
Consequently no comparison on the D folds or the S folds can reach a 0.05 threshold, however
large the effect. We rely on effect size and consistency there, and reserve significance
claims for the 9-fold A split.

---

## 3. Results

### 3.1 Architecture is about one percent of the target

We decompose target variance by sequentially conditioning on monomer A, then monomer B, then
composition, then architecture, computing at each stage the fraction of variance reproduced
by replacing every polymer's value with its group mean.

| Factor | Share of total variance, EA | Share of total variance, IP |
|---|---|---|
| Monomer A | 41.8% | 50.0% |
| Monomer B, given A | 51.1% | 39.6% |
| Composition, given A and B | 6.1% | 8.9% |
| **Architecture, given A, B and composition** | **0.98%** | **1.46%** |
| Unexplained | 0.0% | 0.0% |

The four factors are exhaustive: shares sum to 1.000 on both targets, because the dataset is
an exact factorial with no residual within-cell variation.

Two readings follow, and both matter.

**For averaged metrics, the ceiling is about two points of R².** A model resolving
architecture perfectly and one ignoring it entirely cannot differ by more on an aggregate
score. The 0.88 → 0.90 movement offered as evidence in prior work is not a small effect
detected against a large background; it is close to the entire range available.

**For a residual metric, architecture is not a marginal signal.** Once monomer identity is
known, 7.1% (EA) and 10.3% (IP) of variance remains, and architecture accounts for **13.9%
and 14.1%** of that remainder. This is the justification for ΔR²: the signal is small
relative to the target but substantial relative to the space in which it lives.

### 3.2 A parameter-free lookup table is often competitive

Group-mean R² per fold on the A split, for the null predictor and two trained models.

| Fold | EA null | EA octamer | EA wD-MPNN | IP null | IP octamer | IP wD-MPNN |
|---|---|---|---|---|---|---|
| 0 | 0.694 | 0.958 | 0.882 | **0.969** | 0.748 | 0.496 |
| 1 | 0.487 | 0.991 | 0.982 | 0.509 | 0.970 | 0.982 |
| 2 | **0.961** | 0.986 | 0.981 | −1.019 | 0.995 | 0.976 |
| 3 | **0.953** | 0.995 | 0.979 | −3.206 | 0.982 | 0.962 |
| 4 | 0.884 | 0.962 | 0.947 | −0.251 | 0.960 | 0.918 |
| 5 | 0.676 | 0.994 | 0.970 | −7.528 | 0.977 | 0.755 |
| 6 | −19.069 | 0.937 | 0.878 | 0.569 | 0.996 | 0.986 |
| 7 | 0.098 | 0.982 | 0.846 | 0.410 | 0.988 | 0.482 |
| 8 | 0.428 | 0.989 | 0.992 | −0.034 | 0.995 | 0.987 |
| **Median** | **0.676** | 0.984 | 0.967 | **−0.034** | 0.978 | 0.971 |

Three observations.

**The floor differs by target on the same split.** Median null group-mean R² is 0.676 on EA
and −0.034 on IP. The A split is close to degenerate for EA and genuinely demanding for IP,
and nothing in the split design reveals this — only computing the floor does.

**On some folds the floor is very high.** On EA folds 2 and 3 the lookup table reaches 0.961
and 0.953. A trained model reporting 0.98 on those folds has a skill score of roughly 0.5 and
0.6 respectively; reporting the raw number alone materially overstates what was learned.

**On one fold the floor beats every trained model.** On IP fold 0 the null scores 0.969
against 0.748 and 0.496. We are not aware of prior work on this benchmark reporting such a
comparison.

The extreme negative values (−19.07, −7.53) are genuine: R² is unbounded below, and on those
folds the held-out chemistry is sufficiently unlike the training distribution that the lookup
table is far worse than predicting a constant. We report them rather than clipping them.

### 3.3 The folds of the scaffold split are not exchangeable

The benchmark's 682 B monomers fall into 112 Bemis–Murcko scaffold families, with sizes
distributed extremely unevenly.

| | Count | Share of B monomers |
|---|---|---|
| Largest family | 317 | 46.5% |
| Second largest | 109 | 16.0% |
| **Two largest together** | **426** | **62.5%** |
| Remaining 110 families | 256 | 37.5% |
| — of which singletons | 64 | — |

Because 62.5% of B monomers sit in two families, a balanced scaffold-disjoint split is not
constructible. The consequence is visible in fold composition:

| Fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Fraction of held-out B monomers with a scaffold relative in training | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 | 0.43 | 0.00 | 0.00 |

Folds 0–3 test interpolation within a known scaffold family. Folds 5, 7 and 8 test
extrapolation to scaffolds never seen. These are different experiments, and averaging them
produces a figure describing neither. We therefore report S and D folds separately
throughout.

### 3.4 The two axes separate models that averaged accuracy does not

We compare HPG-octamer and wD-MPNN at its published configuration across all folds and both
targets, using the matched-group construction of Section 2.4. Each row summarises 682 groups.

Median over 9 A-split folds:

| | Placement error (eV), octamer | Placement error (eV), wD-MPNN | Spread ratio, octamer | Spread ratio, wD-MPNN |
|---|---|---|---|---|
| EA | 0.047 | 0.048 | 0.77 | 0.61 |
| IP | 0.029 | 0.045 | 0.95 | 0.55 |

Distance of the spread ratio from 1, median $|\log_2|$ (0 is perfect):

| | Octamer | wD-MPNN |
|---|---|---|
| EA | **0.42** | 0.71 |
| IP | **0.15** | 0.86 |

On EA the two models place chemistry essentially identically (0.047 against 0.048) while
differing substantially in how much architectural variation they reproduce. wD-MPNN errs in
both directions — flattening to 0.25 on EA fold 1 and 0.23 on IP fold 7, exaggerating to 1.39
on EA fold 7 and 1.48 on IP fold 1 — which is why a two-sided summary is required.

### 3.5 Metric choice is immaterial under interpolation and decisive under extrapolation

Paired per-fold differences, HPG-octamer minus wD-MPNN. Positive favours the octamer for the
R² measures; negative favours it for MAE and RMSE. **Note the baseline configuration differs
between the A-split and B-split tables** — see Section 2.2 and Appendix B.

**A split — 9 folds, held-out monomer A — baseline at its PUBLISHED configuration**

| Metric | EA median | EA mean | Wins | p | IP median | IP mean | Wins | p |
|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.017 | +0.030 | 9/9 | **0.004** | +0.021 | +0.070 | 8/9 | 0.039 |
| MAE | −0.018 | −0.023 | 8/9 | 0.039 | −0.026 | −0.026 | 8/9 | 0.039 |
| RMSE | −0.024 | −0.026 | 9/9 | **0.004** | −0.033 | −0.030 | 8/9 | 0.039 |
| Group-mean R² | +0.008 | +0.027 | 8/9 | 0.039 | +0.019 | +0.060 | 7/9 | 0.180 |
| **ΔR²** | **+0.190** | **+0.236** | 8/9 | 0.039 | **+0.278** | **+0.267** | 9/9 | **0.004** |

**B split, S folds — 4 folds, held-out monomer has scaffold relatives in training —
baseline at the HARMONISED configuration**

| Metric | EA median | EA mean | Wins | IP median | IP mean | Wins |
|---|---|---|---|---|---|---|
| Overall R² | +0.014 | +0.012 | 4/4 | +0.005 | +0.004 | 3/4 |
| MAE | −0.022 | −0.019 | 4/4 | −0.013 | −0.013 | 4/4 |
| RMSE | −0.023 | −0.018 | 4/4 | −0.010 | −0.010 | 3/4 |
| Group-mean R² | +0.012 | +0.010 | 3/4 | +0.002 | +0.002 | 3/4 |
| **ΔR²** | **+0.186** | **+0.167** | 4/4 | **+0.122** | **+0.137** | 4/4 |

*(No p-values: with 4 folds the minimum attainable is 0.125.)*

**B split, D folds — 5 folds, held-out monomer has no scaffold relative —
baseline at the HARMONISED configuration**

| Metric | EA median | EA mean | Wins | IP median | IP mean | Wins |
|---|---|---|---|---|---|---|
| Overall R² | +0.003 | **−0.006** | 3/5 | +0.010 | **−0.014** | 3/5 |
| MAE | −0.000 | **+0.001** | 3/5 | −0.010 | −0.003 | 3/5 |
| RMSE | −0.004 | **+0.005** | 3/5 | −0.012 | **+0.005** | 3/5 |
| Group-mean R² | +0.002 | **−0.008** | 3/5 | +0.007 | **−0.016** | 3/5 |
| **ΔR²** | **+0.260** | **+0.225** | 5/5 | **+0.152** | **+0.198** | 5/5 |

Bold marks entries where the mean and median disagree in sign. *(Minimum attainable p with 5
folds is 0.0625; ΔR² attains it on both targets.)*

Three findings.

**Under interpolation, every metric agrees.** On the A split all five metrics favour the
octamer on 7–9 of 9 folds. On the S folds all five point the same way. If the only regimes
examined were these, the choice of metric would be immaterial and the extra machinery of
Section 2.4 would be unnecessary.

**Under extrapolation, the metrics separate completely.** On the D folds the four accuracy
metrics split 3 folds to 2 on both targets — indistinguishable from a coin — while ΔR² splits
5 to 0 on both, with a median difference roughly fifty times larger. These are the *same
predictions* scored two ways.

**Under extrapolation the accuracy metrics are also unstable in a second sense.** On the D
folds, the mean and the median of the same paired differences disagree in sign for four of
four accuracy metrics on EA and three of four on IP. The median suggests the octamer is
marginally ahead; the mean suggests it is marginally behind. ΔR² shows no such disagreement
on either target. A single averaged accuracy number on these folds is therefore not merely
insensitive — its sign depends on which summary statistic is chosen.

---

## 4. Discussion

**What this does and does not say about the wD-MPNN.** It does not challenge the original
ablation, which we reproduce in Section 1.1 and accept: under random splits, adding chain
architecture halves RMSE once stoichiometry is present. The wD-MPNN's representation carries
architecture information and that information helps.

What we show is that this result was obtained in a regime — monomers seen in training — where
architecture is roughly a seventh of the residual signal, and that it is routinely cited in
support of models evaluated in a regime where it is one percent of the target. Those are
different measurements. The ablation has never been run under a held-out-monomer split, and
our results indicate what would happen if it were: on cross-scaffold folds, four accuracy
metrics cannot separate two models that differ by 0.26 in architecture recovery.

The practical consequence is not that published numbers are wrong. It is that a reader cannot
tell, from an aggregate score under a held-out-monomer split, whether a representation
resolves architecture at all — and that is the property such representations are built for.

**Why a residual metric rather than a better-designed dataset.** Both are worth doing, and we
regard the dataset route as the more permanent fix (Section 5). But ΔR² requires no new data
and no retraining: it is computable from any existing set of predictions on this benchmark,
including those already published. It is therefore immediately applicable to the existing
literature.

**On per-fold reporting.** The three phenomena above — a null floor that varies from 0.96 to
−19.07 across folds of one split, folds that test different tasks, and mean–median sign
disagreement — are individually visible only per fold. Each is invisible in a single averaged
number, and each changes the interpretation of that number.

---

## 5. Limitations

**The architecture design space is small.** The benchmark contains only three distinct
transition matrices, and architecture is a two-way discrimination at two of three
compositions. ΔR² therefore measures resolution of a coarse categorical contrast, not of a
continuous architectural axis. Whether the conclusions extend to finer gradations of
blockiness cannot be settled on this data.

**Protocol matching is a confound for one model.** The benchmark's labels were computed on
8-unit chains averaged over up to 32 sampled sequences. HPG-octamer uses an 8-slot chain, 16
sampled sequences, and averages predictions — its structure mirrors the label-generation
procedure. Its advantage may reflect a correct physical inductive bias, or may reflect
alignment with this dataset's construction. **No ablation on this dataset can separate these
readings**, and we do not claim to have done so. This is one reason the paper's conclusions
are framed around measurement rather than around model ranking.

**Sign tests on few folds are weak.** With 5 and 4 folds the minimum attainable p-values are
0.0625 and 0.125. The B-split results are reported as effect sizes with consistency counts,
not as significance claims.

**Two targets, one dataset, one property class.** EA and IP are computed electronic
properties on one copolymer family. Whether the agree-then-diverge pattern of Section 3.5
generalises to experimental properties or other polymer classes is untested.

**We characterise the reported quantity, not run-to-run variance.** Three seeds are averaged
for stability. An SD estimated from three samples is too imprecise to support claims about
variability, and we make none.

---

## 6. Conclusion

On the standard copolymer benchmark, chain architecture accounts for about one percent of
target variance. Averaged accuracy metrics therefore have roughly two points of dynamic range
for the property that polymer representation learning most often claims to capture, and
movements within that range have been read as evidence of architecture awareness.

We propose reporting two axes rather than one, each against a parameter-free null floor, per
fold, with both mean and median. Applied to five models, this shows that the choice of metric
is immaterial while a model interpolates within familiar chemistry, and decisive as soon as it
extrapolates beyond it — which is the regime the field is ultimately interested in.

---

## Appendix A — reproducibility

All figures, tables and manifests: `analysis/paper1_figures/`. Every reported quantity is
accompanied by a manifest recording prediction file paths, seed handling, group counts and
selection rules. Protocol: three seeds (42, 43, 44) averaged at the prediction level;
best-checkpoint predictions; S and D folds never pooled; paired per-fold sign tests.

## Appendix B — to complete before submission

- **Run the published-configuration wD-MPNN on the B split.** This is the most important gap.
  The paper's headline result — Section 3.5's D-fold table, where four accuracy metrics split
  3/5 while ΔR² splits 5/5 — currently uses the *harmonised* baseline, because the
  published-configuration arm covers the A split only (54 runs; a single stray EA B-split
  fold-0 file exists and is unusable). A reviewer can reasonably ask why the headline
  comparison does not use the authors' own settings when the A-split comparison does.
  Estimated cost is small: 9 folds × 2 targets × 3 seeds = 54 runs at 30 epochs, roughly an
  order of magnitude cheaper per run than the 300-epoch harmonised arm. Until this is run,
  either the labelling in Section 3.5 must stand as-is, or the D-fold claim must be stated as
  holding against our harmonised baseline specifically.
- Abstract.
- `[CITE]` markers: field motivation, wD-MPNN paper, adoption of the benchmark by subsequent work.
- Supplementary table: harmonised-configuration wD-MPNN results on the A split, alongside the
  published-configuration results, so the two baselines can be compared directly.
- Supplementary table: HPG-hier, HPG-hier-junction, HPG-hier-junction1 on all splits. Section 1.4
  claims five models; only two are currently tabulated in Section 3.
- Confirm the claim in 1.1 that subsequent work credits architecture-awareness on this basis —
  currently asserted without citation.
- Decide whether the ordering metric (2.4.4) earns its place; it is currently defined but not
  used in Section 3.
