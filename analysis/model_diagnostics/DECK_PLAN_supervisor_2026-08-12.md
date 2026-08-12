# Supervisor meeting deck — plan

**For Wednesday 12 August 2026.** Three sections:

1. The evaluation paper — what it argues and what is written
2. Paper strategy — where it sits, and what dataset Paper 2 builds
3. Why the octamer beats wD-MPNN — what we have excluded and what remains

Component 5 (noise floor) removed throughout, as agreed. The protocol slide says only that
three seeds are averaged for stability, and makes no claim about variance.

---

# Section 1 — The evaluation paper

## Slide 1.1 — What the wD-MPNN paper claims, stated fairly

Their paper has a section headed *"The wD-MPNN captures how polymer properties depend on chain
architecture and monomer stoichiometry."* The supporting ablation, from their Table 2 and
Table S1, on the EA/IP dataset:

| Representation | EA R² | EA RMSE | IP R² | IP RMSE |
|---|---|---|---|---|
| Monomers only | 0.917 | 0.173 | 0.883 | 0.165 |
| + chain architecture | 0.929 | 0.159 | 0.898 | 0.154 |
| + stoichiometry | 0.987 | 0.069 | 0.982 | 0.065 |
| **+ both (full wD-MPNN)** | **0.997** | **0.035** | **0.997** | **0.027** |

**Open by conceding the point.** *"Adding chain architecture on top of stoichiometry halves
the error — 49% on EA, 58% on IP. That is not a marginal result. It looks small in R² only
because R² saturates near 1. Their claim is correct and their evidence supports it. My
argument is not that they were wrong."*

Tactically this matters. Lead with a critique of their numbers and the rest of the talk is a
fight. Lead with a concession and the room hears the actual point.

## Slide 1.2 — But that evidence was obtained in a different regime

**The ablation runs under random 10-fold splits.** Their caption says so.

The dataset has only **9 distinct A monomers** and 682 B monomers. Under a random split all 9
appear in training, so monomer identity is effectively memorised and contributes almost
nothing to test error. The model is scored on what is left — composition and architecture —
and in that space architecture is large.

Under a **held-out-monomer** split, monomer identity becomes the dominant unknown. Their own
figures show it: RMSE goes from 0.027–0.035 eV under random splits to 0.09–0.10 eV when a
monomer is held out, roughly three times worse.

**And the architecture ablation was never run in that second regime.**

One calculation quantifies both:

| | EA | IP |
|---|---|---|
| Architecture, share of **total** variance — the held-out-monomer regime | **0.98%** | **1.46%** |
| Architecture, share of what remains **once monomers are known** — the random-split regime | **13.9%** | **14.1%** |

**The line:** *"Under random splits architecture is about a seventh of what the model still has
to explain, so an averaged metric picks it up. Under held-out-monomer splits it is one percent
of the target, and a model that captured it perfectly would differ from one that ignored it by
about two points of aggregate R². Same benchmark, same property — the metric's ability to see
architecture changes completely with the split."*

## Slide 1.3 — How every number is calculated

State the definitions explicitly. Everything below is computed from **one unchanged set of
predictions** — no retraining.

### The group construction

A **group** *g* is the set of test polymers sharing monomer A, monomer B and composition.
Within a group, only chain architecture differs. Groups with fewer than two architectures are
discarded — they carry no architectural contrast.

For group *g*, let ȳ_g be the mean true value and ŷ̄_g the mean predicted value. Every polymer
*i* then splits into two parts:

```
    y_i  =  ȳ_g(i)   +   δ_i          where  δ_i = y_i − ȳ_g(i)
            └─ where the ─┘   └─ what architecture ─┘
               chemistry sits     does within it
```

and identically for predictions: ŷ_i = ŷ̄_g(i) + δ̂_i.

### Axis 1 — chemistry placement

```
    group_mean_R²  =  R²( { ȳ_g } , { ŷ̄_g } )        over groups g
```

*Did the model put each chemistry-and-composition combination at the right value?* Insensitive
to whether it resolves architecture at all.

### Axis 2 — architecture recovery (our primary quantity)

```
    ΔR²  =  R²( { δ_i } , { δ̂_i } )                  over all retained polymers i
```

*Within a fixed chemistry and composition, does the model reproduce what architecture does?*

**Why this is the right fix, not just another metric:** subtracting the group mean removes
exactly the chemistry-placement term that dominates the aggregate error under held-out-monomer
splits. It puts the model back in the residual space where the original ablation worked — the
14% row of slide 1.2, not the 1% row.

Note ΔR² **cannot be improved by getting the chemistry right**, because the group mean has
been subtracted from both truth and prediction.

### The null floor

A parameter-free lookup table, never trained. For a test polymer it returns:

1. the mean **training** value of polymers sharing the same monomer B, composition and
   architecture; failing that
2. the same monomer B and architecture; failing that
3. the global training mean.

Because monomer A is held out by construction, this predictor **cannot use any information
about monomer A**. It is scored with the identical metric on the identical fold.

```
    skill  =  ( R²_model − R²_null ) / ( 1 − R²_null )
```

0 = the model matched a lookup table. 1 = perfect. A model can post R² = 0.98 and skill ≈ 0.

### Architecture-spread ratio

```
    ratio_g  =  ( max_i∈g ŷ_i − min_i∈g ŷ_i )  /  ( max_i∈g y_i − min_i∈g y_i )
```

*Of the true spread between architectures, how much does the model reproduce?* 1.00 is
perfect; 0 means one value for every architecture; above 1 means exaggeration.

**Two-sided — 1.4 is as wrong as 0.71.** Never summarise with a plain median as though larger
were better. Where one number is needed:

```
    spread_error  =  median_g | log₂( ratio_g ) |          0 = perfect, larger = worse
```

Reported separately for 3-architecture groups (fracA = 0.5) and 2-architecture groups
(fracA = 0.25, 0.75) — a range over three points is not the same statistic as a range over two.

### Conventional metrics, for continuity

```
    overall_R²  =  1 − Σ(y−ŷ)² / Σ(y−ȳ)²       MAE = mean|y−ŷ|       RMSE = √(mean (y−ŷ)²)
```

### Protocol

Three seeds (42/43/44). **Predictions averaged first, then the metric computed once.** Not the
other way round: averaging per-seed metric values gives different answers and is unstable when
one run fails. Best-checkpoint predictions, never the final patience-expired model.

## Slide 1.4 — Baselines blind to the tested property should be standard practice

**The proposal:** whenever a paper claims a representation encodes property X, it should report
a baseline that **cannot** encode X, scored identically, on every fold.

**The wD-MPNN paper already does this — and that is the point.** On the diblock dataset the
authors report, plainly:

| Model | PRC |
|---|---|
| Full wD-MPNN | 0.68 ± 0.01 |
| **Random forest, mole fractions alone** | **0.69 ± 0.01** |
| **Random forest, volume fractions alone** | **0.71 ± 0.01** |
| Best RF (fingerprints + stoichiometry + size) | 0.74 ± 0.01 |

A baseline carrying no architectural information at all scores *above* the architecture-aware
model, on a dataset used to demonstrate architecture awareness.

**The line:** *"They ran the right comparison. They reported it honestly. What did not happen
is anyone drawing the conclusion from it — it appears once, in passing, as a remark about that
one dataset. I want to argue that this is not an optional extra. It should be computed on
every fold and printed next to every score, because without it you cannot tell what a number
means."*

**Framing:** not "we found something they missed" — they didn't miss it — but "**we are
proposing that a comparison they already ran becomes a required part of the protocol**". That
is a much easier argument to win, and it makes the wD-MPNN paper an ally rather than a target.

Our own version of this, systematised across all 9 folds, is slide 1.5.

## Slide 1.5 — The null floor across every fold

Group-mean R² per fold, A split.

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

- *"Same split, same benchmark. On EA a lookup table gets 0.68; on IP it gets −0.03."*
- *"On EA folds 2 and 3 it scores 0.96 and 0.95. A model reporting 0.98 there has learned much
  less than the number suggests."*
- *"And on IP fold 0 the lookup table beats both trained models."*

## Slide 1.6 — The folds are not exchangeable

62.5% of the 682 B monomers sit in two scaffold families (317 and 109), so a balanced
scaffold-disjoint split cannot be built:

| Fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Held-out monomers with a scaffold relative in training | 1.00 | 1.00 | 1.00 | 1.00 | 0.17 | 0.00 | 0.43 | 0.00 | 0.00 |
| | S | S | S | S | D | D | D | D | D |

*"Folds 0–3 are interpolation. Folds 5, 7 and 8 are extrapolation. Averaging them gives a
number that describes neither."*

## Slide 1.7 — The headline: metrics agree until you extrapolate

**A split, 9 folds — baseline at its PUBLISHED configuration**

| Metric | EA median | EA mean | Wins | p | IP median | IP mean | Wins | p |
|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.017 | +0.030 | 9/9 | **0.004** | +0.021 | +0.070 | 8/9 | 0.039 |
| MAE | −0.018 | −0.023 | 8/9 | 0.039 | −0.026 | −0.026 | 8/9 | 0.039 |
| RMSE | −0.024 | −0.026 | 9/9 | **0.004** | −0.033 | −0.030 | 8/9 | 0.039 |
| Group-mean R² | +0.008 | +0.027 | 8/9 | 0.039 | +0.019 | +0.060 | 7/9 | 0.180 |
| **ΔR²** | **+0.190** | **+0.236** | 8/9 | 0.039 | **+0.278** | **+0.267** | 9/9 | **0.004** |

**B split, D folds (cross-scaffold), 5 folds — baseline at PUBLISHED configuration**

| Metric | EA median | EA mean | Wins | IP median | IP mean | Wins |
|---|---|---|---|---|---|---|
| Overall R² | — | — | — | — | — | — |
| MAE | — | — | — | — | — | — |
| RMSE | — | — | — | — | — | — |
| Group-mean R² | — | — | — | — | — | — |
| **ΔR²** | — | — | — | — | — | — |

> **Blank by design — this arm has not been run.** 54 runs at 30 epochs, cheap. See slide 1.8
> for what we expect it to show.

**B split, D folds — baseline at our HARMONISED configuration (what we have today)**

| Metric | EA median | EA mean | Wins | IP median | IP mean | Wins |
|---|---|---|---|---|---|---|
| Overall R² | +0.003 | **−0.006** | 3/5 | +0.010 | **−0.014** | 3/5 |
| MAE | −0.000 | **+0.001** | 3/5 | −0.010 | −0.003 | 3/5 |
| RMSE | −0.004 | **+0.005** | 3/5 | −0.012 | **+0.005** | 3/5 |
| Group-mean R² | +0.002 | **−0.008** | 3/5 | +0.007 | **−0.016** | 3/5 |
| **ΔR²** | **+0.260** | **+0.225** | 5/5 | **+0.152** | **+0.198** | 5/5 |

Bold = mean and median disagree in sign.

**Three things to say:**

1. *"On the A split every metric agrees — the octamer wins on all five. If that were the only
   regime we looked at, none of this machinery would be needed."*
2. *"On the hard cross-scaffold folds, four accuracy metrics split three-to-two, which is what
   a coin does. Architecture recovery splits five-to-zero on both properties. Same predictions."*
3. *"And on those folds mean and median disagree in sign for four of the four accuracy
   metrics. It's not just that they're insensitive — the answer depends on which average you
   pick. ΔR² doesn't do that."*

**Caveat on the slide:** with 5 folds the smallest achievable p is 0.0625. The D-fold result
cannot reach 0.05 by design.

## Slide 1.8 — What we expect the missing arm to show

We can estimate it, because we have **both** wD-MPNN configurations on the A split. Comparing
them there gives the systematic offset between configurations:

**Published config minus 300-epoch config, A split, median over 9 folds:**

| Metric | EA | IP |
|---|---|---|
| Overall R² | −0.002 | +0.006 |
| MAE | +0.000 | −0.003 |
| RMSE | +0.004 | −0.005 |
| Group-mean R² | +0.001 | +0.001 |
| **ΔR²** | **+0.067** | **+0.081** |

**Read this carefully:** on the four accuracy metrics the two configurations are
indistinguishable — offsets within ±0.006. But on **architecture recovery the published
configuration is consistently better**, by 0.067 (EA) and 0.081 (IP). The published baseline
is the *stronger* one on the axis we care about.

Applying that offset to the D-fold ΔR² we already have:

| | fold 4 | fold 5 | fold 6 | fold 7 | fold 8 | median | wins |
|---|---|---|---|---|---|---|---|
| **EA** observed (vs 300-ep) | +0.266 | +0.260 | +0.167 | +0.171 | +0.261 | +0.260 | 5/5 |
| **EA** projected (vs published) | +0.199 | +0.193 | +0.100 | +0.104 | +0.194 | **+0.193** | 5/5 |
| **IP** observed (vs 300-ep) | +0.199 | +0.104 | +0.147 | +0.389 | +0.152 | +0.152 | 5/5 |
| **IP** projected (vs published) | +0.118 | +0.023 | +0.066 | +0.308 | +0.071 | **+0.071** | 5/5 |

**What to say:** *"We expect the accuracy metrics to be essentially unchanged — still around
three of five — because the two configurations are equivalent there. We expect the
architecture gap to shrink but survive: roughly +0.19 on EA and +0.07 on IP, still five of
five. The headline claim should hold."*

**And the honest caveat:** *"IP is the thin case. Fold 5 projects to +0.023, close enough to
zero that the five-of-five could become four-of-five. That's another reason to run it rather
than assume."*

This is a **projection from an offset measured on a different split**, not a result. Label it
as such on the slide.

---

# Section 2 — Paper strategy

## Slide 2.1 — Four papers

| # | Paper | Status | Note |
|---|---|---|---|
| 0 | Review — accepted, *Digital Discovery* | **done** | Argues in the literature that representation quality needs evaluating beyond accuracy |
| 1 | **Evaluation framework** | **write now** | Demonstrates that argument empirically. One cheap run outstanding |
| 2 | **Architecture-aware benchmark dataset** | costed | Builds the data the argument requires — and resolves our own confound. Slide 2.2 |
| 3 | The model — octamer vs baseline | evidence in hand | Held until external validity is tested |

**Why Paper 1 first:** it is writable today; it does not depend on which model wins, which
insulates it from the protocol-matching confound; and the measurement work invalidates most
earlier model comparisons, so it is the stronger contribution.

**What Paper 1 does not claim:** that our model is best. That belongs to Paper 3.

## Slide 2.2 — What dataset Paper 2 builds, and why it does two jobs

The existing benchmark has **three architecture settings** — literally three distinct
transition matrices across all 42,966 rows — and at two of three compositions architecture is
only a two-way choice. So a model never has to do more than tell two or three cases apart.

**The generation pipeline is reusable.** The authors published their code; it encodes a real
Suzuki coupling reaction, and our monomers are the right type for it. We have read it and
verified it runs on monomers we already hold.

### Design — three changes, each with a purpose

| # | Change | Purpose |
|---|---|---|
| 1 | **Blockiness varies continuously** at fixed chemistry and fixed composition | Turns a 3-way classification into a real measurement axis. This is what Paper 1 shows is missing |
| 2 | **Publish the un-averaged per-chain values**, not only the ensemble average | Costs nothing extra — the calculation already produces them — and is strictly more information than any existing benchmark |
| 3 | **Sweep the label chain length** — compute labels at 8, 12, 16 and 24 units for a subset | Resolves the protocol-matching confound. See slide 3.7 |

Note on chain lengths: compositions are quarters, so 6 and 10 units cannot represent them
exactly and would look worse for purely arithmetic reasons. **8, 12, 16 and 24 are the clean
comparisons.**

### Cost

| | |
|---|---|
| Worst case per polymer | 256 structures (32 sequences × 8 conformers) |
| ~2,000 polymers | ≤ 512,000 structures |
| at ~30 CPU-seconds each | **≈ 8–9 kSU of CPU** |

That is a **ceiling**; a one-day pilot would measure the real figure. For scale, reproducing
the full 42,966 would be ~150 kSU — we are proposing about 5%. **This is CPU, not GPU**, so a
different queue from model training.

**The ask:** this is the decision we need from you. It is costed and designed; nothing else
blocks it.

---

# Section 3 — Why does the octamer beat wD-MPNN?

## Slide 3.1 — The result being explained

Median over 9 A-split folds, baseline at its published configuration:

| | EA overall R² | EA MAE | EA ΔR² | IP overall R² | IP MAE | IP ΔR² |
|---|---|---|---|---|---|---|
| HPG-octamer | **0.984** | **0.055** | **0.849** | **0.978** | **0.035** | **0.886** |
| wD-MPNN (published cfg) | 0.967 | 0.070 | 0.397 | 0.971 | 0.050 | 0.565 |

*"The accuracy gap is real but modest. The architecture-recovery gap is more than double."*

## Slide 3.2 — What the difference looks like mechanically

Median across all 682-group folds:

| | Placement error (eV) octamer | wD-MPNN | Spread ratio octamer | wD-MPNN |
|---|---|---|---|---|
| EA | 0.047 | 0.048 | 0.77 | 0.61 |
| IP | 0.029 | 0.045 | 0.95 | 0.55 |

Distance from 1 in either direction (median |log₂|; 0 = perfect):

| | Octamer | wD-MPNN |
|---|---|---|
| EA | **0.42** | 0.71 |
| IP | **0.15** | 0.86 |

*"On EA the two models place the chemistry equally well — 0.047 against 0.048. The whole
difference is in how much architectural variation they reproduce. And wD-MPNN is not
systematically flattening: it flattens to 0.25 on some folds and exaggerates to 1.48 on
others. Unreliable in both directions rather than biased in one."*

## Slide 3.3 — Five differences, tested one at a time

The octamer differs from our own HPG-hier in five ways at once. Each elimination was
**pre-registered before running**.

| # | Factor | Status |
|---|---|---|
| 1 | 8-slot chain instead of a 2-node graph | **open** |
| 2 | Learned position embeddings | **excluded** — slide 3.5 |
| 3 | Attention readout instead of stoichiometry-weighted | **open** |
| 4 | Discards the 16-d port-pair edge features | **open** |
| 5 | 16 sampled sequences averaged, instead of 1 | **excluded** — slide 3.4 |

## Slide 3.4 — Factor 5: the 16-sequence ensemble is not doing the work

**What we ran.** The octamer normally samples 16 sequences per polymer and averages their
predictions. We retrained with **K = 1** — one fixed sequence per polymer, chosen before
training and never resampled — on the B split, both targets, three seeds. Everything else
identical. Pre-registered before running.

**Every metric, not just the new one.** Ablated minus baseline; S and D folds kept separate.

| Metric | EA, S folds | EA, D folds | IP, S folds | IP, D folds |
|---|---|---|---|---|
| Overall R² | −0.001 | −0.000 | −0.001 | −0.000 |
| RMSE | +0.003 | +0.000 | +0.002 | +0.000 |
| MAE | +0.003 | −0.000 | −0.000 | +0.000 |
| Group-mean R² | −0.002 | +0.000 | −0.001 | −0.001 |
| **ΔR²** | **+0.003** | **+0.011** | **−0.002** | **−0.003** |

Medians. Pre-registered ΔR² materiality threshold **±0.024**. No sign test reaches
significance — every p is 0.625 or 1.000, against minimum attainable values of 0.125 (4 folds)
and 0.0625 (5 folds).

**Figure:** per-fold dot plot, one panel per fold group, five metrics on the x-axis, dots at
the per-fold difference, zero line marked. Everything sits on the line.

*"Dropping from sixteen sampled sequences to one changed nothing measurable on any metric —
accuracy, error, chemistry placement or architecture recovery. The averaging is not where the
advantage comes from. That's a negative result and I'm reporting it as one."*

**Disclosed honestly:** the seed-stability half of the pre-registration used a criterion not
defined in advance, so that part is reported as inconclusive rather than as evidence. And IP
fold 7 is a 2-seed cell — one run is missing — so it is not on the intended 3-seed protocol.

## Slide 3.5 — Factor 2: position embeddings are not doing the work either

**What we ran.** The octamer gives each of its 8 chain slots a learned position vector. We
retrained with those removed (`--octamer_position_embeddings off`) on the A split, 9 folds,
both targets, three seeds — 54 runs, completed this week.

**State the mechanism precisely:** without position embeddings, slots holding the same monomer
get identical embeddings, but end slots and interior slots still differ through the chain's
path structure. This is a *reduction* of positional information, not its elimination. **Do not
call the ablated model "position-blind."**

**Pre-registered threshold: ±0.051 on ΔR²**, fixed before running.

**Every metric, per fold — ablated minus baseline**

*EA:*

| Metric | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | **Median** | better | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.035 | −0.008 | +0.003 | −0.006 | −0.002 | −0.005 | −0.002 | +0.007 | −0.000 | **−0.002** | 3/9 | 0.51 |
| RMSE | −0.060 | +0.012 | −0.007 | +0.011 | +0.002 | +0.008 | +0.001 | −0.016 | +0.000 | **+0.001** | 3/9 | 0.51 |
| MAE | −0.051 | +0.011 | −0.010 | +0.010 | +0.001 | +0.009 | +0.003 | −0.015 | +0.000 | **+0.001** | 3/9 | 0.51 |
| Group-mean R² | +0.037 | −0.008 | +0.003 | −0.006 | −0.001 | −0.005 | −0.007 | +0.007 | −0.000 | **−0.001** | 3/9 | 0.51 |
| **ΔR²** | −0.053 | −0.010 | −0.013 | −0.008 | −0.015 | +0.001 | +0.105 | +0.005 | −0.014 | **−0.010** | 3/9 | 0.51 |

*IP:*

| Metric | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | **Median** | better | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall R² | +0.088 | +0.023 | −0.001 | +0.005 | −0.032 | +0.002 | +0.001 | +0.007 | −0.002 | **+0.002** | 6/9 | 0.51 |
| RMSE | −0.028 | −0.032 | +0.003 | −0.004 | +0.013 | −0.002 | −0.002 | −0.014 | +0.002 | **−0.002** | 6/9 | 0.51 |
| MAE | −0.025 | −0.027 | +0.003 | −0.005 | +0.013 | −0.004 | −0.002 | −0.016 | +0.004 | **−0.004** | 6/9 | 0.51 |
| Group-mean R² | +0.092 | +0.023 | −0.001 | +0.006 | −0.032 | +0.000 | +0.001 | +0.007 | −0.003 | **+0.001** | 6/9 | 0.51 |
| **ΔR²** | −0.108 | −0.017 | −0.087 | −0.028 | +0.051 | +0.016 | +0.005 | −0.010 | +0.018 | **−0.010** | 4/9 | 1.00 |

**Figure:** per-fold dot plot, five metrics side by side, EA and IP panels, with the ±0.051
band shaded on the ΔR² panel. Almost every point inside.

*"Removing the position embeddings moved nothing — every metric's median is within four
thousandths of zero except ΔR², which moved by one hundredth against a threshold of five
hundredths I fixed before running. Factor 2 is excluded."*

**Three caveats to state:**

- **Scope is R1 only.** The B split has not been run; do not present this as closing the arm.
- **EA fold 6's +0.105 is not evidence.** It is the largest number in the table, but that
  cell's baseline ΔR² is −0.14 with an across-seed SD of 0.81. It is noise.
- **Fold 0 moves in opposite directions** on the two axes — accuracy improves, architecture
  recovery falls. On a single fold with these magnitudes, that is not a finding.

## Slide 3.5b — Why reporting all five metrics here matters

Notice what just happened across slides 3.4 and 3.5: **on both ablations, every metric agrees —
conventional and new alike, all say "no change".**

Contrast that with slide 1.7, where on the cross-scaffold folds the accuracy metrics say "tie"
and ΔR² says "clear separation".

**The line to say:** *"This is the answer to the obvious objection — that I invented a metric
that flatters my own model. If ΔR² simply reported differences wherever you pointed it, it
would have found something in these two ablations. It didn't. It agrees with accuracy when
there is genuinely nothing there, and separates from it only when there is something accuracy
cannot see."*

That is a stronger defence of the framework than anything in Section 1, and it comes free from
running the ablations properly.
## Slide 3.6 — Correction to flag: three factors remain, not two

Our pre-registration's outcome-3 text says the remaining candidates are factors 1 and 4. That
is an undercount — **factor 3, the attention readout, has never been tested.** I checked every
prediction directory; only two of the four cells exist:

| | Stoichiometry-weighted readout | Attention readout |
|---|---|---|
| **2-node graph** | HPG-hier — *have* | **arm C — never run** |
| **8-slot chain** | **arm D — never run** | octamer — *have* |

So factors 1, 3 and 4 are open, and 1 and 3 are confounded with each other. Arms C and D
separate them; pre-register the reading first. Arm D needs a code patch — `OctamerEncoder` is
only built when the readout is attention, so 8-slot + stoichiometric currently fails silently.

**This needs a dated addendum to the pre-registration**, not an edit.

## Slide 3.7 — The confound, and how Paper 2 resolves it

The benchmark's labels were computed on **8-unit chains averaged over up to 32 sampled
sequences**. The octamer uses an **8-slot chain, 16 sampled sequences, averaged**. Its
structure mirrors how the labels were made.

- **Favourable reading** — we encoded the right physics; the property genuinely is an average
  over chain arrangements.
- **Sceptical reading** — the advantage is alignment with this dataset's recipe. Real chains
  are hundreds of units long.

**No ablation on this dataset can separate these** — every label was made at 8 units, so
there is no variation to exploit.

### But a dataset can, and it is the one we already want to build

This is the answer to "can we design an ablation dataset?" — **yes, and it is change 3 on
slide 2.2.** Generate labels for a common subset of polymers at **8, 12, 16 and 24 units**,
then retrain the octamer at matching and mismatched chain lengths:

| If the octamer's advantage is… | Then as label chain length moves away from 8… |
|---|---|
| **protocol matching** | performance peaks sharply at 8 and degrades at 12, 16, 24 |
| **correct physics** | performance is flat, or improves with longer chains |

This is a clean, pre-registerable discriminator, and it costs nothing beyond the dataset we
are already proposing — the same monomers, the same pipeline, a subset run at four chain
lengths instead of one.

**So Paper 2 does two jobs:** it gives the field a benchmark with a continuous architecture
axis, and it gives us the only experiment that can resolve our own confound. That is a
stronger case for building it than either job alone.

### Cheaper things we can do first, on data we already hold

- **Test on a different dataset.** The glass-transition and block-copolymer phase datasets are
  in the repository. Their labels have nothing to do with 8-unit averaging. Days of GPU.
- **Probe the representation directly.** Freeze the model and test whether architecture can be
  read out of its internal representation by a linear classifier. Separates "predicts better"
  from "represents better", which is the actual claim. One afternoon.

---

## What to ask for

1. Approval to run the published-config wD-MPNN on the B split — 54 runs, closes Paper 1.
2. A decision on the Paper 2 dataset — ~8–9 kSU **CPU**, and it resolves our confound too.
3. A view on arms C and D: now, or after Paper 1 is submitted?

## What not to do

- Do not present the octamer result as the headline. It belongs to Paper 3.
- Do not quote the spread ratio as "higher is better" — it is two-sided.
- Do not claim significance on the B split; the design cannot reach p < 0.05 there.
- Do not present slide 1.8's projection as a result. It is an estimate from another split.
