# Supervisor update — 29 July 2026

*Covers work since the 22 July figures. Chloe Lin.*

---

## Summary

While preparing multi-seed error bars I ran a reproduction check on an existing result. It failed.
Investigating that failure uncovered a **model-selection bug present in every training runner since
each was first written**, and, separately, **run-to-run variance large enough that most of our
reported differences are not measurable at one run per configuration**.

Both are now fixed. All experiments are being regenerated with three seeds and error bars.

The investigation also produced three findings that are independent of the bug and that I think
strengthen the paper: a **null-floor calibration** showing that one of our headline metrics has very
little headroom, a **dataset-design analysis** that identifies which chemical axis each of our splits
actually tests, and the observation that the **B-monomer space is dominated by two scaffold families**
(62.5% of 682 monomers), which changes what any split of this benchmark can claim to measure. Details
in §3.

§4 sets out the new monomer-B split in full — what the data looks like on each axis, why the
9-monomer A axis makes a weak EA test, how the B split is constructed, and how results from the two
splits must be interpreted differently rather than pooled.

**Net position:** the model results shown on 22 July cannot currently be defended and are being
re-measured. The methodological contribution is larger than it was a week ago.

---

## 1. The bug

Every runner (`hpg_hier`, `wdmpnn`, `stage2d` for ChemArch/GlobalArch/frac, and the legacy path)
saved the best-validation checkpoint during training and then **predicted from the final model after
early-stopping patience expired**, discarding the checkpoint it had selected. Confirmed by git blame
to be present in each runner's initial commit, so no existing prediction file predates it.

Measured cost for HPG-hier: **mean +0.048 eV MAE** (median +0.038, max +0.114) relative to the
checkpoint that should have been used. That is larger than most differences we have been reporting
as results.

Fixed in all five runners; best-checkpoint prediction is now the default and both best and final
predictions are saved so the cost can be quantified per model family. Preliminary smoke runs suggest
the bug damaged HPG-hier substantially more than wDMPNN or ChemArch — if that holds at full training,
our previous comparisons **understated** our own model.

## 2. Run-to-run variance

Six V100 runs, same model, same seed, same split, same code, varying nothing but the process:

| EA fold | group-mean R² across 3 runs | MAE (eV) | SD |
|---|---|---|---|
| 0 | 0.962 / 0.982 / 0.986 | 0.084 / 0.055 / 0.052 | 0.018 |
| 1 | **0.450 / 0.790 / 0.978** | 0.226 / 0.146 / 0.045 | **0.091** |

**What this invalidates.** The EA fold-1 "hard monomer" story — that dibenzothiophene sulfone exposes
a Stage-1 chemistry weakness — does not survive. The unmodified baseline reaches 0.978 on that fold
roughly one run in three. The recorded 0.575 was one draw. Consequently:

- the junction-coupling programme, which existed to fix that fold, has no established target;
- the octamer's EA chemistry headline is largely carried by fold 1 (+0.414 of a +0.031 median gain);
- the "pathological folds" we kept returning to (EA 1, EA 6, IP 5, IP 2) are now suspected to be
  simply the high-variance folds rather than chemically interesting ones.

**Fix.** Averaging predictions across repeats improved every metric in every cell tested (6/6 on MAE,
24/24 overall). The regeneration protocol is: unchanged validation design, best checkpoint, and the
**mean prediction of three seeds** as the reported result, applied identically to every model.

I tested two alternative fixes (row-level validation; a minimum-epoch floor) and rejected both —
row-level validation destabilised a different fold and would require defending a validation/test
mismatch; the epoch floor targeted a mechanism the data did not support.

## 3. Two findings that survive, and improve the work

### 3.1 Our EA chemistry metric has almost no headroom

I built a **null predictor that ignores the held-out monomer entirely** — it predicts each test row
from the training mean of its (partner monomer, composition, architecture) cell.

| split | target | null floor (median group-mean R²) | headroom |
|---|---|---|---|
| A-heldout | EA | **0.676** | 0.324 |
| A-heldout | IP | −0.034 | 1.034 |

On EA fold 2 the null **beats HPG-hier** (0.961 vs 0.922). The cause is structural: within an
A-heldout fold the held-out monomer is constant, so all variation across groups comes from the
partner monomer and the composition — all of which the model has seen.

The IP metric is sound. The EA one, on this split, is not, and that includes the "wDMPNN wins EA
chemistry, 0.965 vs 0.925" comparison.

I think reporting group-wise OOD metrics against a null floor is a contribution in its own right —
it is not standard practice and it changed our reading of our own results.

### 3.2 The dataset design tells us which split tests which target

The dataset is an exact factorial: **9 A monomers × 682 B monomers × 7 composition/architecture
cells = 42,966 rows**. Consequences:

- Architecture accounts for **0.98% (EA) / 1.46% (IP)** of total variance; monomer identity ~90%.
  This confirms the figure we had assumed.
- A identity carries 42–50% of variance across only **9** monomers — and because the split holds out
  one for test and one for validation, models train on **7 donor monomers**. Our "unseen chemistry"
  results are 7-example extrapolations. This is the most likely reason three separate representation
  changes (wedge, junction coupling, octamer) all failed to move the architecture axis.
- I have constructed and frozen a **B-heldout split** (76 of 682 B monomers per fold, matched fold
  sizes, both random and scaffold-clustered assignments, with nearest-neighbour similarity calibration
  between splits). Its null floors invert: EA headroom 0.580, IP 0.440.

**So EA chemistry claims belong on the B split and IP chemistry claims belong on the A split** — a
two-regime structure inside data we already have. It also enables a performance-versus-novelty curve,
which the current split cannot produce (one held-out monomer per fold, so no novelty variation).

### 3.3 The B-monomer space is dominated by two scaffold families

Murcko clustering of the 682 B monomers gives 112 scaffolds — but **one has 317 members (46%) and a
second has 109 (16%)**. Together they are 62.5% of the monomer space; 64 scaffolds are singletons.

A strictly scaffold-disjoint nine-fold split is therefore impossible: the largest family alone
exceeds any balanced fold. Our clustered split is capacity-balanced scaffold packing, and the folds
end up testing two different things — folds 0–3 hold out 76 members of the 317-family while 241 of it
remain in training (substituent interpolation), whereas folds 7–8 hold out whole small families with
zero training overlap (genuine new chemistry). The analysis now derives this grouping from the frozen
split and runs paired tests within group rather than pooling nine non-exchangeable folds.

**This is a property of the benchmark, not of our split.** Any random split of the Aldeghi–Coley
EA/IP dataset is mostly measuring substituent interpolation on two familiar cores, regardless of what
it is described as testing. I have not seen this stated anywhere in the literature that uses this
dataset, and I think it is worth reporting in its own right.

## 4. The two splits — data structure, design, and how we read them differently

### 4.1 What the data actually looks like on each axis

|  | monomer A | monomer B |
|---|---|---|
| distinct monomers | **9** | **682** |
| rows per monomer | 4,774 | 63 |
| chemical role | bis-boronic acid donors | comonomer partners |
| share of EA variance | 0.418 | 0.457 |
| share of IP variance | 0.500 | 0.329 |
| spread of monomer means, EA | 1.36 eV (−3.00 → −1.64) | 2.93 eV (−3.07 → −0.14) |
| spread of monomer means, IP | 1.18 eV (0.84 → 2.01) | 1.76 eV (0.47 → 2.23) |
| distinct Murcko scaffolds | 9 closely related | 112, two dominating |

The two axes carry **comparable variance** — A a little more for IP, B a little more for EA — but they
carry it across radically different numbers of monomers. That asymmetry, not the variance share, is
what drives every design decision below. The A and B sets are disjoint, and the design is an exact
factorial: every one of the 6,138 A/B pairs appears exactly 7 times (25/75 block + random, 50/50
alternating + block + random, 75/25 block + random).

### 4.2 Why the A-heldout split is a weak test — specifically for EA

Three consequences follow from having only 9 A monomers:

1. **The training set contains 7 donors.** The split excludes the test A *and* a second A for
   validation. Our "unseen chemistry" results are 7-example extrapolations.
2. **Within a fold, A is constant.** Every test row shares the held-out A, so all variation *across*
   groups comes from B, composition and architecture — all of which the model has seen thousands of
   times. An error on the held-out A therefore enters group-mean R² as a near-constant offset rather
   than as scatter.
3. **Novelty is a single number per fold.** One held-out monomer means no within-fold variation in
   similarity-to-training, so no performance-versus-novelty analysis is possible.

Point 2 is what the A-blind null measures. That null predicts each test row from the training mean of
its (B, composition, architecture) cell — averaging over the other 8 A monomers, using **no**
information about the held-out one.

| split | target | null floor (median group-mean R²) | headroom to 1.0 |
|---|---|---|---|
| A-heldout | EA | **0.676** | 0.324 |
| A-heldout | IP | −0.034 | **1.034** |

**Why EA and IP diverge so sharply.** The null's error is essentially the held-out A monomer's own
offset. Whether that offset matters depends on its size relative to the variation *across* groups
within the fold, which is driven by B. For EA the B spread is large (2.93 eV) and the A offsets are
comparatively small, so the null reconstructs most of the across-group signal and scores 0.676 — on
fold 2 it beats HPG-hier outright (0.961 vs 0.922). For IP the relationship inverts: A carries more
variance (0.500) and the B spread is smaller (1.76 eV), so the missing A offset dominates and the null
collapses to zero. **The A-split is a sound test of IP chemistry and a near-degenerate test of EA
chemistry.**

### 4.3 How the B-heldout split is constructed

- The 682 B monomers are partitioned into **9 folds of 75–76**.
- For fold *k*: test = fold *k*'s monomers, **validation = fold *k+1*'s monomers** (cyclic, so
  validation is B-disjoint from both train and test), train = the remaining ~530.
- Row counts **33,390 / 4,788 / 4,788** — the test size is deliberately matched to the A-split's
  4,774, so fold-level metrics from the two splits sit on comparable footing.
- All 9 A monomers appear in train, validation and test. Only the B axis is held out.
- The split seed is fixed at 42 and is **independent of the model seed**; fold membership is frozen to
  JSON and re-asserted at the start of every training run, so folds cannot drift across seeds.

Two assignments were built and frozen, because monomer count alone doesn't determine difficulty:

| assignment | median nearest-neighbour Tanimoto | worst-case contamination |
|---|---|---|
| A-heldout (reference) | 0.31 – 0.47 | none by construction |
| B random | 0.52 – 0.58 | 4 folds contain a training monomer at Tanimoto **1.00** |
| **B clustered (primary)** | 0.48 – 0.50 | 2 folds contain one at 0.95 |

Holding out 76 of 682 monomers at random leaves close analogues in training, so the random split is
**easier** than the A split and its absolute scores are not directly comparable to it. The clustered
assignment is the difficulty-matched one, which is why it is running first as R3. Six held-out
monomers across the whole dataset have a training near-duplicate at ≥ 0.95, so every B-split metric is
reported twice — full folds and with those monomers filtered out.

### 4.4 The scaffold structure forces an unusual fold design

Murcko clustering of the 682 B monomers gives 112 scaffolds, but **one has 317 members (46%) and a
second has 109 (16%)**; 64 are singletons. A strictly scaffold-disjoint nine-fold split is therefore
**impossible** — the largest family alone exceeds any balanced fold capacity. The split is
capacity-balanced scaffold packing, and the two large families must be split across folds.

The consequence is that the nine folds are **not exchangeable**:

| folds | held out | same-scaffold monomers left in training | what it tests |
|---|---|---|---|
| 0–3 | 76 members of the 317-family each | 241 | new substituents on a familiar core |
| 5 | 76 members of the 109-family | 0 (the rest are in its validation fold) | new chemistry, but **one** family only |
| 4, 6 | 26–29 families each | 13 and 33 | mixed |
| 7, 8 | 28–29 families each | **0** | new chemistry, the cleanest tests |

The analysis derives this grouping from the frozen split rather than assuming it, labels folds
**S (within-scaffold)** or **D (cross-scaffold)** by whether more than half their held-out monomers
have a same-scaffold analogue in training, and flags chemically homogeneous folds separately.

### 4.5 How we interpret the two splits differently

| | A-heldout | B-heldout (clustered) |
|---|---|---|
| valid chemistry test for | **IP** (floor ≈ 0) | **EA** (floor 0.38, headroom 0.62) |
| meaning of "unseen chemistry" | extrapolation from **7** examples | generalisation from **~530** |
| held-out chemistries per fold | 1 | 26–29 (or 1 in folds 0–3, 5) |
| folds exchangeable? | broadly yes — 9 single donors | **no** — two structural groups |
| paired test | one 9-fold sign test, minimum p = 0.0039 | within group: 4–5 folds, minimum p = 0.125 / 0.0625 |
| novelty analysis | impossible | performance-versus-novelty curve across 76 monomers/fold |
| contamination handling | none needed | filtered and unfiltered reporting |

Three practical consequences for how results are read:

1. **We do not quote a single headline number per split.** EA claims come from the B split, IP claims
   from the A split, and every group-mean R² is reported beside its fold-specific null floor together
   with the fraction of available headroom it closes.
2. **The B split trades statistical power for interpretability.** Splitting nine folds into groups of
   five and four means no within-group comparison can reach conventional significance on its own. That
   is a real cost, and it is preferable to a significant-looking result obtained by pooling folds that
   answer different questions.
3. **A win means different things in different folds.** On folds 7–8 it is generalisation to unseen
   scaffolds; on folds 0–3 it is interpolation across substituents on a core the model knows well. Both
   are worth measuring — they are different capabilities — but they must not be averaged into one
   number and called chemistry extrapolation.

## 5. What is running

| stage | description | runs | ~GPU h | status |
|---|---|---|---|---|
| R1 | A-heldout regeneration, 5 models × 2 targets × 9 folds × seeds 42/43/44 | 270 | ~420 | **pilot verified**, remainder submitted |
| R3 | B-heldout, clustered folds | 216 | ~340 | **pilot verified**, remainder submitted |
| R2 | B-heldout, random folds | 216 | ~340 | deferred — clustered is the difficulty-matched split |
| R4 | ChemArch/GlobalArch/frac regeneration | 162 | ~250 | only if we keep that comparison |

R1 restores everything the 22 July figures were based on, with error bars. R3 is new science.

**Pilot verification (14 jobs) passed on both.** Regenerated predictions differ from their predecessors
by 0.18–0.58 eV, confirming the old code path is not being reused; split hashes are identical across
seeds; held-out B monomer sets are properly disjoint. Observed wall time is 1.56 h/job, so the
programme is larger than first estimated — R2 has been deferred on that basis.

One encouraging early signal, on four pilot runs only and not to be read as a result: on the clustered
B split, EA group-mean R² came in at 0.93–0.98 against a B-blind null floor near 0.38. The equivalent
on the A split is 0.925 against a floor of 0.676. If it holds, the B split gives EA chemistry a metric
with real headroom, which is what §3.2 predicted.

A run-quality rule (flag any run whose best epoch is below 10 as potentially undertrained; report
results with and without) was **pre-registered and committed before the bulk results landed**, so the
exclusion cannot be post-hoc.

## 6. What I would like to discuss

1. **Scope of the paper.** Three representation variants have now produced no movement on the
   architecture axis, on a split with 7 training donor monomers and ~1% architecture variance. The
   measurement contributions (diagnostic decomposition, null-floor calibration, split design) look
   stronger than the model contribution. Should the paper lead with the methodology and treat
   HPG-hier as the case study?
2. **Whether to keep the ChemArch/GlobalArch comparison** (R4, 162 runs) or cut that section.
3. **Curtis 2025.** I had planned to port the Stage-2 encoder to it. On reflection I don't think it
   works: its monomers are featureless A/B beads, so Stage-1 collapses to two learned embeddings and
   the model becomes a generic binary-sequence encoder competing against their RNN. It would not
   validate the hierarchy. I propose dropping it.
4. **Timeline.** Roughly one to two weeks of compute and re-analysis before the results are back on
   defensible footing.

## 7. Figures from 22 July that are superseded

Anything derived from single-run seed-42 predictions, in particular the EA LOMO fold-1 analyses
(`15_ea_lomo_fold1_parity_hpg_vs_wdmpnn.png`, `16_ea_lomo_fold1_group_vs_deviation.png`,
`17_lomo_fold1_target_shift_EA_IP.png`) and the chemistry panels (`fig2_chemistry.png`,
`fig2b_chemistry_mean_median.png`, `01_group_mean_r2.png`). The architecture panels
(`fig3_architecture.png`, `02_architecture_delta_r2.png`, `03_ordering_accuracy.png`) are not
disproven, but are single-run and unverified pending R1.
