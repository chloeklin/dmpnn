# Supervisor update — 29 July 2026

*Covers work since the 22 July figures. Chloe Lin.*

---

## Summary

While preparing multi-seed error bars I ran a reproduction check on an existing result. It failed.
Investigating that failure uncovered a **model-selection bug present in every training runner since
each was first written**, and, separately, **run-to-run variance large enough that most of our
reported differences are not measurable at one run per configuration**.

Both are now fixed. All experiments are being regenerated with three seeds and error bars.

The investigation also produced two findings that are independent of the bug and that I think
strengthen the paper: a **null-floor calibration** showing that one of our headline metrics has very
little headroom, and a **dataset-design analysis** that identifies which chemical axis each of our
splits actually tests. Details in §3.

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

## 4. What is running

| stage | description | runs | ~GPU h | status |
|---|---|---|---|---|
| R1 | A-heldout regeneration, 5 models × 2 targets × 9 folds × seeds 42/43/44 | 270 | ~300 | pilot submitted |
| R2 | B-heldout, random folds | 216 | ~240 | queued behind R1 |
| R3 | B-heldout, clustered folds | 216 | ~240 | pending |
| R4 | ChemArch/GlobalArch/frac regeneration | 162 | ~180 | only if we keep that comparison |

R1 restores everything the 22 July figures were based on, with error bars. R2 is new science.

## 5. What I would like to discuss

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

## 6. Figures from 22 July that are superseded

Anything derived from single-run seed-42 predictions, in particular the EA LOMO fold-1 analyses
(`15_ea_lomo_fold1_parity_hpg_vs_wdmpnn.png`, `16_ea_lomo_fold1_group_vs_deviation.png`,
`17_lomo_fold1_target_shift_EA_IP.png`) and the chemistry panels (`fig2_chemistry.png`,
`fig2b_chemistry_mean_median.png`, `01_group_mean_r2.png`). The architecture panels
(`fig3_architecture.png`, `02_architecture_delta_r2.png`, `03_ordering_accuracy.png`) are not
disproven, but are single-run and unverified pending R1.
