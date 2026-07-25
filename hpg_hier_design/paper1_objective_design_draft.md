# Paper 1 — Working Draft / Concept Note

**Working title:** *Objective, Not Representation: Recovering Low-Variance Architecture Effects in Graph Polymer Property Prediction*

**Alternatives:**
- *A Variance-Decomposed Objective Recovers Sequence-Architecture Effects in Copolymer Property Prediction*
- *When the Loss, Not the Model, Is the Bottleneck: Supervising Low-Variance Structure–Property Factors*

**Status:** in-distribution result complete (Group-disjoint, fold 0); full sweep + the decisive Monomer-heldout test pending. This note is the paper skeleton, not a finished manuscript. Numbers marked *(pilot)* are single-fold, no seeds.

**Type:** methods / analysis paper for a cheminformatics or materials-informatics venue (not a SOTA-benchmark paper).

---

## Abstract (draft, ~200 words)

Machine-learning models for copolymer properties are compared almost entirely by aggregate error (R²/MAE). We show that on electron-affinity / ionization-potential prediction this aggregate is a near-pure measure of one factor: monomer chemistry and composition explain 96–99% of the target variance, while sequence architecture — the factor a designer often cares about — accounts for only 1–4% (~0.02 eV). Under a standard mean-squared-error objective the optimizer is therefore rewarded almost exclusively for chemistry, and architecture-induced variation is systematically under-fit. Using a between-/within-group variance decomposition as a diagnostic, we demonstrate that a graph model's weak recovery of architecture effects is a limitation of the *training objective*, not of representational capacity: freezing the representation and adding a within-group residual-variance loss term — derived directly from the decomposition and invariant to the chemistry baseline — recovers most of the architecture-recovery gap to a specialized architecture-conditioned model, at no cost to overall accuracy. We argue that low-variance structure–property factors are systematically attenuated under an aggregate objective even by fully-joint representations, and that preserving them requires explicit supervision rather than greater model expressivity. We release the diagnostic protocol as a reusable tool.

---

## Contributions

1. **Diagnosis (conceptual).** Via a between-/within-group variance decomposition and error attribution, we show that a graph polymer model's weak recovery of sequence-architecture effects is caused by the *training objective*, not by missing representational capacity. The evidence is a freeze-and-swap manipulation: with the representation and network held identical and only the loss changed, most of the recovery gap closes — so the information was already in the representation; the objective was not rewarding its use.
2. **Method.** We introduce a chemistry-baseline-invariant within-group loss, `L_within`, derived directly from the variance decomposition. It equals direct supervision of the within-group deviation (Δŷ→Δy), redistributes prediction within a chemistry group without moving the group mean, and produces a gradient near-orthogonal to the MSE gradient (cosine ≈ 0.04) — i.e. genuinely new optimization signal, not a rescaling of existing error.
3. **Result.** On EA/IP copolymers, freezing a wDMPNN representation and adding `L_within` recovers ~55–70% of the architecture-*ordering* gap to a specialized model, cuts within-group residual error ~47%, drives calibration slope to ≈1.0 (preserving magnitude, not shrinking), and leaves overall R² unchanged. As a single-variable manipulation this is causal/ablation-grade evidence.
4. **Tool + principle.** We release a reusable diagnostic protocol (variance geometry → error attribution → calibration/ordering) that determines, for any property with a groupable factor structure, whether a factor is under-captured and whether the cause is the objective or the representation. We argue the broader principle: low-variance structure–property factors are under-fit under an aggregate objective even by a joint representation, and need explicit supervision.

**Intro-ready one-liner:**
> Our contributions are: (i) we show that a graph polymer model's weak recovery of sequence-architecture effects is a limitation of the training objective, not of representational capacity; (ii) we introduce a chemistry-baseline-invariant within-group loss, derived from a between-/within-group variance decomposition, that explicitly supervises this low-variance factor; (iii) with the representation frozen, this objective recovers most of the architecture-recovery gap to a specialized model at no cost to overall accuracy; and (iv) we release a reusable diagnostic protocol and argue that low-variance structure–property factors require explicit supervision rather than greater model expressivity.

---

## 1. Introduction (framing)

- Polymer property prediction is multiscale: monomer chemistry, composition, and sequence architecture all contribute, in property-dependent proportions.
- Current practice compares representations by aggregate R²/MAE. **Problem:** when one factor dominates the variance, aggregate accuracy is essentially a measure of that factor and is nearly blind to whether the model captured the others.
- **Motivating observation (Fig. 1):** on EA/IP copolymers, chemistry/composition is 96–99% of the variance and sequence architecture is 1–4% (~0.02 eV). So overall R² ≈ a chemistry metric, and a model can win overall while barely capturing architecture.
- **The reframe (the paper's thesis):** the standard response to "our model doesn't capture architecture" is to build a more expressive representation. We show the binding constraint is instead the *objective* — standard MSE allocates optimization pressure in proportion to variance share, so a 1–4% factor is under-fit regardless of whether the representation encodes it. The fix is to supervise the factor explicitly, not to enlarge the model.
- **Why it matters:** architecture (sequence) is exactly the design lever polymer chemists manipulate; a model that predicts the right chemistry regime but flattens architecture is of limited use for design.

## 2. Related work & positioning

- **Polymer representations:** SMILES/BigSMILES descriptors; weighted/probabilistic graphs (wDMPNN); hierarchical polymer graphs; composition-weighted embeddings with global or chemistry-conditioned architecture terms.
- **Evaluation protocols:** random vs structure-disjoint splits; leave-one-monomer-out; how split choice changes what is measured.
- **Adjacent ML methodology — must differentiate explicitly:** multi-task / imbalanced loss balancing (uncertainty weighting [Kendall & Gal], GradNorm, PCGrad / gradient surgery) and imbalanced regression (e.g. Yang et al., *Delving into Deep Imbalanced Regression*, 2021); variance-components / ANOVA decomposition; group-structured and worst-group evaluation.
- **Our differentiation:** `L_within` is not a generic reweighting heuristic — it is *derived from a physically meaningful variance decomposition* in which the under-weighted factor is scientifically named (sequence architecture). The contribution is the diagnosis (objective vs representation) and a factor-structured objective, not the invention of loss weighting.
- **Gap:** no standard way to ask "which *component* of the structure–property map did this representation learn, and if it didn't learn one, is that the model's fault or the loss's fault?"

## 3. Method

### 3.1 Group structure and the decomposition
Matched group `g` = fixed chemistry/composition context, key `(monomer_A, monomer_B, f_A, f_B)`, **architecture excluded**; keep groups with ≥2 architectures.
- True/predicted group means: ȳ_g, ŷ̄_g.
- Within-group (architecture) deviations: Δy = y − ȳ_g, Δŷ = ŷ − ŷ̄_g.
- Exact variance split: SST = SS_between (chemistry) + SS_within (architecture). Report the fraction each factor carries (establishes the 96–99% / 1–4% split).

### 3.2 The objective
```
L = L_overall + λ · L_within
```
- `L_overall` = standard MSE.
- `L_within` = within-group variance of the prediction residuals, normalized by within-group target variance so λ is interpretable.
- **Key identity:** within a group, Var(residual) = mean[(Δŷ − Δy)²], so `L_within` is exactly the MSE of predicted vs true within-group deviations — the loss-form of the between/within error decomposition.

### 3.3 Properties (verified before training)
- Invariant to chemistry-baseline shifts (adding a constant to a group's predictions leaves `L_within` unchanged).
- Within-group gradients sum to ≈ 0 → redistributes prediction within a group without moving the group mean.
- Gradient near-orthogonal to the MSE gradient (cosine ≈ 0.04) → genuinely new signal.
- Backward compatible at λ = 0.
- **Group-aware sampler** keeps whole chemistry groups intact within a batch (required, since `L_within` needs ≥2 architectures of a group in the batch).
- **Normalizer** floored/pooled (per-group within-target variance is estimated from 2–3 points and is noisy).

## 4. Experimental setup

- **Dataset:** `data/ea_ip.csv` — 42,966 copolymers, targets EA and IP vs SHE (eV), DFT labels. Group key `(smiles_A, smiles_B, fracA, fracB)` → 18,414 chemistry/composition contexts. Architecture = `poly_type` ∈ {block, random, alternating}. **Structure (verified):** every group has 2–3 architectures, one sample per architecture (no replicates); 12,276 groups are block+random, 6,138 add alternating; 100% of samples live in ≥2-architecture groups. Consequence: `L_within`'s signal is dominated by the block-vs-random contrast; alternating is a minority third signal.
- **Models:** Frac (composition baseline), wDMPNN (probabilistic graph, implicit architecture), GlobalArch (global architecture embedding), ChemArch (chemistry-conditioned architecture residual). The pilot modifies **wDMPNN only**; ChemArch is the specialized reference for the recovery target.
- **Splits:** Group-disjoint, Pair-disjoint, Monomer-heldout (9 folds).
- **Metrics (defined exactly in Appendix A):** overall R²; group-mean R² (chemistry baseline); architecture-deviation R² (ΔR²) and pairwise ordering accuracy (architecture recovery); calibration slope/dispersion; within-group residual variance.

## 5. Results

### 5.1 Signal geometry (motivation)
Architecture is 1–4% of variance vs 96–99% chemistry; effect sizes ~0.02 eV (median |Δy| = 0.021 eV, EA) vs 0.3–0.6 eV chemistry → overall R² is intrinsically a chemistry metric; a separate architecture metric is necessary, not cosmetic.

### 5.2 The λ-sweep — *(pilot: wDMPNN · EA · Group-disjoint · fold 0; single fold, no seeds)*

| λ | Overall R² | Ordering acc. | Within-grp resid. var. | Calib. slope |
|---|---|---|---|---|
| 0.00 (baseline) | 0.99690 | 0.85356 | 0.000513 | 0.993 |
| 0.03 | 0.99742 | 0.88044 | 0.000316 (−38%) | 1.003 |
| 0.30 | 0.99652 | 0.88777 | 0.000270 (−47%) | 1.000 |
| *ChemArch (ref.)* | — | *0.90259* | — | — |

- Overall prediction unchanged (R² ≈ 0.997 throughout) → chemistry not sacrificed.
- Ordering rises +2.7 to +3.4 pp, closing ~55–70% of the baseline→ChemArch gap — with the representation frozen.
- Within-group residual error cut ~38–47% (the targeted component).
- Calibration slope → ≈1.0 (not below) → magnitude preserved, not shrunk.
- Ordering is the headline metric: magnitude-invariant and noise-robust, so it is the cleanest evidence the model learns real architecture signal rather than fitting label noise.

### 5.3 Planned (to complete the paper)
- Full Group-disjoint sweep, all folds, ≥3 seeds, paired significance vs baseline.
- λ=0.10 (fills the Pareto curve).
- **Monomer-heldout — the decisive test:** does the objective still recover architecture *and* keep chemistry extrapolation on unseen monomer chemistry? This determines whether the claim is "objective recovers architecture in-distribution" or the stronger "objective gives a graph model best-of-both."
- Sampler-confound baseline (λ=0 with the group-aware sampler vs original shuffling).
- Δy noise-floor bound (external DFT/ensemble estimate) — the gate on how far the architecture claim can be pushed.
- Stratified metrics: 2-arch (block/random) vs 3-arch groups.

## 6. Figures & tables plan

- **Fig. 1 (hook):** variance geometry — chemistry 96–99% vs architecture 1–4%. `02_variance_geometry/variance_decomposition_*.png`
- **Fig. 2:** why ChemArch recovers and wDMPNN shrinks — Δŷ vs Δy calibration, wDMPNN vs ChemArch. `pilot_figures/cal_wdmpnn.png`, `cal_chemarch.png`
- **Fig. 3 (main result):** λ-sweep — ordering ↑ toward ChemArch, overall R² flat, within-var ↓. `pilot_figures/figD_lambda_sweep.png`
- **Fig. 4:** gradient diagnostic — cosine(L_within, L_overall) ≈ 0.04 + within-group gradient-sum ≈ 0.
- **Fig. 5:** noise-floor — |Δy| distribution vs measured DFT/label-noise band. `pilot_figures/figF_noise_floor.png` (band currently a placeholder — replace with measured value).
- **Fig. 6:** per-fold Monomer-heldout — combined objective vs baseline vs ChemArch, seed error bars *(pending)*.
- **Table 1:** metric definitions (recipe). **Table 2:** per-split summary. **Table 3:** paired statistics.

## 7. Limitations & threats to validity

- **In-distribution only, so far.** The result is Group-disjoint, where chemistry is already solved — it establishes the *mechanism*, not transfer to unseen chemistry. Monomer-heldout is required before the headline claim generalizes.
- **Noise floor unbounded.** Architecture is ~0.02 eV with no within-architecture replicates, so signal and noise can't be separated from the data alone; `L_within` could partly fit label noise. Ordering (noise-robust) mitigates but does not replace an external noise bound.
- **Single fold / no seeds (pilot).** Current numbers are one fold; the ~0.03 ordering gain needs ≥3 seeds to clear run-to-run noise.
- **Thin, lopsided groups.** 2–3 architectures/group, no replicates, two-thirds block-vs-random only — the objective's signal is dominated by one contrast; alternating is under-represented.
- **One dataset, two properties.** The general principle (contribution 4) is demonstrated, not proven; frame as hypothesis-generating.
- **Proximity to loss-balancing literature.** Novelty must be defended on the physical grounding + the diagnosis, not on loss weighting per se.

## 8. Readiness

≈ 55%. The diagnostic framework, metrics, dataset analysis, and an in-distribution positive result are done. Remaining: full-fold + seeded sweep, Monomer-heldout, noise floor, sampler baseline. Estimated ~6–10 weeks of experiments to a submittable draft (methods/analysis sections come largely from the existing diagnostic report).

## 9. Target venues & timeline

- **Workshop (fall 2026)** for early visibility / citable checkpoint: NeurIPS AI4Science, ICML ML4Materials, or ELLIS ML4Molecules.
- **Journal (Q4 2026 → Q1 2027):** *Digital Discovery* (RSC) or *npj Computational Materials* preferred; *JCIM* as a solid, faster alternative.
- Only target a main ML track if the principle generalizes across multiple datasets (a larger, riskier paper — that's Paper 2 territory).

---

## Appendix A — Metric definitions (exactly as computed)

**Matched group** `g`: fixed `(monomer_A, monomer_B, f_A, f_B)`, architecture excluded, ≥2 architectures. ȳ_g = mean of true y in g; ŷ̄_g = mean of predicted y in g.

- **Group-mean R² (chemistry baseline):** one point per group, `R² = 1 − Σ_g (ȳ_g − ŷ̄_g)² / Σ_g (ȳ_g − ⟨ȳ⟩)²` (`sklearn.r2_score(y_bar_true, y_bar_pred)`; ≥3 groups). *→ why wDMPNN wins overall.*
- **Architecture-deviation R² (ΔR²):** per sample, `Δy = y − ȳ_g`, `Δŷ = ŷ − ŷ̄_g`, `R²(Δŷ vs Δy)`. *→ the recovery claim; the pilot's target.*
- **Ordering accuracy:** fraction of within-group architecture pairs ranked in the correct order (chance 0.5); magnitude-invariant, noise-robust. *→ the headline pilot metric.*
- **Calibration:** regress Δŷ on Δy → slope + dispersion ratio SD(Δŷ)/SD(Δy); ≈1 = faithful magnitude, <1 = shrinkage.
- **Within-group residual variance:** Var over each group of (ŷ − y); the quantity `L_within` minimizes.

*Note:* Overall R² ≈ Group-mean R² on this benchmark because chemistry is 96–99% of the variance — which is exactly why a separate ΔR²/ordering is required.
