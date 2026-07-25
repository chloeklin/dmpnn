# Paper Outline — A Diagnostic Framework for Polymer Representations

**Framing:** a *methods/framework* paper, not a model paper. The contribution is a reusable set of diagnostics that measure **what a polymer representation captures** — decomposed into physically meaningful, separable axes — rather than how accurate it is in aggregate. The EA/IP copolymer benchmark is the primary case study; one **external application** demonstrates the framework transfers beyond our own models and dataset.

**Two headline claims the paper must land:**
1. Aggregate accuracy (R²/MAE) conflates distinct capabilities: on this benchmark overall R² is ~96–99% a measure of *chemistry-baseline* prediction, so it is nearly blind to architecture recovery. A framework that separates these axes is necessary, not cosmetic.
2. The framework has diagnostic *power*: it does not just describe — it overturned one of our own interim conclusions (the residual ablation showed the architecture pathway *rescues* a weak backbone rather than causing failure). A diagnostic that changes a conclusion is the strongest advertisement for the diagnostic.

---

## Working titles
- "Beyond Accuracy: A Diagnostic Framework for What Polymer Representations Capture"
- "Decomposing Representation Quality in Copolymer Property Prediction"
- "Chemistry, Composition, Architecture: Diagnosing Where Polymer Models Succeed and Fail"

---

## Abstract (structure, ~200 words)
- Problem: representation comparisons in polymer ML rely on aggregate error, which conflates chemistry, composition, and architecture — capabilities that a target property weights very unequally.
- Contribution: a diagnostic framework decomposing prediction into a dominant *group-level* (chemistry/composition) factor and a low-variance *within-group* (architecture) factor, with tools for signal geometry, baseline-vs-deviation error attribution, magnitude calibration, rank ordering, effect-size and distribution-shift stressors, and component ablation.
- Case study: four representations (composition-fraction baseline, wDMPNN probabilistic graph, global-architecture embedding, chemistry-conditioned architecture residual) × three generalization splits × two targets (EA, IP).
- Key findings: overall R² is 96–99% chemistry; the best overall model (wDMPNN) is not the best architecture-recovery model (ChemArch); and the two encode partially complementary information.
- External validation: the same decomposition applied to [external model/dataset] recovers interpretable axes, showing the framework generalizes.
- Takeaway: representation design should be evaluated per-factor, and the diagnostics motivate pairing a strong chemistry encoder with explicit supervision of the low-variance factor.

---

## 1. Introduction
- Polymer property prediction is multiscale: monomer chemistry, composition, and sequence architecture (and, more broadly, MW, processing, morphology) all contribute, in property-dependent proportions.
- Current practice compares representations by aggregate R²/MAE. Problem statement: when one factor dominates the variance, aggregate accuracy is essentially a measure of that factor and is blind to whether a model captured the others — so "representation A beats B" can be true on overall accuracy yet reversed on the factor a designer actually cares about.
- Motivating observation (teaser of Fig. 1 result): on our benchmark the best-overall model and the best-architecture-recovery model are *different models*, and aggregate accuracy hides this entirely.
- Contribution list:
  1. A factor-decomposition framework (between-group vs within-group) with a battery of diagnostics that isolate distinct, physically meaningful capabilities.
  2. A full case study on EA/IP copolymers across 4 representations × 3 splits × 2 targets.
  3. Demonstration that the framework has diagnostic power — it corrected one of our own conclusions.
  4. An external application showing transferability, and design implications for the field.
- Scope note: framework applies to any property with a *groupable factor structure* (a dominant "context" factor + one or more lower-variance factors that vary within context). Architecture is our instance; the tools are general.

## 2. Background and related work
- Polymer representations: SMILES/BigSMILES descriptors; probabilistic/weighted graphs (wDMPNN); hierarchical polymer graphs (HPG, Han et al. 2025); composition-weighted embeddings with global or chemistry-conditioned architecture terms (our GlobalArch/ChemArch).
- Evaluation protocols: random vs structure-disjoint splits; leave-one-monomer-out; why split choice changes what is being measured.
- Adjacent methodology: variance-components / ANOVA decomposition; group-structured and worst-group evaluation; calibration and ranking metrics; ablation/attribution. Position our framework as importing these into polymer representation evaluation in a unified, physically grounded way.
- Gap: no standard way to ask "which *component* of the structure–property map did this representation learn?"

## 3. The diagnostic framework (core methods contribution)
Present each tool generically (equation + what it isolates + how to read it), independent of EA/IP.

- **3.1 Canonical decomposition.** Group key g = shared context (here: monomer_A, monomer_B, f_A, f_B; architecture excluded). Group mean ȳ_g; within-group deviation Δy = y − ȳ_g. Predicted analogues use predicted group means. Everything downstream is defined on this split.
- **3.2 Signal geometry.** Exact SS decomposition SST = SS_between + SS_within; report the fraction of variance each factor carries and the physical effect sizes. *Purpose: establishes how hard each sub-task is and whether a separate metric for the low-variance factor is even warranted.*
- **3.3 Baseline prediction.** Group-mean R²/MAE/calibration. *Isolates the dominant-factor capability (chemistry/composition extrapolation).*
- **3.4 Model-error attribution.** Exact split of each model's SSE into between-group and within-group parts. *Answers "where does this model's error live" — the primary tool for explaining aggregate rankings.*
- **3.5 Low-variance-factor fidelity.**
  - Calibration: regress Δŷ on Δy → slope + dispersion ratio (shrinkage vs faithful magnitude).
  - Ordering: within-group pairwise accuracy / Kendall τ (rank recovery; noise-robust, invariant to magnitude scaling).
  - *Note the diagnostic value of using both: calibration and ordering can disagree, and ordering settles ambiguity when the factor's signal-to-noise is low.*
- **3.6 Effect-size-resolved performance.** Bin by |Δy|; per-model error and sign accuracy per bin; pairwise model advantage vs effect size. *Reveals whether a model compresses large effects.*
- **3.7 Generalization stressors.** (a) Chemical novelty (fingerprint distance of held-out unit); (b) target-distribution shift (mean shift, std ratio, Wasserstein). *Distinguishes "novel structure" from "shifted target regime" as the driver of failure.*
- **3.8 Component ablation.** Run inference with/without a model component (here: pre/post architecture residual) to localize a failure to the backbone vs the added pathway. *The mechanism-isolation tool.*
- **3.9 Statistical protocol.** Paired per-fold tests; report distributions (median + mean), effect sizes, and win counts; explicit warnings on pooled-vs-per-fold aggregation and R²-denominator fragility under narrow test variance.
- **Table (methods):** each diagnostic → the question it answers → the quantity reported → how to read it. This table is the paper's reusable "recipe."

## 4. Experimental setup (case study)
- Dataset: copolymer EA/IP vs SHE (eV); DFT labels; matched-group structure (≥2 architectures per chemistry/composition context).
- Representations: Frac (composition baseline), wDMPNN (probabilistic graph, implicit architecture), GlobalArch (global architecture embedding), ChemArch (chemistry-conditioned architecture residual). One paragraph each on the inductive bias.
- Splits: Group-disjoint, Pair-disjoint, Monomer-heldout (9 folds); what each stresses.
- Validation of the evaluation pipeline (same rows, physical units, group key excludes architecture, ≥2-architecture groups) — brief, cite supplement.

## 5. Case study results (organized by the framework)
- **5.1 Signal geometry.** Architecture = 1–2% (EA) / 1.5–4.3% (IP) of variance; effect sizes ~0.02 eV vs 0.3–0.6 eV chemistry. → overall R² is intrinsically a chemistry metric; and IP carries ~2× more architecture signal than EA (seed for property-dependent factor importance). *Fig: variance decomposition; assets `02_variance_geometry/`.*
- **5.2 Aggregate ranking and its cause.** wDMPNN best overall on all splits; group-mean decomposition shows the LOMO advantage is chemistry-baseline extrapolation (group-mean R² 0.93 vs ≤0.02 for composition models), not architecture. *Assets `03_group_mean_prediction/`.*
- **5.3 Error attribution.** Under LOMO, 96% of ChemArch's error is between-group (chemistry) vs 68% for wDMPNN → ChemArch fails on chemistry regime, not architecture. *Assets `02_variance_geometry/model_error_decomposition.csv`.*
- **5.4 Architecture fidelity.** ChemArch best on calibration (slope/dispersion nearest 1) and ordering (pairwise accuracy highest on all six split×property cells); wDMPNN attenuates (LOMO slope 0.57). *Assets `04_architecture_calibration/`, `05_architecture_ordering/`.*
- **5.5 Effect-size dependence.** wDMPNN slightly better for the smallest deviations, ChemArch progressively better as |Δy| grows → wDMPNN compresses large architecture effects. *Assets `06_effect_magnitude/`.*
- **5.6 What makes a fold hard.** Novelty alone does not track difficulty (all held-out monomers OOD; weak Spearman); target-distribution shift does (fold 6 EA: +1.0 eV mean shift, 0.46 std ratio). *Assets `07_monomer_novelty/`, `08_target_shift/`.*
- **5.7 Absolute-error audit.** EA fold 6 is a genuine ~1 eV error (not denominator collapse); IP negatives are mild and partly denominator-driven → resolves how seriously to read negative R². *Assets `11_pathological_folds/`.*
- **5.8 Mechanism (the conclusion-flip).** Pre/post-residual ablation: backbone alone is far worse (LOMO overall R² −4 to −8); the architecture residual *rescues* it on most folds and also improves the group mean → the failure is the composition backbone, not the residual pathway; the "architecture" term is partly a chemistry correction. *Assets `13_chemarch_residual_ablation/`.* **Highlight this as evidence of the framework's diagnostic power.**
- **5.9 Complementarity.** ChemArch vs wDMPNN residuals only partially correlated (r≈0.40 overall, 0.25 on LOMO) → complementary information. *Assets `12_residual_correlation/`.*
- **5.10 Statistics.** Paired Wilcoxon: wDMPNN's overall/group-mean edge significant; ChemArch's ΔR² edge not significant under LOMO (significant only in EA ordering) — the clean architecture claim lives in Group/Pair-disjoint. *Assets `10_summary/statistical_comparisons.csv`.*

## 6. External application (transferability — the section that makes it a *framework*)
- Goal: apply the *same* decomposition to a model/dataset we did not design, and show it yields interpretable factor axes → the tools are not bespoke to our four models.
- Candidate targets (pick one, feasibility-ordered):
  1. **HPG-GAT (Han et al. 2025, public code/data at github.com/spark8ku/HPG)** on its copolymer subset — group by chemistry/composition, treat architecture (block/alternating/branched) as the within-group factor; report signal geometry, group-mean vs deviation attribution, calibration, ordering. Bonus: a hierarchical joint model as a contrast to our residual model.
  2. A public copolymer property dataset with matched architecture groups.
  3. Re-analysis of a published copolymer benchmark's released predictions.
- What to show: (i) the between/within variance split for that property (does the framework reveal a different factor balance?); (ii) whether that model's aggregate win is baseline- or deviation-driven; (iii) at least one case where the decomposition says something the paper's headline metric did not.
- Deliverable: a short "framework applied out-of-the-box" subsection + one figure mirroring Fig. of the case study.
- Honest note if only a partial external analysis is feasible: state scope explicitly; even one external property strengthens the generality claim substantially.

## 7. Discussion
- What the framework reveals that a benchmark cannot: the two-axes result; overall R² ≈ chemistry; failure localized to backbone via ablation; calibration vs ordering distinction.
- Design implications: (i) chemistry extrapolation needs a graph/substructure-sharing encoder; (ii) low-variance factors (architecture) are attenuated under an aggregate objective *even in a fully joint model* (wDMPNN) → preserving them requires explicit supervision or a component-aware objective, not just a "more joint" representation; (iii) factor importance is property-dependent (EA vs IP architecture share) → per-property weighting.
- Relationship to representation choices in the field (additive vs joint vs hierarchical): the diagnostics reframe the question from "which representation" to "which factor, supervised how."
- Limitations: LOMO rests on 9 outlier-prone folds; Δy signal-to-noise vs DFT label noise not yet bounded (ordering used as the noise-robust check); only two properties, so property-dependent importance is indicated, not proven; framework assumes a definable group key.

## 8. Conclusion
- A representation's aggregate accuracy is not its capability profile. The framework measures the profile, transfers across models and datasets, and reframes representation design around per-factor fidelity and supervision. Natural follow-on: a chemistry-strong encoder with explicitly supervised low-variance factors (forward-reference the λ-sweep study).

---

## Figure / table plan (mapping to existing assets)
- **Fig. 1 (hook):** overall vs architecture-recovery R² by split — best-overall ≠ best-architecture. (from generalization figures)
- **Fig. 2 (framework schematic):** group decomposition → the diagnostic battery (new schematic).
- **Fig. 3:** variance geometry, EA & IP. `02_variance_geometry/variance_decomposition_*.png`
- **Fig. 4:** error attribution stacked bars. `02_variance_geometry/error_decomposition_*`
- **Fig. 5:** calibration + ordering summary. `04_architecture_calibration/`, `05_architecture_ordering/`
- **Fig. 6:** target-shift vs difficulty; fold-6 case. `08_target_shift/`, `09_per_fold_case_studies/`
- **Fig. 7:** residual ablation (the conclusion-flip). `13_chemarch_residual_ablation/`
- **Fig. 8:** external application, mirroring Fig. 3–5 on the external model.
- **Table 1 (methods recipe):** diagnostic → question → quantity → interpretation.
- **Table 2:** per-split summary (overall, group-mean, ΔR², slope, dispersion, ordering, SSE fractions). `10_summary/diagnostic_summary.md`
- **Table 3:** paired statistics. `10_summary/statistical_comparisons.csv`

## Positioning / venue notes
- Frame as methodology (evaluation/diagnostics), not SOTA — reviewers reward a tool that changes a conclusion and generalizes.
- The external application and the methods recipe table are the two elements that convert "analysis of our models" into "a framework." Prioritize both.
- Release code as a small library (functions already modularized: validation, grouping, variance_geometry, error_decomposition, calibration, ordering, novelty, target_shift) with a one-call "run diagnostics on your predictions" entry point.
