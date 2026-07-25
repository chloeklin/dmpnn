# HPG-hier: Implementation & Seed-42 Evaluation

*Progress report — [date]. Point estimates from a single seed (42); error bars (seeds 43/44) to follow.*

---

## Summary

I implemented **HPG-hier**, a true two-stage hierarchical polymer model, and evaluated it head-to-head against **wDMPNN** and **ChemArch** on EA/IP across three generalization splits (seed 42, all folds, both targets, one shared pipeline). **Result: HPG-hier recovers wDMPNN's chemistry extrapolation to unseen monomers (IP at parity, EA within ~4% on median) while improving architecture recovery over wDMPNN on every split. It is the strongest all-round model** — unlike ChemArch it does not collapse on unseen chemistry, and unlike wDMPNN it captures architecture well.

---

## 1. What I implemented — HPG-hier

A genuine **two-stage** hierarchy (the original HPG fuses everything into one graph; this separates the two levels into two message-passing stages):

- **Stage 1 — per-monomer chemistry (shared encoder).** Each monomer's atom graph is encoded independently with a shared chemprop D-MPNN encoder → one embedding per monomer. Monomers are featurized from the **repeat units in `WDMPNN_Input`** (with attachment points), *not* `smiles_A/smiles_B` (which are the pre-polymerization monomers with reactive leaving groups that aren't in the polymer). Attachment atoms carry a local `is_attachment` + port flag; their features are computed with the connecting bond counted, matching wDMPNN.
- **Stage 2 — monomer-level architecture (separate encoder).** A small directed graph over the monomers (A→A, A→B, B→A, B→B + self-loops). Edge features come **directly from the `WDMPNN_Input` bond rules** — the same connectivity/architecture information wDMPNN is trained on — split into a port-pair connectivity component and an aggregated transition-weight component. Separate message-passing weights from Stage 1.
- **Readout:** stoichiometry-weighted pooling of the updated monomer embeddings → MLP → EA/IP.
- **Objective:** plain MSE (the within-group loss is a separate line of work, not applied here).
- Built in chemprop (same framework as the baselines) so the diagnostics score all models identically.

## 2. Evaluation setup (seed 42)

- **Models:** HPG-hier, wDMPNN, ChemArch — all trained through the **same seeded generalization pipeline** (uniform provenance) at seed 42.
- **Splits:** Group-disjoint (5 folds), Pair-disjoint (5), **Monomer-heldout** (9 — the decisive test: copolymers whose monomer was never seen in training). Both targets (EA, IP). 114 cells, all completed and validated.
- **Metrics (diagnostic battery):** *group-mean R²* = chemistry-baseline extrapolation (architecture averaged out); *ΔR²* and *pairwise ordering* = within-group architecture recovery; *calibration slope* = architecture-magnitude fidelity. Aggregated across folds as **medians** (Monomer-heldout has known outlier folds, so mean is misleading).
- **Consistency check passed:** fresh wDMPNN Monomer-heldout group-mean R² (EA mean 0.938) reproduces the established ~0.93 — the dedicated runner matches the prior pipeline.

## 3. Results

**Fig 1 — Best-of-both scorecard (Monomer-heldout).** HPG-hier sits in the top-right (high chemistry *and* high architecture) for both targets; wDMPNN has chemistry but weak architecture; ChemArch has architecture but weaker chemistry.

![Fig 1](report_figures/fig1_scorecard.png)

**Fig 2 — Chemistry extrapolation.** On unseen monomers, HPG-hier's group-mean R² is level with wDMPNN (IP) or ~4% behind (EA); ChemArch drops well below both.

![Fig 2](report_figures/fig2_chemistry.png)

**Fig 3 — Architecture recovery.** HPG-hier beats wDMPNN on ΔR² on every split; ChemArch leads in-distribution (Group/Pair-disjoint), but HPG-hier leads on the hardest split (Monomer-heldout).

![Fig 3](report_figures/fig3_architecture.png)

**Fig 2b — Mean vs median (for comparability with the wDMPNN paper).** The wDMPNN paper reports *mean* R² on its splits, so here are both. HPG-hier ≈ wDMPNN on mean and median for both targets; ChemArch's EA *mean* collapses to −0.69 because one fold (benzothiadiazole, a documented pathology) dominates it — which is exactly why the medians are the fairer summary.

![Fig 2b](report_figures/fig2b_chemistry_mean_median.png)

**Key numbers (median across folds, seed 42):**

| | EA | IP |
|---|---|---|
| **Chemistry — Monomer-heldout group-mean R²** | HPG-hier 0.93 · wDMPNN 0.96 · ChemArch 0.81 | HPG-hier 0.97 · wDMPNN 0.97 · ChemArch 0.85 |
| **Architecture — Monomer-heldout ΔR²** | **HPG-hier 0.79** · ChemArch 0.76 · wDMPNN 0.58 | **HPG-hier 0.82** · ChemArch 0.66 · wDMPNN 0.46 |
| **Architecture — Group-disjoint ΔR²** | ChemArch 0.94 · HPG-hier 0.90 · wDMPNN 0.87 | ChemArch 0.97 · HPG-hier 0.94 · wDMPNN 0.92 |

## 4. Verdict

- **HPG-hier vs wDMPNN:** HPG-hier ≥ wDMPNN on essentially everything — matches chemistry extrapolation and is clearly better on architecture (ΔR², ordering, calibration) across all splits. The two-stage abstraction did **not** cost the chemistry-extrapolation strength we were worried about.
- **HPG-hier vs ChemArch:** HPG-hier is far more robust on unseen chemistry (ChemArch collapses on the hardest folds), and it leads architecture recovery on Monomer-heldout — but ChemArch still edges it on *in-distribution* architecture discrimination.
- **Net:** HPG-hier is the best all-round model — strong on both axes, collapses on neither.

## 5. Caveats (stated honestly)

- **Single seed** — these are point estimates. The ~4% EA chemistry gap and the HPG-hier-over-wDMPNN architecture margins (~0.03) are small enough to need seeds 43/44 to confirm they aren't run-to-run noise.
- **Not strictly best-of-both** — ChemArch retains the in-distribution architecture edge.
- **Medians, not means** — Monomer-heldout has known pathological folds (e.g. ChemArch fold 6 / benzothiadiazole collapses, a documented pathology, not a new bug); means are dragged by them.

## 6. Next steps

1. Run **seeds 43/44** (all three models) for error bars and paired significance across folds.
2. Optionally add the remaining baselines (Frac, GlobalArch, original HPG) for the full comparison.
3. Layer the **within-group objective** onto HPG-hier (the representation × objective cell).
4. Re-derive the ChemArch Monomer-heldout summary in the original report (the −0.37 figure is dominated by one catastrophic fold; the per-fold median is positive).
