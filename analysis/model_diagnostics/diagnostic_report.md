# Model Diagnostics Report: wDMPNN vs ChemArch on Overall Prediction and Architecture Recovery

**Scope.** This report interprets the diagnostic pipeline in `analysis/model_diagnostics/`, which was built to answer three linked questions: (1) why wDMPNN achieves the strongest overall EA/IP prediction, (2) why ChemArch often achieves stronger architecture-deviation recovery, and (3) why wDMPNN is strong on Group-disjoint / Pair-disjoint but degrades under Monomer-heldout. The analysis separates *overall property prediction* from *recovery of architecture-induced effects*, so that we can say **where** the models differ rather than only reporting that one has a higher R².

**Models:** `frac`, `wdmpnn`, `globalarch`, `chemarch`. **Splits:** `group_disjoint`, `pair_disjoint`, `monomer_heldout` (9 folds). **Targets:** `EA_vs_SHE_eV`, `IP_vs_SHE_eV`.

---

## Definitions used throughout

A **matched group** is a fixed chemistry/composition context that differs only in sequence architecture:

```
g = (monomer_A, monomer_B, f_A, f_B)      # architecture is NOT in the key
```

- True group mean: `ȳ_g = mean(y_true in g)`
- Predicted group mean: `ŷ̄_g = mean(y_pred in g)`
- **True architecture deviation:** `Δy = y_true − ȳ_g` (uses the *true* group mean)
- **Predicted architecture deviation:** `Δŷ = y_pred − ŷ̄_g` (uses the *predicted* group mean)

Group centering is done within each fold/split. Only matched groups with ≥2 architectures are used for the architecture metrics. The `Δy` metric therefore isolates the within-group, architecture-induced signal after removing chemistry/composition.

---

## Step 1 — Evaluation validation

The pipeline first confirmed all four models are scored on **exactly the same test rows** for every split/target/fold, that predictions are in physical units (eV), that the group key excludes architecture, and that only ≥2-architecture groups feed the deviation metrics. The inventory (`01_validation/evaluation_inventory.csv`) records `n_test`, matched-group counts, and prediction scale for every cell.

One validation warning arose (IP, Monomer-heldout fold 2) but was traced to an over-strict validation *threshold*, not a training or prediction fault — predictions were already stored correctly in eV, so no reruns were needed. After adjusting the heuristic the pipeline passed cleanly.

**What it tells us.** All downstream comparisons are apples-to-apples: differences between models are real, not row-misalignment or unit artifacts.

---

## Step 2 — Variance geometry: how much signal is architecture?

Decomposing the *true* target variance into between-group (chemistry/composition) and within-group (architecture) sums of squares (`SST = SS_between + SS_within`), averaged over folds:

| Split | Target | SS_between | SS_within (architecture) | SD(group means) | SD(Δy) | median \|Δy\| |
|---|---|---|---|---|---|---|
| Group-disjoint | EA | 99.0% | **1.0%** | 0.598 | 0.059 | 0.021 eV |
| Group-disjoint | IP | 98.5% | **1.5%** | 0.481 | 0.058 | 0.019 eV |
| Pair-disjoint | EA | 99.0% | **1.0%** | 0.598 | 0.059 | 0.021 eV |
| Pair-disjoint | IP | 98.5% | **1.5%** | 0.481 | 0.058 | 0.019 eV |
| Monomer-heldout | EA | 97.9% | **2.1%** | 0.443 | 0.059 | 0.021 eV |
| Monomer-heldout | IP | 95.7% | **4.3%** | 0.325 | 0.055 | 0.021 eV |

**What it tells us.** Architecture accounts for only **1–4% of the total variance**; chemistry/composition accounts for the other 96–99%. In physical terms, chemistry moves the baseline by several tenths of an eV (SD of group means ≈ 0.3–0.6 eV) while a typical architecture shift is a few hundredths of an eV (median |Δy| ≈ 0.02 eV). This is the foundational result: **overall R² is almost entirely a measure of chemistry-baseline prediction**, and recovering architecture is an intrinsically much harder, low-variance task. It is the quantitative justification for reporting a separate architecture-deviation metric — the two metrics cannot be redundant because one is dominated by a signal 25–100× larger than the other.

---

## Step 3 — Group-mean prediction: why wDMPNN wins overall

Each prediction was split into its predicted group mean and its within-group deviation, and the group means were scored against the true group means. Median group-mean R²:

| Split | wDMPNN | GlobalArch | ChemArch | Frac |
|---|---|---|---|---|
| Group-disjoint | 0.998 | 0.999 | 0.999 | 0.998 |
| Pair-disjoint | 0.998 | 0.998 | 0.999 | 0.998 |
| **Monomer-heldout** | **0.927** | 0.024 | **−0.366** | −0.018 |

**What it tells us.** Under Group-disjoint and Pair-disjoint, **chemistry is essentially solved by everyone** (group-mean R² ≈ 0.998–0.999), so any model difference on these splits must come from architecture, not chemistry.

Everything changes under Monomer-heldout. Only wDMPNN still extrapolates the chemistry baseline to an unseen monomer (0.93); the three composition-based models collapse (GlobalArch ≈ 0, ChemArch strongly negative). This is the mechanism behind wDMPNN's headline overall-R² advantage on Monomer-heldout: **it is winning on the chemistry baseline, not on architecture.** The likely reason is representational: wDMPNN's atom-level message passing lets a novel monomer decompose into already-seen substructures, whereas the composition models represent each monomer more holistically and have no compositional handle on a monomer they never saw.

Note also that ChemArch is *worse than its own siblings* (Frac, GlobalArch) on the baseline — see Step 10 for the likely cause (its chemistry-conditioned architecture residual receives out-of-distribution inputs and corrupts the baseline).

---

## Step 4 — Error decomposition: where each model's error lives

Each model's total SSE was split exactly into a between-group (chemistry) part and a within-group (architecture) part. Fraction of total error from each component:

| Split | Model | Between-group (chemistry) | Within-group (architecture) |
|---|---|---|---|
| Group-disjoint | wDMPNN | 57% | 43% |
| Group-disjoint | ChemArch | 61% | 39% |
| Pair-disjoint | wDMPNN | 64% | 37% |
| Pair-disjoint | ChemArch | 70% | 30% |
| **Monomer-heldout** | **wDMPNN** | **68%** | 32% |
| **Monomer-heldout** | **ChemArch** | **96%** | 4% |
| Monomer-heldout | GlobalArch | 97% | 3% |

**What it tells us.** This is the primary diagnostic and it is decisive. Under Monomer-heldout, **96% of ChemArch's error is chemistry-baseline error** and only 4% is architecture error. ChemArch is not failing because it mishandles architecture — it is failing because it predicts the wrong absolute property regime for unseen chemistry. wDMPNN's between-group error fraction is much smaller (68%), confirming its overall advantage comes almost entirely from better chemistry prediction, not from better architecture recovery. (On Group/Pair-disjoint, where chemistry is solved, the error is much more evenly split, and ChemArch actually pushes a *larger* share into the between-group bucket precisely because its within-group error is smaller.)

---

## Step 5 — Architecture calibration: does the model preserve deviation magnitude?

Regressing predicted deviation on true deviation (`Δŷ = a + b·Δy`) gives a slope `b` and a dispersion ratio `SD(Δŷ)/SD(Δy)`. Slope and dispersion near 1 mean the architecture magnitude is preserved; below 1 means shrinkage toward the group mean.

| Split | wDMPNN slope / disp | ChemArch slope / disp |
|---|---|---|
| Group-disjoint | 0.894 / 0.943 | **0.945 / 0.969** |
| Pair-disjoint | 0.893 / 0.943 | **0.945 / 0.969** |
| Monomer-heldout | 0.565 / 0.740 | **0.836 / 1.026** |

**What it tells us.** ChemArch preserves architecture magnitude best on every split — slope and dispersion closest to 1. wDMPNN increasingly **attenuates** architecture effects, mildly on the in-distribution splits (slope ≈ 0.89) and strongly under unseen chemistry (slope 0.57, dispersion 0.74) — i.e. it shrinks Δŷ toward zero when the chemistry is novel.

**Caveat worth stating in the manuscript.** Because architecture variance is tiny (Step 2), part of the true Δy may be label/estimation noise. When the target is noisy, a slope below 1 is the *statistically correct* shrinkage, so wDMPNN's 0.57 is not automatically "wrong" and ChemArch's dispersion ratio slightly *above* 1 under Monomer-heldout (1.026) could indicate mild over-dispersion. This is why the ordering result (Step 6) matters — it is robust to magnitude scaling and settles the ambiguity.

---

## Step 6 — Architecture ordering: does the model rank architectures correctly?

For each matched group, the pipeline compared the true vs predicted ordering of architectures (full-ranking accuracy, pairwise accuracy, Spearman, Kendall τ). Pairwise ordering accuracy (chance = 0.5):

| Split | wDMPNN | GlobalArch | ChemArch | Frac |
|---|---|---|---|---|
| Group-disjoint | 0.847 | 0.859 | **0.894** | 0.500 |
| Pair-disjoint | 0.843 | 0.855 | **0.892** | 0.500 |
| Monomer-heldout | 0.751 | 0.737 | **0.778** | 0.500 |

A bug was fixed here: tied predictions (mainly the Frac baseline, which predicts identical values across architectures) had been counted as reversed orderings. After the fix, Frac correctly sits at chance (0.5); the wDMPNN/GlobalArch/ChemArch results were unchanged.

**What it tells us.** ChemArch ranks architectures within matched groups most reliably on **all six split/property combinations**. Because pairwise ordering is invariant to how much the magnitudes are scaled, this is the cleanest, noise-robust evidence that ChemArch genuinely captures the *direction* of architecture effects rather than merely being less shrunk. This is the metric I would lead with for the architecture claim.

---

## Step 7 — Performance vs architecture-effect magnitude

Samples were binned by |Δy| and the per-sample advantage `|err_ChemArch| − |err_wDMPNN|` computed (positive = wDMPNN better).

**What it tells us.** On Group-disjoint and Pair-disjoint, ChemArch beats wDMPNN across all effect-size bins. On Monomer-heldout EA a clear pattern emerges: wDMPNN is slightly better for the *smallest* architecture effects, but ChemArch becomes progressively better as |Δy| grows, and substantially better for the largest deviations. This is the calibration story made concrete: **wDMPNN compresses large architecture effects while ChemArch preserves them**, so the two models' relative strength depends on how big the architecture effect actually is.

---

## Step 8 — Chemical novelty of held-out monomers

Using Morgan fingerprints, every held-out monomer is genuinely out-of-distribution (max Tanimoto to any training monomer averages ≈ 0.41 across folds; fold 6, benzothiadiazole diboronic acid, is the most novel at 0.31). But novelty does **not** track fold difficulty: Spearman(max Tanimoto, overall R²) = 0.37 (EA) and −0.42 (IP), both non-significant at n = 9.

**What it tells us.** Generic structural novelty is insufficient to explain why some folds are hard. Every held-out monomer is novel, yet difficulty varies widely, and it does not line up with fingerprint distance. With only 9 folds this is low-powered, so the honest phrasing is *"target-distribution shift (Step 9) is a more direct predictor of fold difficulty than fingerprint distance"* rather than "novelty doesn't matter."

---

## Step 9 — Target-distribution shift

For each Monomer-heldout fold, train vs test target statistics were compared (mean shift, std ratio, Wasserstein distance). Selected folds:

| Fold | Monomer | Target | Mean shift (eV) | Std ratio (test/train) | Wasserstein |
|---|---|---|---|---|---|
| 6 | benzothiadiazole diboronic acid | **EA** | **+1.02** | **0.46** | 1.02 |
| 5 | bithiophene diboronic acid | IP | −0.70 | 0.59 | 0.70 |
| 3 | DTT trithiophene | IP | −0.39 | 0.46 | 0.41 |
| 2 | difluorobenzene diboronic acid | IP | +0.63 | 1.22 | 0.63 |

**What it tells us.** Hard folds are characterised by a large **shift in the target mean** and often a **narrower** test distribution than training. Fold 6 (EA) is the extreme case: the held-out chemistry sits ≈ +1 eV outside the training EA range with less than half the spread. This is the direct driver of the collapse in overall R² on those folds — a model that cannot extrapolate the shifted baseline is penalised heavily, and the narrow test variance further deflates R² through a small denominator (so R² overstates how bad the absolute error is; MAE in eV should be reported alongside).

---

## Step 10 — Per-fold case study: fold 6 (benzothiadiazole, EA)

Fold 6 is the cleanest illustration of the whole story. Overall EA R² on this fold:

| Model | Overall EA R² | Group-mean EA R² | ΔEA R² (architecture) |
|---|---|---|---|
| wDMPNN | **0.85** | 0.88 | 0.34 |
| ChemArch | **−17.3** | −16.4 | **0.42** |
| GlobalArch | −7.3 | −7.0 | 0.36 |
| Frac | −9.5 | −9.0 | 0.00 |

**What it tells us.** On the exact same fold, ChemArch recovers architecture deviations *better* than wDMPNN (ΔR² 0.42 vs 0.34) yet has a catastrophic overall R² because it completely mispredicts the shifted chemistry baseline (group-mean R² −16.4). wDMPNN predicts the shifted baseline well (0.88) and therefore keeps a strong overall R². The two failure modes are fully decoupled here: **architecture recovery and chemistry-baseline extrapolation are independent axes of model quality**, and this single fold shows a model can win one while losing the other.

The pre-vs-post-residual ablation (Step 13) shows fold 6 EA is the **one exception** to an otherwise favourable pattern: here the architecture-conditioned residual *overcorrects* an already-poor backbone (backbone group-mean R² −7.7 → full −17.4). Everywhere else the residual rescues the backbone rather than corrupting it — so ChemArch's baseline failure is fundamentally a backbone (composition-extrapolation) problem, with fold 6 the single case where the residual makes it worse. See Step 13.

---

## Step 11 — Statistical comparisons (Monomer-heldout, paired by fold, n = 9)

Paired Wilcoxon signed-rank tests, wDMPNN vs ChemArch:

| Metric | Median diff (wDMPNN − ChemArch) | p | Folds won (wDMPNN : ChemArch) |
|---|---|---|---|
| Overall EA R² | **+0.294** | **0.004** | 9 : 0 |
| Group-mean EA R² | +0.271 | 0.004 | 9 : 0 |
| ΔEA R² (architecture) | −0.121 | 0.129 | 1 : 8 |
| Overall IP R² | **+0.283** | **0.020** | 7 : 2 |
| Group-mean IP R² | +0.252 | 0.008 | 8 : 1 |
| ΔIP R² (architecture) | −0.063 | 0.820 | 4 : 5 |
| EA pairwise ordering | −0.067 | **0.027** | 2 : 7 |

**What it tells us.** wDMPNN's overall and group-mean advantage under Monomer-heldout is **statistically significant** (p < 0.05, wins nearly every fold). ChemArch's architecture-deviation advantage under Monomer-heldout is **not** significant (ΔR² p = 0.13 EA, 0.82 IP), though it wins the median and most folds — the exception is EA pairwise *ordering*, where ChemArch's edge is significant (p = 0.027). In other words: under Monomer-heldout the strong, defensible claim is that wDMPNN extrapolates chemistry better; ChemArch's architecture edge on this split is real in ordering but too noisy in ΔR² to establish across 9 monomers. The clean, well-separated architecture claim lives in Group-disjoint and Pair-disjoint (ΔR² 0.951/0.949 vs 0.898/0.896; ordering 0.894/0.892 vs 0.847/0.843).

---

## Step 12 — Absolute errors on pathological folds: real error vs denominator collapse

The negative overall R² values raise a question: are they genuine eV-scale errors, or artifacts of a small R² denominator (narrow test variance)? The pipeline recomputed R², MAE, RMSE, and percentile errors alongside the test-set spread for every Monomer-heldout fold.

**EA — fold 6 (benzothiadiazole), the pathological case:**

| Model | R² | MAE (eV) | RMSE (eV) | Max AE (eV) | Test SD (eV) |
|---|---|---|---|---|---|
| wDMPNN | 0.85 | 0.070 | 0.093 | 0.57 | 0.243 |
| Frac | −9.48 | 0.694 | 0.786 | 1.31 | 0.243 |
| GlobalArch | −7.33 | 0.609 | 0.701 | 1.18 | 0.243 |
| ChemArch | **−17.34** | **1.027** | 1.040 | 1.44 | 0.243 |

**What it tells us.** ChemArch's fold-6 EA collapse is a **genuinely large error, not a denominator artifact**: a 1.03 eV mean absolute error against a target SD of 0.24 eV. wDMPNN, on the same fold, sits at 0.07 eV. Across EA, the R²–test-SD correlation is only 0.49 and the median MAE in failing folds is 0.69 eV, so the verdict is that **real error dominates** the EA negatives.

For **IP**, the picture is milder. The hardest IP fold is fold 5 (bithiophene diboronic acid), where ChemArch has R² = −0.17 but MAE = 0.27 eV against a test SD of 0.26 eV — i.e. the error is comparable to the spread, so an R² near zero is partly denominator-driven. Across IP the R²–test-SD correlation is 0.28 and median failing-fold MAE is only 0.28 eV, giving a **mixed** verdict: the IP negatives reflect both modest denominator shrinkage and somewhat elevated error, not catastrophe.

This refines the earlier caveat: report MAE in eV alongside R² — the EA fold-6 failure is real and worth highlighting, whereas the IP negatives are much less dramatic than their R² suggests.

---

## Step 13 — ChemArch pre-residual vs post-residual (the residual is a rescue, not the culprit)

Using existing ChemArch checkpoints only (no retraining), inference was run at two points: the composition backbone `h_mix` (before the architecture residual) and the full model `h_mix + α·r_arch` (after). This isolates whether ChemArch's Monomer-heldout failure originates in the backbone or is introduced by the residual pathway. Mean over the 9 Monomer-heldout folds:

| Target | Stage | Overall R² | MAE (eV) |
|---|---|---|---|
| EA | Backbone only | −4.03 | 0.796 |
| EA | Full ChemArch | −1.22 | 0.292 |
| IP | Backbone only | −7.71 | 0.765 |
| IP | Full ChemArch | +0.22 | 0.219 |

Per-fold, the residual **rescues a poor backbone** on 8/9 EA folds and 6/9 IP folds; it never turns a good backbone bad; and there is exactly **one fold where it hurts — fold 6 EA** (backbone R² −7.4 → full −16.7). Backbone-only architecture-deviation R² is ≈0 everywhere (as expected — the backbone has no architecture pathway), and adding the residual is what produces ChemArch's architecture recovery, improving ΔR² by +0.46 (EA) and +0.48 (IP) on the in-distribution splits.

**What it tells us — this revises the earlier interpretation.** ChemArch's Monomer-heldout failure is **already present, and far worse, in the composition backbone** (overall R² −4 to −8 without the residual). The chemistry-conditioned residual is not corrupting the baseline in general — it is *substantially recovering* it. Two consequences:

1. Because the residual improves the **group mean** (not just the deviation), the "architecture residual" is really a broader chemistry-conditioned correction: it takes monomer embeddings as input and, under unseen chemistry, primarily patches the mis-placed composition baseline, carrying architecture signal as a secondary effect. Removing it collapses the group mean.
2. The root cause of ChemArch's overall LOMO weakness is the **composition-weighted backbone's inability to extrapolate to unseen monomers** — exactly the capability wDMPNN's graph encoder provides. Fold 6 EA is the lone case where the residual overcorrects an extreme target shift and makes matters worse.

This strengthens, rather than weakens, the best-of-both hypothesis: the failing component is the backbone, and it is the component wDMPNN would replace.

---

## Step 14 — Residual correlation: are wDMPNN and ChemArch complementary?

Residuals for each model were correlated against wDMPNN's, separately for overall, group-mean, and architecture-deviation residuals, across all splits/targets/folds. ChemArch vs wDMPNN (mean correlation):

| Residual type | Pearson r | Spearman ρ |
|---|---|---|
| Overall | 0.40 | 0.35 |
| Group-mean | 0.34 | 0.31 |
| Architecture-deviation | 0.57 | 0.51 |
| Overall, Monomer-heldout only | **0.25** | — |

**What it tells us.** The two models' errors are only **partially correlated** — they are not redundant. Correlation is lowest on the group-mean/chemistry component (r ≈ 0.34) and lowest of all on Monomer-heldout specifically (r ≈ 0.25), meaning the two representations **diverge most exactly where it matters** — on unseen monomer chemistry. This is direct, quantitative support for combining them: a wDMPNN-quality graph backbone (better chemistry extrapolation) with ChemArch's explicit architecture-conditioned residual (which the ablation shows carries genuine architecture signal, +0.46–0.48 ΔR²) would draw on complementary, not overlapping, information.

---

## Overall interpretation

The diagnostics support a two-axis conclusion rather than a single ranking:

**On Group-disjoint and Pair-disjoint** (chemistry solved by all models, group-mean R² ≈ 0.999), differences arise almost entirely from architecture representation, and **ChemArch is best on every architecture metric** — highest ΔR² (0.95), calibration slope and dispersion closest to 1, and highest ordering accuracy — consistent across EA and IP and (for ΔR²/ordering) clearly separated from the others.

**On Monomer-heldout** the task changes character. Predicting the unseen chemistry baseline becomes the dominant challenge (95–98% of variance, and it drives 68–96% of each model's error). **wDMPNN extrapolates the chemistry-dependent group mean far better** (group-mean R² 0.93 vs ≤0.02 for the composition models) and this — not architecture — is why it wins overall, significantly. ChemArch still recovers and orders architecture effects at least as well, but its overall prediction is capped by poor chemistry extrapolation. Crucially, the ablation (Step 13) shows this cap lives in the **composition backbone**, not the residual pathway: without the residual ChemArch's overall R² is far worse (−4 to −8), and the residual actually rescues it on most folds — the lone exception being fold 6 EA, where it overcorrects an extreme target shift. When ChemArch produces a genuinely large eV-scale error, it is fold 6 EA (MAE 1.03 eV, Step 12); the IP negatives are milder and partly denominator-driven.

So the representation best for overall prediction is **not** the representation that best preserves architecture-induced effects:

- **wDMPNN** — strongest for extrapolating the dominant chemistry-dependent baseline, especially under unseen monomer chemistry.
- **ChemArch** — strongest for recovering, calibrating, and ordering architecture-induced deviations once the chemistry baseline is established.

This is a stronger scientific story than a headline benchmark, and it directly motivates the obvious next design: an **explicit architecture-conditioned residual head on a wDMPNN-style graph backbone**, combining wDMPNN's chemistry extrapolation with ChemArch's architecture resolution — best-of-both rather than an either/or. The follow-up diagnostics give this concrete support: (i) the failing ChemArch component is the composition backbone, precisely what the graph encoder would replace (Step 13); (ii) the residual head carries real architecture signal (+0.46–0.48 ΔR²) independent of the backbone; and (iii) wDMPNN and ChemArch residuals are only partially correlated, most weakly on unseen chemistry (r ≈ 0.25, Step 14), so the two encode complementary information. This remains a hypothesis requiring a retraining experiment to confirm.

## Caveats to carry into the Discussion

1. **Δy signal-to-noise.** Architecture is 1–4% of variance and ≈0.02 eV; establish that this exceeds the DFT label noise before reading calibration slopes as physics. Lean on ordering (noise-robust) as the primary architecture evidence.
2. **R² denominator fragility — resolved by Step 12.** Report MAE/RMSE in eV alongside R². The EA fold-6 collapse is a *genuine* ~1 eV error, not an artifact, and is worth highlighting. The IP negatives, by contrast, are mild (MAE ≈ test SD) and partly denominator-driven, so they should not be overstated.
3. **n = 9 folds.** Monomer-heldout conclusions rest on 9 outlier-prone folds. Keep the strong ChemArch architecture claim on Group/Pair-disjoint; present Monomer-heldout architecture recovery as directional (significant only in EA ordering).
4. **Aggregation.** Pooled overall R² (e.g. EA LOMO −1.29 for ChemArch) is dominated by one or two folds; report the per-fold distribution (median as well as mean), not just the pooled number.
