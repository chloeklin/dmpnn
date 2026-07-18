# Progress Update — Diagnostics on the wDMPNN vs ChemArch Asymmetry & Refined Next Step

**To:** [Supervisor]  **From:** Chloe Lin  **Date:** 15 July 2026
**Re:** Why wDMPNN wins overall and ChemArch wins architecture recovery — and a first pilot result showing the gap is largely an *objective* problem, not a representation one.

> **TL;DR.** Diagnostics resolved the asymmetry into two causes: wDMPNN's recovery gap is a *training-objective* problem, ChemArch's overall gap is a *representation* (backbone) problem. A first pilot (one fold) adds a within-group loss term to wDMPNN with the representation frozen: **architecture ordering rose 0.854 → 0.888 (~55–70% of the gap to ChemArch closed), within-group residual variance fell ~47%, calibration became near-ideal, and overall R² was unchanged (0.997).** Single fold, no seeds yet — promising, not proven.

---

## Where we left off

Last update: benchmark of wDMPNN vs Frac / GlobalArch / ChemArch on overall R² and architecture-deviation R² *(Table 1/2; `fig_overall_vs_arch_recovery_*`)*. Two reads came out of it: (i) wDMPNN's weak architecture recovery looks like a **training** issue, not a representation deficiency — it isn't trained to preserve architecture; (ii) to improve ChemArch, a **hierarchical representation** was the proposed next step. You then asked me to pin down **why** the asymmetry exists before we commit. That is what the diagnostics did.

## What the diagnostics found (the "why")

The asymmetry has **two different causes**, not one:

- **Why wDMPNN wins overall:** it extrapolates the *chemistry baseline* to unseen monomers. Under Monomer-heldout its group-mean R² is **0.93** vs ≤0.02 for the composition models — its graph encoder decomposes a novel monomer into seen substructures; composition models have no handle on a monomer they never saw. wDMPNN's overall win is a **chemistry-extrapolation** win, not an architecture win. *(Fig. B)*
- **Why ChemArch wins recovery:** explicit architecture conditioning. It leads on calibration and ordering on Group-/Pair-disjoint (the clean, well-separated claim); under Monomer-heldout its recovery edge is real but noisy (significant only in EA ordering).
- **The key mechanistic result:** ChemArch's poor *overall* score under unseen monomers comes from its **composition backbone**, not its architecture residual — the residual actually *rescues* the backbone (ablation, `13_...`). *(Fig. C)*
- **Why standard training under-rewards architecture:** chemistry is **96–98%** of the variance, architecture only **2–4%** (~0.02 eV) — so MSE is almost entirely a chemistry objective. *(Fig. A)* This confirms read (i) from the last update with a quantitative mechanism.

**Net:** wDMPNN's *recovery* gap ≈ an **objective** problem (not trained to preserve architecture); ChemArch's *overall* gap ≈ a **representation** problem (composition backbone can't extrapolate — exactly what a graph backbone would fix).

## How this refines the plan — two levers, cheapest first

- **Lever A — objective (test first, no new architecture):** does explicitly training wDMPNN to minimise within-group error recover architecture? If yes, we may not need a new representation for the *recovery* side at all.
- **Lever B — representation (the hierarchical step, now precisely targeted):** a graph/hierarchical backbone to fix ChemArch's chemistry extrapolation. Still on the table — but gated on Lever A's outcome so we don't rebuild the representation to solve a training problem.

## Pilot for Lever A

Freeze wDMPNN's representation and network; change **only** the objective:

**L = L_overall + λ · L_within**, where L_within penalises within-group residual variance (algebraically = MSE of predicted vs true within-group deviations — the loss-form of our error decomposition). Chemistry groups kept intact per batch via a group-aware sampler; L_within normalised by within-group target variance so λ is interpretable.

Pre-flight checks passed: baseline-shift invariance; within-group gradients sum to ≈0 (redistributes within group, doesn't move the group mean); exact backward compatibility at λ=0. Gradient analysis confirmed the within-group term is **genuinely new signal, not a reweighting of existing error** — cosine similarity to the MSE gradient ≈ **0.04** (near-orthogonal).

Pilot matrix: wDMPNN · EA · Group-disjoint · fold 0 · **λ ∈ {0, 0.03, 0.1, 0.3}**; identical schedule/optimiser/eval to the corrected benchmark; scored with existing diagnostics. Status: λ = 0, 0.03, 0.30 **done**; λ = 0.10 **running**.

## Pilot results — Group-disjoint, fold 0 (single fold, no seeds yet)

The representation was frozen; only the objective changed. All metrics on the held-out test fold.

| λ | Overall R² | MAE (eV) | Ordering acc. | Within-grp resid. var. | Calib. slope |
|---|---|---|---|---|---|
| 0.00 (baseline) | 0.99690 | 0.02354 | 0.85356 | 0.000513 | 0.993 |
| 0.03 | 0.99742 | 0.02129 | 0.88044 | 0.000316 (−38%) | 1.003 |
| 0.30 | 0.99652 | 0.02456 | 0.88777 | 0.000270 (−47%) | 1.000 |
| *ChemArch (ref.)* | — | — | *0.90259* | — | — |

Reading: **overall prediction is unchanged** (R² ≈ 0.997 throughout — chemistry not sacrificed), while **every architecture metric improves** — ordering +2.7 pp (λ=0.03) to +3.4 pp (λ=0.30), closing ~55–70% of the baseline→ChemArch ordering gap; within-group residual variance down ~38–47%; calibration slope moving to ≈1.0. Crucially the slope going *to* 1 (not below) means this is **not shrinkage** — the model preserves architecture magnitude better, not by damping it. A mild Pareto knee is emerging: λ=0.03 is nearly free (overall R² even ticks up); λ=0.30 buys more architecture at a negligible overall cost. Ordering is the headline result because it is magnitude-invariant and noise-robust — the cleanest evidence the model is learning real architecture signal rather than fitting label noise.

## Why this ordering is worth it — two-way payoff (evidence now leans "λ helps")

- **If λ helps** (ordering/calibration ↑, overall flat) — *which fold 0 shows*: objective design is a real lever → paper on *objective design for imbalanced variance factors*, representation fixed; and it partly de-scopes the hierarchical build.
- **If λ does not help:** we've ruled out the objective and **earned** a firm justification for the hierarchical representation (Lever B).

Either way, one experiment resolves whether the hierarchical representation is actually necessary — before we invest in it. Fold 0 is the encouraging first data point; folds + seeds decide whether it holds.

## Risks & mitigations (scoped)

1. **Sampler confound** — grouped batches ≠ original shuffling → run two baselines (original vs λ=0-with-sampler).
2. **Thin, deterministic groups (verified against `data/ea_ip.csv`, 42,966 rows / 18,414 groups):** **100% of samples live in ≥2-arch groups** (zero singletons — every row drives L_within), but each group has only **2–3 architectures with no replicates**. Structure is asymmetric: all groups have **block+random**; **alternating appears in just 33%** (and never alone). So two-thirds of groups offer a single block-vs-random contrast, and L_within's gradient is dominated by it. Mitigations: sampler keeps whole groups intact per batch; floor/pool the within-variance normalizer (don't estimate per-group from 2 points); **stratify ΔR²/ordering into 2-arch (block/random) vs 3-arch groups** so gains on the common contrast don't mask flat performance on the rarer alternating signal; specify *which* architectures in any recovery claim.
3. **Δy noise floor** — if architecture ≈ DFT label noise, L_within fits noise. With **no within-architecture replicates**, signal and noise can't be separated from the data alone → **bound the noise floor externally as a gate before trusting the sweep.**
4. **Normalizer instability** — floor the within-variance denominator.
5. **Small effect ceiling** (~+0.05 ΔR² on Group-disjoint) → ≥3 seeds so "moved" beats seed noise.

## Decisions I'm asking for

(1) Endorse **objective-first (Lever A), representation-second (Lever B)**; (2) agree the **noise-floor bound as the gate**; (3) sign off on target venue/timeline; (4) approve compute for multi-seed sweeps.

## Next steps

Finish λ=0.10 (fills the Pareto curve) → repeat all Group-disjoint folds with ≥3 seeds → paired per-fold significance vs baseline → if consistent, extend to Pair-disjoint then Monomer-heldout. In parallel: sampler-confound baseline and the external noise-floor bound (the gate).

## Timeline

Fold 0 done (this update). Remaining Group-disjoint folds + seeds + noise floor (~2 wks) → full-split sweep (Aug) → workshop abstract (fall) → journal draft (Q4 2026; Digital Discovery / npj Comp Mat / JCIM).

---

## Suggested figures

**Recap of last update (paper folder):**
- **Fig. 0 — Overall vs architecture recovery by model:** `paper/fig_overall_vs_arch_recovery_mean.png` (per-fold: `..._per_fold.png`). *The asymmetry that prompted the diagnostics.*

**Existing diagnostic assets (use as-is / lightly re-styled):**
- **Fig. A — Variance geometry:** `02_variance_geometry/variance_decomposition_EA.png` (+ IP). *Chemistry 96–98% vs architecture 2–4% — why MSE under-rewards architecture.*
- **Fig. B — Where wDMPNN's win comes from:** `03_group_mean_prediction/group_mean_foldwise_EA.png` + `.../group_mean_scatter_EA_monomer_heldout.png`. *Group-mean R² 0.93 vs ≤0.02 → chemistry-extrapolation win, not architecture.*
- **Fig. C — Backbone vs residual (the mechanism):** `13_chemarch_residual_ablation/chemarch_backbone_vs_full_EA_R2_overall.png` + `.../chemarch_delta_r2.png`. *ChemArch's overall gap = composition backbone; residual rescues it.*
- **Supporting:** calibration/ordering `04_architecture_calibration/delta_slope_summary_EA.png`, `05_architecture_ordering/architecture_ordering_EA.png`; complementarity `12_residual_correlation/residual_scatter_chemarch_vs_wdmpnn_EA.png`.

**New figures from the pilot (fold 0 data now available):**
- **Fig. D — λ-sweep (the money shot):** x = λ (0 → 0.03 → 0.10* → 0.30); twin axes = ordering accuracy (rising, with ChemArch reference line) vs overall R² (flat). Fold 0 points exist now; add remaining folds + seed band as they land. *λ=0.10 pending.*
- **Fig. E — Gradient diagnostic:** cosine similarity of L_within vs L_overall gradients ≈ 0.04 (near-orthogonal) + within-group gradient-sum ≈ 0 trace.
- **Fig. F — Noise-floor gate:** distribution of |Δy| vs estimated DFT/label-noise band — how much architecture signal is learnable above noise.
