# Paper 2 — outline and evidence map

*Working title:* **What does a held-out-monomer split actually measure? Null-floor
calibration and two-axis evaluation for copolymer property prediction**

Banked separately: the ChemArch / GlobalArch / fusion draft. Its model set was never
regenerated under the fixed protocol, so it cannot share a table with anything below.

---

## The claim chain

Each row is one thing the paper asserts, the evidence for it, and where it comes from.
Nothing goes in the paper that isn't in this table.

| # | Claim | Evidence | Figure / source |
|---|---|---|---|
| 1 | Architecture is ~1% of the variance; monomer identity is ~90% | EA 0.98%, IP 1.46% within (A,B,fracA); A+B identity 0.93 / 0.90 | `fig2_variance_by_axis` |
| 2 | A split can be degenerate for a target, and you cannot tell without a null | A-blind null median group-mean R² **0.676** on EA vs **−0.034** on IP; on **IP fold 0** the null (0.969) beats **both** trained models | `F3_null_floor` (`analysis/paper1_figures/`) |
| 3 | The two chemical axes need different splits | headroom: A-split EA 0.32 / IP 1.03; B-split EA 0.62 / IP 0.44 | design audit §0.5–0.6 |
| 4 | The benchmark's B space is dominated by two scaffold families | 112 Murcko scaffolds, largest 317 (46%) and 109 (16%) = 62.5% | `fig3_scaffold_cluster_sizes` |
| 5 | Therefore a balanced scaffold-disjoint split is impossible, and the folds are not exchangeable | folds 0–3: 76/76 held-out monomers have a same-core relative in training; folds 7–8: 0/75 | fold-composition table |
| 6 | Single-run benchmark numbers are not measurable at this scale | three identical runs, EA fold 1: group-mean R² 0.450 / 0.790 / 0.978, **SD 0.268** (`_noise_floor_results.md`, `group_mean_r2_sd`) | `fig1_run_to_run_variance` |
| 7 | **HPG-hier recovers architecture better than wDMPNN on both chemical axes** | ΔR² A-split 0.803/0.808 vs 0.433/0.450; B-split S 4–0, D 5–0, both targets | `fig_ab_comparison` |
| 8 | Explicit sequence *may* help where the chemistry is new — **suggestive, not established** | octamer vs HPG-hier: D +0.019/+0.032, consistent in direction (5–0) but **4 of 5 (EA) and 2 of 5 (IP) per-fold differences are smaller than the measured seed SD** (`folds_smaller_than_measured_seed_sd`); S −0.001/+0.000 (2–2). Report as consistent in direction, within run-to-run variation, not established. | `fig_r3_architecture` |
| 9 | Bottom-line accuracy hides the architecture difference | wDMPNN competitive on B-split D EA RMSE while 0.16 behind on ΔR² | `fig_r3_overall_performance` |

Claim 7 is the headline. Claims 2–5 are the methodological contribution and are what
makes this a different paper from the banked one.

---

## Section plan

1. **Introduction** — copolymer property prediction; the comparison that matters is
   between architectures at fixed (A, B, fᴀ), so the shared chemistry term cancels.
2. **Dataset structure** — the exact factorial (9 × 682 × 7); variance by axis; the A
   axis has 9 monomers and models train on 7 of them. *Claims 1.*
3. **Metrics** — group mean / deviation decomposition (carried from methods reference);
   **the null-floor predictor**; skill against the null; absolute error in eV. *Claim 2.*
4. **Splits** — A-heldout as it exists; the new B-heldout construction; Murcko packing
   and why disjointness is impossible; the S/D fold grouping derived from the frozen
   split. *Claims 3–5.*
5. **Protocol** — best-checkpoint prediction, three seeds averaged, the measured
   run-to-run variance that motivates it. *Claim 6.*
6. **Results** — architecture recovery on both axes; the S/D contrast; absolute error.
   *Claims 7–9.*
7. **Limitations** — see below.

---

## Reporting conventions to fix now, once

Set these in the methods section and never deviate:

- Per-fold values, aggregated as **median and mean**, both reported. The gap between
  them is informative (A-split EA: 0.803 vs 0.634; B-split: 0.778 vs 0.746).
- **Never pool the nine B folds.** Report S and D separately; the pooled test is a
  footnote at most.
- Every group-mean R² appears **beside its fold-specific null floor**.
- Every model gets the **same** three-seed averaging. State it wherever numbers appear.
- Paired per-fold sign tests with the attainable minimum p stated (0.0039 pooled;
  0.063 and 0.125 within group). Report effect sizes with intervals, not p-values alone.

> The banked draft currently reports ChemArch ΔEA as **0.43** in one table and **0.924**
> in another, both labelled LOMO — pooled versus median-per-fold. That is exactly the
> failure these conventions prevent.

---

## Open before submission

### Claim 8 is confounded five ways, not three

Baseline Stage-2 versus octamer Stage-2 differ in *all* of the following at once:

| # | factor | HPG-hier | octamer |
|---|---|---|---|
| 1 | topology | 2 nodes | 8-slot chain |
| 2 | positional embeddings | – | 8 learned vectors |
| 3 | readout | stoichiometry-weighted sum | attention pooling |
| 4 | **edge features** | 16-d port-pair + transition weight | **none — a loss** |
| 5 | replicas (random rows only) | 1 | 16, averaged *inside the loss* |

Factor 5 is not test-time ensembling. `hpg_hier.py` forward returns `pred_sum / replica_counts`
and `_loss` takes MSE against that mean, so the model is **trained** as a 16-member ensemble
with a variance-reduced gradient. That is the leading explanation for its low seed SD.

### The 2×2 that resolves factors 1 and 3

| | stoich / mean readout | attention readout |
|---|---|---|
| **2-node graph** | HPG-hier — *have* | **arm C** — runs today |
| **8-slot chain** | **arm D** — needs a patch | octamer — *have* |

Reading, fixed in advance:

- D ≈ octamer, C ≈ baseline → the **sequence** did it. Claim 8 stands.
- C ≈ octamer, D ≈ baseline → the **readout** did it. Claim 8 goes.
- both midway → they interact; report, do not attribute.
- both ≈ baseline → it was factor 2, 4 or 5.

On the 8-slot chain the stoichiometric readout **is** mean pooling over positions: because
`n_A = round(8 · fᴀ)`, the slot counts already encode composition, so the mean weights A and B
in proportion to fᴀ and f_B. It is the exact analogue of `f_A·h_A + f_B·h_B`, not a substitute.
Pooling happens after message passing, so arrangement information survives — only the *learned
weighting of positions* is removed, which is the factor being isolated.

Only one readout variant. Adding max / sum / gated turns attribution into a search, and with
nine folds they could not be distinguished anyway.

**Arm D does not currently exist and fails silently.** `OctamerEncoder` is instantiated only when
`stage2_mode == "octamer_sequence" and stage2_readout == "attention"` (`hpg_hier.py:201`), and
`data/hpg_hier.py:37` hard-codes `+ 2 * polymer_idx` for the transition graph. So passing
`--stage2_mode octamer_sequence --stage2_readout stoich_weighted` today runs the **exact baseline
model** with no error. Assert against that combination regardless of whether arm D is built.

### Open items

| Item | Why it matters | Cost |
|---|---|---|
| **Arm C** — 2-node Stage-2 + attention pooling | Isolates factor 3. Can falsify the sequence claim; cannot on its own establish it. | 54 runs, ~2.0 kSU |
| **Arm D** — 8-chain + mean pooling | Completes the 2×2. Needs the patch above. | 54 runs, ~2.2 kSU |
| **Octamer K=1** (`--n_random_samples 1`) | Isolates factor 5. Pre-registered prediction: if seed SD jumps but ΔR² holds, the stability was the ensembling and the architecture claim survives. Affects **random rows only** — report that subset separately. | 54 runs, ~2.0 kSU |
| **Q1 wedge never regenerated** | Dropped on single-seed pre-fix evidence, the same class we invalidated. The paper should not say four variants were tested when one was never re-measured. | 54 runs, or state plainly that it is untested |
| **Group D is 5 folds (4 excluding the homogeneous fold 5)** | Claim 8 rests on few independent chemistries. Report it as such. | none — wording |
| **Factors 2 and 4 stay confounded** inside the 8-chain arm | Positional embeddings and the dropped port features move together. Separating them is two more arms. | state as a limitation, not runs |

### Training-cost claim — already measured, zero further SU

From 481 provenance sidecars, identical protocol (batch 64, patience 15, same LR, gpuvolta):

| model | n | median epochs to best | median s/epoch | median wall | SU/run |
|---|---|---|---|---|---|
| wDMPNN | 108 | **64** | 134 | 2.36 h | 85.0 |
| HPG-hier | 107 | **31** | 115 | 1.01 h | 36.5 |
| octamer | 107 | 34 | 119 | 1.14 h | 41.1 |
| junction | 106 | 31 | 119 | 1.03 h | 37.1 |

Per-epoch cost is flat across all five (115–134 s), so the 2.3× wall-clock gap is **entirely
epochs-to-converge**, not per-step cost. The hierarchy is 2.3× cheaper to train *and* better at
architecture recovery. Report epochs-to-best and s/epoch separately — wall time alone confounds them.

Charging: gpuvolta is 3 SU per resource-hour and jobs request `ncpus=12`, so **36 SU per GPU-hour**.
The 481 completed cells account for 25.7 kSU of the 28.33 kSU spent on um09 to date.

Data-scaling learning curves (advantage vs training fraction) are a *different* claim, ~3 kSU, and
cannot rescue Claim 8. Attribution first.

---

## What is deliberately NOT in this paper

- ChemArch, GlobalArch, frac, the fusion ablation — banked, never regenerated.
- Curtis 2025 — its monomers are featureless A/B beads, so Stage-1 collapses to two
  learned embeddings and the model becomes a generic binary-sequence encoder. It would
  not validate the hierarchy. Name as future work only.
- Any claim of statistical significance. With nine folds nothing survives correction;
  the paper reports consistency and effect size instead, and says so.
