# Experiment Plan — Objective vs Representation for Polymer Architecture Effects

*A single map of what we're doing and why. Read section 0 first; the rest is reference.*

---

## 0. The whole thing in one paragraph

We found that for copolymer EA/IP, **chemistry explains ~96–99% of the property and sequence architecture only ~1–4%**. Under standard training (MSE), a model is rewarded almost entirely for getting chemistry right, so it under-learns architecture. We think there are **two ways to fix that**: change the **training objective** so it explicitly rewards architecture (Lever A), or change the **representation** so architecture is encoded structurally (Lever B, a hierarchical graph). The clean way to test both is a **2×2 grid: {representation} × {objective}**. We've done the first cell (objective on the existing graph model — it worked in-distribution). Now we're building the hierarchical representation and will run the full grid. The whole point is to learn **whether the objective alone is enough, or whether we also need a new representation.**

---

## 1. The finding that started this

- Overall accuracy (R²) is almost a pure **chemistry** measure — architecture is a tiny slice of the variance.
- Our two models disagree on *which* thing they're good at:
  - **wDMPNN** wins **overall** — because it extrapolates the **chemistry baseline** to unseen monomers.
  - **ChemArch** wins **architecture recovery** — because it has an explicit architecture term.
- Diagnostics showed these are **two different problems**:
  - wDMPNN's weak architecture recovery = a **training-objective** problem (it *can* represent architecture, MSE just doesn't reward preserving it).
  - ChemArch's weak overall score on unseen chemistry = a **representation** problem (its composition backbone can't extrapolate).

## 2. The two levers

| | **Lever A — Objective** | **Lever B — Representation** |
|---|---|---|
| Idea | Add a loss term (`L_within`) that explicitly rewards getting within-group (architecture) variation right | Build a hierarchical polymer graph (HPG) that encodes composition + sequence structurally |
| Fixes | wDMPNN's architecture-recovery gap | ChemArch's chemistry-extrapolation gap |
| Cost | Cheap (loss change, existing model) | Expensive (new model, engineering) |
| Status | **Pilot done** (worked in-distribution) | **Being built now** |

`L_within` = penalise the within-group variance of the prediction residuals; it equals directly supervising the architecture deviation (Δŷ→Δy), and leaves the chemistry baseline untouched.

## 3. The unifying design — a 2×2 factorial

|                          | **Objective = MSE** | **Objective = MSE + λ·L_within** |
|--------------------------|---------------------|----------------------------------|
| **Representation = wDMPNN** | baseline | **the pilot (done)** — objective on the flat graph |
| **Representation = HPG**    | HPG baseline (building now) | HPG + explicit supervision (the payoff cell) |

- **Rows** tell us what the representation buys. **Columns** tell us what the objective buys. The **interaction** tells us if they're complementary.
- This is why running both "in parallel" is fine: a factorial *isolates* each effect. What we must NOT do is change representation and objective at once inside one model.

## 4. The models being compared

| Model | What it is | Expected strength |
|---|---|---|
| **Frac** | composition-only baseline (no architecture) | reference floor |
| **wDMPNN** | one flat atom-level weighted graph; architecture implicit | chemistry extrapolation |
| **GlobalArch / ChemArch** | encode each monomer separately, mix by fixed rule, add an explicit architecture-label residual | architecture recovery |
| **HPG (original)** | faithful reproduction: one fused graph (atoms + virtual monomer nodes + chain edges), one message-passing stack | baseline hierarchical model |
| **HPG-hier (new — the model we're building)** | **true two-stage**: Stage 1 encodes each monomer's atom graph independently (shared encoder) → monomer embedding; Stage 2 is a separate message-passing over a monomer-level graph whose edges carry junction chemistry + architecture transition weights | (goal) best-of-both |

The three-way comparison on EA/IP is: **original wDMPNN vs original HPG vs HPG-hier** (with ChemArch/GlobalArch/Frac kept as context). All scored by the same diagnostics.

**Design decision (the junction):** a junction bond couples the two levels, so it can live in only one place. HPG-hier is the **strict two-stage** design (option 2): monomers are encoded independently and the junction re-enters at Stage 2 as an edge feature (bond order + port-pair + transition weight). This is the deliberate bet — it tests whether an explicit hierarchy + coarse junction is enough, or whether atom-level cross-junction context (which wDMPNN has) is needed. If HPG-hier recovers architecture well but trails wDMPNN on Monomer-heldout chemistry extrapolation, the junction abstraction is the cause, and the fallback is "option 3" (a few atom-level steps across the junction before pooling).

**Shelved:** the stochastic-chain-edge HPG variant (`chain_edge_mode='stochastic'`) is **built but will NOT be run** as a standalone experiment — it's superseded by HPG-hier. Its `compute_stochastic_ff_weights` function is *reused* inside HPG-hier as the Stage-2 transition-weight edge feature.

## 5. The metrics (and which claim each supports)

| Metric | Measures | Supports the claim |
|---|---|---|
| **Overall R² / MAE** | total accuracy (≈ chemistry) | headline number |
| **Group-mean R²** | chemistry baseline (architecture averaged out) | "wDMPNN wins overall / does HPG extrapolate chemistry?" |
| **Architecture-deviation R² (ΔR²)** | within-group architecture recovery | "which model captures architecture" |
| **Ordering accuracy** | rank of architectures within a group (noise-robust) | the **headline architecture metric** |
| **Calibration slope** | does it preserve architecture *magnitude* (≈1) or shrink it (<1)? | shrinkage vs faithful |

## 6. What we're actually running — the phases

| Phase | What | Status |
|---|---|---|
| **1** | Objective pilot on wDMPNN (Group-disjoint fold 0): ordering 0.854→0.888, within-var −47%, calibration→1, overall R² flat | **done (1 fold, no seeds)** |
| **2** | Original HPG baseline (fused graph) benchmarked under our splits | **built, needs to run** |
| **3** | **Build HPG-hier** — the true two-stage model — and run the three-way (wDMPNN / HPG / HPG-hier) on EA/IP | **building now** |
| **4** | Full factorial — apply `L_within` to the best representation (the cell that unites Lever A + B) | **not started** |
| — | *Stochastic-chain-edge HPG variant* | **shelved — will NOT be run** (function reused inside HPG-hier) |

## 7. Experimental grid & infrastructure

- **Splits:** Group-disjoint (5 folds), Pair-disjoint (5), Monomer-heldout (9). Monomer-heldout is the decisive test (unseen chemistry).
- **Seeds:** 42, 43, 44. Predictions saved **per seed** (no averaging). Metrics are seed-averaged per fold; significance is **paired Wilcoxon across folds** (not fold×seed — that would fake significance); seed spread = error bars.
- **Predictions** saved in physical eV, in the exact format the diagnostics pipeline reads; Monomer-heldout test indices asserted byte-identical to the stored split.
- **Diagnostics pipeline** scores every run (the same battery from the report). Run once per seed, then aggregated.
- **Grid size (updated — stochastic-edge dropped):** HPG-hier + original HPG (2 targets × 19 folds × 3 seeds each) + baseline re-seeds (wDMPNN/ChemArch/GlobalArch/Frac × 2 × 19 × 3) ≈ **680 runs**. Run seed 42 across the whole grid first (point estimates + the story), then seeds 43/44 for error bars.

## 8. How we run it on Gadi (fast)

Order matters for getting answers ASAP:

1. **Gates first** (small, short-walltime jobs): (a) full fold-0 run — confirm HPG's group-mean R² ≈ 0.99 like the others; (b) 2-seed run — confirm per-fold seed std is non-zero (catches a config bug). Cheap insurance against re-running the whole grid.
2. **Seed-42 job array over the full grid** → all point estimates and the whole story at 1/3 the jobs.
3. **Seeds 43/44 array** for error bars / significance, launched while reviewing step 2.

Job **array** (not one qsub per run) to stay under Gadi's per-user job cap.

## 9. Decision gates — what tells us what

- **Objective pilot on Monomer-heldout:** does `L_within` recover architecture on *unseen* chemistry while keeping wDMPNN's chemistry extrapolation? If **yes** → the objective alone gives best-of-both, and the hierarchical representation may be unnecessary. If **no** → representation is the bottleneck, Lever B justified.
- **HPG-hier on Monomer-heldout:** does the true two-stage model extrapolate chemistry as well as wDMPNN (group-mean R² ≈ 0.93) *and* recover architecture as well as ChemArch? Best-of-both = high on both axes.
- **HPG-hier vs wDMPNN:** if HPG-hier trails wDMPNN on chemistry extrapolation, the junction abstraction (independent Stage-1 encoding) is the likely cause → motivates option 3 (atom-level junction coupling).

## 10. Current status & immediate next action

- **Built & verified:** original HPG, within-group loss, seeded diagnostics, runners. HPG-hier (true two-stage) **built and gate-passed locally** (fold-0 group-mean R² ~0.9965, converged, non-zero seed std). Stochastic-edge featurizer **shelved (not run)**.
- **Legacy-baseline finding:** all legacy baselines were seed-42, and LOMO coverage is **complete (9/9 EA and IP** for wDMPNN — an earlier "1/9 IP" count was a manifest artifact, retracted). So the report's LOMO numbers (wDMPNN 0.93 / ChemArch −0.37) are backed by full data. **But provenance is mixed:** legacy LOMO came from the `train_graph.py` path while legacy group/pair came from the generalization runners, and HPG-hier uses the current dedicated runner. **Decision: regenerate ALL baselines fresh at seed 42 through the current runners** — for uniform provenance / same pipeline as HPG-hier, not because legacy is incomplete.
- **Next action (Gadi):** confirm gate1/gate2 on the cluster → launch the **seed-42 diagnosis slice** (`submit_seed42_diagnosis.sh`: all 7 models × 2 targets × 19 folds, seed 42 — regenerates baselines cleanly) → read HPG-hier LOMO group-mean vs the fresh baselines → gate seeds 43/44 (all 7 models) on the result.
- **Consistency check (now meaningful):** fresh seed-42 wDMPNN LOMO group-mean should reproduce ~0.93 (ChemArch ~−0.37). Match → the current runner ≈ the legacy path, comparison sound. Materially off → the training-path difference is real; investigate before trusting the numbers.
- **What we're waiting to learn:** whether objective-alone suffices (pilot Monomer-heldout) and whether the two-stage representation reaches best-of-both.

---

### Glossary
- **Group / matched group** = a fixed chemistry+composition context `(monomer_A, monomer_B, f_A, f_B)`; architecture varies within it.
- **Architecture** = sequence type: block / random / alternating.
- **Δy (deviation)** = a sample's value minus its group mean = the architecture signal after removing chemistry.
- **`L_within`** = the loss term that supervises Δŷ→Δy.
- **Monomer-heldout (LOMO)** = test on copolymers whose monomer was never seen in training = the hard, decisive generalization test.
