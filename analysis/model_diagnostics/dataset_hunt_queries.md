# Dataset hunt — finding an architecture-diverse, sequence-controlled ML dataset

**Goal:** find a dataset that can host a *novel sequence-aware architecture* paper — i.e. one where the **same chemistry (and ideally the same composition) appears with different sequences/architectures**, with a measured property, at ML-ready size. Your diblock `block.xlsx` can't do this (99.7% diblock ≈ Arora 2021, already ~90%-solved). The literature's one open gap is **multiblock / sequence-controlled**, so the question is purely: *does the data exist?*

## What counts as a hit (acceptance criteria)

A dataset is usable only if it has **all** of:

1. **Architecture/sequence variation** — the same monomer pair appears as ≥2 of {random, gradient, alternating, block, multiblock, star, cyclic, graft}. (Composition-only variation is *not* enough — that's the diblock case, already done.)
2. **A measured property** — Tg, morphology/phase, modulus/mechanical, LCST, etc. (architecture-dominant properties per your Q4 search).
3. **Controlled confounds** — ideally composition and chain length held ~fixed while sequence varies (this is the Tao-2022 design; it's what makes the architecture signal clean).
4. **ML-ready size** — ≥ a few hundred rows, machine-readable, or a paper with SI tables you can scrape.

If a candidate lacks #1, it's another diblock/composition dataset — skip it.

---

## Consensus queries (natural-language; run each, read the cited datasets)

**A. Pin down the Tao-2022 lead + its dataset**
1. `Which datasets vary monomer sequence (random, gradient, block, alternating) at fixed copolymer composition and report glass transition temperature?`
2. `What machine learning studies predict copolymer glass transition temperature from monomer sequence rather than composition alone, and what data did they use?`

**B. Sequence-controlled / precision polymers with measured properties**
3. `Are there datasets of sequence-defined or sequence-controlled polymers with measured thermal, mechanical, or self-assembly properties suitable for machine learning?`
4. `Do experimental datasets exist where the same monomers are arranged in different sequences and a physical property is measured for each arrangement?`

**C. Multiblock / segmented / topology-diverse**
5. `What machine learning datasets exist for multiblock copolymer property or morphology prediction beyond diblock systems?`
6. `Are there datasets comparing linear, star, cyclic, or graft copolymers of identical chemistry and composition with a measured property?`

**D. Simulation-generated (fallback if experimental data is too thin)**
7. `Have coarse-grained or SCFT simulations generated labeled datasets of copolymer sequence or architecture versus self-assembled morphology usable for machine learning?`
8. `What computational polymer datasets vary chain architecture at fixed composition and report phase behavior or thermal properties?`

---

## Google Scholar boolean strings (for dataset/SI hunting)

```
("sequence-controlled" OR "sequence-defined" OR "monomer sequence") copolymer dataset ("glass transition" OR morphology OR modulus) "machine learning"

copolymer (random OR gradient OR block OR alternating) "fixed composition" (Tg OR "glass transition") dataset prediction

multiblock copolymer ("phase behavior" OR morphology) dataset ("machine learning" OR "neural network") -diblock

(star OR cyclic OR graft OR "linear") copolymer "same composition" architecture property dataset

"sequence-controlled polymers" (thermal OR mechanical OR self-assembly) "training data" OR "data set" OR "supporting information"
```

Tip: append `filetype:csv OR filetype:xlsx` or search the paper's SI directly — architecture-diverse data usually lives in SI tables, not a released benchmark.

---

## Known leads to check first (from your 4 Consensus searches)

| Lead | Why | What to verify |
|---|---|---|
| **Tao et al. 2022** (sequence-varying Tg, RNN) | Exact design you want: composition fixed, sequence varied, property measured | Is the dataset released / in SI? Size? Chemistries? |
| **Xing et al. 2023** (gradient copolymer sequence → morphology / Tg) | Sequence-distribution → property, architecture-dominant | Is it a dataset or a few samples? |
| **Williams et al. 2024** (explicit full-chain BCP graphs, TPE stereochem) | Uses full-chain graphs → likely has sequence-resolved data | Data availability, size, property |
| **Kimmig et al. 2026** (structure-aware GCN, MMD, R²=0.89 Tg) | Monomer-as-node + dispersity; may include architecture variation | Does it vary architecture or just chemistry/MMD? |
| **Zhu et al. 2026** (mixing-vector, 10 copolymer datasets) | Aggregates 10 copolymer datasets — some may be sequence-varying | Which of the 10 vary sequence at fixed composition? |

Start with **Tao 2022** — if its data is available and architecture-diverse, that's the fastest path to the novel-architecture paper, and your octamer/sequence-aware machinery slots straight in.

---

## Decision rule after the hunt

- **Hit found (meets all 4 criteria):** that dataset becomes the home for the sequence-aware architecture paper; port HPG-hier/octamer to it, baselines = composition-only + RNN.
- **Only simulation data found:** viable but weaker novelty (label is a model, not experiment); still publishable as a method paper.
- **Nothing meets #1:** the architecture-dominant paper is data-gated with no data → bank the EA/IP diagnostic paper, list architecture-dominant as future work pending data.

---

# RESULTS — what the hunt found (2026-07-26)

**Headline:** experimental architecture-dominant data at ML scale **does not exist** — every one of the four searches names data scarcity as the field's central bottleneck, and true fixed-composition sequence sweeps are described as "rare / small / not yet standardized for ML." So the novel-architecture paper's realistic home is **simulation-generated data**, where architecture/sequence *is* varied at fixed composition and a property is labeled. That's a method-paper novelty (label = a model, not experiment) but the data is real and ML-ready.

## Tao 2022 repo — AUDITED, and it's OUT (verified 2026-07-26, `Desktop/Copolymer/`)

The repo (iScience 2022, *not* STAR Protocols) contains 4 datasets. None is a fixed-composition architecture sweep with measured labels:

| Tao dataset | Source | Size | Label | Architecture variation? |
|---|---|---|---|---|
| Dataset 1 | Wilbraham 2019 | 47,988 trimers | EA/IP/excitation | **None** — CNN "sequence" input is a constant tile (even==odd positions identical); it's oligomer property prediction, *same optoelectronic space as your EA/IP work* |
| Dataset 2 | Reis 2021 (¹⁹F-MRI) | 411 | ¹⁹F-NMR SNR | **Synthetic random only** — sequence is a random draw from composition; no block/gradient comparison |
| Dataset 3 | Pilania 2019 (PHA Tg) | **131** | Tg (K) | Real `Random/Block` column, but **67 homo / 56 random / 8 block**, composition not matched → useless for training |
| Dataset 4 | NIMS PolyInfo | — | — | **Not in repo** (`"Need query data from PolyInfo"`) |

**Conclusion:** the field's flagship "sequence-aware copolymer ML" paper is trained on composition/oligomer labels; its random/block/gradient handling is a *model capability demonstrated on synthetic sequences*, not fitted to architecture-resolved data. This **personally confirms** the Consensus finding — no experimental fixed-composition architecture-varied labeled dataset exists.

## Curtis 2025 & Webb 2020 — AUDITED (2026-07-26, local repos)

**Curtis 2025 (`stochastic-sequences/`) — the best architecture-dominant benchmark found, with one structural catch.**
- **270 unique sequences, ALL 12A/8B → composition held EXACTLY fixed.** Only arrangement varies → composition cannot leak. This is the clean architecture-at-fixed-composition test.
- Task: 20-bead A/B sequence → 2D self-assembled morphology embedding (Z0, Z1), regression. 270 seq × 11 stochasticity p × 5 runs = 14,843 rows. ML-ready (`no_avg_dataset.csv`, `nn_input.npy`).
- **Catch: monomers are featureless A/B beads — NO monomer chemistry.** So HPG-hier's Stage-1 (chemistry D-MPNN) is null; the model collapses to its Stage-2 sequence encoder, competing head-on with Curtis's RNN. Validates *half* the model.

**Webb 2020 (`supporting_information/`)** — 1,540 sequences (ClassI) → single-chain Rg; coarse-grained beads; composition varies. Same "no chemistry" limitation, smaller, weaker label. Fallback only.

## The real structural gap (stated plainly)
**No ML-ready dataset has BOTH rich monomer chemistry AND dominant architecture signal.** Your two datasets are complementary opposites:
- **EA/IP** — chemistry-dominant, architecture 1–4% → stresses **Stage-1** only.
- **Curtis** — architecture-dominant, zero chemistry → stresses **Stage-2** only.

Neither alone exercises the full two-stage hierarchy. The only way to test both simultaneously is to **generate a chemistry-×-architecture dataset** (MD/SCFT) — a separate project.

## Ranked candidate datasets (after all audits)

| Rank | Dataset | Size | Label | Role for you | Caveat |
|---|---|---|---|---|---|
| **1** | **Curtis 2025** (audited, local) | 270 seq / 14.8k rows | 2D morphology embedding | Clean Stage-2 (architecture) benchmark, fixed composition | No chemistry → tests Stage-2 only |
| **2** | **Webb 2020** (audited, local) | 1,540 | radius of gyration | Fallback Stage-2 benchmark | No chemistry; smaller; Rg |
| **3** | **Jiang 2024** | 1,342 | rheology / topological | topology-diverse at fixed node/edge counts | Topology descriptors, not chemistry |
| ~~Tao 2022~~ | — | — | — | **OUT** (8 block samples) | — |
| ~~block.xlsx (BCDB)~~ | — | — | — | **OUT** (99.7% diblock, solved) | — |

## Confirmed dead ends
- **Your `block.xlsx` = BCDB (Rebello et al. 2024)** — 5,400+ experimental di/multiblock phase measurements, BigSMILES-encoded. It's a real published benchmark, but 99.7% diblock in your extract, and diblock phase is ~solved (Arora 2021 RF ~90%). Dead for the *architecture-dominant* angle.
- **Experimental sequence-Tg** (PLGA, styrene/isoprene, styrene/4-hydroxystyrene; Kim 2006, Wong 2007, Patil 2024, Bocharova 2026) — real effects, but tens-of-samples scale, not ML-ready.

## The genuinely open gap (your novelty target)
> *"No ML paper trains on datasets where random, gradient, block, and alternating sequence are varied at fixed composition and encoded as the predictive signal."* (Which_datasets… search, verbatim finding)

That is the hole a sequence-aware **hierarchical GNN** (HPG-hier / octamer) can fill — Tao 2022 used only stacked Morgan-fingerprint vectors + shallow ML, not a hierarchical message-passing architecture. So: *same task, better architecture* is a clean, defensible contribution.

## Recommendation (after all audits) — the two-regime paper

**Strongest paper buildable with data you already have:** validate the hierarchy component-wise across two complementary regimes.
1. **EA/IP (chemistry-dominant):** Stage-1 handles chemistry; model still recovers the small (1–4%) architecture signal. → your existing diagnostic decomposition.
2. **Curtis (architecture-dominant, fixed composition):** Stage-2 sequence encoder dominates and should beat the RNN baseline where architecture *is* the whole signal.

**Contribution =** (a) the diagnostic decomposition that identifies which regime a property lives in, and (b) a hierarchical representation whose two stages are each validated in the regime where they matter. Novel (no one has the decomposition or the two-regime validation), fully data-supported, no new simulation required.

**Caveats to be honest about in the paper:**
- On Curtis, HPG-hier = Stage-2 only (no chemistry beads) → frame as validating the architecture *encoder*, not the full hierarchy.
- Testing both stages *simultaneously* needs a chemistry-×-architecture dataset → name as future work (MD/SCFT generation).

**Fallback:** if the two-regime framing isn't wanted, bank the EA/IP diagnostic paper alone; use Curtis as a separate "sequence-to-morphology encoder" contribution.
