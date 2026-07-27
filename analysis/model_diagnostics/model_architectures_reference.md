# Model Architectures — Reference

Full input / representation / architecture / output for every model compared on EA/IP, plus the two proposed HPG-hier variants (Q1, Q2). All models predict two scalar targets (EA vs SHE, IP vs SHE, in eV) from a binary copolymer.

---

## Glossary

- **Repeat unit** — a monomer's structure *as it appears in the polymer*, with attachment points `[*:n]` where it bonds to neighbours (parsed from the `WDMPNN_Input` column, **not** `smiles_A/smiles_B`, which are the pre-polymerisation monomers with reactive leaving groups).
- **Port** — a numbered attachment point `[*:n]`. Binary copolymer: ports **1,2 → monomer A**, ports **3,4 → monomer B**.
- **Transition weight `w_ij`** — first-order Markov probability that a monomer of type *i* is followed by type *j*. Encodes the architecture: alternating = off-diagonal, block = diagonal-dominant, random = rows equal the fractions. Derived from the `WDMPNN_Input` bond rules.
- **Group mean / architecture deviation** — for a fixed chemistry+composition context, the group mean = chemistry baseline; the within-group deviation = the architecture-induced signal.
- **Stochastic / weighted edge** — a bond weighted by its probability of occurring in the ensemble; the weight *scales the message* (wDMPNN).

---

## 1. wDMPNN  (weighted directed MPNN — Aldeghi & Coley)

- **Input:** one *flat* atom graph built from `WDMPNN_Input` — all atoms of **both** repeat units in a single graph, joined by **probability-weighted stochastic bonds** at the attachment points.
- **Nodes = atoms.** Features = chemprop atom features (element, degree, formal charge, #H, hybridisation, aromaticity, mass).
- **Edges = bonds.** Features = bond features (type, conjugated, in-ring, stereo). Inter-monomer junction bonds carry a **probability weight**.
- **Message passing = directed, bond-based (D-MPNN), T steps.** Messages live on directed bonds: `h⁰(u→v)=ReLU(W_i·[x_u ; e_uv])`; each step sums incoming bond messages, **messages along stochastic edges are scaled by the edge weight**; then aggregate bond messages → atoms.
- **Readout:** atom pooling **weighted by monomer stoichiometry** → polymer vector.
- **Output head:** FFN → EA, IP.
- **Key idea:** one connected atom graph; architecture is implicit in **edge weights**; junction chemistry is kept; **1 message-passing pass**.

## 2. HPG (original — Han et al.)

- **Input:** one *fused* graph combining atom nodes + virtual monomer nodes + three edge types (not two separate graphs).
- **Nodes (two types):**
  - **Atom nodes** — real atom features (49-d original / 130-d in our chemprop reimplementation).
  - **Virtual monomer nodes** (one per monomer) — **initialised to an all-ones vector** (no chemistry; acquire meaning through message passing).
- **Edges (three types, 1-D scalar feature each):** atom–atom = bond order (1/1.5/2/3); atom→monomer = 1.0 (abstraction); monomer–monomer = `degree` (chain edge). **Junction atoms are dropped.**
- **Message passing = one shared GAT stack (6 layers) over ALL nodes.** Attention `= LeakyReLU(src·a_src + dst·a_dst + W_edge(e))`, edge-softmax over incoming edges, `message = attention × W_node(h_src)` (edge in attention only), aggregate = sum, mean over heads.
- **Readout:** **sum-pool over all nodes** (atoms + monomer nodes) → linear.
- **Output head:** FFN → EA, IP.
- **Key idea:** atoms + virtual monomer nodes fused in one graph, **1 shared pass**, junction dropped, sum-pool everything.

## 3. ChemArch  (Stage2D `2d1_arch`)

- **Input:** each monomer encoded **separately** by a Stage-1 GNN → holistic monomer embeddings `h_A`, `h_B`; plus fractions `f_A`, `f_B`; plus a **discrete architecture label** (alternating/random/block = 0/1/2).
- **Composition mixing (fixed rule):** `h_mix = f_A·h_A + f_B·h_B` — a hand-specified linear pool.
- **Architecture (residual):** `h_poly = h_mix + α_arch · r_arch`, where `r_arch` = MLP of a learned embedding of the discrete architecture label; `α_arch` a per-architecture learnable scalar.
- **Readout:** `h_poly`.
- **Output head:** MLP → EA, IP.
- **Key idea:** monomer-level chemistry + **fixed linear composition** + explicit **discrete-label** architecture residual. Strong on architecture recovery; weak on chemistry extrapolation (the composition backbone can't extrapolate to unseen monomers).

## 4. HPG-hier  (ours — current)

**Two genuinely separate message-passing stages.**

- **Stage 1 — per-monomer chemistry (shared D-MPNN encoder):**
  - Each repeat unit (from `WDMPNN_Input`) as its **own** atom graph — **no inter-monomer bonds**.
  - Atom features = chemprop features **+ `is_attachment` flag + local port id (2-bit)**; the attachment atom is featurised **with its connecting bond counted** (degree matches wDMPNN).
  - Encode with the chemprop D-MPNN (shared weights for A and B) → pool atoms → **one embedding per monomer, `m_A`, `m_B`**.
- **Stage 2 — architecture graph (separate weights):**
  - Nodes = `m_A`, `m_B`, each concatenated with its stoichiometry `f_i`.
  - Edges = directed A→A, A→B, B→A, B→B **+ self-loops**. Edge feature = `[16-d port-pair connectivity | 1 transition weight w_ij]` (both from `WDMPNN_Input`).
  - Layer: `message = MLP( [h_source ; edge_feature] )`; **summed (unweighted)**; residual; ~2 layers. → the transition weight is used as a **feature**, not a multiplier.
- **Readout:** **stoichiometry-weighted pool** of updated monomer embeddings.
- **Output head:** MLP → EA, IP.
- **Key idea:** chemistry (Stage 1) and architecture (Stage 2) in **2 separate passes**; junction represented at Stage 2 as an edge feature. Best architecture recovery on unseen chemistry; chemistry at parity with wDMPNN except where the isolated Stage-1 encoding mis-places a hard monomer's baseline.

## 5. HPG-hier + weighted Stage-2 edges  (Q1 variant)

- **Same as HPG-hier**, with **one change in the Stage-2 layer:** the transition weight `w_ij` becomes a **multiplier** on the message rather than just a feature:
  `message = w_ij · MLP( [h_source ; port_pair_feature] )`, then summed.
- Because the transition weights are row-normalised, Stage-2 mixing becomes a proper **architecture-weighted (convex) combination** — a block copolymer's A→A message is scaled up, A→B scaled down — exactly the wDMPNN inductive bias, now at the monomer level.
- **Key idea:** structurally **enforce** the architecture-weighted mixing instead of letting the MLP learn it from a scalar. Targets block-vs-random discrimination (which currently rests on one number). Cheap (≈ one-line change). Ablation: weight as feature / as multiplier / as both.

## 6. HPG-hier + explicit octamer Stage-2  (Q2 variant)

- **Stage 1 unchanged** (per-monomer → `m_A`, `m_B`).
- **Stage 2 replaced** by an **explicit length-8 (octamer) sequence** encoder:
  - Build a path of 8 monomer-instance nodes, each = `m_A` or `m_B` by its position in the sequence.
  - **Block / alternating:** the octamer is deterministic (AAAABBBB / ABABABAB) → one sequence.
  - **Random:** no single sequence — **sample K octamers** (the wDMPNN paper used up to 32) and **average the predictions**.
  - Encode the octamer with a small sequence model (path GNN / mini-transformer / RNN over 8 positions) → sequence representation → pool over positions.
- **Output head:** MLP → EA, IP (for random, mean over K sampled octamers).
- **Key idea:** replace the 2-node *ensemble-average* representation with the **explicit sequence**. Tests a core wDMPNN assumption: `property(average structure)` (weighted edges) vs `average(property(sampled sequences))` (octamers) — which differ because the model is nonlinear, potentially most for random copolymers. Captures positional/neighbour effects the transition matrix cannot.
- **As built:** sequences are **architecture-conditioned from the `WDMPNN_Input` bond rules** (an early bug that used fractions only — making all architectures identical — was fixed). Block/alternating use the deterministic canonical sequence; random uses K=16 transition-weighted samples with **prediction averaging**. Encoder = path-GNN with **positional embeddings + attention pooling** (order-aware).

## 7. HPG-hier + junction coupling  (chemistry fix — Stage 1 change)

- **Stage 2 unchanged** (baseline transition-graph, feature).
- **Stage 1 modified:** after encoding each monomer's atoms independently, **insert the inter-monomer junction bonds** (connecting the real attachment atoms of A and B, from the `WDMPNN_Input` cross-monomer bond rules, weighted by connection probability) and run **`n_coupling_steps`** extra message-passing steps on the **combined graph (intra-monomer bonds + junction edges)** — so cross-junction context propagates *into* each monomer — then pool to `m_A`, `m_B`.
- **Output head:** MLP → EA, IP.
- **Key idea:** give the isolated Stage-1 encoding a **limited atom-level view across the junction** (what wDMPNN has), to fix the chemistry-baseline mis-placement for hard monomers — *without* collapsing into a flat graph. This is the targeted fix for the fold-1 (dibenzothiophene sulfone) EA failure. Tunable via `n_coupling_steps` (2 over-couples; 1 pending).

## 8. HPG-hier + wDMPNN ensemble  (no training — post-hoc average)

- **Not a new model:** the arithmetic mean of the `HPG-hier` and `wDMPNN` predictions.
- **Key idea:** the two representations fail on *different* held-out monomers (complementary blind spots), so averaging their predictions covers both — tests and exploits the complementarity directly.

---

## Summary comparison

| | **wDMPNN** | **HPG (orig)** | **ChemArch** | **HPG-hier** | **+Q1 weighted** | **+Q2 octamer** |
|---|---|---|---|---|---|---|
| Graph | 1 flat atom graph | 1 fused (atoms + monomer nodes) | monomer-level (fixed mix) | 2 monomer graphs + 2-node graph | same as HPG-hier | 2 monomer graphs + 8-node sequence |
| Message-passing passes | 1 | 1 (shared) | Stage-1 GNN + residual | **2 (separate)** | 2 | 2 |
| Monomer chemistry | atom-level, joint | atom→monomer pooling | separate, holistic | separate D-MPNN (Stage 1) | same | same |
| Architecture encoding | weighted junction edges | chain edges (degree) | discrete label residual | Stage-2 edge: port-pair + `w_ij` (feature) | Stage-2 edge: `w_ij` as **multiplier** | **explicit octamer sequence** |
| Junction chemistry | kept (weighted bond) | dropped | none (fixed mix) | Stage-2 edge feature | Stage-2 edge feature | Stage-2 (sequence neighbours) |
| Readout | stoich-weighted atoms | sum over all nodes | `h_poly` | stoich-weighted monomers | same | pool over 8 positions |
| Output | FFN → EA/IP | FFN → EA/IP | MLP → EA/IP | MLP → EA/IP | MLP → EA/IP | MLP → EA/IP (mean over K for random) |

**Where each is strong/weak (seed-42):** wDMPNN — best chemistry extrapolation, weak architecture; ChemArch — best in-distribution architecture, collapses on unseen chemistry; HPG-hier — best architecture on unseen chemistry, chemistry ≈ wDMPNN except hard-monomer baseline placement. Q1/Q2 target the *architecture* axis; the chemistry-baseline gap needs Stage-1 **junction coupling**, which is a separate change.

---

## Variant status (Phase 1, seed 42)

| Variant | Axis targeted | Status | One-line result |
|---|---|---|---|
| **HPG-hier** (baseline) | — | ✅ done | best architecture on unseen chemistry; chemistry ≈ wDMPNN except fold-1 |
| **+Q1 wedge** (weighted edge) | architecture | ❌ **dropped** | helped in-distribution, but architecture **collapsed OOD** (IP ΔR² 0.82→0.56) |
| **+Q2 octamer** (explicit sequence) | architecture | ⏳ **re-gated, LOMO pending** | construction bug fixed; in-distribution ΔR² recovered to 0.92; unseen-chemistry unknown |
| **+junction coupling (n=2)** | chemistry | ✅ **works, over-couples** | fold-1 EA 0.575→0.925; EA LOMO **beats wDMPNN**; IP mixed + slight architecture cost |
| **+junction coupling (n=1)** | chemistry | ⏳ **pending** | tuning run — keep EA gain, recover IP/architecture cost? |
| **HPG-hier + wDMPNN ensemble** | both | ✅ done | beats both on almost everything (complementary blind spots) |

Full numbers, per-fold breakdowns and verdicts: see `variant_results_report.md`.
