# Design Note — Hierarchical Representation (Lever B) & how it fits with the objective work

**Purpose:** answer three things — (1) what actually differs between the wDMPNN and ChemArch encoders, (2) where the HPG paper (Han et al., *Chem. Eng. J.* 2025) sits and what it leaves open, (3) a concrete hierarchical implementation to try, and how to run it alongside the objective pilot *without* losing attribution.

---

## 1. wDMPNN vs ChemArch — the encoders are structurally different

They are not two flavours of the same message passing. They differ in **where composition mixing happens** and **how architecture enters**.

| | **wDMPNN** | **ChemArch** (Stage2D `2d1_arch`) |
|---|---|---|
| Stages | Single-stage, flat | Two-stage / factored |
| Graph | One weighted directed MPNN over the **whole copolymer ensemble graph** (Coley-style stochastic/weighted edges) | Stage 1: a GNN encodes **each monomer separately** → holistic embeddings h_A, h_B |
| Composition mixing | Implicit, **at the atom/bond level**, inside one message-passing pass | Explicit, **at the monomer level**, by a *fixed linear rule*: `h_mix = f_A·h_A + f_B·h_B` |
| Architecture (sequence) | Implicit — encoded in edge weights / stochastic connectivity | Explicit — a **discrete architecture label** (block/random/alternating ∈ {0,1,2}) → embedding → residual `h_poly = h_mix + α_arch·r_arch` |
| Consequence (from your diagnostics) | Novel monomers decompose into **shared substructures** → strong chemistry extrapolation. But architecture is never *supervised*, so MSE lets it shrink. | Architecture is explicit and separately supervised → best recovery. But composition is a **hand-specified linear pool of holistic monomer embeddings** with no atom-level sharing → a novel monomer has no compositional handle → the backbone can't extrapolate (the failing component). |

**One-line version:** wDMPNN mixes chemistry *at the atoms* inside a single graph and leaves architecture implicit; ChemArch mixes chemistry *at the monomers* by a fixed linear rule and bolts architecture on as a discrete label. That is precisely why wDMPNN wins chemistry extrapolation and ChemArch wins architecture recovery — and why neither wins both.

*(Confirmed in code: `chemprop/nn/stage2d.py` — `h_mix = f_A*h_A + f_B*h_B`, `ARCH_LABEL_MAP = {alternating:0, random:1, block:2}`, residual variants 2d0/2d1.)*

---

## 2. Where HPG fits — and the gap it leaves open

HPG (Han et al. 2025) is essentially **ChemArch's two-stage idea done properly**:
- **Stage 1 (monomer-level graph):** each repeating unit is a molecular graph; a GAT/AFP encodes it → monomer embedding, then abstracted to a **virtual monomer node** with designated connection points.
- **Stage 2 (polymer chain-level graph):** virtual monomer nodes connected by **bidirected edges that encode the actual sequence/topology** (block sequence, branching, degree of polymerization); message passing runs at this level.

So instead of ChemArch's *fixed linear mixture + discrete architecture label*, HPG uses a **learned message-passing graph whose topology is the architecture**. That is a genuinely richer representation of both composition mixing and sequence — it is the natural fix for ChemArch's backbone.

**The gap HPG explicitly leaves open (quote the paper):** HPG works for *regularly-structured* polymers (homopolymer, block, alternating, periodic, branched) but "still faces limitations when applied to certain complex polymers including **random copolymers** and network-type polymers." Their suggested fix: "**probabilistic representations for random copolymers**."

Two things follow, both directly relevant to us:
1. **Our dataset is ~1/3–2/3 random** (12,276 of 18,414 groups are block+random; random is a first-class architecture here). So HPG applied naively would be weakest on exactly the copolymers we have most of.
2. **wDMPNN *is* the probabilistic representation HPG asks for** — Coley-style stochastic edges are cited in the HPG paper itself (ref [52]) as the probabilistic-edge approach. We already have it.

That is the opening: HPG's stated future work is a representation we already own. Combining them is well-motivated and not incremental.

---

## 3. What I propose to try (in priority order)

**Baseline first:** reproduce HPG-GAT under *our* protocol. HPG scaffolding already exists in the repo (`chemprop/nn/hpg.py` HPG-GAT layers incl. a Phase-2A "edge-in-message" variant; `chemprop/data/hpg.py` `BatchHPGMolGraph`). Train it on EA/IP with our Group-/Pair-/Monomer-heldout splits and score with the diagnostic battery. Note: HPG's own generalization claims are on random test splits for ionic conductivity — nobody has stress-tested it under strict monomer-heldout OOD or on electronic properties. Doing so is itself a contribution.

Then, the improvements — each is a distinct, publishable increment:

1. **Stochastic/probabilistic chain-level edges for random copolymers (the flagship improvement).** Build the Stage-2 chain graph with weighted/stochastic edges (à la wDMPNN) so random copolymers get a proper probabilistic sequence representation, while block/alternating/branched keep deterministic topology. This directly resolves HPG's own stated limitation using our existing machinery. Strongest single idea.

2. **Cross-monomer coupling at connection points (matters specifically for EA/IP).** HPG abstracts each monomer to a virtual node and encodes monomers independently — so it loses the atom-level electronic coupling *across* the junction bond. For conjugated copolymers, EA/IP depend on conjugation that spans the monomer–monomer junction; ionic conductivity (HPG's target) is far less sensitive to this. Proposal: allow limited message passing across connection-point atoms between adjacent monomer subgraphs before/after abstraction. This is a property-motivated refinement HPG doesn't need for its task but we do for ours.

3. **Apply the within-group objective (Lever A) to the hierarchical model.** Your pilot's whole lesson is that even a good representation under MSE under-fits the 1–4% architecture factor. So `L_within` (already implemented, `chemprop/nn/within_group_loss.py`) belongs on HPG too. Representation gives the *capacity/structure*; the objective ensures it's *supervised*. This is how the two threads combine rather than compete.

4. **Architecture-aware readout instead of concatenation.** HPG concatenates component embeddings + numeric features into a final MLP. A learned, sequence-aware pooling of the chain-level graph (attention readout keyed on architecture) is a cheap, natural upgrade.

---

## 4. How to run Lever A and Lever B in parallel *without* losing attribution

Your supervisor wants both at once. That is fine — provided you don't build **one** model that changes representation *and* objective simultaneously (that's the confound I flagged before). The clean way to honour "do both" is a **2×2 factorial**:

|                | **MSE only** | **MSE + λ·L_within** |
|----------------|--------------|----------------------|
| **wDMPNN**     | baseline | the objective pilot (done, fold 0) |
| **HPG (hierarchical)** | HPG reproduced under our splits | hierarchical + explicit supervision |

This design *preserves* attribution rather than muddying it: the row effect isolates **representation**, the column effect isolates **objective**, and the interaction tells you whether they're complementary or redundant. It directly answers the deeper thesis question — *how should a model allocate capacity vs. supervision across variance components* — instead of just "which model wins." So the supervisor's parallel request, framed as a factorial, makes the paper **stronger**, not messier.

The decision gate from the roadmap still applies to *what you claim*: the pilot's Monomer-heldout result tells you whether the objective alone suffices; the HPG row tells you what representation buys on top. If the objective alone already gives best-of-both, HPG becomes the "can we push further / is capacity ever the bottleneck" arm. If it doesn't, HPG is the answer — and you'll have it in hand.

---

## 5. Concrete implementation steps

1. **Data:** extend `BatchHPGMolGraph` construction to emit chain-level edges for our three architectures; add a stochastic-edge mode for random. (`chemprop/data/hpg.py`.)
2. **Model:** two-stage module — Stage-1 monomer GAT (reuse `HPGGATLayer`), Stage-2 chain-level GAT over virtual monomer nodes; optional connection-point coupling. (`chemprop/nn/hpg.py`.)
3. **Objective:** wire `within_group_loss` into the HPG trainer (group-aware sampler already exists from the pilot).
4. **Protocol:** identical splits/folds/seeds/eval as the corrected benchmark; run all four factorial cells.
5. **Scoring:** the existing diagnostic battery (gm-R², ΔR², ordering, calibration, per-fold stats) on every cell.

## 6. Risks / honest flags

- **HPG on random copolymers is unproven** — the stochastic-edge extension is genuine research, not a port; budget for it not working first try.
- **Cross-monomer coupling** blurs the "monomer = clean substructure" abstraction; validate it actually helps EA/IP before committing.
- **Compute/time:** a 2×2 × 3 splits × folds × seeds is a large grid — stage it (Group-disjoint first, as with the pilot).
- **Attribution only holds if the four cells share everything except the varied factor** — same encoder dims, schedule, data, eval. Keep them identical.
- The noise-floor gate still governs how hard you can push any architecture-recovery claim, HPG included.

---

*Refs: Han, Yokoo, Park, Oyaizu, Park, "Deep learning prediction of ionic conductivity in polymer electrolytes using hierarchical polymer graphs," Chem. Eng. J. 521 (2025) 166829. Coley et al. stochastic-edge weighted D-MPNN (HPG ref [52]) = our wDMPNN. Code: `chemprop/nn/stage2d.py`, `chemprop/nn/hpg.py`, `chemprop/nn/within_group_loss.py`, `chemprop/data/hpg.py`.*
