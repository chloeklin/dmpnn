# Review of the attribution programme — what we are actually explaining

Written 12 August 2026. Purpose: check whether the five-factor ablation programme can answer
the question "why does the octamer beat wD-MPNN", and specify what a complete decomposition
would require.

**Headline: it cannot, as currently designed. The five factors address 16% of the gap on EA
and 24% on IP. The remaining 76–84% is a step no ablation currently touches.**

---

## 1. The problem: we have been decomposing the wrong difference

The five factors in HANDOFF §7 separate **HPG-octamer from HPG-hier**. But the claim we make
in the paper, on slides, and to supervisors is about **HPG-octamer versus wD-MPNN**. Those are
different quantities, and the difference between them is most of the effect.

A-split ΔR², median over 9 folds:

| Model | EA ΔR² | IP ΔR² |
|---|---|---|
| wD-MPNN (published config) | 0.397 | 0.565 |
| wD-MPNN (our config, 300 ep) | 0.433 | 0.450 |
| **HPG-hier (2-node)** | **0.776** | **0.808** |
| HPG-hier + junction coupling | 0.797 | 0.761 |
| **HPG-octamer** | **0.849** | **0.886** |

Decomposing the gap we actually claim:

| Step | EA | share | IP | share |
|---|---|---|---|---|
| wD-MPNN → HPG-hier — **different model family** | **+0.379** | **84.0%** | **+0.243** | **75.7%** |
| HPG-hier → octamer — **the five factors** | +0.072 | 16.0% | +0.078 | 24.3% |
| **Total** | **+0.452** | | **+0.321** | |

Every ablation run so far — K=1, position embeddings, and the planned arms C and D — operates
inside the 16–24% slice. **Nothing has been run on the 76–84% step.**

This is not a reason to discard the work done. The five-factor arms are correct, and two
factors are properly excluded. But the programme cannot answer the question we are posing,
and we should stop describing it as if it can.

---

## 2. Full factor inventory, both levels

### Level 1 — wD-MPNN → HPG-hier (84% / 76% of the gap, entirely untested)

| # | Factor | wD-MPNN | HPG-hier | Status |
|---|---|---|---|---|
| L1 | **Graph topology** | one flat graph containing both monomers, joined by stochastic edges | monomers encoded **independently**, then a 2-node graph | **untested** |
| L2 | **Where architecture enters** | stochastic edge weights inside atom-level message passing | explicit stage-2 edge features between monomer nodes | **untested** |
| L3 | **Where stoichiometry enters** | atom feature weighting | stage-2 readout weighting | **untested** |
| L4 | **Cross-monomer atom messages** | inherent — stochastic edges carry them | off by default; optional junction coupling | **partly tested** — see below |
| L5 | **Codebase** | chemprop 1.4.0 | chemprop 2.2.0 | **untestable directly** |
| L6 | **Training configuration** | batch 50, 30 epochs, no early stopping | batch 64, 100 epochs, patience 15 | **bounded** — see below |

**L4 is partly answered already.** `hpg_hier` (junction off) versus `hpg_hier_junction`
(2 coupling steps) gives EA ΔR² 0.776 → 0.797 and IP 0.808 → 0.761. So cross-monomer atom
messages move ΔR² by about ±0.02–0.05 — small, and opposite in sign on the two targets. This
is an existing ablation nobody has written up as one.

**L6 is bounded already.** wD-MPNN at its published config versus our 300-epoch config differs
by 0.036 (EA) and −0.115 (IP) in ΔR². So training configuration accounts for at most ~10% of
the family gap on EA. It is not the explanation.

**L5 is the hard one.** Two different codebases implement message passing, featurisation,
initialisation and optimisation differently. You cannot ablate "the codebase". The only way
through is to reimplement one model inside the other's codebase — see §4.

### Level 2 — HPG-hier → octamer (16% / 24% of the gap)

| # | Factor | Status |
|---|---|---|
| 1 | 8-slot chain instead of 2-node graph | **open** — confounded with factor 4 |
| 2 | Learned position embeddings | **excluded** (A split, pre-registered, outcome 3) |
| 3 | Attention readout instead of stoichiometry-weighted | **open** — arms C and D |
| 4 | Discards the 16-d port-pair edge features | **open** — confounded with factor 1 |
| 5 | 16 sampled sequences averaged instead of 1 | **excluded** (B split, pre-registered, outcome C) |

**Factors 1 and 4 cannot be separated by arms C and D.** The octamer branch in
`hpg_hier.py:265` returns before the `Stage2Layer` loop, so any 8-slot model also loses the
edge features. Both arm D and the octamer sit in the "no edge features" row; both HPG-hier and
arm C sit in the "has edge features" row. Moving between rows always changes two things.

---

## 3. What is genuinely resolved, and what a complete decomposition needs

**Resolved:** factors 2 and 5, both pre-registered, both negative. L6 bounded. L4 partly
characterised but unwritten.

**Arms C and D will resolve:** factor 3 (readout), cleanly.

**Will still be open after arms C and D:** factors 1 and 4 (confounded with each other), and
all of L1–L3 and L5 — which is three-quarters of the effect.

### The missing experiments, in order of how much of the gap they address

| # | Experiment | Addresses | Runs | Needs code? |
|---|---|---|---|---|
| **A** | **Flat-graph wD-MPNN equivalent implemented in our codebase** | L1–L3 and L5 jointly — **the 76–84%** | 54 | **yes, substantial** |
| **B** | 8-slot chain that *also* receives the port-pair edge features | separates factor 1 from factor 4 | 54 | yes, moderate |
| C | Arms C and D | factor 3 | 108 (12 pilot) | done |
| D | Write up the existing junction on/off/1-step comparison | L4 | **0** | no |
| E | Factor 5 on the A split, factor 2 on the B split | symmetry | 108 | no |

**Experiment A is the one that matters and the one nobody has scoped.** The idea: implement, in
chemprop 2.2.0, a model that represents the polymer as a single flat graph with stochastic
inter-monomer edges — i.e. wD-MPNN's representation inside our training stack. Then:

- **flat-in-our-codebase vs wD-MPNN (theirs)** isolates **L5**, the codebase, with the
  representation held fixed
- **flat-in-our-codebase vs HPG-hier** isolates **L1–L3**, the hierarchy, with the codebase
  held fixed

That is the only construction that attributes the 76–84%. It is a genuine implementation task,
not a config change — but it is the difference between "we know why our model wins" and "we
know why one of our models beats another of our models."

**Experiment B** is a smaller patch: allow `OctamerEncoder` to consume the stage-2 edge
features, so the 8-slot row can exist both with and without them. That completes the Level-2
2×2 into a 2×2×2.

---

## 4. What this means for the compute decisions in front of us

**Arms C and D are worth less than they looked.** Even resolved perfectly, factor 3 explains a
share of a 16–24% slice. Spending 3.9 kSU on the full R1 arm to attribute part of one-fifth of
the effect is poor value while three-quarters sits untouched. **Run the 12-run pilot** — it is
cheap and the code is already written — but do not commit to the full 108 until experiment A
is scoped.

**Experiment D costs nothing.** The junction on/off/1-step comparison is already run; it just
needs analysing and writing up. Do this first — it is free and it is the only Level-1 factor
with any data.

**Revised priority:**

| | Job | Runs | Rationale |
|---|---|---|---|
| 1 | Published-config wD-MPNN on the B split | 54 | Only thing blocking Paper 1. PBS files already exist |
| 2 | Analyse the existing junction variants | **0** | Free; the only Level-1 evidence we have |
| 3 | Arms C and D **pilot only** | 12 | Cheap, code written, resolves factor 3 |
| 4 | **Scope experiment A** | 0 for now | Design and cost the flat-graph model |
| 5 | Experiment B | 54 | Separates factors 1 and 4 |
| 6 | Full arms C/D, symmetry arms | 162 | Only if the timeline allows |

---

## 5. What to say tomorrow

Do not present the five-factor programme as "why the octamer beats the baseline". Present it
accurately:

> *"I've been careful about attribution, and that care just told me something uncomfortable.
> The five factors I've been eliminating explain the difference between two of my own models.
> Against the published baseline they account for about a sixth of the gap on EA and a quarter
> on IP. The other three-quarters is the step from wD-MPNN to my two-stage design, and I have
> not tested that at all — because it isn't a single knob, it's a different architecture in a
> different codebase."*

Then the constructive half:

> *"Two of the five are properly excluded, pre-registered, and I'll report them as negative
> results. But if I want to claim I understand why my model wins, I need to build a
> flat-graph version of the baseline representation inside my own codebase. That's the
> experiment that attributes the three-quarters, and it's the one I'd want your view on."*

This is a stronger position than it sounds. Finding that your own attribution programme
addresses the minority of the effect — before a reviewer does — is exactly the kind of
self-audit that Paper 1 argues the field should be doing.
