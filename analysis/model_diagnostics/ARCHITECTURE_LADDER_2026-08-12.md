# Architecture-only attribution: wD-MPNN → HPG-octamer

Written 12 August 2026, revised same day after computing the decomposition on every metric.
Scope restricted to **model architecture design**. Codebase and training configuration are
held fixed by construction, not argued about.

Two corrections to earlier documents are recorded in §1 and §2. **The revision in §2 reverses
the priority advice given in `ABLATION_PROGRAMME_REVIEW_2026-08-12.md`.**

---

## 1. Correction — the codebase confound does not exist

`ABLATION_PROGRAMME_REVIEW` listed "chemprop 1.4.0 vs 2.2.0" as an untestable factor worth
most of the gap. That was wrong.

**Our wD-MPNN is our own reimplementation inside chemprop 2.2.0.**
`scripts/python/run_wdmpnn_generalization.py:289` builds it as:

```python
mp  = nn.WeightedBondMessagePassing()      # chemprop/nn/message_passing/base.py:225
agg = nn.WeightedMeanAggregation()
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)
```

Same repository, same featurisation, same Lightning loop, same data loaders as HPG-hier and
the octamer. The `wdmpnn_original` arm is also ours; it differs only in hyperparameters.

So wD-MPNN → HPG-hier is **already an architecture comparison within one codebase.**

---

## 2. Correction — the "84% is the family step" figure holds only for ΔR²

The earlier decomposition was computed on ΔR² alone. Recomputed on every metric, against
`wdmpnn_original` (published configuration) as the anchor, median over 9 A-split folds:

### EA

| Metric | wD-MPNN (pub) | HPG-hier | octamer | family step | octamer step |
|---|---|---|---|---|---|
| Overall R² | 0.967 | 0.966 | 0.984 | **−0.001** | +0.018 |
| MAE | 0.070 | 0.067 | 0.055 | +0.002 | +0.012 |
| RMSE | 0.088 | 0.083 | 0.066 | +0.005 | +0.017 |
| Group-mean R² | 0.982 | 0.971 | 0.986 | **−0.012** | +0.015 |
| **ΔR²** | 0.397 | 0.776 | 0.849 | **+0.379** | +0.072 |

### IP

| Metric | wD-MPNN (pub) | HPG-hier | octamer | family step | octamer step |
|---|---|---|---|---|---|
| Overall R² | 0.971 | 0.890 | 0.978 | **−0.081** | +0.088 |
| MAE | 0.050 | 0.068 | 0.035 | **−0.019** | +0.033 |
| RMSE | 0.064 | 0.078 | 0.048 | **−0.015** | +0.030 |
| Group-mean R² | 0.976 | 0.898 | 0.982 | **−0.078** | +0.085 |
| **ΔR²** | 0.565 | 0.808 | 0.886 | **+0.243** | +0.078 |

*Family step = HPG-hier − wD-MPNN. Octamer step = octamer − HPG-hier. Signs oriented so
positive is better on every row.*

### The finding

**HPG-hier is worse than wD-MPNN on every accuracy metric on IP, and level on EA. Its only
advantage is architecture recovery.** The octamer step accounts for *all* of the accuracy gain
— more than 100% of it, since it must first undo the family step's loss.

> **The two-stage hierarchy buys architecture recovery at the cost of accuracy.
> The octamer recovers the accuracy while keeping — and extending — the architecture recovery.**

This is the paper's own thesis applied to our own model: the attribution depends on which
metric you attribute. It is a better result than "84% is the family step", and it should be
reported as such.

### What it changes

`ABLATION_PROGRAMME_REVIEW` §4 advised holding the full arms C/D campaign because those
factors address "a 16–24% slice". **That advice was based on ΔR² alone and is withdrawn.** On
accuracy the octamer-specific factors are the entire story. Arms C and D are worth more than
that review said, not less.

---

## 3. Holding configuration fixed without touching the baseline

**Decision: `wdmpnn_original` is the only wD-MPNN we report. Its configuration is never
changed, and `regen_v1/wdmpnn` is not used anywhere**, because it does not follow the original
paper's settings.

That leaves one configuration difference across the ladder:

| | wD-MPNN (published) | HPG family |
|---|---|---|
| batch size | **50** | **64** |
| epoch cap | **30** | **100** |
| patience | 30 (cannot fire) | 15 |

**Resolution: run M1 at both configurations.** M1 is the cheapest model in the ladder and
becomes the bridge that separates configuration from architecture at exactly one point:

| Comparison | Isolates |
|---|---|
| wD-MPNN(published) → M1(published) | **architecture only**, at the published config |
| M1(published) → M1(ours) | **configuration only**, architecture held fixed |
| M1(ours) → HPG-hier → … → octamer | **architecture only**, at our config |

The baseline is never modified and `regen_v1/wdmpnn` is never used. Cost is one extra 54-run
block of the cheapest model, ≈ 0.35 kSU.

---

## 4. The architectural degrees of freedom

| | Dimension | wD-MPNN | HPG-hier | octamer |
|---|---|---|---|---|
| **D1** | Where the polymer vector is formed | atoms → polymer directly | atoms → **monomer vectors** → polymer | atoms → monomer vectors → **8 slots** → polymer |
| **D2** | Level at which inter-monomer information flows | **atom level** (stochastic edges inside message passing) | **monomer level** (stage-2 graph) | monomer level |
| **D3** | What carries the architecture signal | edge weights on atom–atom edges | 17-d port-pair + transition features, 2-node graph | an explicit 8-slot sequence |
| **D4** | Readout over the assembled object | stoichiometry-weighted mean over atoms | stoichiometry-weighted sum over 2 monomers | attention over 8 slots |
| D5 | Position embeddings | — | — | 8 learned vectors — **excluded** |
| D6 | Sequence sampling K | — | — | 16 sampled, averaged — **excluded** |

**D2 and D3 are the same choice seen twice.** If monomers are encoded independently there are
no cross-monomer atom edges left to carry weights. One rung, not two.

---

## 5. The ladder

Rungs 1 onward: chemprop 2.2.0, batch 64, 100 epochs, patience 15, seeds 42/43/44 averaged at
the prediction level, frozen protocol. Rung 0 and the M1(published) bridge run at the published
configuration.

| Rung | Model | Vector formed at | Inter-monomer info at | Architecture via | Readout | Status |
|---|---|---|---|---|---|---|
| **0** | wD-MPNN (published cfg) | atoms | atom level | edge weights | weighted mean | **exists** |
| **0b** | **M1 (published cfg)** | monomers | atom level | edge weights | stoich-weighted | **NEW — config bridge** |
| **1** | **M1 (our cfg)** | **monomers** | atom level | edge weights | stoich-weighted | **NEW — small patch** |
| **2** | HPG-hier | monomers | **monomer level** | 17-d edge features | stoich-weighted | exists |
| **3** | arm C | monomers | monomer level | 17-d edge features | **attention** | pilot ready |
| **4** | **M2** | **8 slots** | monomer level | 17-d **+ sequence** | mean | **NEW — moderate patch** |
| **5** | arm D | 8 slots | monomer level | **sequence only** | mean | pilot ready |
| **6** | octamer | 8 slots | monomer level | sequence only | **attention** | exists |

### What each step isolates

| Step | Isolates |
|---|---|
| 0 → 0b | **D1** — whether an explicit monomer-level representation exists at all, at the published config |
| 0b → 1 | **configuration**, architecture held fixed |
| 1 → 2 | **D2 + D3** — inter-monomer information moves from atom level to monomer level |
| 2 → 3 | **D4** — readout, at 2-node topology |
| 2 → 4 | **D1** — 2-node → 8-slot, edge features held |
| 4 → 5 | **factor 4** — the 17-d edge features, topology held |
| 5 → 6 | **D4** — readout, at 8-slot topology |
| 6 → posemb-off | factor 2 — **excluded** |
| 6 → K=1 | factor 5 — **excluded** |

Rungs 4 and 5 finally separate **topology from edge features** — the confound arms C and D
cannot break alone, because the octamer path returns before the `Stage2Layer` loop, so any
8-slot model also loses the features.

### Dropped: HPG-hier + junction coupling

Not a rung. It is a side branch off rung 2, and the octamer does not use junction coupling.

**But it is already run, and it answers the obvious follow-up to step 1 → 2** — *"if moving
inter-monomer information to the monomer level matters, does adding atom-level messaging back
help?"* Report it as one supporting paragraph using existing data. Zero compute.

---

## 6. The two new models

### M1 — two-level aggregation on the flat graph

Smallest patch in the programme. Keep the wD-MPNN graph and `WeightedBondMessagePassing`
exactly as they are. Replace only the aggregation:

- current: `WeightedMeanAggregation` pools **all atoms → one polymer vector**
- M1: pool atoms **per monomer** using the ownership the featurizer already tracks, giving two
  monomer vectors, then combine as `f_A·h_A + f_B·h_B`

Same message passing, same edge weights, same features. An aggregation swap.

**Answers:** does simply *having* a monomer-level representation account for the jump,
independent of how inter-monomer information flows?

### M2 — 8-slot chain that also receives the edge features

Extend `OctamerEncoder` to consume `stage2_edge_features`. For adjacent slots *i*, *i+1*
holding monomers *(m_i, m_{i+1})*, index the existing `pairs[m_i, m_{i+1}]` 16-d port-pair
vector plus the transition weight. `_stage2_edges` already computes all four monomer-pair
combinations, so this is indexing, not new featurisation.

**Answers:** of the octamer's advantage over HPG-hier, how much is the 8-slot topology and how
much is the loss of junction regiochemistry?

---

## 7. Cost and order

Per-run, measured: wD-MPNN family ≈ 6 SU, HPG family ≈ 36.5 SU.

| | Job | Runs | ~SU | Code | Why here |
|---|---|---|---|---|---|
| **1** | Analyse existing junction on / 1-step / off | **0** | **0** | none | Free, already run |
| **2** | Arms C + D pilot | 12 | 0.45 k | done | **Promoted.** Resolves D4, which §2 shows carries the accuracy gain |
| **3** | M1 pilot (both configs) | 24 | 0.2 k | small | Cheapest new rung; attacks the largest ΔR² step and the config bridge together |
| **4** | M1 full (both configs) | 108 | 0.7 k | — | If the pilot is clean |
| **5** | M2 pilot → full | 12 → 54 | 0.45 → 2.0 k | moderate | Separates topology from edge features |
| **6** | Full arms C + D | 108 | 3.9 k | done | If the pilot warrants it |

Steps 1–4 cost roughly **1.35 kSU** and cover both the accuracy story and the ΔR² story. That
is the package to take to a supervisor, not the full 8 kSU.

---

## 8. What this will and will not establish

**After steps 1–4**, the wD-MPNN → HPG-hier jump is attributed to either "having a
monomer-level representation" or "moving inter-monomer information to the monomer level", with
configuration separated out. On ΔR² that is the largest step; on accuracy it is where the
*loss* occurs, which is equally worth understanding.

**After step 5**, the octamer-side factors are fully separated: topology, edge features,
readout, position embeddings and sampling each get their own comparison.

**Outside this ladder entirely: the protocol-matching confound.** Every rung trains against
labels computed on 8-unit chains, so a model whose structure mirrors that construction may be
flattered at every step. Only the chain-length sweep in the Paper 2 dataset addresses it. Keep
saying so.

**One caution carried forward.** The decomposition in §2 is a median over 9 folds on the A
split. Before it goes in a paper, check that the pattern — family step negative on accuracy,
positive on ΔR² — holds per fold and on the B split, rather than being driven by a few cells.
