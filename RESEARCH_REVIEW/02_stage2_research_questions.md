# Stage 2 — From chronology to research questions

*9 August 2026. Builds on `01_stage1_inventory.md` and its §8 answers.*
*Still no thesis structure proposed — that is Stage 5. This document reorganises what exists.*

---

## 0. The reframing that Stage 1 forces

Two facts, established in Stage 1, change how the year should be read:

1. **Every Feb–May number is void.** Not weakened — void. The checkpoint bug (all runners, all
   experiments) plus single-run measurement plus, for wDMPNN specifically, an input parser that
   never fired. There is no salvageable *number* from the first four months.
2. **The experimental designs, negative results and diagnostics from that period are not void.**
   Neither are the questions.

So the year did not produce the body of results it looks like it produced. What it produced was
**a sequence of research questions that got progressively sharper, and — in the last three months —
a measurement apparatus capable of answering them.** That is a different and, I think, more
defensible story, but it has to be told deliberately rather than by presenting the old tables.

---

## 1. The six research questions the work actually addresses

I can identify six. They are not equally weighted, and their chronological order is not their
logical order.

---

### RQ1 — What kind of signal does a polymer benchmark actually contain, and when do learned structural representations beat descriptors or bare identity?

**Question → experiments → evidence → conclusion → uncertainty**

| | |
|---|---|
| **Experiments** | E2 (HTPMD architecture benchmark), E3 (descriptor fusion), E4 (descriptor-only RF/XGB), E5 (identity baseline on PAE Tg / Block / EA-IP), plus the `insulator` / `opv` / `polyinfo` / `tc` sweeps in `results/` |
| **Evidence** | All single-run, checkpoint-bugged, splits undocumented (C1). **Numerically void.** |
| **Conclusion at the time** [YOU] | Dominant signal is dataset-dependent: PAE Tg → chemical structure; Block → identity + composition; EA/IP → identity |
| **What survives** | The *design* — a descriptor-only baseline and an identity baseline as controls — and one quantitative corroboration: the July dataset audit independently measures **monomer identity ≈ 90% of EA/IP variance**, which is the same finding E5 reached by a completely different route |
| **Remaining uncertainty** | Whether E5's ~0.988 R² was a random split. If it was, "identity suffices" is a statement about interpolation, not about the benchmark |

**[ME] This question has quietly changed character.** In February it was a *modelling* question
("do GNNs help?"). By August it has become a *benchmark-critique* question ("what is this dataset
even testing?"). The July/August work answers the second version rigorously and the first version
not at all. I would stop treating these as one theme.

**Status: numerically void; conceptually absorbed into RQ5.**

---

### RQ2 — How should monomer representations be combined into a polymer representation?

The question you returned to more than any other. Asked **four times, in four independent
frameworks**, each time after the previous framework was abandoned:

| Framework | Experiments | Winner |
|---|---|---|
| Backbone GNNs (DMPNN/GIN/GAT) | E7, E8, E9, E10, E11 | fraction-weighted **mixture** |
| Original HPG | E12 (`HPG_frac`), E14 (`HPG_attnPool`) | fraction-weighted **mixture** |
| HPG-2Stage | E16 (Stage 2A), E17 (Stage 2B) | fraction-weighted **mixture** |
| HPG-hier | E26–E27 | stoichiometry-weighted sum (octamer converts it to slot counts) |

**Consistent findings across all four** [DOC]:
- Mean pooling is substantially worse than fraction weighting.
- Learned adaptive weighting (`attnPool`, vanilla attention, fraction-aware attention) **never
  beats** the fixed compositional prior.
- Composition must be supplied explicitly; models do not recover it.

**[ME] This is the most robust empirical claim in the entire body of work**, and it is robust for
an unusual reason: the individual numbers are all void, but the *direction* survived four complete
rewrites of the encoder, the featuriser, the aggregation code and — after July — the training
protocol and the metric. A result that survives that much churn is not a checkpoint-bug artefact.

**Two caveats I would not drop:**
- The effect has never been measured under the fixed protocol with error bars. "Mixture > mean"
  is qualitatively secure; the ~0.09 eV magnitude from E8 is not.
- The octamer complicates it: `n_A = round(8·frac_A)` turns composition from a multiplier into
  structure. That is arguably a *fifth* answer to RQ2, not a variant of the fourth
  (`HANDOFF_2026-08-05.md` §5 flags this explicitly).

**Status: reasonably well supported in direction; unmeasured in magnitude.**

---

### RQ3 — Does sequence architecture carry recoverable signal beyond composition, and how should it be represented?

The spine of the project, and the question whose history is most misleading if read chronologically.

**The trajectory** — note that the reversal is a *metric* event, not a *finding* event:

| When | Claim | What was actually happening |
|---|---|---|
| Mar (E12) | "polytype gives inconsistent gains, degrades EA" | judged on overall R² |
| Apr (E13) | "architecture information introduces noise rather than signal" | judged on overall R² |
| May (E19) | architecture = **~1% of total** but **~50–60% of post-composition residual** variance | ΔR² metric introduced |
| May (E20) | explicit architecture ≫ implicit graph encoding | judged on ΔR² |
| Jul (E30) | 0.98% (EA) / 1.46% (IP) — independent re-measurement | confirms E19 |
| Aug (E27/E31) | octamer beats wDMPNN on ΔR² by +0.190 / +0.278, 8–9 folds of 9 | fixed protocol, 3 seeds |

**The reconciliation, now confirmed** (Q8): overall R² is mathematically incapable of resolving a
1% variance component against run-to-run noise of the size later measured. Phase 2B did not find
that architecture is unhelpful; it found that its instrument was blind. **Record as superseded by
metric change.**

**[ME] This is the strongest narrative asset you have**, and it is stronger than "we built a better
model": *we could not see the effect we were studying until we built a metric that could, and the
field's default metric still cannot.* That is a reusable methodological contribution and it is
already written down as Claim 9 in `writing/paper2_outline.md`.

**What is now open, not closed:**
- **The 2D0/2D1 decomposition is confounded** (§8.1 of Stage 1). The global-vs-chemistry-conditioned
  contrast was never actually tested. The real measurement is 0.21 → 0.85, which is a *much* bigger
  and cleaner claim than the one the May document makes.
- **Why the octamer works is unknown.** Four candidate factors remain live (topology, positional
  embeddings, attention readout, absence of edge features); the replica-ensembling hypothesis was
  pre-registered and **falsified** (E32). The positional-embedding ablation (E33) is the current probe.
- **The octamer's mechanism is bounded**: with `stage2_depth = 2` a slot sees at most ±2 positions,
  so it captures local blockiness, not global chain arrangement. Any "explicit sequence" claim must
  be qualified.

**Status: the phenomenon is well supported; the mechanism and the model comparison are not.**

---

### RQ4 — How should non-additive monomer–monomer interaction be modelled?

**Experiments**: E9–E11 (attention families), E14 (Phase 3B/3C), E15 (Phase 4 gating),
E17 (Stage 2B), E18 (Stage 2C diagnostics).

This theme is almost entirely a **cluster of well-designed negative results**, which is why it looks
thin in the documents and is actually one of the more publishable parts:

| Finding | Experiment | Type |
|---|---|---|
| Naive uniform pairwise interaction ≈ no gain | E10, E14 | negative |
| Changing the fusion operator (sum/concat/gated/scalar) changes nothing | E10 | negative |
| Per-sample adaptive gating collapses; λ stays small and narrow regardless of context | E15 | negative, **with mechanism** |
| Attention-weighted pairwise gives consistent but modest gains | E10, E14 | weak positive |
| A single learned **scalar** interaction weight is stable and helps IP | E17 | positive |
| BB/BF/FF typed channels are **not identifiable**: \|r\| 0.83–0.97, PC1 = 97.3%, VIF ≈ 8 | E18 | negative, **pre-implementation** |

**The surviving positive claim is the EA/IP asymmetry**: EA behaves additively, IP benefits from
interaction. It appears in E14, E15, E17 and E18, and — this is the part that matters — it survived
a change of encoder, framework and metric. [YOU, May §11.2] *"This pattern survives new architecture,
new encoder, lower variance. Thus it likely reflects real chemistry rather than model noise."*

**[ME] I would push back on one thing.** That claim was made before the July null-floor analysis,
which showed the **A-split EA metric is near-degenerate** (null floor 0.676, and the null beats the
model on fold 2). An EA/IP asymmetry measured on a split whose EA arm barely functions is a weaker
observation than it looked in May. It may still be real — the B split exists precisely to test this
— but "likely reflects real chemistry" is currently over-stated.

**[ME] E18 deserves separate billing.** Running an identifiability diagnostic *before* building the
model, and killing the model on the evidence, is the single best-executed piece of scientific
reasoning in the corpus. It belongs in the thesis as a **method**, regardless of whether typed
interactions are ever revisited.

**Status: negative results well supported and cheap to defend; the one positive claim (EA/IP
asymmetry) needs re-establishing on a non-degenerate metric.**

---

### RQ5 — What does a held-out-monomer copolymer benchmark actually measure?

**Entirely post-May. Not in the original proposal. Not in any uploaded document.**

**Experiments**: E21 (three-split design, May), E28 (noise floor), E29 (checkpoint bug + 3-seed
regeneration), E30 (null floors, dataset audit, B-heldout split), E31 (protocol parity),
E32 (pre-registered K=1 ablation).

**Findings** [DOC, `_dataset_design_audit.md`, `_groupmean_metric_floor.md`, `_noise_floor_results.md`]:

1. The EA/IP benchmark is an **exact factorial**: 9 A × 682 B × 7 cells = 42,966. Nobody has said
   this in print.
2. The A-held-out split trains on **7 donor monomers**. "Unseen chemistry" is a 7-example
   extrapolation.
3. An **A-blind null** scores median group-mean R² of **0.676 on EA** vs **−0.034 on IP**, and beats
   the model outright on one fold. The EA arm of the standard split is near-degenerate.
4. **Three identical runs** give EA fold-1 group-mean R² of 0.450 / 0.790 / 0.978.
5. The B-monomer space is **62.5% two Murcko scaffold families** (317 + 109 of 682), so a balanced
   scaffold-disjoint 9-fold split is *impossible* and the folds are not exchangeable.
6. Therefore **EA claims belong on the B split and IP claims on the A split** — a two-regime
   structure inside data you already had.

**[ME] This is the strongest evidence base in the project by a wide margin** — three seeds,
pre-registered ablations, frozen splits re-asserted per run, null floors, and a documented record of
its own corrections. It is also the only theme where you have *falsified your own hypotheses in
public* (K=1; the compute claim; the "hard monomer" story).

**[ME] It is also the theme that most cleanly discharges an original thesis objective.** Proposal
Objective 1 — *"Create a benchmarking dataset and evaluation framework for polymer ML, enabling
standardised comparisons, reproducibility, and performance tracking"* — was scheduled as a
side-deliverable and has been achieved almost by accident, in a stronger form than planned.

**Status: reasonably well supported. The most defensible thing you have.**

---

### RQ6 — Does data volume or chemical diversity limit polymer property prediction?

**Experiments**: E22 (learning curves) only, plus the diagnosis in E30 (§4.2) that 7 training
donor monomers is the likely reason *three separate representation changes all failed to move the
architecture axis*.

**Current position** [DOC]: data volume is **not** the dominant limitation; chemical diversity
probably is. The proposed answer — generate a DFT copolymer dataset (~2,000 polymers, ~8–9 kSU) —
is live and costed.

**Status: promising but barely evidenced. One qualitative learning-curve experiment.**

---

### RQ7 — 2D vs 3D structural information

**Zero experiments.** Nothing in the eight documents or the repository touches conformations,
3D descriptors or geometry. See §3.

---

## 2. Cross-cutting analysis

### 2.1 Experiments that address the same question

| Question | Asked in | Note |
|---|---|---|
| How to aggregate monomers? | E7/E8, E9–E11, E12, E14, E16, E17 | four frameworks, same answer |
| How to represent architecture? | E12 (polytype), E13 (archGraph), E20 (2D0/2D1), E27 (octamer) | four attempts, escalating explicitness |
| Do interactions help? | E10, E14, E15, E17, E18 | five attempts, converging on "constrained yes, adaptive no" |
| Is wDMPNN a strong baseline? | E2, E7, E8, E20, E31 | five attempts; **only E31 is valid** |

**[ME]** Four independent attempts at the same question is not waste — it is the reason the
directional findings are trustworthy. But it should be *presented* as replication, not as four
separate contributions.

### 2.2 Logical build chains

```
E6  read the released HPG implementation, found 4 defects
 └─ E12–E15  patch each defect in place                       → all four patches failed
     └─ E16–E20  rebuild cleanly as HPG-2Stage                → composition dominates; architecture is the residual
         └─ E26–E27  HPG-hier, then explicit octamer sequence → current lead model
```

```
E21  three splits of increasing difficulty (May)
 └─ E28  measure run-to-run variance                          → most differences unmeasurable
     └─ E29  find checkpoint bug, regenerate with 3 seeds
         └─ E30  null floors + dataset audit + B split        → the metric itself was partly degenerate
             └─ E31/E32  protocol parity, pre-registered ablation
```

**[ME]** The first chain is the model story. The second is the measurement story. **The second chain
undermines every number in the first, and is itself the stronger contribution.** That tension is the
central strategic issue for the thesis, and you flagged it yourself in the supervisor update §6.1.

### 2.3 Contradictions, and their resolutions

| Apparent contradiction | Resolution |
|---|---|
| E13 "architecture doesn't help" vs E20 "architecture is the main residual" | **Metric artefact.** Overall R² cannot see a 1% component. Confirmed (Q8). |
| E7 "wDMPNN ≫ baselines" vs E8 "DMPNN+mixture > wDMPNN" | **Both void** — pre-18-June wDMPNN was not a wD-MPNN. |
| E2 "wDMPNN weakest architecture" vs E31 "wDMPNN competitive, beats HPG-hier on IP group-mean" | E2 void; E31 valid. The baseline is **stronger** than a year of documents suggests. |
| May §11.1 "Stage 2A outperforms wDMPNN massively" | Void on both sides (Stage 2A pre-checkpoint-fix; wDMPNN pre-input-fix). |
| E15 "adaptive gating collapses" vs E17 "learned scalar weight is stable" | Not a contradiction — **per-sample** gating vs a **global scalar**. The constrained version works; this is RQ4's actual finding. |
| §8.2.3 global offsets R² 0.21/0.25 vs §8.3.5 2D0 ΔR² 0.847/0.908 | **Non-linear readout** (Stage 1 §8.1). 2D0 is not global. |

### 2.4 Abandoned and dormant directions

| Direction | Last seen | Status |
|---|---|---|
| HTPMD / homopolymers entirely | Mar group meeting | **Abandoned.** Was the whole of Experiment 1. |
| Descriptor-fusion programme (FiLM, aux supervision) | Feb | **Abandoned**, with a usable negative result |
| PAE Tg, Block Copolymer | Mar | **Dormant** — data and results present, protocol void |
| Level 3 supergraph / virtual node | Mar (proposed) | **Never implemented** |
| Stage 2C typed interactions | May | **Deliberately killed on diagnostics** — the good kind |
| Stage 2E property-adaptive interaction | May (specified) | **Never implemented** |
| Junction coupling | Jul | **Target dissolved** — existed to fix EA fold 1, which turned out to be a variance artefact |
| Curtis 2025 port | Jul | **Explicitly dropped** — featureless A/B beads would collapse Stage 1 |
| ChemArch / GlobalArch comparison | Jul | **Banked** — never regenerated under the fixed protocol |
| 2D vs 3D comparison | — | **Never started** (§3) |

### 2.5 Negative results worth keeping

1. Descriptor injection into message passing (FiLM) degrades performance (E3).
2. Fusion-operator choice is irrelevant once mixture and interaction terms are fixed (E10).
3. Per-sample adaptive interaction gating collapses, with a diagnosed mechanism (E15).
4. BB/BF/FF interaction channels are not identifiable on this benchmark (E18).
5. In-loss replica ensembling does not explain the octamer's stability (E32) — **pre-registered
   and falsified**.
6. Enriching edge descriptors does not help when edges only bias attention (E12/E13).

**[ME]** Items 4 and 5 are the two strongest, because both were pre-committed and both went against
you. That is unusually good practice and it is worth making visible rather than burying.

### 2.6 Debugging vs scientific experiments — a reclassification

The obvious split is: E6, E24, E28, E29, E31 = engineering; everything else = science.
**I think that split is wrong.**

| Experiment | Looks like | Actually is |
|---|---|---|
| E6 (HPG code study) | debugging | **method critique** — it is the stated justification for the whole April–May programme |
| E28 (noise floor) | QA | **a measurement result**: single-run benchmark numbers are not measurable at this scale |
| E29 (checkpoint bug) | bug fix | bug fix — but the +0.048 eV cost quantifies a failure mode that is probably widespread in the literature |
| E30 (null floor) | QA | **a genuine methodological contribution** — you say yourself it is not standard practice and it changed your reading of your own results |
| E31 (protocol parity) | reproduction | **closes the "under-tuned baseline" objection permanently**, which most papers never do |

---

## 3. Alignment against the Year-1 proposal and the review paper

Now that I have both documents, two things stand out.

### 3.1 The Year-2 decision gate was never executed

Proposal §5.2.1 lists six Year-2 activities. Against the evidence:

| Planned Year-2 activity | Status |
|---|---|
| 1. Benchmark tabular vs graph representations | **Done** (E2–E5) — but numerically void |
| 2. **Comparative study of 2D vs 3D molecular features** | **Not started** |
| 3. Encode chain architecture and stoichiometry | **Done, and became the whole year** (RQ2, RQ3) |
| 4. Scalability of conformation generation | **Not started** |
| 5. Data-efficient learning (few-shot, transfer, active) | **Not started** (one mention of "pretraining" in E8 that I cannot trace) |
| 6. **Select trajectory: 3D structure vs data scarcity** | **Gate never reached** |

Activity 3 was one line in the plan. It consumed the year and produced everything of value.
Activities 2, 4, 5 — and therefore the decision gate that was supposed to determine the shape of
Year 3 — did not happen.

**[ME] I do not think this is a failure, but it does need a decision rather than a drift.** The
Year-3 plan in the proposal branches on a gate you never passed. Either you pass it (a scoped 2D vs
3D study), or you retire it explicitly and re-derive Year 3 from where the work actually is. Doing
neither means the annual report has to explain a plan that no longer describes the project.

### 3.2 The work has moved from Paper 2 to Papers 3 and 4

Proposal §5.3 publication plan vs reality:

| Planned paper | Planned content | Actual status |
|---|---|---|
| P1 — *Learning Polymers* (review) | representations + dataset gaps | **Accepted, Digital Discovery.** Year 1 delivered. |
| P2 — *From Descriptors to Geometry* | tabular vs graph, **2D vs 3D** | Half of it (tabular vs graph) is void; the 2D-vs-3D half doesn't exist. **This paper is not currently writable.** |
| P3 — *A Generalisable Representation for Polymer Architectures* | new representation method | This is what Year 2 actually built (HPG-2Stage → HPG-hier → octamer). Evidence is 3 weeks old and still moving. |
| P4 — *polyBench* | benchmark datasets, standardised splits, baselines | This is what the July–August work **actually is**, arriving ~a year early and in a sharper form (null floors, factorial audit, S/D fold structure). `writing/paper2_outline.md` is already this paper. |
| P5 — data scarcity **or** 3D generation | — | untouched |

**[ME]** Your `paper2_outline.md` is titled *"What does a held-out-monomer split actually measure?"*
That is **P4 content, not P2 content**. The naming mismatch is cosmetic, but the strategic point is
not: the project's centre of gravity has moved from *representation comparison* to *representation
evaluation*, and the proposal has not been updated to say so.

### 3.3 The review paper already commits you to this

The accepted review's Future Perspectives contains, as its own headings:

- *Data Infrastructure: Scarcity, Heterogeneity and Benchmarking*
- **Evaluating Representation Quality Beyond Predictive Accuracy**

**[ME]** The second one is, almost verbatim, what the July–August work demonstrates. A thesis in
which Year 1 argues in the literature that evaluation must go beyond predictive accuracy, and
Year 2 shows empirically that the field's standard copolymer benchmark cannot see the effect it
claims to study, is a coherent line of argument that neither piece has on its own. That is the
strongest structural link I can find between what you have published and what you have measured.

---

## 4. Where each question stands (preview of Stage 3)

| RQ | Classification | One-line reason |
|---|---|---|
| RQ1 signal type / structure vs identity | **Numerically void; conceptually absorbed into RQ5** | splits unknown, protocol void; the surviving insight is about the benchmark |
| RQ2 aggregation | **Reasonably supported in direction, unmeasured in magnitude** | survived four framework rewrites; no error bars ever |
| RQ3 architecture representation | **Phenomenon supported; model comparison invalidated** | ΔR² result replicated under fixed protocol; 2D0/2D1 confounded; octamer mechanism open |
| RQ4 interaction modelling | **Negative results supported; the positive claim needs re-establishing** | EA/IP asymmetry rests partly on a degenerate EA metric |
| RQ5 what the benchmark measures | **Reasonably well supported — the strongest evidence you have** | 3 seeds, null floors, frozen splits, pre-registration, self-falsification |
| RQ6 data volume vs diversity | **Promising, barely evidenced** | one qualitative learning curve; DFT dataset costed but not run |
| RQ7 2D vs 3D | **Not attempted** | the Year-2 decision gate |

---

## 5. Three things I would challenge before Stage 3

1. **"The EA/IP interaction asymmetry reflects real chemistry."** Stated in May §11.2. It was
   measured largely on the A split, whose EA arm has a null floor of 0.676. Until it reproduces on
   the B split I would downgrade this from a finding to a hypothesis. It is currently doing load-bearing
   work in the Stage 2B/2C/2E narrative.

2. **"Stage 2A / the clean hierarchy is state-of-the-art."** Void on both sides of every comparison
   that supports it. The August evidence points somewhere different: the octamer is the lead model,
   HPG-hier loses to a properly-configured wDMPNN on IP group-mean R², and the hierarchy-versus-flat
   question has not been tested under the fixed protocol at all.

3. **The model contribution and the measurement contribution are now competing, not complementary.**
   You raised this in the supervisor update §6.1 and it has not been resolved. Three representation
   variants have produced no movement on the architecture axis; meanwhile the measurement work has
   produced five defensible claims. Stage 5 will have to pick a lead, and I think the honest answer
   may not be the one you expected in May.

---

## 6. Questions before I do Stage 3

Fewer this time, and they are judgement calls rather than facts.

1. **Is the 2D-vs-3D gate live or retired?** This single answer changes the whole Year-3 plan. If
   retired, say so explicitly in the annual report and re-derive the trajectory from RQ3/RQ5.
2. **Do you accept the reclassification of the July–August work as a scientific contribution
   rather than remediation?** I think it is your strongest material; if you disagree, Stage 5 looks
   very different.
3. **How much of the Feb–May work do you intend to re-run at all?** Given that everything is void,
   "rerun" is a budgeting decision across roughly 15 experiment groups, not a wDMPNN question. I
   would rather scope this with you than guess.
4. **Is `data/pae_tg.csv` / `data/block.csv` a live thesis thread or a closed one?** They are your
   only non-EA/IP copolymer evidence and your only classification task. You mentioned an
   "annual-plan chat" proposing an off-protocol test — I do not have that conversation, so tell me
   what was proposed if it should factor in.
5. **What is the actual Year-3 milestone you are assessed against?** The proposal says "develop and
   refine a polymer-specific representation framework and support it with a curated benchmark
   dataset." If that is still the formal commitment, the DFT dataset and the B-split benchmark
   together satisfy it — which is worth knowing before we design the plan.

---

*Next: Stage 3 — evidence classification per claim, then Stage 4 — rerun prioritisation, which is
now a whole-corpus question rather than a wDMPNN one.*
