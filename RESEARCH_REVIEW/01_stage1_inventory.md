# Stage 1 — Materials, chronological inventory, and open questions

*Compiled 9 August 2026 from the eight uploaded PDFs and the `dmpnn/` repository.*
*No thesis structure or Year-3 plan is proposed here. This document is a reconstruction only.*

---

## 0. Evidence tags used throughout

| Tag | Meaning |
|---|---|
| **[DOC]** | Documented result — explicitly stated in your notes |
| **[NUM]** | Raw/numerical evidence — I opened a saved results file or CSV |
| **[CODE]** | Code/config evidence — inferred from source, config, git history |
| **[FIG]** | Figure-based observation — visible only in a plot, treated with caution |
| **[YOU]** | Your interpretation/conclusion at the time |
| **[ME]** | My interpretation — a proposal, not established |
| **[RECON]** | My reconstruction — inferred from several pieces of evidence, not stated by you |
| **[?]** | Unclear / needs clarification |
| **[CONFLICT]** | Sources disagree; I have not chosen between them |
| **[RERUN]** | Existing numbers likely invalid; experimental design may still stand |

---

## 1. What I found

### 1.1 Uploaded experiment documentation (8 PDFs)

| # | File | Pages | Period covered | Nature |
|---|---|---|---|---|
| 1 | `Feb16-22.pdf` | 7 | Feb 2026 | HTPMD homopolymer benchmark + descriptor-fusion study. Self-contained report. |
| 2 | `Mar02-17.pdf` | 6 | Mar 2026 | Copolymer datasets, representation "levels" (0–4), first PAE Tg / Block / EA-IP results. Ends with a to-do plan. |
| 3 | `2026 march group meeting.pdf` | 9 | ~Mar 2026 | **Presentation script, not an experiment log.** Restates Feb + early-Mar results for a chemistry audience; ends with a dataset-generation proposal. |
| 4 | `Mar16-22.pdf` | 16 | Mar 2026 | Two distinct halves: (a) a code-reading analysis of the *released* HPG implementation; (b) EA/IP experiments — polytype features, wDMPNN comparison, representation-strategy comparison. |
| 5 | `Mar23-29.pdf` | 18 | Mar 2026 | Copolymer aggregation methods: fraction-aware attention, pairwise interaction, fusion strategies, self-attention. Ends with the proposal for an interaction-aware HPG. |
| 6 | `Mar30-Apr19.pdf` | 12 | Mar–Apr 2026 | HPG Phases 1–3 (frac / polytype / edge typing / arch-aware / relMsg / fragGraph / attnPool / pairInteract / pairInteractAttn). |
| 7 | `Apr20-26.pdf` | 5 | Apr 2026 | HPG Phase 4 — adaptive interaction gating. |
| 8 | `May.pdf` | 35 | May 2026 (dated "May 15") | **HPG-2Stage design document.** Part specification, part results report. Stages 2A–2E, architecture diagnostics, Stage 2D results, generalisation splits, learning curves. |

Organisation as I read it: files 1–7 are weekly/fortnightly lab-notebook entries in chronological order; file 8 is a consolidated framework document that supersedes much of 4–7 conceptually. File 3 is a communication artefact and should not be treated as an independent evidence source — every claim in it traces back to files 1 and 2.

### 1.2 The `dmpnn/` repository

Relevant top-level structure:

```
chemprop/                  model code (fork of chemprop 2.2.0)
  nn/message_passing/mixins.py   <- _WeightedBondMessagePassingMixin (wDMPNN message)
  models/hpg_hier.py             <- HPG-hier / octamer / junction models
  featurizers/molgraph/hpg_hier.py
polymer_input/featurizers/wdmpnn.py
scripts/python/            runners + analysis + plotting (run_wdmpnn_generalization.py,
                           run_hpg_generalization.py, analyze_*, plot_*)
scripts/shell/             PBS submission scripts
configs/                   wdmpnn_{a_held_out,group_disjoint,pair_disjoint}.yaml
results/                   per-model CSVs of per-fold test metrics (the Feb–May era)
predictions/               NPZ prediction trees (the June–Aug era: regen_v1, octamer_k1,
                           wdmpnn_original, noise_floor, octamer_posemb, ...)
experiments/               diagnostics, hpg, hpg2stage, tabular, wdmpnn_diagnostics
hpg_hier_design/           EXPERIMENT_PLAN.md, seed_42_diagnostic/, design notes
analysis/model_diagnostics/  the June–Aug evidence base (~40 .md + .csv reports)
29-07-2026 supervisor_update/  SUPERVISOR_UPDATE_29-07-2026.md + evidence/
22-07-2026 report_figures/
writing/paper2_outline.md
```

Git history runs from July 2025 to 8 August 2026, ~200 commits. Commit messages are mostly
uninformative (`fix`, `res`, `scripts`), so I traced work by file path and by the dated
markdown reports rather than by message.

### 1.3 The single most important structural observation

**The uploaded documentation stops on 15 May 2026. The repository contains roughly three
months of subsequent work — June, July, August — that materially changes or invalidates a
large fraction of what the Feb–May documents conclude.** [CODE][DOC]

The key post-May artefacts are:

- `analysis/model_diagnostics/HANDOFF_2026-07-29.md`, `HANDOFF_2026-08-05.md`
- `29-07-2026 supervisor_update/SUPERVISOR_UPDATE_29-07-2026.md` and its `evidence/` folder
- `analysis/model_diagnostics/_regen_v1_results.md`, `_noise_floor_results.md`,
  `_wdmpnn_original_results.md`, `_octamer_k1_r3_results.md`, `_dataset_design_audit.md`,
  `_groupmean_metric_floor.md`, `_code_drift_investigation.md`
- `writing/paper2_outline.md` — a claim-by-claim evidence map for a paper whose framing is
  quite different from the May document's

**This is not a minor caveat.** Your instructions framed the validity problem as "old wDMPNN
implementation". The repository documents at least three *additional*, broader validity
problems (§3 below), one of which affects every single number in every Feb–May document,
including the ones that have nothing to do with wDMPNN. I have flagged these rather than
quietly folding them in, because they change what "needs rerunning" means.

---

## 2. Chronological experiment inventory

Format: what I can support, what I cannot, and where to look.

---

### E1 — Pre-February: graph vs tabular across homopolymer datasets
**Not in the uploaded documents.** Referenced only as "Previous Observation" in the group
meeting script. [DOC]

- **Repository evidence** [NUM]: `results/{AttentiveFP,DMPNN,GAT,GIN,PPG,DMPNN_DiffPool,tabular}/`
  contain results for `insulator`, `opv_camb3lyp`, `polyinfo`, `tc` (thermal conductivity),
  including learning-curve size sweeps (`__size128` … `__size3367`, `__size250` … `__size12000`)
  and RDKit-descriptor variants. Also `graph_vs_tabular_improvement.png`,
  `scripts/python/plot_graph_vs_tabular.py`.
- **Claim made** [YOU]: "Across many datasets, graph models generally performed better."
- **Status**: I have **no written protocol** for this work. Dataset construction, splits,
  metric, and tuning are undocumented in anything you gave me.
- **[?]** Is this Year-1 work that is already written up elsewhere, or does it need reconstructing too?

---

### E2 — Feb 16–22: GNN architecture benchmark on HTPMD (homopolymers)
`Feb16-22.pdf` §3, §5

- **Question** [DOC]: which GNN mechanisms improve representation learning for polymers?
- **Dataset** [DOC]: HTPMD, 6,270 rows, SMILES with `[Au]`/`[Cu]` placeholders converted to
  explicit connection points via RDKit. Targets: conductivity, TFSI/Li/polymer diffusivity,
  transference number.
- **Models** [DOC]: AttentiveFP, D-MPNN, D-MPNN+DiffPool, GIN, GAT, wD-MPNN, PPG.
- **Result** [DOC][YOU]: ranking `AttentiveFP > D-MPNN+DiffPool > GAT ≈ GIN > D-MPNN > PPG > wD-MPNN`.
- **Conclusion drawn** [YOU]: "representation learning capacity dominates polymer-specific encodings";
  "hierarchical pooling is beneficial"; "in homopolymers wD-MPNN's probabilistic edge weights
  collapse to unity".
- **Establishes**: a relative ordering on one dataset under one (undocumented) split, single run.
- **Does not establish**: anything about copolymers; anything about *why* AttentiveFP wins
  (attention vs capacity vs hyperparameters are confounded — no parameter-count control is described).
- **[?] Split protocol is not stated anywhere in the document.** Random? Scaffold? Polymer-held-out?
  §7.1 refers to a "polymer-held-out evaluation" as a *future* step, which suggests the reported
  results were **not** polymer-held-out.
- **[CONFLICT]** I can find **no saved wD-MPNN results file for HTPMD** in `results/`.
  `results/PPG/htpmd_results.csv` exists; there is no equivalent wDMPNN file anywhere under
  `results/`. The ranking's last two positions therefore cannot currently be traced to numbers
  in the repo.
- **[RERUN]** wDMPNN involved. See §4.
- **Code**: `results/{AttentiveFP,DMPNN,DMPNN_DiffPool,GAT,GIN,PPG}/htpmd_results.csv`;
  `scripts/python/plot_htpmd_*.py`.

---

### E3 — Feb 16–22: descriptor-integration strategies on HTPMD
`Feb16-22.pdf` §4, §6

- **Question** [DOC]: how should polymer-level descriptors (DoP, density, molality, monomer MW)
  be integrated into GNNs?
- **Conditions** [DOC]: graph-only / late concat / FiLM-all-layers / FiLM-last-layer /
  auxiliary supervision, on GIN, GAT, D-MPNN backbones.
- **Results** [DOC][YOU]: late concat ≈ neutral; FiLM-all consistently degrades; FiLM-last
  mitigates but rarely beats graph-only; auxiliary supervision helps **transference number**
  specifically and consistently.
- **Establishes**: within this dataset and setup, naive descriptor injection into message
  passing hurts. That is a clean, useful **negative result**.
- **Does not establish**: that descriptors are uninformative — §6.6 shows the opposite.
- **Limitation** [ME]: "within typical seed variance" is asserted in §6.1 but **no seed variance
  was measured** anywhere in the Feb–May period (§3.2 below). The claim is not supported by
  anything I can find.
- **Code** [NUM]: `results/{DMPNN,GAT,GIN}/htpmd__desc_results.csv`, `__desc__film_results.csv`,
  `__desc__film__fllast_results.csv`, `__aux_results.csv`; `chemprop/nn/message_passing/mixins.py`
  gained FiLM mode in commit `dfd8ea2` (2026-02-19).
- No wDMPNN involvement.

---

### E4 — Feb 16–22: descriptor-only tabular baselines
`Feb16-22.pdf` §6.6

- **Models** [DOC]: RF and XGB on DoP, density, molality, monomer MW only.
- **Result** [CONFLICT]: the Feb document says descriptor-only models "achieved moderate
  predictive performance"; the March group meeting script quantifies this as "R² between roughly
  0.3 and 0.5". The Feb document gives **no numbers**. I have not verified the 0.3–0.5 figure
  against `results/tabular/htpmd_descriptors.csv`.
- **Conclusion** [YOU]: descriptors carry real but insufficient signal; structure adds information.
- **Code**: `results/tabular/htpmd_descriptors.csv`, `htpmd_rdkit*.csv`.

---

### E5 — Mar 02–17: copolymer datasets, representation levels, identity baseline
`Mar02-17.pdf`; restated in the group meeting script

- **Question** [DOC]: is chemical structure actually necessary, or do monomer identity and
  composition suffice?
- **Datasets** [DOC]: PAE Tg (2,794), Block Copolymer (5,371), EA/IP (42,966).
- **Design** [DOC]: Level 0 tabular (RF/XGB) · Identity baseline (`nn.Embedding` per unique
  monomer SMILES) · Level 1 mixture `z = αz_A + (1−α)z_B` · Level 2 interaction concatenation.
  Levels 3 (supergraph + virtual node) and 4 (wDMPNN as polymer-aware upper bound) are described
  as **to-dos**, not results.
- **Results** [DOC][YOU]:
  - PAE Tg — graph > tabular > identity; mixture best; interaction adds little.
  - Block — identity best on accuracy, **worst on F1** ⇒ learns majority class.
  - EA/IP — everything ≈ 0.988–0.989 R²; identity alone 0.985.
- **Conclusion** [YOU]: "dominant signal" differs per dataset: structure (PAE Tg) / identity+composition (Block) / identity (EA/IP).
- **[?] Split.** No split is stated. The EA/IP numbers (~0.988) are far above the monomer-disjoint
  numbers reported one week later in `Mar16-22`, which strongly suggests this was a **random split**.
  Please confirm — the whole "EA/IP is easy / identity suffices" conclusion depends on it, and
  it reads very differently under a random split than under a monomer-disjoint one.
- **[?]** Level 3 (supergraph/virtual node) — was it ever implemented? I find no `supergraph` or
  `virtual node` code or results. If not, this is an **abandoned direction**.
- **Code** [NUM]: `results/IdentityBaseline/*`, `results/tabular/ea_ip_*`, `results/DMPNN/ea_ip__copoly_*`,
  `results/{GIN,GAT}/…`, `scripts/python/plot_block_identity_vs_structure.py`.

---

### E6 — Mar 16–22 (first half): analysis of the released HPG implementation
`Mar16-22.pdf` pp. 1–8

**This is not an experiment.** It is a code-reading study of the published HPG code. [DOC]

- **Findings** [DOC][YOU]: flattened hierarchy; 49-d node features with fragment nodes as
  `ones(49)`; single scalar edge channel so attachment edges and single bonds both = 1.0;
  fragment–fragment edges carry only `degree`; block and alternating collapse to the same graph;
  edge features bias attention only, not message content.
- **Why it matters**: every design decision from April onward is justified by reference to these
  four limitations. It is load-bearing for the whole narrative.
- **[?]** Was this verified by execution/instrumentation, or by reading source only? Section 8
  reads as source analysis. If by reading only, the "edge features only bias attention" claim
  — later used to explain the Phase 2A degradation — should be stated as such.

---

### E7 — Mar 16–22: polytype features and wDMPNN on EA/IP, monomer-disjoint
`Mar16-22.pdf` §2.1, §2.2

- **Setting** [DOC]: EA/IP, monomer-disjoint (A-held-out) split, 5 CV folds, targets EA and IP.
- **§2.1 result** [DOC]: wDMPNN RMSE ≈ 0.09 vs mean-pooled `+polytype` baselines ≈ 0.16 —
  "nearly a twofold reduction", consistent across folds.
- **§2.2 result** [DOC]: adding `+PT` to DMPNN/GIN/GAT consistently improves, but only by
  ~0.005–0.01 eV RMSE / 0.005–0.014 R².
- **Conclusion** [YOU]: the bottleneck is the **aggregation** of monomer embeddings, not the
  absence of polymer-type descriptors.
- **[RERUN] — this is the highest-stakes wDMPNN result in the corpus.** It is the comparison
  that motivated the entire subsequent research programme.
- **[CONFLICT]** §2.1 says wDMPNN beats the baselines roughly 2:1 on RMSE. §2.3, in the *same
  document*, says DMPNN+mixture **outperforms** wDMPNN. Both are stated as monomer-split results.
  These are consistent only if the mixture representation closes a 2× gap entirely — possible,
  but it should be stated explicitly, and it makes the wDMPNN number the pivot of both claims.
- **Code**: `results/DMPNN/ea_ip__copoly_mean__a_held_out_results.csv` (I opened this: 5 folds),
  `results/HPG2Stage_LOMAO/ea_ip__wDMPNN__a_held_out__target_*.csv` (**9 folds, added
  2026-07-10** — i.e. *after* the June wDMPNN fixes, so this file is probably **not** the one
  behind the March claim).
- **[?] Where are the March wDMPNN numbers stored?** I cannot locate a 5-fold wDMPNN EA/IP
  results file. Without it the §2.1 comparison cannot be re-derived.

---

### E8 — Mar 16–22: copolymer representation strategy comparison
`Mar16-22.pdf` §2.3

- **Conditions** [DOC]: mean pooling vs mixture vs explicit interaction, × {DMPNN, GIN, GAT} × {EA, IP}.
- **Result** [DOC]: mixture best; beats mean pooling by ~0.085–0.093 eV RMSE and 0.06–0.12 R²,
  improving in **5/5 folds**. Interaction beats mean but is below mixture and higher variance.
- **Conclusion** [YOU]: "representation is the dominant factor"; simple GNNs with the right
  representation match or exceed specialised models.
- **This is, in my reading, the strongest and cleanest empirical result of the Feb–May period.**
  It is a within-architecture, within-fold, 5/5 paired comparison with a large effect size.
  [ME] It is also the only Feb–May comparison whose effect size (≈0.09 eV) clearly exceeds the
  run-to-run noise later measured in July (MAE SD up to 0.091 eV on the worst fold) — though see §3.2.
- **[?] "the effect of representation is significantly larger than that of pretraining… pretraining
  yields modest improvements (~0.007 eV RMSE)".** No pretraining experiment appears anywhere in
  the eight documents or, as far as I can find, in the repository. What was pretrained, on what?
- **[RERUN]** partially — the wDMPNN comparison arm only, not the mean/mixture/interaction contrast.

---

### E9–E11 — Mar 23–29: aggregation mechanisms
`Mar23-29.pdf`

Three related experiment groups sharing one setup (EA/IP, monomer-disjoint, DMPNN/GIN/GAT backbones):

| | Conditions | Result [DOC] | Conclusion [YOU] |
|---|---|---|---|
| **E9** Mixture vs attention | fixed mixture, vanilla attention, fraction-aware attention | mixture strongest; vanilla attention much worse; frac-attn recovers most of the gap but doesn't beat mixture | composition is a physical prior that must not be discarded; the missing ingredient is interactions |
| **E10** Interaction & fusion | pairwise-fixed, pairwise-attention; sum / concat / gated / scalar-residual fusion | naive pairwise ≈ no gain; **attention-weighted** pairwise gives consistent gains (GAT, GIN; more on IP); DMPNN barely moves. Changing fusion does **nothing** consistent | interaction relevance must be *learned*; the limit is the representation, not the fusion |
| **E11** Self-attention | full self-attention with fraction prior | strong but high variance; best method is model- and target-dependent | expressivity/stability trade-off |

- **Hypotheses were explicitly stated** here (§4, five numbered hypotheses) — the only document
  in the corpus that does this cleanly.
- **[FIG]** Much of §5 is read off scatter plots ("points below the diagonal"). Underlying numbers
  exist as `results/DMPNN/ea_ip__copoly_*_meta__poly_type*` directories, which I have not opened
  fold-by-fold.
- **Establishes**: a consistent ordering of aggregation mechanisms under one protocol.
- **Does not establish**: that any of these differences exceed run-to-run variance (§3.2).
- **[ME]** E10's fusion sub-experiment is a well-designed **negative result** and, in my view, is
  worth keeping in the thesis precisely because it closes a door cheaply.
- No wDMPNN involvement except as a reference point carried from E7.

---

### E12–E14 — Mar 30–Apr 19: HPG Phases 1–3
`Mar30-Apr19.pdf`. All on EA/IP, A-held-out.

**Phase 1 (E12)** — `HPG_baseline` → `HPG_frac` → `HPG_frac_polytype` → `HPG_frac_edgeTyped` → `HPG_frac_archAware`

- **Results** [DOC]: fraction-weighted pooling gives "the largest and most consistent improvement";
  polytype degrades EA and is mixed on IP; explicit edge typing consistently degrades
  (win rate 40% EA / 20% IP); mean-field arch-aware degrades EA badly.
- **Interpretation** [YOU]: edge features only bias attention (from E6), so enriching them cannot help.
- **[CONFLICT] — please check this one.** I opened
  `results/HPG/ea_ip__hpg_frac__a_held_out__target_EA vs SHE (eV)_results.csv` (added 2026-04-07).
  It contains **5 folds** with test R² of **0.957, 0.088, 0.267, 0.408, −0.038**. The document
  describes HPG_frac as showing "improved R²" and "stable behavior across CV". A fold spread of
  0.96 → −0.04 is not stable in any usual sense. Either (a) the claim is relative to an even worse
  `HPG_baseline`, (b) this CSV is not the file behind the figures, or (c) the figures were
  generated from a different run. I cannot tell which.

**Phase 2 (E13)** — `HPG_relMsg` (edge features enter the message), `HPG_fragGraph`/`archGraph`
(architecture-dependent adjacency weights w_AA, w_AB, w_BA, w_BB, zero-initialised)

- **Results** [DOC]: both degrade. archGraph win rate ≈ 10%, negative R² on EA for some folds.
- **Conclusion** [YOU]: "the primary limitation of HPG is not local graph structure or polymer
  topology, but the aggregation mechanism".
- **[CONFLICT] — the important one.** Six weeks later, the May document (E20) concludes almost the
  opposite: that explicit architecture is the largest recoverable residual signal and that
  architecture-aware models (2D0/2D1) clearly beat composition-only. [ME] The most likely
  reconciliation is that Phase 2 was judged on **overall R²** — where architecture is ~1% of
  variance and therefore invisible — while Stage 2D introduced the **architecture-deviation metric
  R²(Δy)**, which is designed to see it. If that is right, Phase 2's conclusion is a *metric
  artefact*, not a finding, and the "architecture doesn't help" statement should be retracted
  rather than carried forward. **I would like you to confirm or reject this reading before I build
  anything on it.**

**Phase 3 (E14)** — `HPG_attnPool`, `HPG_pairInteract`, `HPG_pairInteractAttn`

- **Results** [DOC]: attention pooling degrades; pairwise interaction helps IP (win rate ≈ 80%)
  and hurts EA in all folds; pair-attention adds variance without mean gain.
- **Conclusion** [YOU]: interactions are beneficial but **target-specific**; EA is well approximated
  by a linear mixture.
- **This EA/IP asymmetry is the most-repeated finding in the corpus** — it appears in E14, E15,
  E17, E18 and survives a change of framework. That repetition is itself evidence.
- **Code** [NUM]: `results/HPG/ea_ip__hpg_{attnPool,pairInteract,pairInteractAttn,relMsg,archGraph,fragGraph,frac_edgeTyped,frac_archAware,frac_polytype}__a_held_out__target_*.csv`.

---

### E15 — Apr 20–26: Phase 4, adaptive interaction gating
`Apr20-26.pdf`

- **Hypothesis** [DOC]: property ≈ additive composition + adaptive interaction correction; the
  gate λ should go to ~0 for EA and >0 for IP.
- **Diagnostics planned** [DOC]: per-polymer ‖h_int‖/‖h_mix‖ ratio and learned λ.
- **Result** [DOC][YOU]: gating does not improve on either `HPG_frac` or fixed `HPG_pairInteract`;
  large fold-to-fold instability; **λ stays small and narrowly distributed** — the model does not
  learn when to use interactions.
- **[ME]** This is a genuine negative result with a clear mechanism, and it directly motivated
  Stage 2E. The λ-distribution diagnostic is good practice and worth reusing.
- **[FIG]** The results section contains three empty figure references ("As shown in ,"). The
  claims about per-fold collapse and the λ distribution are currently **unsupported by anything
  I can see** — the figures did not survive into the PDF text layer.
- **Code**: `results/HPG/ea_ip__hpg_pairInteractGate__a_held_out__target_*.csv`;
  `scripts/python/report_hpg_gate.py`.

---

### E16–E22 — May 15: HPG-2Stage
`May.pdf`. A framework document containing both specification and results; the two are interleaved,
which makes it hard to tell what was run from what was planned.

**E16 — Stage 2A, composition-only aggregation**
- **Design** [DOC]: clean two-stage factorisation — DMPNN monomer encoder with full chemprop
  featurisation (72-d atom / 14-d bond), then `h_poly = Σ f_i h_i`.
- **Result** [DOC]: EA RMSE 0.077 / R² 0.983; IP RMSE 0.067 / R² 0.980. "Significantly outperforms
  original HPG and wDMPNN"; "fold-to-fold variance drops dramatically".
- **[CONFLICT]** §5.4 gives Frac EA R² = **0.983**. §8.3.5, Table 1, in the same document, gives
  Frac overall EA = **0.9741** and IP = 0.9642. Both are presented as the composition-only model.
  Neither states its split or aggregation. Separately, the saved 9-fold CSV
  `results/HPG2Stage_LOMAO/ea_ip__copoly_stage2d_frac__a_held_out__target_EA…csv` (added 2026-07-05)
  has per-fold R² of 0.705, 0.333, 0.797, 0.863, 0.754, 0.870, **−9.48**, 0.255, 0.676 — median
  ≈ 0.75, nowhere near either figure. **Three sources, three different numbers.**
  I cannot determine which produced the table.

**E17 — Stage 2B, residual pairwise interaction**
- **Result** [DOC]: a learned **scalar** interaction weight is stable and improves IP; minimal effect
  on EA. Explicitly contrasted with Phase 4's per-sample gate collapse.
- **Conclusion** [YOU]: interaction effects are low-dimensional; constrained beats adaptive.

**E18 — Stage 2C, typed interaction decomposition — ABANDONED, and correctly so**
- **Diagnostics run before implementing** [DOC]: (1) interaction benefit — IP 5/5 folds,
  mean ΔRMSE ≈ +0.0047; EA 2/5, ≈ +0.0003. (2) Identifiability — BB/BF/FF channels correlate
  |r| ≈ 0.83–0.97, **PC1 = 97.3% of variance**, VIF ≈ 8. (3) Residual explanation — incremental
  ΔR² ≈ 0.013 ± 0.013 (EA), 0.014 ± 0.006 (IP), with unstable, sign-flipping coefficients.
- **Decision** [YOU]: postpone; the benchmark supports only low-rank interaction corrections;
  future decomposition should use electronic (donor/acceptor) rather than structural channels.
- **[ME] This is the best-executed piece of scientific reasoning in the corpus** — a pre-implementation
  identifiability check that killed a model before the compute was spent. It belongs in the thesis
  as a method, independent of whether Stage 2C is ever built.

**E19 — Architecture diagnostics (§8.2)**
- **Design** [DOC]: matched groups g = (A, B, f_A, f_B); Δy = y − ȳ_g isolates architecture effect.
- **Results** [DOC]: σ(ΔEA) = 0.059 eV, σ(ΔIP) = 0.058 eV, p95 > 0.13 eV — larger than DFT precision.
  Systematic direction: alternating ↓EA ↑IP; block the reverse; random intermediate.
  Architecture explains ~51% (EA) / ~60% (IP) of the **residual** variance after Frac, but only
  ~0.8% / ~1.3% of **total** variance. Architecture-label-only offsets transfer at R² ≈ 0.21/0.25;
  adding monomer embeddings + composition raises this to ≈ 0.61/0.72.
- **Conclusion** [YOU]: architecture effects are real, chemistry-dependent, and should modulate
  interactions rather than act as a categorical feature.
- **[NUM]** Independently confirmed in July: `_dataset_design_audit.md` gives 0.98% / 1.46%.
  **This is the most robustly corroborated quantitative finding in the whole body of work.**
- **Code**: `experiments/diagnostics/pre_2d_architecture_diagnostic.py`,
  `feature_conditioned_architecture_transfer.py`, `summary_metrics.csv`, `fold_metrics.csv`.

**E20 — Stage 2D, architecture-aware representations (§8.3)**
- **Conditions** [DOC]: Frac / wDMPNN / 2D0 (global arch embedding; fixed, per-arch, gated variants)
  / 2D1 (chemistry-conditioned arch residual; same three variants).
- **Headline table** [DOC]:

  | Model | Overall EA | Overall IP | ΔEA | ΔIP |
  |---|---|---|---|---|
  | Frac | 0.9741 | 0.9642 | 0.0 | 0.0 |
  | wDMPNN | 0.9700 | 0.9523 | 0.674 | 0.709 |
  | 2D0 | 0.9810 | 0.9789 | 0.847 | 0.908 |
  | 2D1 | 0.9820 | 0.9794 | 0.865 | 0.917 |

- **Conclusion** [YOU]: explicit architecture beats implicit graph encoding; chemistry-conditioning
  adds a small consistent further gain.
- **[RERUN] The wDMPNN row is pre-fix and is the comparison the chapter's argument rests on.**
- **[?] A numerical tension I cannot resolve.** §8.2.3 measured architecture-label-only global
  offsets at R²(Δy) ≈ 0.21 / 0.25. 2D0 is described as adding a *global* architecture embedding
  `h_poly = h_mix + α·e_arch`, and yet scores 0.847 / 0.908 on the same deviation metric. Since
  h_mix is constant within a matched group, a linear readout over that representation should
  reproduce roughly the §8.2.3 global-offset ceiling. Getting 4× the ΔR² implies the readout is
  non-linear in a way that lets chemistry leak into the deviation prediction — in which case
  "2D0 = global architecture" is a misnomer and the 2D0/2D1 contrast is not testing what the
  section says it tests. **Please tell me which it is.**
- **[?]** Fold count. `arch_metric_discrepancy_explanation.md` (18 June) says Stage 2D metrics were
  "pooled across all folds… 5-fold CV covers all data". The saved Stage 2D CSVs have **9 folds**
  and were added 5 July. So the May table and the July CSVs are different run sets.

**E21 — Generalisation across splits (§8.3.7)**
- **Design** [DOC]: three splits of increasing difficulty — group-disjoint (A,B,f_A,f_B),
  pair-disjoint (A,B), A-held-out (entire monomer A). Good design; clearly motivated.
- **Result** [DOC]: 2D1 ΔEA 0.9381 / 0.9346 and ΔIP 0.9649 / 0.9637 on group- vs pair-disjoint —
  near-identical, so not pair-memorisation. A-held-out is where performance drops.
- **Conclusion** [YOU]: architecture transfers across pairings; the open problem is extrapolation
  to unseen chemistry.
- **[ME] This conclusion survived** — it becomes the central design premise of the July/August work.
- **Code**: `configs/wdmpnn_{group_disjoint,pair_disjoint,a_held_out}.yaml`,
  `scripts/python/run_wdmpnn_generalization.py`, `scripts/shell/submit_wdmpnn_generalization.sh`.

**E22 — Learning curves (§8.4.3)**
- **Result** [DOC]: overall EA/IP barely depend on training-set size; ΔR² keeps improving with more
  matched groups but does not plateau.
- **Conclusion** [YOU]: data volume is not the dominant limitation, but more data would still help.
- **[FIG]** No numbers are given; this is read from curves. Treated as qualitative.
- **Code**: `experiments/hpg2stage/output/learning_curve_final`,
  `scripts/python/plot_multi_model_learning_curves.py`.

**E23 — Stage 2E (§9): specified, not run.** Three candidate strategies for property-adaptive
interaction. No results. [DOC]

---

### E24–E33 — June to August 2026: undocumented in the uploads

I am listing these because they are where the current state of the project actually is. Details
are in the repository markdown, not in anything you uploaded.

| # | Date | Work | Where |
|---|---|---|---|
| **E24** | 18 Jun | **wDMPNN implementation fixes** — `PolymerDatapoint.from_smi` + `PolymerDataset` replacing `MoleculeDatapoint`/`MoleculeDataset`; `models.MPNN` import; second `nn.MPNN` call in `train_wdmpnn_fold`. Plus an architecture-deviation metric audit. | `026d5cd`, `af6ac5c`, `f52d499`, `0676f49`, `7d63dfe` |
| **E25** | Jul | **Lever A** — within-group loss `L_within` (λ) pilot on wDMPNN; "worked in-distribution" | `chemprop/nn/within_group_loss.py`, `scripts/shell/submit_pilot_wdmpnn_lambda.sh`, `14_lambda_pilot_comparison/` |
| **E26** | 18 Jul | **HPG-hier** introduced (the successor to HPG-2Stage) | `e70ffc3`, `chemprop/models/hpg_hier.py` |
| **E27** | 25–26 Jul | **Octamer sequence** variant (8-slot chain, `n_A = round(8·frac_A)`, 16 replicas, positional embeddings, attention pooling) and **junction coupling** | `6b0c226`, `fea3a8f`, `7524e67`, `5e51504` |
| **E28** | Jul | **Run-to-run variance study** — 6 identical runs; EA fold 1 group-mean R² = 0.450 / 0.790 / 0.978 | `_noise_floor_results.md` |
| **E29** | 29 Jul | **Model-selection bug found in every runner** (best checkpoint saved, final model used) + full regeneration with 3 seeds | `SUPERVISOR_UPDATE_29-07-2026.md` §1–2, `_regen_v1_results.md` |
| **E30** | 29 Jul | **Null-floor calibration**, **dataset design audit** (9 × 682 × 7 exact factorial), **B-held-out split** construction with Murcko scaffold packing | `_groupmean_metric_floor.md`, `_dataset_design_audit.md`, `windsurf_spec_b_heldout_split.md` |
| **E31** | 5 Aug | **wDMPNN at the original paper's configuration** (batch 50, 30 epochs, no early stopping) — protocol parity closed on the A split | `HANDOFF_2026-08-05.md` §4, `_wdmpnn_original_results.md` |
| **E32** | 5 Aug | **Octamer K=1 ablation** — the 16-replica ensembling hypothesis is *not* supported | `PREREG_octamer_k1_2026-07-30.md`, `_octamer_k1_r3_results.md` |
| **E33** | 8 Aug | **Octamer positional-embedding ablation** pre-registered | `PREREG_octamer_posemb_2026-08-05.md` |

---

## 3. Validity problems that are broader than wDMPNN

These are documented in your own repository. I am surfacing them because they change the scope
of "what needs rerunning" well beyond the caveat you gave me.

### 3.1 The model-selection bug (affects everything)
`SUPERVISOR_UPDATE_29-07-2026.md` §1 [DOC]

> "Every runner (`hpg_hier`, `wdmpnn`, `stage2d` for ChemArch/GlobalArch/frac, and the legacy path)
> saved the best-validation checkpoint during training and then predicted from the final model after
> early-stopping patience expired… Confirmed by git blame to be present in each runner's initial
> commit, so no existing prediction file predates it."

Measured cost for HPG-hier: **mean +0.048 eV MAE**, max +0.114 — larger than most reported differences.

**[?] Critical scoping question: does "the legacy path" cover the runners used for E2–E15?**
If yes, then every number in `Feb16-22`, `Mar02-17`, `Mar16-22`, `Mar23-29`, `Mar30-Apr19`
and `Apr20-26` was produced by a model that was not the one selected. That is a far larger
invalidation than the wDMPNN issue and it would need to lead any thesis-planning discussion.

### 3.2 Run-to-run variance exceeds most reported effects
`_noise_floor_results.md`; `SUPERVISOR_UPDATE_29-07-2026.md` §2 [DOC]

Six runs, same model, same seed, same split, same code: EA fold 1 group-mean R² came out
**0.450 / 0.790 / 0.978** (SD 0.091 on MAE).

Every Feb–May experiment I can identify is **single-run**. Statements such as "within typical seed
variance" (E3), "consistent across folds" (E7), "stable behavior across CV" (E12) were made without
any variance measurement. Effects smaller than ~0.05 eV MAE are, on this evidence, not measurable
at one run per configuration.

**[ME]** The E8 mixture-vs-mean effect (~0.09 eV, 5/5 folds) is one of very few Feb–May results
large enough to plausibly survive this. Most Phase 1–4 win-rate arguments are not.

### 3.3 The evaluation metric itself was partly degenerate
`_groupmean_metric_floor.md`; supervisor update §3.1 [DOC]

An **A-blind null predictor** — one that ignores the held-out monomer entirely — achieves median
group-mean R² of **0.676 on EA** (vs −0.034 on IP) under the A-held-out split, and on EA fold 2 it
**beats** the model (0.961 vs 0.922).

Consequence, in your own words: *"The IP metric is sound. The EA one, on this split, is not, and
that includes the 'wDMPNN wins EA chemistry' comparison."*

**[ME]** This retro-actively weakens every EA claim made on the A-held-out split from E7 onward —
which is a large share of the March–May corpus.

### 3.4 The A-held-out split trains on 7 donor monomers
`_dataset_design_audit.md` [DOC]

The EA/IP benchmark is an exact factorial: 9 A × 682 B × 7 composition/architecture cells = 42,966.
The A-held-out split holds out one A for test and one for validation, so models train on **7**
monomer-A chemistries. Every "unseen chemistry" claim in the March–May documents is a
7-example extrapolation.

---

## 4. wDMPNN register — every place wDMPNN appears

| Exp | Document | Role of wDMPNN | Does a conclusion depend on it? | Post-fix rerun exists? |
|---|---|---|---|---|
| E2 | Feb16-22 §5.1 | last in the architecture ranking; "polymer-specific modifications don't help in homopolymers" | **Yes** — the claim is *about* wDMPNN | **No.** No HTPMD wDMPNN result file found at all |
| E5 | Mar02-17 "Level 4" | proposed as polymer-aware upper bound | No — never run at that point | n/a |
| E7 | Mar16-22 §2.1 | **the reference model**; 2× RMSE advantage over mean+PT baselines | **Yes — this is the load-bearing one.** It motivated the entire aggregation programme | Unclear; the March source file is not locatable |
| E8 | Mar16-22 §2.3 | comparison target; DMPNN+mixture beats it | **Yes** — "simple GNNs can match specialised models" | Partially (E31) |
| E9–E11 | Mar23-29 §1 | inherited framing ("can match or outperform wDMPNN") | Indirect | via E7 |
| E20 | May §8.3.5 | one of four representations; ΔEA 0.674 / ΔIP 0.709 | **Yes** — "explicit beats implicit architecture encoding" | **Yes** — E29/E31 |
| E21 | May §8.3.7 | cross-split comparator | Yes | Yes (A split only) |
| E24 | Jun | *the fix itself* | — | — |
| E25 | Jul | Lever A pilot host model | — | current code |
| E31 | Aug | original-paper configuration parity | — | current code |

**Post-fix status, from `HANDOFF_2026-08-05.md` §4 [DOC]:** re-running wDMPNN at the published
configuration made the baseline **better**, not worse — group-mean R² rose from 0.970/0.962 to
0.982/0.976. It now places chemistries better than HPG-hier does on IP. Its architecture recovery
(ΔR² 0.397/0.565) was unchanged. Your own note: *"'The baseline was under-tuned' is closed off."*

**[ME]** That matters for prioritisation. The reruns done so far have *strengthened* the baseline,
which means the March-era claim "wDMPNN is beaten by simple GNNs with better aggregation" (E8) is
the wDMPNN-dependent claim most at risk, not the Stage 2D architecture claim.

---

## 5. Conflicts and gaps, consolidated

| # | Issue | Sources in tension |
|---|---|---|
| C1 | Split protocol unstated for E2, E3, E4, E5 | all four documents |
| C2 | No HTPMD wDMPNN results file exists | `Feb16-22` §5.1 vs `results/` |
| C3 | wDMPNN "2× better" (§2.1) vs "beaten by DMPNN+mixture" (§2.3) | `Mar16-22`, same document |
| C4 | A "pretraining" experiment is cited but never described | `Mar16-22` §2.3 |
| C5 | `HPG_frac` described as "stable across CV"; saved CSV has R² 0.957→−0.038 | `Mar30-Apr19` vs `results/HPG/…hpg_frac…csv` |
| C6 | Phase 2 "architecture doesn't help" vs Stage 2D "architecture is the main residual signal" | `Mar30-Apr19` vs `May` |
| C7 | Stage 2A/Frac EA R²: 0.983 (§5.4) vs 0.9741 (§8.3.5) vs median 0.75 in the saved CSV | `May` ×2 vs `results/HPG2Stage_LOMAO/` |
| C8 | 2D0 ΔR² 0.847 vs global-offset ceiling 0.21 measured in §8.2.3 | `May` §8.2.3 vs §8.3.5 |
| C9 | Fold count: 5-fold (Mar–May text) vs 9-fold (saved CSVs, July onward) | throughout |
| C10 | Phase 4 results reference figures that are absent from the PDF | `Apr20-26` |
| C11 | Level 3 supergraph/virtual-node: proposed, no code or results found | `Mar02-17` |
| C12 | PAE Tg and Block Copolymer datasets disappear entirely after March | `Mar02-17`, `Mar16-22` vs everything later |
| C13 | The March group-meeting dataset-generation proposal (DFT bandgap/EA/IP) — status unknown | `2026 march group meeting` |

---

## 6. Running lists (v1 — will be revised as you answer)

**Reasonably established (my current read):**
1. Architecture is ~1% of EA/IP total variance but ~50–60% of the post-composition residual.
   Corroborated twice, independently, four months apart. (E19, E30)
2. Composition-weighted pooling is a very strong baseline for EA/IP. (E8, E16, E20)
3. Interaction modelling helps IP and not EA. Survived a change of framework, encoder and metric.
   (E14, E15, E17, E18)
4. The BB/BF/FF interaction decomposition is not identifiable on this benchmark. (E18)
5. The A-held-out split gives a near-degenerate EA chemistry metric. (E30)

**Tentative:**
6. Mixture aggregation > mean aggregation by a large margin. (E8) — large effect, single run.
7. Explicit architecture representation > implicit graph encoding. (E20) — pre-fix wDMPNN arm.
8. Fusion mechanism choice does not matter. (E10) — negative result, single run.

**Currently unsupported or superseded:**
9. "wDMPNN is the weakest architecture" (E2) — no traceable numbers, pre-fix.
10. "Architecture/topology modelling does not help HPG" (E13) — probably a metric artefact (C6).
11. Everything in the 22 July figure set derived from single-run seed-42 predictions
    (your own §7 in the supervisor update).

**Implementation concerns on the record:**
12. Model-selection bug in all runners (E29).
13. Run-to-run variance ≫ most reported effects (E28).
14. wDMPNN dataset-class bug pre-18-June (E24).
15. wDMPNN protocol deviated from the published configuration in three fields (E31).
16. HPG runs with `num_workers=0` — compute comparisons withdrawn (E32/§7).

---

## 7. Questions I need answered before Stage 2

Batched by topic. The first four are the ones that block me most.

### A. Scope of the invalidation
1. **Which fix do you mean by "we recently fixed the WDMPNN implementation"?** I can see four
   candidates and they have very different blast radii:
   (a) 18 June — `PolymerDatapoint`/`PolymerDataset` replacing the Molecule classes (`026d5cd`).
       This one would mean pre-June wDMPNN wasn't consuming polymer structure correctly at all.
   (b) 29 July — the best-checkpoint model-selection bug, which hit *every* model.
   (c) 30 July — the message-passing vectorisation (`4d3eb86`), which is documented as
       numerically equivalent, so a performance fix not a correctness one.
   (d) 5 August — the original-paper protocol parity (batch 50 / 30 epochs), a configuration
       change rather than an implementation fix.

2. **Does the model-selection bug (§3.1, "the legacy path") cover the runners that produced
   E2–E15?** If yes, the Feb–May numbers are all affected and wDMPNN is not the main problem.

3. **Were any Feb–May experiments run with more than one seed?** I have found no multi-seed
   artefacts before the July regeneration.

### B. Traceability of specific results
4. **Where are the March wDMPNN EA/IP numbers stored** (the ≈0.09 RMSE in `Mar16-22` §2.1)?
   I can only find a 9-fold file added on 10 July, which post-dates the fix.

5. **C7 — which number is the real Stage 2A/Frac EA result:** 0.983 (§5.4), 0.9741 (§8.3.5), or
   the ~0.75 median in `results/HPG2Stage_LOMAO/ea_ip__copoly_stage2d_frac…csv`? What are the
   other two?

6. **C5 — is `results/HPG/ea_ip__hpg_frac__a_held_out__target_EA…csv` the file behind the Phase 1
   figures?** Its fold spread (0.957 → −0.038) does not match "stable behavior across CV".

### C. Design intent
7. **C8 — in Stage 2D, is 2D0's readout linear or non-linear in `h_poly`?** If non-linear, "2D0 =
   global architecture effect" is not what the model implements, and the 2D0-vs-2D1 contrast is
   not a clean test of the global-vs-chemistry-conditioned hypothesis.

8. **C6 — do you agree that Phase 2B's "architecture doesn't help" conclusion is an artefact of
   judging on overall R² rather than R²(Δy)?** If so I will record Phase 2B as *superseded by
   metric change* rather than as a contradictory finding.

9. **C1 — what split was used for the HTPMD experiments (E2–E4) and for `Mar02-17` (E5)?**
   I suspect random for both; the "EA/IP is easy, identity suffices" conclusion reads very
   differently under a random split.

### D. Abandoned or unclear directions
10. **PAE Tg and Block Copolymer** (C12): deliberately dropped, paused, or still live? They carry
    the only non-EA/IP copolymer evidence you have, and the only classification task.
11. **The March group-meeting proposal to generate a DFT copolymer dataset** (C13): did this go
    anywhere? Given §3.4 (7 training donor monomers), a chemically more diverse dataset is the
    obvious structural answer to the extrapolation problem — so its status matters a lot for
    Year 3.
12. **E1** — is the pre-February graph-vs-tabular work across `insulator` / `opv` / `polyinfo` /
    `tc` already written up somewhere, or does it also need reconstructing?

---

---

## 8. Answers received — 9 August 2026

| Q | Answer | Effect on this document |
|---|---|---|
| 1 | Fix (a), `026d5cd`, 18 June. The featurizer received raw SMILES rather than Mol objects, so `WDMPNN_Input` was never parsed; pre-18-June wDMPNN had **neither stochastic edge weights nor stoichiometric atom weights**. | Every pre-18-June wDMPNN number is **void by construction**, not merely suspect. It was not a wD-MPNN. Retag E2, E7, E8, E20, E21. |
| 2 | **Yes.** Legacy helpers in `scripts/python/utils.py` — three `ModelCheckpoint` sites, all `monitor="val_loss", save_top_k=1, save_last=True` — never pass `ckpt_path` to predict. | Every Feb–May number is affected, wDMPNN or not. §3.1 confirmed, not hypothesised. |
| 3 | **No multi-seed runs before July.** Every Feb–May CSV has exactly `test/mae, test/rmse, test/r2, split, target`; there is no seed dimension in the artefact set. | §3.2 confirmed. Single-run is documented, not inferred. |
| 4 | Not locatable, and irrelevant — any March wDMPNN run predates 18 June. | **E7 §2.1 recorded as void by construction**, superseding "untraceable". |
| 5 | None of the three Frac numbers is "real"; they are three unlabelled (split, protocol, code-version) combinations, all superseded. The 9-fold CSV was added 5 July, before the 29 July checkpoint fix, so it is void too. | C7 reclassified from *conflict* to **provenance failure**. Note: fold 6 R² = **−9.479**, a catastrophic single-fold failure unremarked at the time — exactly the pathology the July noise-floor work later characterised. |
| 6 | Cannot determine; moot, pre-fix either way. | C5 closed. The substantive point stands: a fold spread of 0.957 → −0.038 is not "stable behaviour across CV" and that description must not survive into the thesis. |
| 7 | **Resolved by reading `chemprop/nn/stage2d.py` — see below.** | C8 closed. |
| 8 | Agreed — Phase 2B is a **metric artefact**, corroborated by the ~1% total / ~50–60% residual variance split measured twice, four months apart. | Phase 2B recorded as **superseded by metric change**, not as a contradictory finding. |
| 9 | Cannot determine from artefacts; random split suspected. | C1 recorded as **unknown**. The conclusion "identity suffices for EA/IP" is not carried forward. |
| 10 | Data still present (`data/pae_tg.csv`, `data/pae_tg/`, `data/block.csv`, results across GAT/GIN/DMPNN). Availability is not the constraint. | C12 reclassified from *abandoned* to **dormant but available**. |
| 11 | Live and costed: pipeline portable, monomers fit, ~8–9 kSU CPU for ~2,000 polymers. | C13 closed. |
| 12 | E1 **is** Feb16-22. Year 1 was literature review, written up as a review paper now accepted at *Digital Discovery*. | E1 merged into E2/E3/E4. Year 1 output = the review paper, not experiments. |

### 8.1 Q7 resolved — the 2D0 readout is non-linear, and the §8.3 framing does not hold

**[CODE]** `chemprop/nn/stage2d.py`:

```python
# lines 211–219 — prediction head, shared by ALL variants including `frac`
self.heads = nn.ModuleList([
    nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(),
                  nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    for _ in range(n_targets)])

# lines 291–299 — 2D0
h_poly = h_mix + alpha * e_arch          # 2d0_fixed
h_poly = h_mix + alpha[arch] * e_arch    # 2d0_arch

# line 361
preds = torch.cat([head(h_poly) for head in self.heads], dim=1)
```

The readout is `Linear → ReLU → Dropout → Linear`. **Non-linear.**

Why that settles it: inside a matched group `(A, B, f_A, f_B)` every input is constant except
`arch`, so `h_mix` is fixed and `e_arch` takes one of three learned values. If the head were
**linear**, the architecture deviation would be exactly `Δŷ = α·W·(e_arch − ē)` — a constant
offset per architecture class, independent of chemistry. That is precisely the quantity §8.2.3
measured, and its ceiling was **R²(Δy) ≈ 0.21 (EA) / 0.25 (IP)**.

2D0 reports **0.847 / 0.908**. The ReLU is what makes that possible: it lets `h_mix` modulate how
the architecture shift is decoded, so chemistry enters the deviation prediction implicitly.

**Consequences — three sentences in `May.pdf` §8.3 need rewriting, not just re-running:**

1. "2D0 = architecture contributes a **transferable global effect**" is **false as implemented**.
   2D0 is already chemistry-conditioned; the conditioning is hidden in the readout instead of
   being declared in the model definition.
2. The 2D0-vs-2D1 contrast is therefore **not** a test of global-vs-chemistry-conditioned
   architecture. Both are chemistry-conditioned. 2D1 only moves the conditioning from the readout
   into the residual input `z = [h_A, h_B, |h_A−h_B|, h_A⊙h_B, f_A, f_B, e_arch]`.
3. The small 2D0 → 2D1 gap (0.847 → 0.865 EA, 0.908 → 0.917 IP) was read as *"architecture is
   mostly global, with a modest chemistry-dependent component."* **The opposite reading is the
   supported one.** The real global-only ceiling is 0.21/0.25; the jump to ~0.85 is the
   chemistry-conditioning effect and it is enormous. Whether that conditioning is expressed
   explicitly (2D1) or absorbed by the readout (2D0) turns out to matter very little.

**[ME]** This is good news for the science and bad news for one paragraph of prose. The Section 8.2
diagnostic conclusion — *"architecture effects are strongly chemistry-dependent"* — is **more**
strongly supported than the document claims, because the 0.21 → 0.85 gap is the cleanest measurement
of it in the corpus. What has to go is the model-comparison narrative built on top.

**Clean fix if you want the 2D0 arm to mean what it says**: add a linear-readout variant, or report
the §8.2.3 global-offset predictor as the true "global architecture" baseline row in the table and
relabel 2D0 as *implicitly conditioned*. The first is a one-line change; the second costs nothing.

---

*Next: Stage 2 — `02_stage2_research_questions.md`.*
