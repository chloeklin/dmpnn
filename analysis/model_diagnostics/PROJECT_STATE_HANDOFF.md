# Project State — Handoff Summary

*A single-file context dump to start a fresh chat. Written 2026-07-26.*

---

## 0. Who / what / how

- **You:** Chloe, 2nd-year PhD, polymer molecular representation learning.
- **Working model:** you run **Windsurf** (AI coding assistant) in your repo and paste its plans/outputs here for critical review; I write Windsurf prompts and audit its plans/code before you run them.
- **Compute:** Gadi (NCI), PBS/qsub, project `ng76`, storage `scratch/um09` + `gdata/dk92`, `gpuvolta` queue, venv `/home/659/hl4138/dmpnn-venv`.
- **Repo:** `Desktop/experiments/dmpnn` (chemprop fork). Key folders: `analysis/model_diagnostics/` (reports, figures), `chemprop/models/`, `chemprop/featurizers/molgraph/`, `scripts/`.

---

## 1. The core scientific idea

Reframe copolymer property prediction from **"which model wins"** to **"what are the sources of variation, and how should a representation allocate capacity to them."**

**Diagnostic decomposition** (your main methodological contribution): split any property into
- **group-mean R²** = *chemistry baseline* (how well you place a monomer-pair's average property), vs
- **ΔR² + pairwise ordering** = *architecture recovery* (how well you capture the deviation due to sequence/architecture).

Key empirical fact on EA/IP: **architecture is only 1–4% of the variance** — chemistry dominates.

---

## 2. Models compared (all on EA/IP copolymer data)

- **wDMPNN** (Aldeghi & Coley) — flat atom graph, stochastic/weighted junction edges, stoichiometry-weighted readout. Strong chemistry baseline.
- **HPG (original, Han et al. 2025)** — fused single graph (atoms + virtual monomer nodes + 3 edge types), one GAT stack, sum-pool. Junction atoms dropped. (You verified the reimplementation: atom features are 130-d not the claimed 49-d; junction atoms dropped vs original Mg pseudo-atoms; LeakyReLU 0.2 vs 0.01.)
- **ChemArch** (`2d1_arch`) — per-monomer encoding + fixed linear mix `h_mix = f_A·h_A + f_B·h_B` + discrete architecture-label residual. Best architecture recovery *in-distribution*, collapses on unseen chemistry.
- **HPG-hier (your new model)** — TRUE two-stage hierarchy:
  - **Stage-1:** per-monomer D-MPNN → monomer embeddings m_A, m_B (chemistry encoder).
  - **Stage-2:** monomer graph, edge features `[16-d port-pair | 1 transition weight]`, message passing (transition weight is a **feature**, not a multiplier). Transition weights = first-order Markov matrix (alternating=off-diagonal, block=diagonal-dominant, random=uniform), derived from `WDMPNN_Input` bond rules in `ea_ip.csv`.

**Model files:** `chemprop/models/hpg_hier.py`, `chemprop/featurizers/molgraph/hpg_hier.py`, `chemprop/data/hpg_hier.py`. Runner: `scripts/python/run_hpg_generalization.py`. Config flags: `stage2_edge_weight={feature,multiplier,both}`, `stage2_mode={transition_graph,octamer_sequence}`, `junction_coupling={off,on}`, `n_coupling_steps`, `octamer_len=8`, `n_random_samples=16`.

---

## 3. Results so far (seed 42, single seed; medians across folds for LOMO)

**Headline:** HPG-hier is the **best architecture-recovery model on unseen chemistry**, matches wDMPNN on chemistry **except one hard monomer** (EA fold-1 dibenzothiophene sulfone, where isolated Stage-1 mis-places the baseline by ~0.2 eV).

Monomer-heldout architecture ΔR² (median): **HPG-hier EA 0.79 / IP 0.82** vs wDMPNN 0.58 / 0.46 vs ChemArch 0.76 / 0.66. Chemistry (group-mean R²): wDMPNN wins EA (0.965), ties IP; ChemArch collapses.

**Ensemble** (HPG-hier + wDMPNN, simple average) **beats both** — complementary blind spots (they fail on *different* held-out monomers).

**Phase-1 variants:**
- **Q1 (weighted edges / "wedge")** — ❌ DROPPED. Helped in-distribution, collapsed OOD (IP architecture ΔR² 0.820→0.558). Learned-feature version is more robust.
- **Junction coupling (Stage-1 cross-junction message passing)** — ✅ fixes EA fold-1 (group-mean 0.575→0.925, MAE 0.213→0.085) and EA LOMO now beats wDMPNN on chemistry. But at `n=2` it **over-couples**: IP fold-5 chemistry collapse (0.770→0.494), EA fold-6 architecture collapse. `n_coupling_steps=1` tuning pending. **Concern you raised:** if junction coupling is just "HPG + wDMPNN combined," it's not novel enough.
- **Q2 (octamer / explicit length-8 sequence)** — construction bug fixed (builder ignored bond rules → block/alt/random were identical; now architecture-conditioned, deterministic block/alt + K=16 sampled random, **averages predictions not embeddings**). Re-gated ΔR² 0.0→0.92; LOMO run pending.

**Reports:** `variant_results_report.md` (full results table), `model_architectures_reference.md` (all variants), `report_figures/` (SVG + PNG figures).

---

## 4. The novelty problem & the dataset hunt (most recent work)

You want to **propose a novel architecture**, and worried the model-combination framing is weak. So we asked: is there an **architecture-dominant** property/dataset where a sequence-aware model can shine? Ran Consensus literature searches + audited candidate datasets locally.

**What the literature said:**
- The ensemble-average limitation is already acknowledged (Aldeghi & Coley themselves) and partly addressed (Tao 2022 RNN beats composition-only for sequence-dependent Tg). Hierarchical polymer GNNs are **crowded** (Kimmig, PU-Graph, HiMol, etc.). Diblock BCP phase is **done** (Arora 2021 RF ~90%).
- **The one open gap:** *"no ML paper trains on datasets where random/gradient/block/alternating sequence are varied at fixed composition and encoded as the predictive signal."* Experimental architecture-labeled data at ML scale **does not exist** (field's central bottleneck).

**Datasets audited (local folders):**
- **`block.xlsx` = BCDB (Rebello 2024)** — ❌ OUT. 5,400 rows but 99.7% diblock; = Arora's ~90%-solved dataset. Chemistry-disjoint baseline = majority baseline (χ is chemistry-specific, not in features).
- **Tao 2022 (`Desktop/Copolymer/`)** — ❌ OUT. 4 datasets; none is a fixed-composition architecture sweep. Dataset 1 (47,988 trimers → EA/IP) has *no* sequence axis (CNN "sequence" input is a constant tile). Dataset 3 (PHA Tg) has a real Random/Block column but only **8 block samples**. The flagship "sequence-aware" paper is trained on composition/oligomer labels.
- **Curtis 2025 (`Desktop/stochastic-sequences/`, example sims `Downloads/polymers_p0.0/`)** — ✅ **best architecture-dominant benchmark.** 270 sequences **all 12A/8B (composition EXACTLY fixed)**; task = 20-bead A/B sequence → 2D morphology embedding (Z0,Z1); 14,843 rows, ML-ready. **Catch:** monomers are featureless A/B beads → NO chemistry → tests only HPG-hier's Stage-2.
- **Webb 2020 (`Downloads/supporting_information/`)** — 1,540 sequences → single-chain Rg; also bead-only, smaller. Fallback.

**The real structural gap:** **no ML-ready dataset has BOTH rich monomer chemistry AND dominant architecture signal.** EA/IP = chemistry-dominant (stresses Stage-1); Curtis = architecture-dominant, no chemistry (stresses Stage-2). Neither alone exercises the full hierarchy.

---

## 5. Current recommended direction — the "two-regime" paper

Validate the hierarchy **component-wise** across two complementary regimes, using data you already have:
1. **EA/IP (chemistry-dominant):** Stage-1 handles chemistry; model still recovers the small architecture signal → your diagnostic decomposition.
2. **Curtis (architecture-dominant, fixed composition):** Stage-2 sequence encoder should beat the RNN baseline where architecture is the whole signal.

**Contribution =** (a) the diagnostic decomposition that identifies which regime a property lives in, + (b) a hierarchical representation whose two stages are each validated where they matter. Novel, fully data-supported, no new simulation needed.

**Honest caveats for the paper:** on Curtis, HPG-hier = Stage-2 only (frame as validating the *architecture encoder*); testing both stages simultaneously needs a chemistry-×-architecture dataset → name as future work (MD/SCFT generation).

**Banked fallback:** the **EA/IP diagnostic paper** stands alone as a no-data-risk contribution regardless.

---

## 6. Open tasks / pending runs

- Gadi: junction `n_coupling_steps=1` (tune to keep EA win, recover IP/architecture cost); octamer 38-cell LOMO; failure-fold diagnosis (IP fold-5, EA fold-6); **seeds 43/44** for error bars (all current numbers are seed-42 point estimates).
- **Next concrete step discussed:** draft a Windsurf spec to **port the Stage-2/octamer encoder onto Curtis 2025** and benchmark against their RNN (their trained model `embedding_model_noavg_absolute.pth` is in the repo for comparison) — with a fair fixed-composition train/val split.

---

## 7. Key files (all in `analysis/model_diagnostics/` unless noted)

- `PROJECT_STATE_HANDOFF.md` — **this file.**
- `dataset_hunt_queries.md` — full dataset-hunt results, audits (Tao/Curtis/Webb/BCDB), ranked candidates, two-regime recommendation.
- `variant_results_report.md` — all model/variant results.
- `model_architectures_reference.md` — architecture definitions + variant catalog.
- `EXPERIMENT_PLAN.md` — master plan (levers, phases, seed-42 results).
- `critical_assessment_and_roadmap.md` — original brutal assessment + roadmap.
- `hpg_implementation_verification.md` — HPG reimplementation audit.
- `hierarchical_representation_design_note.md`, `paper1_objective_design_draft.md` — design notes.
- `report_figures/` — figures.
- Supervisor materials: `supervisor_update_memo.md`, `pilot_update.pptx`.

---

## 8. Working preferences (for the new chat)

- Wants **honesty over encouragement** — brutal, critical assessment; call out weak novelty.
- Concise and direct; minimal fluff.
- Reviews Windsurf's plans/code via me before running; I write Windsurf prompts and insist on correctness checks (several real bugs caught this way: junction pre/post-deletion indexing, octamer sequence-construction collapse, predict-on-average vs average-of-predictions).
