# Context pack — seed any new chat with this

Written 9 August 2026. Purpose: stop each new conversation re-deriving the same facts and
repeating the same three errors. Paste this whole file, plus the documents listed in §5.

---

## 1. Corrections other conversations have already got wrong

These have each been made at least once by an assistant working without full context. Correct
them up front.

| claim | status |
|---|---|
| "Is the K=1 arm re-sampling sequences each epoch?" | **No.** `_build_octamer_sequences` runs inside the featurizer (`chemprop/featurizers/molgraph/hpg_hier.py:251`), called once from `_build_hier_graphs` (`run_hpg_generalization.py:439`) before training starts. Sequences are fixed per polymer for the whole run. K=1 genuinely trained on one fixed sequence. The averaging limb of the protocol-matching hypothesis **is** excluded. |
| "An 8-bin composition histogram is the missing null model." | **Its ΔR² is 0 by construction.** At fixed composition every polymer has the same histogram, so within-group deviations are identically zero. Derivable without running it. Useful only as an overall-R² ceiling, which the existing A-blind/B-blind null floor already provides. |
| "If position embeddings don't matter, the model is a multiset / composition histogram." | **Does not follow.** Path message passing reads local neighbourhoods, so `AABB` and `ABAB` differ even with identical slot embeddings. Removing position embeddings makes the model invariant to **reversal** (`AABB ≡ BBAA`), not to arbitrary permutation. Reversal-invariance is arguably physically correct for a chain read from either end. |
| "Run-to-run SD is 0.091." | 0.091 is the fold-1 **MAE** SD in eV. The **group-mean R² SD is 0.268** for the three values 0.450 / 0.790 / 0.978. Quote both, attached to the right quantities. Corrected in `HANDOFF_2026-07-29.md` §4 and §9. |
| "The noise floor was six runs of one configuration." | 2 folds × 3 repeats. The 0.450/0.790/0.978 triple is fold 1 only. |
| "HPG is ~2.3× cheaper to train than wDMPNN." | **Reversed and withdrawn.** The wDMPNN timings were inflated by a Python loop over a CUDA tensor in `_WeightedBondMessagePassingMixin.message`. After vectorisation wDMPNN runs at 20.2 s/epoch and 6.0 SU/run against HPG-hier's 115.4 s/epoch and 36.5 SU/run. Do not publish the inverse either — HPG uses `num_workers=0` against wDMPNN's 4. **Report no cross-family compute comparison.** |

---

## 2. Validity findings — verified against the repository, 9 August

**The 18 June fix is the one with the largest blast radius for wDMPNN.** Commit `026d5cd`:
the featurizer was receiving raw SMILES instead of Mol objects, so `WDMPNN_Input` was never
parsed. Pre-18-June wDMPNN received neither stochastic edge weights nor stoichiometric atom
weights — the two features that define the method. **Every pre-18-June wDMPNN number is void,
not merely suspect.** This includes the Feb HTPMD ranking, the March "2× better RMSE" claim,
and the wDMPNN row of the May Stage-2D table.

**The model-selection bug covers the legacy runners too.** `scripts/python/utils.py` has three
`ModelCheckpoint` sites, all `monitor="val_loss", save_top_k=1, save_last=True`, and **no
`ckpt_path=` is passed to predict anywhere in the file**. So training saved the best checkpoint
and then predicted from the final patience-expired model. Every February–May number is affected,
wDMPNN or not.

**Feb–May was single-run throughout.** Every results CSV in `results/` carries exactly
`test/mae, test/rmse, test/r2, split, target`. There is no seed dimension anywhere in the
artefact set. This is established, not inferred.

**The three conflicting "Frac EA" numbers are not a contradiction.** 0.983 (May §5.4), 0.9741
(May §8.3.5) and median 0.754 (`results/HPG2Stage_LOMAO/…frac…csv`, 9 folds) are three
unlabelled combinations of split, protocol and code version. The CSV was added 5 July, before
the 29 July checkpoint fix, so it is void independently. Record as "three unlabelled
configurations, all superseded." Note fold 6 in that file gives **R² = −9.479**, a catastrophic
single-fold failure that went unremarked at the time.

**Phase 2B's "architecture doesn't help" is a metric artefact.** Architecture is ~1% of total
variance but 51–60% of post-composition residual variance — corroborated independently four
months apart. Overall R² cannot resolve a 1% effect. Record Phase 2B as *superseded by metric
change*, not as a contradictory finding.

**PAE Tg and Block datasets are still present** — `data/pae_tg.csv`, `data/pae_tg/`,
`data/block.csv`, with results files. An off-protocol test is a scoping decision, not a
data-availability problem.

---

## 3. Current results, as of 9 August

A split, median over 9 folds, three seeds averaged at the prediction level. `hpg_hier_octamer`
takes every column on both targets.

| model | overall R² EA / IP | MAE eV EA / IP | ΔR² EA / IP |
|---|---|---|---|
| wDMPNN, published config | 0.967 / 0.971 | 0.070 / 0.050 | 0.397 / 0.565 |
| HPG-hier | 0.966 / 0.890 | 0.067 / 0.068 | 0.776 / 0.808 |
| **HPG-octamer** | **0.984 / 0.978** | **0.055 / 0.035** | **0.849 / 0.886** |

Paired per-fold, octamer versus wDMPNN at its published configuration: EA overall R² **9/9
folds, p = 0.004**; IP ΔR² **9/9, p = 0.004**; four more at p = 0.039. **Two comparisons reach
the minimum attainable p for 9 folds** — `HANDOFF_2026-07-29.md` §4's "nothing reaches
significance" is superseded. Fix the comparison family *before* quoting any p-value.

**On the B split's cross-scaffold folds the two models tie on accuracy** (3/5 folds, p = 1.0 on
overall R², MAE and group-mean R²) **and separate by 0.15–0.26 on ΔR²** (5/5 folds). Same
predictions, two metrics pointing opposite ways. This is the strongest form of the measurement
argument.

**Dataset structure.** Exact factorial, 6,138 rows per cell: block and random at fracA
0.25/0.50/0.75, alternating at 0.50 only. So architecture recovery is a **2-way discrimination
at fracA 0.25 and 0.75, 3-way at 0.50**. There are only **three distinct transition matrices**
in all 42,966 rows, and they do not vary with composition — architecture and composition are
independent axes.

**Open attribution.** Five factors separate the octamer from HPG-hier. Factor 5 (the 16-replica
ensemble) is **excluded** by the K=1 arm, pre-registered outcome C. Factors 1 (8-slot topology),
2 (positional embeddings), 3 (attention readout) and 4 (discarded 17-d edge features) remain.
Effect sizes sit at or below the noise floor, so **no ablation on this dataset can separate
them** — a limit of the data, not of effort.

**The confound to state first, before anyone else finds it.** Labels were computed on
**octamers**, over up to 32 sampled sequences, then averaged (paper SI p. S4). The best model
uses an 8-slot chain, 16 sampled sequences, and averages predictions. Its structure mirrors the
label-generation protocol. This cannot be resolved on this dataset by any ablation.

---

## 4. Standing methodological commitments

Three seeds (42/43/44) averaged at the **prediction** level; best-checkpoint predictions under
`--frozen_protocol`; paired per-fold sign tests with Holm correction; every group-mean R²
reported beside its fold-specific null floor; S and D B-split fold groups never pooled;
pre-registration before submission with post-hoc changes disclosed as **dated addenda, never
edits**. Dated documents are immutable — supersede, don't rewrite.

Minimum attainable two-sided sign-test p: **0.0039** at 9 folds, **0.0625** at 5, **0.125** at 4.

---

## 5. Documents to attach alongside this pack

- `analysis/model_diagnostics/HANDOFF_2026-08-05.md` — current state, supersedes the 29 July handoff where they differ
- `analysis/model_diagnostics/HANDOFF_2026-07-29.md` — methodology, splits, the corrections table
- `analysis/model_diagnostics/PREREG_octamer_k1_2026-07-30.md` — pre-registration practice, with two disclosed addenda
- The Stage 1 audit document — the Feb–May inventory
- `writing/paper2_outline.md` — claim → evidence map
- `29-07-2026 supervisor_update/SUPERVISOR_UPDATE_29-07-2026.md`

---

## 6. What not to spend time on

Re-running February–May. Between the 18 June input bug, the checkpoint bug in every runner,
single-run everything and a degenerate EA metric on the A split, that work cannot be repaired
selectively. **Write it up as the phase that produced the methodology** — the null-floor
predictor, ΔR², the three-seed protocol and pre-registration all exist because it failed in
diagnosable ways. That is a real contribution and it is more honest than a table of numbers
requiring caveats into meaninglessness.
