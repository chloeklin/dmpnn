# Variant Results — Report

Running record of all model results on EA/IP. Last revised 2026-07-26 after three provenance/metric
audits. **Read §0 before using any number in this file.**

Metrics: group-mean R² = chemistry baseline; ΔR² + pairwise ordering = architecture recovery;
overall R²/MAE = bottom line. Architecture definitions: `model_architectures_reference.md`.

---

## 0. Status and reliability — read first

Three things limit every comparison below. None of them was known when the earlier version of this
report was written.

**0.1 MEASURED: training is unstable, and the noise exceeds every effect in this report.**
Six Gadi V100 runs of `hpg_hier`, EA, A-heldout, seed 42, current code, varying only the repeat label
(`_code_drift_investigation.md`, noise-floor analysis):

| fold | group-mean R² across 3 repeats | MAE (eV) across 3 repeats | MAE SD |
|---|---|---|---|
| 0 | 0.962 / 0.982 / 0.986 | 0.084 / 0.055 / 0.052 | 0.018 |
| 1 | **0.790 / 0.450 / 0.978** | 0.146 / 0.226 / 0.045 | **0.091** |

Committed code drift was excluded (post-20-July commits are gated behind the octamer and junction
paths). This is run-to-run instability at fixed seed. Wall times spanned 2,866–5,214 s, so runs
terminate at very different points; the leading hypothesis is that early stopping is driven by a
validation set consisting of a single held-out A monomer, making the stopping signal noisy and
monomer-specific (`windsurf_prompt_training_stability.md`).

> **Consequence: no single-run comparison in this report is interpretable.** Under the optimistic
> fold-0 noise estimate (±2 SD = ±0.035 eV) roughly half the octamer's per-fold MAE gains survive;
> under the fold-1 estimate (±0.182 eV) one of nine does. Every conclusion below that rests on
> differences smaller than these bands is withdrawn pending repeat-measured results.

**0.1b The "pathological folds" are probably variance, not chemistry.** Fold-1 MAE SD is 5× fold-0's.
The folds this project has repeatedly treated as chemically interesting — EA 1, EA 6, IP 5, IP 2 —
are candidates for simply being the high-variance folds. This must be tested by repeating them before
any fold-specific chemical interpretation stands.

**0.2 Single seed.** Everything here is seed 42. Seeds 43/44 do not exist yet.

**0.3 Nothing survives multiple-comparison correction.** The largest effects reach an uncorrected
two-sided sign-test p of 0.039 across 9 folds; the comparison family has ~20 members. With 9 folds the
minimum attainable two-sided p is 0.0039.

**Convention change.** Headline numbers are now the **median of paired per-fold differences**, not the
difference of separately ranked medians. The old convention overstated the octamer's EA chemistry gain
by roughly 2× (+0.064 reported vs +0.031 actual).

---

## 1. Headline

- **HPG-hier recovers architecture on unseen chemistry better than wDMPNN.** Monomer-heldout ΔR²
  0.79/0.82 vs 0.58/0.46, ordering 0.80/0.83 vs 0.75/0.76. This has survived every audit and is the
  result of record — subject to §0.1.
- **No Phase-1 variant has moved the architecture axis.** Q1 (wedge) collapsed OOD; junction coupling
  at n=1 and n=2 moves incoherently across pathological folds; the octamer changed EA ΔR² by −0.001
  (5 wins / 4 losses across folds) and IP ΔR² by an amount not statistically supported (7/2, p = 0.18).
  Three interventions aimed at architecture recovery, three misses.
- **The octamer's real effect is on chemistry placement, not architecture** — and its provenance is
  clean (§5.3). Whether that gain is the explicit sequence or the attention readout is untested.
- **The EA chemistry metric on the A-heldout split is close to degenerate** (§2.3). Claims resting on
  it, including "wDMPNN wins EA chemistry," are not safe.

---

## 2. Dataset design and what each split actually tests  *(new — from `_dataset_design_audit.md`)*

### 2.1 The design

Exact factorial: **9 A monomers × 682 B monomers × 7 composition/architecture cells = 42,966 rows.**
Every A/B pair appears exactly 7 times. A and B monomer sets are disjoint. The 7 cells are
25/75 block+random, 50/50 alternating+block+random, 75/25 block+random.

Rows per A monomer: 4,774. Rows per B monomer: 63.

### 2.2 Where the signal lives

| target | A identity | B identity | A + B | fracA | poly_type | within (A,B,fracA) |
|---|---|---|---|---|---|---|
| EA | 0.418 | 0.457 | 0.929 | 0.001 | 0.003 | **0.98%** |
| IP | 0.500 | 0.329 | 0.897 | 0.019 | 0.005 | **1.46%** |

Architecture accounts for ~1–1.5% of variance, confirming the figure the project has assumed.
Monomer identity accounts for ~90%.

### 2.3 The A-heldout split is a 7-example extrapolation, and its EA metric has little headroom

The A-heldout generator excludes the test A **and a second A for validation**, so models train on
**7 donor monomers** while A identity carries 42–50% of total variance.

An **A-blind null** — a lookup predicting each test row from the training mean of its
`(smiles_B, fracA, poly_type)` cell, using no information about the held-out monomer — scores:

| split | target | null floor (median group-mean R²) | headroom to 1.0 |
|---|---|---|---|
| A-heldout | EA | **0.676** | 0.324 |
| A-heldout | IP | −0.034 | **1.034** |
| B-heldout (random) | EA | 0.420 | **0.580** |
| B-heldout (random) | IP | 0.560 | 0.440 |
| B-heldout (clustered) | EA | 0.384 | 0.616 |
| B-heldout (clustered) | IP | 0.556 | 0.444 |

On EA the A-blind null **beats HPG-hier on fold 2** (0.961 vs 0.922) and comes within 0.09 on fold 1.
The reason is structural: within an A-heldout fold the held-out A is constant, so all across-group
variation comes from B, composition and architecture — all seen in training.

> **Consequence: EA chemistry claims belong on the B-heldout split; IP chemistry claims belong on the
> A-heldout split.** The architecture metrics (ΔR², ordering) measure within-group deviation recovery
> and are unaffected by these floors; they remain valid on both splits.

### 2.4 The two splits are not difficulty-matched

Maximum Tanimoto (Morgan r=2, 2048-bit) from each held-out monomer to the training monomers of the
same role — median across each fold:

| split | median nearest-neighbour similarity |
|---|---|
| A-heldout (1 monomer/fold) | 0.308 – 0.474 |
| B-heldout, random (76/fold) | 0.519 – 0.579 |
| B-heldout, clustered (76/fold) | 0.483 – 0.500 |

Random B folds are **easier** than A-heldout; the clustered assignment is the difficulty-matched
comparator. Absolute scores must not be compared across splits without this caveat.

Six held-out B monomers (of 682) have a training near-duplicate at Tanimoto ≥ 0.95, four of them
≥ 0.99. The folds were not rebuilt; all B-split metrics are to be reported twice, full and filtered.

### 2.5 What the B split enables that the A split cannot

With 76 held-out monomers per fold, B-split runs support a **performance-versus-novelty curve**
(binning held-out monomers by similarity to training). The A split has one held-out monomer per fold,
so its novelty is a single constant per fold and no such curve exists.

**Status: no models have been trained on the B split yet.** 144 cells (4 models × 2 targets × 9 folds
× random and clustered assignments, seed 42) are specified and generated but unsubmitted —
`windsurf_spec_b_heldout_split.md`.

---

## 3. Main three-way comparison — wDMPNN / HPG-hier / ChemArch (A-heldout, seed 42, medians)

### Chemistry extrapolation — group-mean R², shown against the A-blind null floor
| Target | HPG-hier | wDMPNN | ChemArch | **A-blind null** |
|---|---|---|---|---|
| EA | 0.925 | 0.965 | 0.815 | **0.676** |
| IP | **0.969** | 0.968 | 0.853 | **−0.034** |

The EA row spans 0.815–0.965 against a floor of 0.676 — the previously reported "wDMPNN wins EA
chemistry" is a 0.04 gap inside the low-information band, on a split where the null beats HPG-hier
outright on one fold. **Do not cite the EA row.** The IP row is sound.

### Architecture recovery — ΔR² (median)
| Split · Target | HPG-hier | wDMPNN | ChemArch |
|---|---|---|---|
| Group-disjoint · EA | 0.90 | 0.87 | **0.94** |
| Group-disjoint · IP | 0.94 | 0.92 | **0.97** |
| **Monomer-heldout · EA** | **0.79** | 0.58 | 0.76 |
| **Monomer-heldout · IP** | **0.82** | 0.46 | 0.66 |

### Ordering accuracy — monomer-heldout (median)
| Target | HPG-hier | wDMPNN | ChemArch |
|---|---|---|---|
| EA | **0.799** | 0.753 | 0.790 |
| IP | **0.826** | 0.758 | 0.792 |

**Read:** in-distribution ChemArch leads architecture; on unseen chemistry HPG-hier leads architecture.
The chemistry half of this comparison is only interpretable for IP.

## 4. Complementarity — HPG-hier + wDMPNN ensemble (A-heldout, seed-42 median)

| | HPG-hier | wDMPNN | **Ensemble** |
|---|---|---|---|
| EA MAE (eV) | 0.094 | 0.069 | **0.066** |
| IP R² | 0.963 | 0.957 | **0.973** |
| IP MAE (eV) | 0.051 | 0.062 | **0.046** |

The models fail on different held-out monomers, so a free average beats both at seed 42. Not
multi-seed confirmed. The EA-MAE margin (0.003) is far inside any plausible run-to-run noise band
(§0.1) and should not be cited until §0.1 is resolved.

## 5. Phase-1 variants (A-heldout, seed 42)

### 5.1 Q1 — wedge (transition weight as multiplier) — ❌ DROPPED
- In-distribution: small ΔR² gain (EA GD 0.898 → 0.913).
- Monomer-heldout: **IP architecture collapsed** (ΔR² 0.820 → 0.558, ordering 0.826 → 0.753).
- Finding: hard-weighting helps in-distribution, hurts OOD. The learned-feature version is more robust.

### 5.2 Q3 — junction coupling depth — ⚠️ UNDER-DETERMINED (was "n=1 dropped")

| Target | Model | group-mean R² | ΔR² | ordering | overall R² | MAE (eV) |
| --- | --- | --- | --- | --- | --- | --- |
| EA | Baseline | 0.925 (0.900) | 0.790 (0.686) | 0.799 (0.781) | 0.924 (0.893) | 0.094 (0.104) |
| EA | Junction n=1 | 0.963 (0.883) | 0.785 (0.722) | 0.803 (0.771) | 0.961 (0.872) | 0.059 (0.085) |
| EA | Junction n=2 | 0.971 (0.949) | 0.788 (0.613) | 0.788 (0.761) | 0.969 (0.937) | 0.086 (0.077) |
| IP | Baseline | 0.969 (0.921) | 0.819 (0.734) | 0.826 (0.818) | 0.963 (0.917) | 0.051 (0.072) |
| IP | Junction n=1 | 0.952 (0.868) | 0.762 (0.719) | 0.810 (0.803) | 0.942 (0.862) | 0.058 (0.077) |
| IP | Junction n=2 | 0.962 (0.894) | 0.775 (0.766) | 0.809 (0.811) | 0.961 (0.890) | 0.050 (0.074) |

Medians, fold means in parentheses.

| Target | Fold | Group-mean R²: baseline / n=1 / n=2 | ΔR²: baseline / n=1 / n=2 | Ordering: baseline / n=1 / n=2 | Notable move |
| --- | --- | --- | --- | --- | --- |
| EA | 0 | 0.925 / 0.989 / 0.915 | 0.734 / 0.785 / 0.772 | 0.745 / 0.753 / 0.788 | n=1 chemistry gain |
| EA | 1 | 0.575 / 0.962 / 0.925 | 0.790 / 0.813 / 0.765 | 0.812 / 0.806 / 0.724 | n=1 sulfone rescue |
| EA | 2 | 0.922 / 0.904 / 0.978 | 0.845 / 0.672 / 0.831 | 0.806 / 0.751 / 0.775 | n=1 chemistry/architecture loss |
| EA | 3 | 0.975 / 0.985 / 0.986 | 0.837 / 0.912 / 0.803 | 0.799 / 0.867 / 0.869 | n=1 architecture gain |
| EA | 4 | 0.951 / 0.955 / 0.980 | 0.347 / 0.583 / 0.561 | 0.578 / 0.598 / 0.536 | both coupling depths help |
| EA | 5 | 0.969 / 0.976 / 0.974 | 0.882 / 0.869 / 0.884 | 0.847 / 0.809 / 0.841 | broadly neutral |
| EA | 6 | 0.917 / 0.232 / 0.844 | 0.243 / 0.291 / -0.688 | 0.796 / 0.740 / 0.673 | n=1 chemistry collapse; n=2 architecture collapse |
| EA | 7 | 0.902 / 0.987 / 0.965 | 0.650 / 0.783 / 0.788 | 0.795 / 0.803 / 0.820 | both coupling depths help |
| EA | 8 | 0.963 / 0.963 / 0.971 | 0.842 / 0.787 / 0.805 | 0.854 / 0.817 / 0.821 | architecture loss |
| IP | 0 | 0.928 / 0.872 / 0.981 | 0.828 / 0.923 / 0.877 | 0.827 / 0.863 / 0.831 | n=1 chemistry loss |
| IP | 1 | 0.981 / 0.981 / 0.962 | 0.737 / 0.786 / 0.775 | 0.826 / 0.810 / 0.818 | n=1 near baseline chemistry |
| IP | 2 | 0.769 / 0.915 / 0.871 | 0.211 / 0.564 / 0.672 | 0.757 / 0.775 / 0.765 | both coupling depths help |
| IP | 3 | 0.975 / 0.914 / 0.845 | 0.819 / 0.854 / 0.733 | 0.822 / 0.824 / 0.809 | coupling chemistry loss |
| IP | 4 | 0.944 / 0.952 / 0.927 | 0.581 / 0.560 / 0.596 | 0.764 / 0.738 / 0.801 | mixed |
| IP | 5 | 0.770 / 0.240 / 0.494 | 0.896 / 0.565 / 0.767 | 0.917 / 0.843 / 0.880 | n=1 worsens bithiophene collapse |
| IP | 6 | 0.975 / 0.992 / 0.993 | 0.880 / 0.859 / 0.802 | 0.846 / 0.809 / 0.805 | chemistry gain, architecture loss |
| IP | 7 | 0.969 / 0.974 / 0.980 | 0.773 / 0.762 / 0.831 | 0.731 / 0.704 / 0.748 | mixed |
| IP | 8 | 0.981 / 0.968 / 0.990 | 0.880 / 0.602 / 0.839 | 0.873 / 0.862 / 0.845 | n=1 architecture loss |

**Paired per-fold sign tests vs baseline (uncorrected):** n=1 EA group-mean 6W/3L p = 0.51; n=1 IP ΔR²
4W/5L p = 1.0; n=2 EA group-mean 7W/2L p = 0.18; n=2 EA ΔR² 4W/5L p = 1.0. **Nothing is significant.**

**Revised decision.** The n=1 vs n=2 vs baseline pattern is non-monotonic in coupling depth and
concentrated on folds where the baseline is already unstable (EA 6, IP 5, IP 2). At one seed, with the
noise floor unmeasured, this is **not distinguishable from run-to-run variation**. The previous verdict
("n=1 dropped") is withdrawn as under-determined. n=2's EA fold-1 rescue (0.575 → 0.925) is the one
effect large enough to be plausibly real.

### 5.3 Q2 — octamer (explicit sequence) — ⚠️ REAL BUT MIS-ATTRIBUTED

**Provenance (audited, `_octamer_provenance_check.md`):** identical held-out sets, `y_true` hashes and
row identifiers across all models in all 18 cells; identical held-out monomer SMILES; no row
expansion (ratio 1.000, max identifier multiplicity 1); `n_train` 33,418 / `n_val` 4,774 /
`prediction_scale` matched to baseline in every cell. **Clean.**

**The K=16 averaging confound is ruled out.** The featurizer samples K candidates only for uniform
transition weights; `block` and `alternating` are deterministic K=1. Those rows improve as much as
`random` rows (EA fold 1 bias: baseline −0.204/−0.224/−0.205 → octamer −0.002/−0.001/+0.007).

| Target | Model | group-mean R² | ΔR² | ordering | overall R² | MAE (eV) |
| --- | --- | --- | --- | --- | --- | --- |
| EA | Baseline | 0.925 (0.900) | 0.790 (0.686) | 0.799 (0.781) | 0.924 (0.893) | 0.094 (0.104) |
| EA | Octamer | 0.989 (0.973) | 0.789 (0.684) | 0.818 (0.795) | 0.986 (0.966) | 0.043 (0.049) |
| EA | wDMPNN | 0.965 (0.938) | 0.580 (0.511) | 0.753 (0.722) | 0.956 (0.928) | 0.069 (0.085) |
| IP | Baseline | 0.969 (0.921) | 0.819 (0.734) | 0.826 (0.818) | 0.963 (0.917) | 0.051 (0.072) |
| IP | Octamer | 0.970 (0.952) | 0.868 (0.798) | 0.827 (0.820) | 0.967 (0.947) | 0.054 (0.051) |
| IP | wDMPNN | 0.968 (0.864) | 0.460 (0.549) | 0.758 (0.758) | 0.957 (0.847) | 0.062 (0.078) |

| Target | Fold | Group-mean R²: baseline / octamer / wDMPNN | ΔR²: baseline / octamer / wDMPNN | Ordering: baseline / octamer / wDMPNN | Notable move |
| --- | --- | --- | --- | --- | --- |
| EA | 0 | 0.925 / 0.995 / 0.760 | 0.734 / 0.708 / 0.684 | 0.745 / 0.744 / 0.796 | chemistry gain; architecture neutral |
| EA | 1 | 0.575 / 0.989 / 0.945 | 0.790 / 0.834 / 0.436 | 0.812 / 0.862 / 0.788 | sulfone chemistry and architecture gain |
| EA | 2 | 0.922 / 0.995 / 0.980 | 0.845 / 0.858 / 0.485 | 0.806 / 0.838 / 0.702 | octamer gain |
| EA | 3 | 0.975 / 0.984 / 0.990 | 0.837 / 0.913 / 0.726 | 0.799 / 0.879 / 0.777 | architecture gain |
| EA | 4 | 0.951 / 0.897 / 0.946 | 0.347 / 0.139 / 0.145 | 0.578 / 0.572 / 0.512 | architecture/chemistry loss |
| EA | 5 | 0.969 / 0.991 / 0.969 | 0.882 / 0.847 / 0.580 | 0.847 / 0.834 / 0.734 | chemistry gain; slight ΔR² loss |
| EA | 6 | 0.917 / 0.925 / 0.894 | 0.243 / 0.280 / 0.339 | 0.796 / 0.818 / 0.645 | chemistry/ordering gain |
| EA | 7 | 0.902 / 0.989 / 0.965 | 0.650 / 0.789 / 0.580 | 0.795 / 0.788 / 0.753 | chemistry/ΔR² gain |
| EA | 8 | 0.963 / 0.994 / 0.994 | 0.842 / 0.785 / 0.627 | 0.854 / 0.818 / 0.793 | chemistry tie; ΔR² loss |
| IP | 0 | 0.928 / 0.927 / 0.270 | 0.828 / 0.921 / 0.713 | 0.827 / 0.868 / 0.816 | ΔR²/ordering gain |
| IP | 1 | 0.981 / 0.970 / 0.964 | 0.737 / 0.763 / 0.303 | 0.826 / 0.795 / 0.743 | small chemistry/order loss |
| IP | 2 | 0.769 / 0.994 / 0.977 | 0.211 / 0.517 / 0.424 | 0.757 / 0.657 / 0.717 | chemistry gain; ordering loss |
| IP | 3 | 0.975 / 0.933 / 0.968 | 0.819 / 0.885 / 0.825 | 0.822 / 0.875 / 0.870 | architecture gain; chemistry loss |
| IP | 4 | 0.944 / 0.910 / 0.712 | 0.581 / 0.519 / 0.445 | 0.764 / 0.809 / 0.758 | ordering gain; chemistry/ΔR² loss |
| IP | 5 | 0.770 / 0.876 / 0.946 | 0.896 / 0.941 / 0.460 | 0.917 / 0.922 / 0.829 | octamer chemistry/architecture gain |
| IP | 6 | 0.975 / 0.986 / 0.984 | 0.880 / 0.912 / 0.385 | 0.846 / 0.827 / 0.603 | ΔR² gain; slight ordering loss |
| IP | 7 | 0.969 / 0.988 / 0.983 | 0.773 / 0.859 / 0.610 | 0.731 / 0.757 / 0.660 | gain |
| IP | 8 | 0.981 / 0.984 / 0.977 | 0.880 / 0.868 / 0.776 | 0.873 / 0.873 / 0.828 | near neutral |

**Paired per-fold sign tests vs baseline (uncorrected, 9 folds):**

| target | metric | W/L | median paired Δ | p |
|---|---|---|---|---|
| EA | group-mean R² | **8/1** | **+0.031** | 0.039 |
| EA | overall R² | 8/1 | +0.031 | 0.039 |
| EA | MAE | **8/1** | **+0.051 eV** | 0.039 |
| EA | ΔR² | 5/4 | −0.001 | 1.00 |
| EA | ordering | **4/5** | — | 1.00 |
| IP | ΔR² | 7/2 | +0.045 | 0.18 |
| IP | group-mean R² | 5/4 | — | 1.00 |
| IP | MAE | 5/4 | — | 1.00 |

**Correction to the previous version of this report:** it claimed the octamer "has better ordering" on
EA. Paired folds are 4 wins, 5 losses. The median moved only because medians of independently ranked
columns move. It also reported the EA chemistry gain as 0.925 → 0.989; the paired median gain is
**+0.031**, and seven of the eight wins are +0.008 to +0.088 — the headline is carried by fold 1 (+0.414).

**Pooled and across-fold placement** (`_groupmean_metric_floor.md`) — these are not vulnerable to the
within-fold constant-A degeneracy of §2.3:

| model | target | pooled group-mean R² | placement R² | placement slope | fold-bias SD |
|---|---|---|---|---|---|
| hpg_hier | EA | 0.949 | 0.923 | 0.913 | 0.103 |
| octamer | EA | **0.989** | **0.989** | 1.031 | **0.037** |
| wdmpnn | EA | 0.967 | 0.946 | 0.994 | 0.088 |
| hpg_hier | IP | 0.952 | 0.939 | 0.806 | 0.083 |
| octamer | IP | **0.984** | **0.980** | 1.051 | **0.041** |
| wdmpnn | IP | 0.954 | 0.936 | 1.003 | 0.071 |

Slopes near 1 rule out shrinkage toward the global mean as the explanation — the *baseline* is the
model that shrinks (IP slope 0.806). Paired |fold bias|: **EA 8W/1L (p = 0.039); IP 5W/4L (not
supported)**. The IP pooled gain is driven by two folds where the baseline blows up. Eight of nine
octamer IP fold biases are positive (~+0.04 eV) — a small global over-prediction.

**Attribution is untested.** The octamer changes three things at once versus baseline Stage-2:
explicit 8-instance sequence, positional embeddings, and **attention pooling replacing the
stoichiometry-weighted readout**. The gain is entirely on chemistry placement while the architecture
axis is flat — the signature of a readout effect, not a sequence-encoding effect. The decisive
ablation (2-node Stage-2 + attention pooling) is specified in
`windsurf_prompt_readout_ablation_spec.md` and has not been run.

**Consequence for §5.4 below:** if the fold-1 gap is fixable by a Stage-2-only change, the
"missing cross-junction conjugation in Stage-1" diagnosis cannot be the whole story.

### 5.4 The fold-1 chemistry weakness — ❌ WITHDRAWN, it is run-to-run variance

**The baseline solves fold 1 unaided in roughly one run out of three.** Three repeats of the plain
baseline at seed 42 gave group-mean R² 0.790 / 0.450 / **0.978** (§0.1). The canonical 0.575 sits
inside that spread, and the best repeat (0.978) is comparable to the octamer (0.989) and better than
junction n=2 (0.925).

There is no established systematic chemistry gap on this fold. The "isolated Stage-1 encoding misses
cross-junction conjugation" diagnosis, the junction-coupling programme built to fix it, and the
octamer's largest single EA contribution (fold 1, +0.414) all rest on a single unstable draw. The
table below is retained only as a record of what those single runs produced.

**EA fold 1 (dibenzothiophene sulfone):**
| Model | group-mean R² | MAE (eV) | bias (eV) |
|---|---|---|---|
| Baseline HPG-hier | 0.575 | 0.213 | −0.213 |
| A-blind null | 0.487 | 0.217 | −0.212 |
| HPG-hier + junction (n=2) | 0.925 | 0.085 | −0.081 |
| **HPG-hier + octamer** | **0.989** | **0.031** | **+0.002** |
| wDMPNN | 0.945 | 0.076 | — |

Two observations that were not available when this was diagnosed as a Stage-1 conjugation deficit:

1. **The baseline barely beats a predictor that ignores the held-out monomer entirely** (0.575 vs 0.487).
2. **A Stage-2-only change closes the gap more completely than the Stage-1 fix does**, with Stage-1
   untouched. Missing encoder physics cannot be restored by the readout.

Either the gap is a readout/extrapolation-calibration artefact rather than a conjugation deficit, or
the metric is dominated by the constant offset described in §2.3. **The mechanism claim is withdrawn
pending the readout ablation.**

## 6. Variant status

| Variant | Status | Result |
|---|---|---|
| HPG-hier (baseline) | ✅ reference | best architecture recovery OOD; chemistry claims limited by §2.3 |
| +Q1 wedge | ❌ dropped | OOD architecture collapse (IP ΔR² 0.820 → 0.558) |
| +Q2 octamer | ⚠️ real, mis-attributed | chemistry-placement gain, provenance clean; **zero architecture gain**; readout vs sequence untested |
| +junction (n=2) | ⚠️ mechanism-only | EA fold-1 rescue is the one large effect; IP fold-5 and EA fold-6 costs |
| +junction (n=1) | ⚠️ under-determined | previously "dropped"; not distinguishable from noise at one seed |
| HPG-hier + wDMPNN ensemble | ⚠️ seed-42 only | EA-MAE margin 0.003 is inside plausible noise |

## 7. Open questions / next runs

**Blocking everything:**

1. **Code drift vs training noise** (`windsurf_prompt_code_drift_vs_noise.md`). One GPU run on the
   pre-20-July commit decides it; 6 runs measure the noise floor. Until the noise floor is known, no
   variant difference in this report can be called a finding.
2. **Regenerate the 20-July A-split baselines under current code** if drift is confirmed (36 runs).
   Required regardless before seeds 43/44 are averaged in — you cannot average across code versions.

**Then:**

3. **B-heldout runs, 144 cells** (`windsurf_spec_b_heldout_split.md`) — the only split where EA
   chemistry is measurable, and the first test of whether the architecture-recovery advantage holds
   when the unseen monomer is a B. Includes the performance-versus-novelty curve (§2.5).
4. **Readout ablation** (`windsurf_prompt_readout_ablation_spec.md`) — attribution of the Q2 gain.
5. **Seeds 43/44** for everything retained.

**Not recommended:** further Phase-1 variants. Three interventions have produced no architecture-axis
movement on a split with 7 training donor monomers, ~1% architecture variance, and an unmeasured noise
floor. New variants cannot be evaluated until items 1–2 are resolved.

## 8. Conventions and caveats

- Headline differences are **medians of paired per-fold differences**; the difference of separately
  ranked medians is not reported as a result.
- Paired comparisons use exact two-sided sign tests across the 9 folds; minimum attainable p = 0.0039;
  **no p-value in this report survives correction for the ~20-member comparison family.**
- LOMO aggregates use medians; means are given parenthetically and are distorted by pathological folds.
- Group-mean R² is reported against its null floor wherever the floor is known (§2.3).
- Single seed (42) throughout. No non-finite predictions were found in the 90 seed-42 A-split NPZs.
- Audit trail: `_phase1_metrics_scratch.md`, `_octamer_provenance_check.md`,
  `_groupmean_metric_floor.md`, `_dataset_design_audit.md`, `_a_heldout_bitwise_reproduction.md`.
