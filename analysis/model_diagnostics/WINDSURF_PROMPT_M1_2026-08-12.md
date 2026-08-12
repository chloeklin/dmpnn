# Windsurf task — implement M1 (monomer-level readout on the flat graph) + Gadi pilot scripts

Rung 1 of `ARCHITECTURE_LADDER_2026-08-12.md`. **Implementation and PBS generation only —
do not train, do not submit, do not log in to Gadi.**

---

## 0. Pilot definition has changed — read this first

The arms C/D pilot used **2 folds × 1 target × 3 seeds**. The two folds pointed in opposite
directions and the pilot could not be read. That design is retired.

**A pilot is now: all 9 folds × one target (EA) × three seeds = 27 runs.**

This is protocol-compliant, supports a proper 9-fold paired sign test (minimum attainable
two-sided p = 0.0039), and costs roughly half a full arm. Never pilot on a subset of folds
again — fold-to-fold variation on this benchmark is larger than most effects being tested.

---

## 1. What M1 is

M1 sits between wD-MPNN and HPG-hier and isolates **one architectural choice: whether an
explicit monomer-level representation exists at all.**

| | Polymer vector formed at | Inter-monomer info at | Architecture via | Readout |
|---|---|---|---|---|
| wD-MPNN | atoms → polymer directly | atom level | edge weights | weighted mean over atoms |
| **M1** | **atoms → monomer vectors → polymer** | atom level | edge weights | stoichiometry-weighted over 2 monomers |
| HPG-hier | monomer vectors | **monomer level** | 17-d edge features | stoichiometry-weighted |

**Everything about wD-MPNN stays identical** — the flat graph containing both monomers, the
stochastic inter-monomer edges, `WeightedBondMessagePassing`, the edge weights, the features.
The *only* change is the readout: instead of pooling all atoms straight to one polymer vector,
pool atoms **per monomer** first, then combine the two monomer vectors as `f_A·h_A + f_B·h_B`.

Current wD-MPNN construction, `scripts/python/run_wdmpnn_generalization.py:289`:

```python
mp  = nn.WeightedBondMessagePassing()
agg = nn.WeightedMeanAggregation()      # <- this is the only thing M1 replaces
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)
```

---

## 2. The obstacle — check this before writing anything

I described this as "an aggregation swap" in the ladder document. **That was optimistic and I
was wrong.** Verify the following before you start:

`PolymerMolGraph` (`chemprop/data/molgraph.py`) carries `atom_weights`, `edge_weights`,
`edge_index`, `rev_edge_index`, `degree_of_polym` — **but no per-atom monomer index.**
`atom_weights` is populated from the RDKit property `w_frag`
(`chemprop/featurizers/molgraph/molecule.py:434`), i.e. the monomer's mole fraction.

**Do not try to recover monomer identity from `atom_weights`.** It works only when
fracA ≠ fracB, and fails exactly at **fracA = 0.5** — which is where the three-architecture
groups live and where ΔR² is most informative. A model that silently mis-pools at fracA = 0.5
would be worse than no model.

So M1 needs a genuine **`monomer_index` array**: shape `[num_atoms]`, values 0 or 1, plumbed
through:

1. the featurizer that builds `PolymerMolGraph` — the RDKit fragment each core atom came from
   is known at construction (see the `w_frag` / `core` property handling around lines 399–440
   and the fragment duplication at 513–540)
2. the `PolymerMolGraph` dataclass
3. whatever batches these into `BatchPolymerMolGraph`, with the usual per-graph offsetting so
   monomer indices do not collide across a batch
4. a new aggregation that consumes it

**First deliverable is a short report on the cleanest route**, before implementing. If there is
an existing fragment or ownership field I have missed, say so and use it.

---

## 3. The new aggregation

Add `MonomerLevelStoichAggregation` (name negotiable) alongside `WeightedMeanAggregation` in
`chemprop/nn/agg.py`:

```
for each polymer p:
    h_A = mean of atom embeddings where monomer_index == 0
    h_B = mean of atom embeddings where monomer_index == 1
    g_p = f_A * h_A + f_B * h_B
```

`f_A`, `f_B` are the mole fractions already available per atom in `atom_weights` — take the
value shared by that monomer's atoms, and **assert** it is constant within each monomer.

Two design points to get right:

- **Use a plain mean within each monomer, not a weighted mean.** The stoichiometric weighting
  is applied once, at the monomer level. Applying it twice would double-count composition and
  make M1 incomparable to HPG-hier, which weights once.
- **A monomer with zero atoms must not produce NaN.** It should not occur — every polymer has
  both monomers — but assert it rather than discovering it in fold 7 of a live arm.

Register a model token `wdmpnn_monomer_readout` (or similar) in `evaluation/naming.py`, and
wire it into `run_wdmpnn_generalization.py` behind an explicit flag so the default wD-MPNN path
is untouched.

---

## 4. Verification before any job is generated

1. **Parameter count.** M1 must differ from wD-MPNN by however many parameters the new
   aggregation introduces — if it introduces none, the counts must be **identical**. Report
   both counts. A difference you cannot explain means something else changed.
2. **Monomer index correctness.** On a fracA = 0.5 polymer, confirm the two monomers receive
   different indices and that the atom counts per monomer match the SMILES. This is the case
   `atom_weights` cannot distinguish, so test it explicitly.
3. **Gradient flows** to both monomer branches — non-zero gradient on the aggregation output
   with respect to atoms of each monomer.
4. **Sanity forward pass** on one real batch, comparing M1 and wD-MPNN outputs. They should
   differ; if they are identical, the aggregation is not being used.

---

## 5. Pre-registration — write before generating jobs

Create `analysis/model_diagnostics/PREREG_M1_2026-08-12.md`, following
`PREREG_octamer_posemb_2026-08-05.md`. It must state, before any result:

**Question.** Does having an explicit monomer-level representation, with inter-monomer
information still flowing at the atom level, account for the wD-MPNN → HPG-hier gap?

**Primary quantity:** ΔR² on `all` rows, A split, per fold.

**Reference points:** wD-MPNN at its published configuration, and HPG-hier. Both exist.

**Pre-registered readings:**

| Outcome | Reading |
|---|---|
| M1 ≈ HPG-hier | the monomer-level representation did it; where inter-monomer information flows is secondary |
| M1 ≈ wD-MPNN | having the representation is not enough; the level at which information flows is what matters |
| M1 midway | both contribute; report the split, do not attribute to one |

**Materiality threshold:** derive it the same way as the posemb pre-registration — median
per-cell across-seed SD of ΔR² for the relevant comparator. State the value and its source
before running.

**Also state:** that M1 runs at *our* configuration (batch 64, 100 epochs, patience 15) while
the wD-MPNN reference runs at its published configuration, so the M1 ↔ wD-MPNN comparison
carries a configuration difference. The bridge run that removes it is §6.

---

## 6. The configuration bridge

The published-config baseline must never be modified. So M1 is run **twice**:

| Run | Config | Purpose |
|---|---|---|
| M1 (ours) | batch 64, 100 epochs, patience 15 | comparable to HPG-hier and the rest of the ladder |
| M1 (published) | batch 50, 30 epochs, patience 30 | comparable to wD-MPNN |

`M1(published) → M1(ours)` then isolates configuration with architecture held fixed, and every
other rung comparison is architecture-only.

**Generate both, but mark the published-config set as the second batch** — it is only needed
once M1(ours) looks sane.

---

## 7. PBS generators

Follow `scripts/shell/generate_octamer_posemb_r1.sh` exactly, including its header stating it
generates only and does not submit.

Write `scripts/shell/generate_m1_pilot.sh` producing:

- **Pilot:** EA only, folds 0–8, seeds 42/43/44, our config = **27 jobs**
- **Bridge:** EA only, folds 0–8, seeds 42/43/44, published config = **27 jobs**, in a separate
  manifest so they can be submitted independently

PBS header: `gpuvolta`, `ncpus=12`, `ngpus=1`, `mem=100GB`, `jobfs=100GB`,
`walltime=06:00:00`, project `hm62`, storage `scratch/um09+gdata/dk92`, modules
`python3/3.12.1 cuda/12.0.0`, venv `/home/659/hl4138/dmpnn-venv/bin/activate`,
`PROJECT_DIR=/scratch/um09/hl4138/dmpnn`.

Pre-flight: refuse to emit a job whose output exists; verify the HPG-hier and wD-MPNN
comparators are present for every fold.

Output tokens: `__m1` and `__m1pub`. Prediction directory `predictions/m1/`.

**Do not generate the IP half or the B split.** Those come after the pilot is read.

---

## 8. Protocol

Three seeds averaged at the **prediction** level, metric computed once. `y_pred`, never
`y_pred_final`. `--frozen_protocol` — the CUDA guard must be in force. Do not modify
`evaluation/metrics.py`, any dated document, or any existing prediction.

---

## 9. Report back

1. The §2 route report — how monomer identity is plumbed, or what existing field was used.
2. The §4 verification numbers, especially the fracA = 0.5 monomer-index test.
3. The pre-registration file.
4. Job counts for both manifests, and confirmation nothing was submitted or trained.
5. `py_compile` clean; tests; every file written.
