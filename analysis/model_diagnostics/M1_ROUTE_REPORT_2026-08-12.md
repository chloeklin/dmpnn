# M1 monomer-identity plumbing route report

Date: 2026-08-12  
Scope: route chosen for adding an explicit per-atom monomer index to wD-MPNN.

## What exists

- `PolymerMolGraph` carries `V`, `E`, `atom_weights`, `edge_weights`, `edge_index`,
  `rev_edge_index`, `degree_of_polym` — but no per-atom monomer ownership field.
- `atom_weights` is built from the RDKit `w_frag` atom property (mole fraction of the
  fragment each atom came from). It is identical for every atom of monomer A and
  identical for every atom of monomer B, but at `fracA = 0.5` the two monomers have the
  same weight, so it cannot disambiguate them.
- No existing fragment/ownership field carries a robust 0/1 monomer label through the
  featurizer and batching path.

## Chosen route

Add a genuine `monomer_index` array and plumb it through three layers.

### 1. Featurizer — `chemprop/featurizers/molgraph/molecule.py`

After `remove_wildcard_atoms(rwmol)` the R groups have been stripped, leaving the two
monomer cores as the connected components of the remaining molecule. Compute

```python
frags = Chem.GetMolFrags(rwmol, asMols=False, sanitizeFrags=False)
```

and assert `len(frags) == 2` (every polymer has both monomers). Build a lookup
`atom_idx -> 0/1` from `frags`. When iterating `rwmol.GetAtoms()` with the same
filter used to build `V` (`core is True`), emit a matching `monomer_index` array.

Why this is robust:

- It uses connectivity, not stoichiometry, so it works at `fracA = 0.5`.
- It follows the same atom order as `V` because it is computed on the same `rwmol`
  after wildcard removal.
- Ring wildcards that survive `remove_wildcard_atoms` as simple `[*]` atoms sit on a
  single monomer fragment and do not bridge the two monomers; the inter-monomer
  bonds are added later to a temporary copy (`cm`) and are not present in `rwmol`.

### 2. Graph dataclass — `chemprop/data/molgraph.py`

Add `monomer_index: np.ndarray` to `PolymerMolGraph`, shape `[num_atoms]`, dtype int,
values 0 or 1.

### 3. Batch dataclass — `chemprop/data/collate.py`

In `BatchPolymerMolGraph.__post_init__`, concatenate per-graph `monomer_index` arrays
into a `Tensor` of shape `[num_atoms_in_batch]`. The existing `batch` tensor already
gives the graph index, so a global `monomer_index` of 0/1 is sufficient; no
offsetting of the 0/1 values themselves is needed, only correct concatenation so the
entries align with `bmg.batch` and `bmg.atom_weights`.

### 4. Readout — `chemprop/nn/agg.py`

Add `MonomerLevelStoichAggregation` (registered as `monomer_level_stoich`). For each
polymer `p` and monomer `m in {0, 1}`:

```text
h_m[p] = mean(H[batch == p & monomer_index == m], dim=0)
```

Then read the stoichiometric weight `f_m[p]` from `atom_weights` for any atom of that
monomer (asserted constant within the monomer) and compute

```text
g[p] = f_0[p] * h_0[p] + f_1[p] * h_1[p]
```

If a monomer has zero atoms, the mean is defined as a zero vector and an assertion
fires.

### 5. Model dispatch — `chemprop/models/model.py`

`MPNN.fingerprint` currently passes the full `bmg` only when the aggregator is a
`WeightedMeanAggregation`. The check is expanded to include
`MonomerLevelStoichAggregation` so the new aggregator receives `bmg.monomer_index`,
`bmg.atom_weights`, and `bmg.batch`.

## What was not used

- `atom_weights` / `w_frag` — retained for the stoichiometric weights `f_A` and `f_B`,
  but not used to infer monomer identity.
- The temporary `OrigMol` boolean set on `cm` during inter-monomer bond construction —
  this lives on the duplicated scratch molecule, not on the returned graph, and would
  require a more invasive refactor to persist.

## Files touched

1. `chemprop/data/molgraph.py`
2. `chemprop/featurizers/molgraph/molecule.py`
3. `chemprop/data/collate.py`
4. `chemprop/nn/agg.py`
5. `chemprop/models/model.py`
6. `scripts/python/run_wdmpnn_generalization.py`
7. `evaluation/naming.py`
