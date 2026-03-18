# polymer_input — Canonical Polymer Specification & Featurization

A model-agnostic abstraction layer that sits between raw CSV data and Chemprop's graph featurizers. It provides a canonical `PolymerSpec` schema, validation, serialization, parsing, and concrete featurizer implementations for DMPNN, PPG, and wDMPNN.

## Package Structure

```
polymer_input/
  __init__.py          # Public API re-exports
  schema.py            # PolymerSpec, FragmentSpec, PolymerConnection dataclasses
  validation.py        # RDKit-based SMILES/topology validation
  serialization.py     # to_dict / from_dict, save_jsonl / load_jsonl
  parsing.py           # PolymerParser with configurable SchemaMapping
  graph_data.py        # HPGGraphData, HPGEdgeType, AtomNode, FragmentNode
  mol_utils.py         # Fragment combination, wildcard removal, scalar extraction
  test_polymer_input.py
  featurizers/
    base.py            # BasePolymerFeaturizer ABC
    hpg.py             # HPGFeaturizer (hierarchical polymer graph)
    dmpnn.py           # DMPNNFeaturizer → chemprop SimpleMoleculeMolGraphFeaturizer
    ppg.py             # PPGFeaturizer  → chemprop PPGMolGraphFeaturizer
    wdmpnn.py          # WDMPNNFeaturizer → chemprop PolymerMolGraphFeaturizer
  sample_specs/
    example_specs.py   # Pre-built PolymerSpec examples for testing
```

## Relationship to Chemprop

`polymer_input` is **not** a fork — it is a thin adapter that converts structured polymer data into inputs accepted by existing Chemprop utilities. Where Chemprop already provides the functionality, `polymer_input` delegates:

| polymer_input function | Delegates to (chemprop) | Purpose |
|---|---|---|
| `remove_wildcards_and_cap()` | `chemprop.featurizers.molgraph.molecule.remove_wildcard_atoms` | Strip wildcard atoms from combined mol |
| `build_wdmpnn_mol()` | `chemprop.utils.utils.make_polymer_mol` | Combine fragments + assign `w_frag` atom props |
| `DMPNNFeaturizer.featurize()` | `SimpleMoleculeMolGraphFeaturizer.__call__` | Standard molecular graph |
| `PPGFeaturizer.featurize()` | `PPGMolGraphFeaturizer.__call__` | Periodic polymer graph (bond-length bins) |
| `WDMPNNFeaturizer.featurize()` | `PolymerMolGraphFeaturizer.__call__` | Weighted polymer graph (stochastic edges) |

**What polymer_input adds (not in chemprop):**

- `PolymerSpec` canonical schema (fragments + connections + scalars + target)
- `PolymerParser` for loading CSV/dict/JSONL into `PolymerSpec` objects
- `parse_fragment_mol()` — wildcard-safe SMILES parsing (extends `make_mol` with partial sanitization that skips kekulization)
- `combine_fragments()` — lightweight fragment combiner for DMPNN/PPG (no weight assignment needed)
- `extract_scalar_features()` / `collect_scalar_keys()` — scalar-to-`x_d` mapping
- `HPGFeaturizer` — hierarchical polymer graph scaffold (atom + fragment level)
- Validation, serialization, schema mapping

## Full Pipeline

### 1. Data Loading → PolymerSpec

```python
from polymer_input import PolymerParser, SchemaMapping

# Define how CSV columns map to PolymerSpec fields
mapping = SchemaMapping(
    polymer_id_col="id",
    smiles_cols=["smiles_A", "smiles_B"],
    fragment_name_cols=["A", "B"],
    connection_cols=[],          # auto-inferred for linear polymers
    scalar_cols=["mw", "temperature", "fracA", "fracB"],
    target_col="property_value",
)

parser = PolymerParser(mapping)
specs = parser.parse_csv("data/my_polymers.csv")
```

### 2. Validation

```python
from polymer_input import validate_polymer_spec

for spec in specs:
    errors = validate_polymer_spec(spec)
    if errors:
        print(f"Invalid: {spec.polymer_id}: {errors}")
```

### 3. Featurization → Chemprop MolGraph

```python
from polymer_input.featurizers.dmpnn import DMPNNFeaturizer
from polymer_input.featurizers.ppg import PPGFeaturizer
from polymer_input.featurizers.wdmpnn import WDMPNNFeaturizer

# Choose based on model:
featurizer = DMPNNFeaturizer()     # Standard D-MPNN
# featurizer = PPGFeaturizer()     # Periodic Polymer Graph
# featurizer = WDMPNNFeaturizer()  # Weighted D-MPNN

molgraphs = [featurizer.featurize(spec) for spec in specs]
```

### 4. Scalar Features → x_d

```python
from polymer_input import extract_scalar_features, collect_scalar_keys

scalar_keys = collect_scalar_keys(specs)  # consistent ordering
x_d_list = [extract_scalar_features(spec, scalar_keys) for spec in specs]
```

### 5. Chemprop Datapoints & Training

```python
from chemprop import data

# Create datapoints from SMILES + targets + descriptors
all_data = [
    data.MoleculeDatapoint.from_smi(
        ".".join(spec.fragment_smiles),
        [spec.target],
        x_d=x_d
    )
    for spec, x_d in zip(specs, x_d_list)
]

# Standard chemprop training flow
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
dataset = data.MoleculeDataset(all_data, featurizer)
# ... split, normalize, train
```

## Featurizer Details

### DMPNNFeaturizer
- **Input**: `PolymerSpec` with any number of fragments
- **Pipeline**: `combine_fragments()` → `remove_wildcards_and_cap()` → `SimpleMoleculeMolGraphFeaturizer`
- **Output**: `MolGraph(V=(n, 72), E=(m, 14))`
- **Use case**: Treat polymer as a single molecule (wildcards removed)

### PPGFeaturizer
- **Input**: `PolymerSpec` with wildcard attachment points
- **Pipeline**: `combine_fragments()` (keep wildcards) → `PPGMolGraphFeaturizer`
- **Output**: `MolGraph(V=(n, 72), E=(m, 24))` — 14 bond features + 10 bond-length bins
- **Use case**: Periodic polymer graph with 3D-coordinate-based periodic bonds

### WDMPNNFeaturizer
- **Input**: `PolymerSpec` with connections and optional composition ratios
- **Pipeline**: `build_wdmpnn_mol()` (via `make_polymer_mol`) → `PolymerMolGraphFeaturizer`
- **Output**: `PolymerMolGraph` with per-atom weights and inter-fragment stochastic edges
- **Use case**: Weighted message passing for copolymers with composition fractions

### HPGFeaturizer
- **Input**: `PolymerSpec`
- **Output**: `HPGGraphData` (hierarchical: atom-level + fragment-level nodes)
- **Use case**: Graph-of-graphs representation (experimental)

## Integration with Training Scripts

The training scripts (`train_graph.py`, `train_identity_baseline.py`) can optionally use `polymer_input` for structured data loading and validation. The existing SMILES-based flow remains the default.

### train_graph.py

For homopolymers and standard models (DMPNN, PPG, wDMPNN):
1. CSV → `load_and_preprocess_data()` → DataFrame with SMILES + targets
2. `process_data()` → SMILES array + descriptor array
3. Featurizer selection: `SimpleMoleculeMolGraphFeaturizer` / `PPGMolGraphFeaturizer` / `PolymerMolGraphFeaturizer`
4. `create_all_data()` → `MoleculeDatapoint` or `PolymerDatapoint` list
5. Split → preprocess descriptors → normalize → train

**polymer_input integration** (optional): Use `PolymerParser` for step 1 to get validated `PolymerSpec` objects, then `extract_scalar_features()` for step 2.

### train_identity_baseline.py

For copolymer identity-embedding baselines:
1. CSV → DataFrame with smilesA/B + fracA/B
2. Build monomer vocabulary → integer IDs
3. MLP with shared embedding table
4. No graph featurization (categorical IDs only)

**polymer_input integration**: `PolymerParser` can validate the copolymer schema, and `PolymerSpec.scalars` can provide the composition fractions.

## Running Tests

```bash
python -m polymer_input.test_polymer_input
```

Covers: PolymerSpec creation, validation, serialization, HPG featurization, DMPNN/PPG/wDMPNN end-to-end, and scalar extraction.

## Key Design Decisions

- **Model-agnostic schema**: `PolymerSpec` stores fragments + connections + scalars; featurizer-specific logic lives in the featurizer classes
- **No unrolling**: One `FragmentSpec` per abstract fragment; DP/MW stay as scalars
- **Featurizer-specific edge types**: `HPGEdgeType`, `PPGEdgeType`, `DMPNNEdgeType`, `WDMPNNEdgeType`
- **Fragment weights from scalars**: Keys like `ratio_A`, `frac_B` auto-extracted; defaults to uniform
- **wDMPNN wildcards**: Auto-numbered `[*:1]`, `[*:2]` if fragments use plain `[*]`
