"""Usage example and smoke tests for the polymer_input package.

Run with:
    python -m polymer_input.test_polymer_input

Or directly:
    python polymer_input/test_polymer_input.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """End-to-end usage example demonstrating the full pipeline."""

    print("=" * 70)
    print("  polymer_input  —  Usage Example & Smoke Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Create a PolymerSpec from code
    # ------------------------------------------------------------------
    print("\n--- 1. Create PolymerSpec objects ---")

    from polymer_input.schema import FragmentSpec, PolymerConnection, PolymerSpec

    spec = PolymerSpec(
        polymer_id="peo_001",
        fragments=[FragmentSpec(smiles="[*]OCC[*]", name="A")],
        connections=[],
        topology_type="homopolymer",
        scalars={"mw": 50000.0, "temperature": 298.15},
        target=1.23,
    )
    print(f"  Created: {spec.polymer_id} ({spec.topology_type})")
    print(f"  Fragments: {spec.fragment_smiles}")
    print(f"  Scalars: {spec.scalars}")

    # ------------------------------------------------------------------
    # 2. Validate
    # ------------------------------------------------------------------
    print("\n--- 2. Validate ---")

    errors = spec.validate()
    if errors:
        print(f"  ERRORS: {errors}")
    else:
        print("  Valid!")

    # Test validation catches bad input
    bad_spec = PolymerSpec(
        polymer_id="bad",
        fragments=[FragmentSpec(smiles="not_a_smiles")],
        connections=[PolymerConnection(src=0, dst=5)],  # out of bounds
    )
    bad_errors = bad_spec.validate()
    print(f"  Bad spec errors ({len(bad_errors)}):")
    for e in bad_errors:
        print(f"    - {e}")

    # ------------------------------------------------------------------
    # 3. Serialize / deserialize
    # ------------------------------------------------------------------
    print("\n--- 3. Serialize / Deserialize ---")

    d = spec.to_dict()
    print(f"  to_dict keys: {list(d.keys())}")

    json_str = json.dumps(d)
    print(f"  JSON length: {len(json_str)} chars")

    spec_round = PolymerSpec.from_dict(json.loads(json_str))
    assert spec_round.polymer_id == spec.polymer_id
    assert spec_round.fragment_smiles == spec.fragment_smiles
    assert spec_round.target == spec.target
    print("  Round-trip OK!")

    # ------------------------------------------------------------------
    # 4. Parse from dict row (simulating CSV row)
    # ------------------------------------------------------------------
    print("\n--- 4. Parse from dict row ---")

    from polymer_input.parsing import PolymerParser, SchemaMapping

    # Simulate a CSV row where JSON fields are stored as strings
    csv_row = {
        "sample_id": "block_csv_001",
        "fragment_smiles": json.dumps([
            {"smiles": "[*]CC[*]", "name": "A"},
            {"smiles": "[*]OCC[*]", "name": "B"},
        ]),
        "connections": json.dumps([
            {"src": 0, "dst": 1, "edge_type": "polymer_link"},
        ]),
        "topology": "block",
        "scalars": json.dumps({"mw": 100000.0}),
        "y": "0.85",
    }

    mapping = SchemaMapping(
        polymer_id="sample_id",
        fragments="fragment_smiles",
        connections="connections",
        topology_type="topology",
        scalars="scalars",
        target="y",
    )
    parser = PolymerParser(mapping)
    parsed = parser.parse_row(csv_row)
    print(f"  Parsed: {parsed.polymer_id}, {parsed.topology_type}")
    print(f"  Fragments: {parsed.fragment_smiles}")
    print(f"  Target: {parsed.target}")

    # ------------------------------------------------------------------
    # 5. JSONL round-trip
    # ------------------------------------------------------------------
    print("\n--- 5. JSONL round-trip ---")

    from polymer_input.serialization import save_jsonl, load_jsonl
    from polymer_input.sample_specs import ALL_EXAMPLES

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        tmp_path = Path(f.name)

    save_jsonl(ALL_EXAMPLES, tmp_path)
    loaded = load_jsonl(tmp_path)
    assert len(loaded) == len(ALL_EXAMPLES)
    for orig, reloaded in zip(ALL_EXAMPLES, loaded):
        assert orig.polymer_id == reloaded.polymer_id
    print(f"  Saved {len(ALL_EXAMPLES)} specs -> loaded {len(loaded)} specs. OK!")
    tmp_path.unlink()

    # ------------------------------------------------------------------
    # 6. HPG Featurizer
    # ------------------------------------------------------------------
    print("\n--- 6. HPG Featurizer ---")

    from polymer_input.featurizers.hpg import HPGFeaturizer

    hpg = HPGFeaturizer()

    for ex in ALL_EXAMPLES:
        graph = hpg.featurize(ex)
        print(f"  {graph.summary()}")

    # Detailed inspection of block copolymer graph
    from polymer_input.sample_specs import BLOCK_COPOLYMER

    graph = hpg.featurize(BLOCK_COPOLYMER)
    print(f"\n  Detailed block copolymer graph:")
    print(f"    Atom nodes: {graph.n_atoms}")
    for node in graph.atom_nodes:
        attach = " [ATTACH]" if node.is_attachment else ""
        print(f"      atom {node.global_idx}: {node.symbol} "
              f"(frag={node.fragment_idx}, local={node.local_idx}){attach}")

    print(f"    Fragment nodes: {graph.n_fragments}")
    for fnode in graph.fragment_nodes:
        print(f"      frag {fnode.fragment_idx}: {fnode.name!r} "
              f"({fnode.smiles}) — {fnode.n_atoms} atoms")

    print(f"    Atom-bond edges: {graph.n_atom_bonds}")
    for edge in graph.atom_bond_edges:
        print(f"      {edge.src} -> {edge.dst}  type={edge.edge_type.name}")

    print(f"    Atom-fragment edges: {graph.n_attachment_edges}")
    for edge in graph.atom_fragment_edges:
        print(f"      atom {edge.atom_idx} <-> frag {edge.fragment_idx}  "
              f"type={edge.edge_type.name}")

    print(f"    Fragment-fragment edges: {graph.n_polymer_edges}")
    for edge in graph.fragment_fragment_edges:
        print(f"      frag {edge.src} <-> frag {edge.dst}  "
              f"type={edge.edge_type.name} ({edge.connection_label})")

    print(f"    Scalar features: {graph.scalar_features}")
    print(f"    Target: {graph.target}")

    # ------------------------------------------------------------------
    # 7. DMPNN Featurizer (end-to-end)
    # ------------------------------------------------------------------
    print("\n--- 7. DMPNN Featurizer ---")

    from polymer_input.featurizers.dmpnn import DMPNNFeaturizer

    dmpnn = DMPNNFeaturizer()

    for ex in ALL_EXAMPLES:
        mg = dmpnn.featurize(ex)
        print(f"  {ex.polymer_id}: V={mg.V.shape}, E={mg.E.shape}, "
              f"edges={mg.edge_index.shape}")

    # ------------------------------------------------------------------
    # 8. PPG Featurizer (end-to-end)
    # ------------------------------------------------------------------
    print("\n--- 8. PPG Featurizer ---")

    from polymer_input.featurizers.ppg import PPGFeaturizer

    ppg = PPGFeaturizer()

    for ex in ALL_EXAMPLES:
        mg = ppg.featurize(ex)
        print(f"  {ex.polymer_id}: V={mg.V.shape}, E={mg.E.shape}, "
              f"edges={mg.edge_index.shape}")

    # ------------------------------------------------------------------
    # 9. wDMPNN Featurizer (end-to-end, needs numbered wildcards)
    # ------------------------------------------------------------------
    print("\n--- 9. wDMPNN Featurizer ---")

    from polymer_input.featurizers.wdmpnn import WDMPNNFeaturizer

    wdmpnn = WDMPNNFeaturizer()

    # wDMPNN needs numbered wildcards for edge mapping
    wdmpnn_homo = PolymerSpec(
        polymer_id="wdmpnn_homo",
        fragments=[FragmentSpec(smiles="[*:1]OCC[*:2]", name="A")],
        connections=[],
        topology_type="homopolymer",
        scalars={"mw": 50000.0},
        target=1.23,
    )
    pmg = wdmpnn.featurize(wdmpnn_homo)
    print(f"  wdmpnn_homo: V={pmg.V.shape}, E={pmg.E.shape}, "
          f"atom_weights={pmg.atom_weights.shape}, "
          f"edge_weights={pmg.edge_weights.shape}, "
          f"DP={pmg.degree_of_polym:.2f}")

    wdmpnn_block = PolymerSpec(
        polymer_id="wdmpnn_block",
        fragments=[
            FragmentSpec(smiles="[*:1]CC[*:2]", name="A"),
            FragmentSpec(smiles="[*:3]OCC[*:4]", name="B"),
        ],
        connections=[PolymerConnection(src=0, dst=1)],
        topology_type="block",
        scalars={"ratio_A": 0.5, "ratio_B": 0.5},
        target=0.85,
    )
    pmg = wdmpnn.featurize(wdmpnn_block)
    print(f"  wdmpnn_block: V={pmg.V.shape}, E={pmg.E.shape}, "
          f"atom_weights={pmg.atom_weights.shape}, "
          f"edge_weights={pmg.edge_weights.shape}, "
          f"DP={pmg.degree_of_polym:.2f}")

    # ------------------------------------------------------------------
    # 10. Scalar feature extraction
    # ------------------------------------------------------------------
    print("\n--- 10. Scalar feature extraction ---")

    from polymer_input.mol_utils import extract_scalar_features, collect_scalar_keys

    # Collect all scalar keys across examples
    all_keys = collect_scalar_keys(ALL_EXAMPLES)
    print(f"  All scalar keys: {all_keys}")

    for ex in ALL_EXAMPLES:
        x_d = extract_scalar_features(ex, scalar_keys=all_keys)
        if x_d is not None:
            print(f"  {ex.polymer_id}: x_d shape={x_d.shape}, values={x_d}")
        else:
            print(f"  {ex.polymer_id}: x_d=None (no scalars)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  All checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
