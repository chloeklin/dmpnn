"""Smoke test for the HPG-GAT implementation in chemprop.

Tests the full pipeline:
  1. HPGMolGraphFeaturizer  — fragment SMILES → HPGMolGraph
  2. BatchHPGMolGraph       — batching multiple graphs
  3. HPGMessagePassing      — GAT forward pass
  4. HPGMPNN                — full model forward pass
"""

import numpy as np
import torch

from chemprop.data.hpg import HPGMolGraph, BatchHPGMolGraph
from chemprop.featurizers.molgraph.hpg import HPGMolGraphFeaturizer
from chemprop.nn.hpg import HPGGATLayer, HPGMessagePassing
from chemprop.models.hpg import HPGMPNN


def test_featurizer_homopolymer():
    """Single fragment (homopolymer): [*]OCC[*]"""
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]OCC[*]"])

    assert mg.n_fragments == 1
    # [*]OCC[*] → wildcards removed → 3 real atoms (O, C, C)
    assert mg.n_atoms == 3
    print(f"  homopolymer: n_frags={mg.n_fragments}, n_atoms={mg.n_atoms}")
    print(f"  V shape: {mg.V.shape}, E shape: {mg.E.shape}")
    print(f"  edge_index shape: {mg.edge_index.shape}")
    assert mg.V.shape[0] == mg.n_fragments + mg.n_atoms
    assert mg.V.shape[1] == feat.d_v
    assert mg.E.shape[1] == 1


def test_featurizer_copolymer():
    """Two fragments with a connection."""
    feat = HPGMolGraphFeaturizer()
    mg = feat(
        fragment_smiles=["[*]CC[*]", "[*]OCC[*]"],
        connections=[(0, 1, 2.0)],  # fragment 0 → 1 with degree 2
    )

    assert mg.n_fragments == 2
    print(f"  copolymer: n_frags={mg.n_fragments}, n_atoms={mg.n_atoms}")
    print(f"  V shape: {mg.V.shape}, E shape: {mg.E.shape}")
    print(f"  edge_index shape: {mg.edge_index.shape}")
    assert mg.V.shape[0] == mg.n_fragments + mg.n_atoms
    # Should have: 1 frag-frag edge + atom-atom bonds (bidirectional) + atom→frag edges
    assert mg.edge_index.shape[1] > 0


def test_batching():
    """Batch two graphs together."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])

    bmg = BatchHPGMolGraph([mg1, mg2])
    print(f"  batch: V={bmg.V.shape}, E={bmg.E.shape}, "
          f"edge_index={bmg.edge_index.shape}, batch={bmg.batch.shape}")
    assert len(bmg) == 2
    assert bmg.V.shape[0] == mg1.V.shape[0] + mg2.V.shape[0]
    assert bmg.frag_mask.sum().item() == mg1.n_fragments + mg2.n_fragments


def test_gat_layer():
    """Single GAT layer forward pass."""
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    bmg = BatchHPGMolGraph([mg])

    layer = HPGGATLayer(in_feats=feat.d_v, out_feats=128, edge_feats=1, num_heads=8)
    with torch.no_grad():
        out = layer(bmg.V, bmg.edge_index, bmg.E)
    print(f"  GAT layer output: {out.shape}")
    assert out.shape == (bmg.V.shape[0], 128)


def test_message_passing():
    """Full HPG message passing stack."""
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    bmg = BatchHPGMolGraph([mg])

    mp = HPGMessagePassing(d_v=feat.d_v, d_h=128, depth=3, num_heads=4)
    with torch.no_grad():
        H = mp(bmg)
    print(f"  MP output: {H.shape}")
    assert H.shape == (bmg.V.shape[0], 128)


def test_full_model():
    """Full HPGMPNN forward pass."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    bmg = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v,
        d_h=64,
        d_ffn=32,
        depth=2,
        num_heads=4,
        n_tasks=1,
        d_xd=0,
    )
    with torch.no_grad():
        preds = model(bmg)
    print(f"  Model output: {preds.shape}")
    assert preds.shape == (2, 1)  # 2 graphs, 1 target


def test_full_model_with_xd():
    """Full HPGMPNN with scalar features."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    bmg = BatchHPGMolGraph([mg1, mg2])

    X_d = torch.randn(2, 5)  # 2 graphs, 5 scalar features

    model = HPGMPNN(
        d_v=feat.d_v,
        d_h=64,
        d_ffn=32,
        depth=2,
        num_heads=4,
        n_tasks=1,
        d_xd=5,
    )
    with torch.no_grad():
        preds = model(bmg, X_d=X_d)
    print(f"  Model with X_d output: {preds.shape}")
    assert preds.shape == (2, 1)


if __name__ == "__main__":
    print("=" * 60)
    print("  HPG-GAT Smoke Tests")
    print("=" * 60)

    tests = [
        ("1. Featurizer (homopolymer)", test_featurizer_homopolymer),
        ("2. Featurizer (copolymer)", test_featurizer_copolymer),
        ("3. Batching", test_batching),
        ("4. GAT Layer", test_gat_layer),
        ("5. Message Passing", test_message_passing),
        ("6. Full Model", test_full_model),
        ("7. Full Model + X_d", test_full_model_with_xd),
    ]

    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  All HPG smoke tests complete!")
    print("=" * 60)
