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


def test_frag_fracs_in_graph():
    """frag_fracs stored in HPGMolGraph and batched correctly."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])

    # Attach fractions
    mg1 = mg1._replace(frag_fracs=np.array([1.0], dtype=np.float32))
    mg2 = mg2._replace(frag_fracs=np.array([0.3, 0.7], dtype=np.float32))

    bmg = BatchHPGMolGraph([mg1, mg2])
    assert bmg.frag_fracs is not None
    assert bmg.frag_fracs.shape == (3,)  # 1 + 2 fragment nodes
    assert torch.allclose(bmg.frag_fracs, torch.tensor([1.0, 0.3, 0.7]))
    print(f"  frag_fracs batched: {bmg.frag_fracs.tolist()}")


def test_frag_fracs_none_when_absent():
    """frag_fracs is None when not provided."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    bmg = BatchHPGMolGraph([mg1, mg2])
    assert bmg.frag_fracs is None
    print("  frag_fracs correctly None when absent")


def test_hpg_frac_model():
    """HPG_frac: fraction-weighted pooling forward pass."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])._replace(
        frag_fracs=np.array([1.0], dtype=np.float32))
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])._replace(
        frag_fracs=np.array([0.4, 0.6], dtype=np.float32))
    bmg = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, hpg_variant="HPG_frac",
    )
    with torch.no_grad():
        preds = model(bmg)
    print(f"  HPG_frac output: {preds.shape}")
    assert preds.shape == (2, 1)


def test_hpg_frac_polytype_model():
    """HPG_frac_polytype: fraction-weighted pooling + polytype X_d."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])._replace(
        frag_fracs=np.array([1.0], dtype=np.float32))
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])._replace(
        frag_fracs=np.array([0.4, 0.6], dtype=np.float32))
    bmg = BatchHPGMolGraph([mg1, mg2])

    polytype_dim = 3
    X_d = torch.randn(2, polytype_dim)  # polytype one-hot

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=polytype_dim, hpg_variant="HPG_frac_polytype",
    )
    with torch.no_grad():
        preds = model(bmg, X_d=X_d)
    print(f"  HPG_frac_polytype output: {preds.shape}")
    assert preds.shape == (2, 1)


def test_fraction_sensitivity():
    """Changing fractions should change the output for HPG_frac."""
    feat = HPGMolGraphFeaturizer()
    frags = ["[*]CC[*]", "[*]OCC[*]"]
    conn = [(0, 1, 1.0)]

    mg_a = feat(frags, conn)._replace(
        frag_fracs=np.array([0.2, 0.8], dtype=np.float32))
    mg_b = feat(frags, conn)._replace(
        frag_fracs=np.array([0.8, 0.2], dtype=np.float32))

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, hpg_variant="HPG_frac",
    )
    with torch.no_grad():
        bmg_a = BatchHPGMolGraph([mg_a])
        bmg_b = BatchHPGMolGraph([mg_b])
        pred_a = model(bmg_a)
        pred_b = model(bmg_b)
    diff = (pred_a - pred_b).abs().item()
    print(f"  Prediction diff for swapped fracs: {diff:.6f}")
    assert diff > 1e-6, "Different fractions should produce different outputs"


def test_invalid_variant_rejected():
    """Unknown hpg_variant should raise ValueError."""
    feat = HPGMolGraphFeaturizer()
    try:
        HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                n_tasks=1, hpg_variant="INVALID")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")


def test_baseline_unchanged():
    """HPG_baseline should produce same output regardless of frag_fracs presence."""
    feat = HPGMolGraphFeaturizer()
    mg_no_frac = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    mg_with_frac = mg_no_frac._replace(
        frag_fracs=np.array([0.5, 0.5], dtype=np.float32))

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, hpg_variant="HPG_baseline",
    )
    model.eval()  # disable dropout for deterministic comparison
    with torch.no_grad():
        bmg_no = BatchHPGMolGraph([mg_no_frac])
        bmg_yes = BatchHPGMolGraph([mg_with_frac])
        pred_no = model(bmg_no)
        pred_yes = model(bmg_yes)
    assert torch.allclose(pred_no, pred_yes, atol=1e-6), \
        "HPG_baseline should ignore frag_fracs"
    print(f"  HPG_baseline correctly ignores frag_fracs")


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
        ("6. Full Model (baseline)", test_full_model),
        ("7. Full Model + X_d", test_full_model_with_xd),
        ("8. frag_fracs in graph", test_frag_fracs_in_graph),
        ("9. frag_fracs None when absent", test_frag_fracs_none_when_absent),
        ("10. HPG_frac model", test_hpg_frac_model),
        ("11. HPG_frac_polytype model", test_hpg_frac_polytype_model),
        ("12. Fraction sensitivity", test_fraction_sensitivity),
        ("13. Invalid variant rejected", test_invalid_variant_rejected),
        ("14. Baseline unchanged with fracs", test_baseline_unchanged),
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
