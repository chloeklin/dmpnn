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


# =====================================================================
#  Phase 1 variant tests
# =====================================================================

def _make_copolymer_batch_with_fracs(fracA1=0.6, fracB1=0.4, fracA2=0.3, fracB2=0.7):
    """Helper: build a 2-graph copolymer batch with explicit fragment fractions."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(
        ["[*]CC[*]", "[*]OCC[*]"],
        connections=[(0, 1, 1.0)],
        frag_fracs=np.array([fracA1, fracB1], dtype=np.float32),
    )
    mg2 = feat(
        ["[*]OCC[*]", "[*]CC[*]"],
        connections=[(0, 1, 1.0)],
        frag_fracs=np.array([fracA2, fracB2], dtype=np.float32),
    )
    bmg = BatchHPGMolGraph([mg1, mg2])
    return bmg


def test_frag_fracs_featurizer():
    """Featurizer stores frag_fracs when provided."""
    feat = HPGMolGraphFeaturizer()
    mg = feat(
        ["[*]CC[*]", "[*]OCC[*]"],
        connections=[(0, 1, 1.0)],
        frag_fracs=np.array([0.6, 0.4], dtype=np.float32),
    )
    assert mg.frag_fracs is not None
    assert mg.frag_fracs.shape == (2,)
    np.testing.assert_allclose(mg.frag_fracs, [0.6, 0.4])
    print(f"  frag_fracs stored: {mg.frag_fracs}")

    # Without fracs
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    assert mg2.frag_fracs is None
    print("  frag_fracs=None when not provided")


def test_frag_fracs_batching():
    """BatchHPGMolGraph correctly batches fragment fractions."""
    bmg = _make_copolymer_batch_with_fracs(0.6, 0.4, 0.3, 0.7)
    assert bmg.frag_fracs is not None
    # 2 fragments per graph × 2 graphs = 4
    assert bmg.frag_fracs.shape == (4,)
    expected = torch.tensor([0.6, 0.4, 0.3, 0.7])
    torch.testing.assert_close(bmg.frag_fracs, expected)
    print(f"  Batched frag_fracs: {bmg.frag_fracs.tolist()}")


def test_frag_fracs_batching_mixed():
    """Batch with one graph having fracs and one without → uniform fallback."""
    feat = HPGMolGraphFeaturizer()
    mg_with = feat(
        ["[*]CC[*]", "[*]OCC[*]"],
        connections=[(0, 1, 1.0)],
        frag_fracs=np.array([0.8, 0.2], dtype=np.float32),
    )
    mg_without = feat(["[*]OCC[*]"])  # homopolymer, no fracs
    bmg = BatchHPGMolGraph([mg_with, mg_without])
    assert bmg.frag_fracs is not None
    # mg_with: [0.8, 0.2], mg_without: [1.0] (uniform for 1 frag)
    expected = torch.tensor([0.8, 0.2, 1.0])
    torch.testing.assert_close(bmg.frag_fracs, expected)
    print(f"  Mixed batch frag_fracs: {bmg.frag_fracs.tolist()}")


def test_hpg_frac_forward():
    """HPG_frac: frac_weighted pooling produces correct output shape."""
    bmg = _make_copolymer_batch_with_fracs()
    model = HPGMPNN(
        d_v=HPGMolGraphFeaturizer().d_v,
        d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    with torch.no_grad():
        preds = model(bmg)
    print(f"  HPG_frac output: {preds.shape}")
    assert preds.shape == (2, 1)


def test_hpg_frac_polytype_forward():
    """HPG_frac_polytype: frac pooling + polytype X_d."""
    bmg = _make_copolymer_batch_with_fracs()
    d_polytype = 3  # e.g. 3 polytype categories
    X_d = torch.randn(2, d_polytype)

    model = HPGMPNN(
        d_v=HPGMolGraphFeaturizer().d_v,
        d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=d_polytype, pooling_type="frac_weighted",
    )
    with torch.no_grad():
        preds = model(bmg, X_d=X_d)
    print(f"  HPG_frac_polytype output: {preds.shape}")
    assert preds.shape == (2, 1)


def test_frac_weighted_differs_from_sum():
    """Fraction-weighted pooling gives different results from sum pooling."""
    bmg = _make_copolymer_batch_with_fracs(0.9, 0.1, 0.1, 0.9)
    d_v = HPGMolGraphFeaturizer().d_v

    torch.manual_seed(42)
    model_sum = HPGMPNN(
        d_v=d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="sum",
    )
    torch.manual_seed(42)
    model_frac = HPGMPNN(
        d_v=d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    with torch.no_grad():
        pred_sum = model_sum(bmg)
        pred_frac = model_frac(bmg)
    # Should differ because pooling strategy is different
    assert not torch.allclose(pred_sum, pred_frac, atol=1e-6), \
        "sum and frac_weighted should produce different predictions"
    print(f"  sum preds:  {pred_sum.flatten().tolist()}")
    print(f"  frac preds: {pred_frac.flatten().tolist()}")


def test_frac_sensitivity():
    """Changing fractions changes the frac_weighted output."""
    feat = HPGMolGraphFeaturizer()
    d_v = feat.d_v
    torch.manual_seed(0)
    model = HPGMPNN(
        d_v=d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    # Use single-graph batches so there's no cross-graph batch interaction
    mg_a = feat(
        ["[*]CC[*]", "[*]OCC[*]"],
        connections=[(0, 1, 1.0)],
        frag_fracs=np.array([0.9, 0.1], dtype=np.float32),
    )
    mg_b = feat(
        ["[*]CC[*]", "[*]OCC[*]"],
        connections=[(0, 1, 1.0)],
        frag_fracs=np.array([0.1, 0.9], dtype=np.float32),
    )
    bmg_a = BatchHPGMolGraph([mg_a])
    bmg_b = BatchHPGMolGraph([mg_b])
    with torch.no_grad():
        pred_a = model(bmg_a)
        pred_b = model(bmg_b)
    # Swapped fractions on identical structure → different prediction
    assert not torch.allclose(pred_a, pred_b, atol=1e-6), \
        "Different fractions should produce different predictions"
    print(f"  pred(0.9/0.1)={pred_a.item():.6f}, pred(0.1/0.9)={pred_b.item():.6f}")
    print(f"  Fraction sensitivity confirmed")


def test_frac_weighted_requires_fracs():
    """frac_weighted pooling raises if frag_fracs is None."""
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])  # no fracs
    bmg = BatchHPGMolGraph([mg])
    assert bmg.frag_fracs is None

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    try:
        with torch.no_grad():
            model(bmg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised: {e}")


def test_invalid_pooling_type():
    """Invalid pooling_type raises ValueError at construction."""
    try:
        HPGMPNN(
            d_v=130, d_h=64, d_ffn=32, depth=2, num_heads=4,
            n_tasks=1, d_xd=0, pooling_type="invalid",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised: {e}")


def test_baseline_unchanged():
    """HPG_baseline (pooling_type='sum') still works with existing API."""
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]OCC[*]"])
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)])
    bmg = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="sum",
    )
    with torch.no_grad():
        preds = model(bmg)
    assert preds.shape == (2, 1)
    print(f"  Baseline still works: {preds.shape}")


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
        # Phase 1 variant tests
        ("8.  frag_fracs featurizer", test_frag_fracs_featurizer),
        ("9.  frag_fracs batching", test_frag_fracs_batching),
        ("10. frag_fracs mixed batch", test_frag_fracs_batching_mixed),
        ("11. HPG_frac forward", test_hpg_frac_forward),
        ("12. HPG_frac_polytype forward", test_hpg_frac_polytype_forward),
        ("13. frac_weighted differs from sum", test_frac_weighted_differs_from_sum),
        ("14. Fraction sensitivity", test_frac_sensitivity),
        ("15. frac_weighted requires fracs", test_frac_weighted_requires_fracs),
        ("16. Invalid pooling_type", test_invalid_pooling_type),
        ("17. Baseline unchanged", test_baseline_unchanged),
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
