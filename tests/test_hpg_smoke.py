"""Smoke test for the HPG-GAT implementation in chemprop.

Tests the full pipeline:
  1. HPGMolGraphFeaturizer  — fragment SMILES → HPGMolGraph
  2. BatchHPGMolGraph       — batching multiple graphs
  3. HPGMessagePassing      — GAT forward pass
  4. HPGMPNN                — full model forward pass
"""

import numpy as np
import torch
import torch.nn as nn

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


# ---------------------------------------------------------------------------
#  Tests 18-24: HPG_frac_edgeTyped
# ---------------------------------------------------------------------------

def test_encode_hpg_edge():
    """encode_hpg_edge returns correct 4-dim vectors for all three edge types."""
    from chemprop.featurizers.molgraph.hpg import encode_hpg_edge
    import numpy as np

    ab  = encode_hpg_edge('atom_bond', 2.0)
    atf = encode_hpg_edge('atom_to_fragment', 1.0)
    ftf = encode_hpg_edge('fragment_to_fragment', 3.0)

    assert ab.shape  == (4,), f"Expected (4,), got {ab.shape}"
    assert atf.shape == (4,), f"Expected (4,), got {atf.shape}"
    assert ftf.shape == (4,), f"Expected (4,), got {ftf.shape}"

    assert list(ab[:3])  == [1.0, 0.0, 0.0], f"atom_bond one-hot wrong: {ab[:3]}"
    assert list(atf[:3]) == [0.0, 1.0, 0.0], f"atom_to_fragment one-hot wrong: {atf[:3]}"
    assert list(ftf[:3]) == [0.0, 0.0, 1.0], f"fragment_to_fragment one-hot wrong: {ftf[:3]}"

    assert ab[3]  == 2.0, f"atom_bond scalar wrong: {ab[3]}"
    assert atf[3] == 1.0, f"atom_to_fragment scalar wrong: {atf[3]}"
    assert ftf[3] == 3.0, f"fragment_to_fragment scalar wrong: {ftf[3]}"
    print(f"  encode_hpg_edge: all 4 cases correct")


def test_encode_hpg_edge_invalid():
    """encode_hpg_edge raises ValueError for unknown edge type."""
    from chemprop.featurizers.molgraph.hpg import encode_hpg_edge
    try:
        encode_hpg_edge('unknown_type', 1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised: {e}")


def test_typed_featurizer_edge_shape():
    """HPGMolGraphFeaturizerEdgeTyped produces E of shape [n_edges, 4]."""
    from chemprop.featurizers.molgraph.hpg import HPGMolGraphFeaturizerEdgeTyped
    import numpy as np

    feat = HPGMolGraphFeaturizerEdgeTyped()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 2.0)],
              frag_fracs=np.array([0.4, 0.6], dtype=np.float32))

    assert mg.E.shape[1] == 4, f"Expected E.shape[1]=4, got {mg.E.shape[1]}"
    print(f"  E shape: {mg.E.shape}  (expected [n_edges, 4])")

    # Every edge must have exactly one hot 1 in positions 0-2
    onehot_part = mg.E[:, :3]
    assert np.all(onehot_part.sum(axis=1) == 1.0), "One-hot constraint violated"
    print(f"  One-hot constraint satisfied for all edges")

    # Spot-check: the first edge is fragment-fragment, so one-hot should be [0,0,1]
    ff_edge = mg.E[0]
    assert list(ff_edge[:3]) == [0.0, 0.0, 1.0], f"First edge not fragment_to_fragment: {ff_edge}"
    print(f"  Fragment-fragment edge: {list(ff_edge)}")


def test_typed_vs_scalar_featurizer():
    """Typed featurizer gives E[*,4] while scalar gives E[*,1]; node features identical."""
    from chemprop.featurizers.molgraph.hpg import HPGMolGraphFeaturizerEdgeTyped
    import numpy as np

    scalar_feat = HPGMolGraphFeaturizer()
    typed_feat  = HPGMolGraphFeaturizerEdgeTyped()

    smi_list = ["[*]CC[*]", "[*]OCC[*]"]
    conns    = [(0, 1, 1.0)]

    mg_scalar = scalar_feat(smi_list, conns)
    mg_typed  = typed_feat(smi_list, conns)

    assert mg_scalar.E.shape[1] == 1, f"scalar E col dim wrong: {mg_scalar.E.shape}"
    assert mg_typed.E.shape[1]  == 4, f"typed  E col dim wrong: {mg_typed.E.shape}"
    assert mg_scalar.E.shape[0] == mg_typed.E.shape[0], "Edge count mismatch"
    assert np.allclose(mg_scalar.V, mg_typed.V), "Node features differ between variants"
    print(f"  scalar E: {mg_scalar.E.shape}, typed E: {mg_typed.E.shape}  — node features identical")


def test_hpg_frac_edgeTyped_forward():
    """HPG_frac_edgeTyped: full forward pass with d_e=4 produces correct output shape."""
    from chemprop.featurizers.molgraph.hpg import HPGMolGraphFeaturizerEdgeTyped
    import numpy as np

    feat = HPGMolGraphFeaturizerEdgeTyped()
    ff1  = np.array([0.6, 0.4], dtype=np.float32)
    ff2  = np.array([0.3, 0.7], dtype=np.float32)
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"],  connections=[(0, 1, 1.0)], frag_fracs=ff1)
    mg2  = feat(["[*]CCC[*]", "[*]OCCC[*]"], connections=[(0, 1, 1.0)], frag_fracs=ff2)
    bmg  = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v, d_e=4, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    with torch.no_grad():
        preds = model(bmg)
    assert preds.shape == (2, 1), f"Expected (2,1), got {preds.shape}"
    print(f"  HPG_frac_edgeTyped output: {preds.shape}")


def test_d_e_mismatch_raises():
    """Using typed featurizer (d_e=4) with model d_e=1 raises a RuntimeError."""
    from chemprop.featurizers.molgraph.hpg import HPGMolGraphFeaturizerEdgeTyped
    import numpy as np

    feat = HPGMolGraphFeaturizerEdgeTyped()
    ff   = np.array([0.5, 0.5], dtype=np.float32)
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)], frag_fracs=ff)
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)], frag_fracs=ff)
    bmg  = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v, d_e=1, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    try:
        with torch.no_grad():
            _ = model(bmg)
        assert False, "Should have raised RuntimeError due to d_e mismatch"
    except RuntimeError:
        print(f"  Correctly raised on d_e mismatch: RuntimeError")


def test_existing_variants_unaffected():
    """Existing variants (baseline, frac) still work with default d_e=1."""
    feat = HPGMolGraphFeaturizer()
    ff   = torch.tensor([0.6, 0.4])
    import numpy as np
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.3, 0.7], dtype=np.float32))

    for pooling_type in ("sum", "frac_weighted"):
        bmg   = BatchHPGMolGraph([mg1, mg2])
        model = HPGMPNN(
            d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
            n_tasks=1, d_xd=0, pooling_type=pooling_type,
        )
        with torch.no_grad():
            preds = model(bmg)
        assert preds.shape == (2, 1)
        print(f"  pooling_type={pooling_type!r} with default d_e=1: OK")


# ──────────────────────────────────────────────────────────────────
# Phase 1E — HPG_frac_archAware tests
# ──────────────────────────────────────────────────────────────────

def test_hpg_frac_archAware_forward():
    """HPG_frac_archAware produces correct output shape."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.3, 0.7], dtype=np.float32))
    bmg  = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_arch_aware",
    )
    with torch.no_grad():
        preds = model(bmg)
    assert preds.shape == (2, 1), f"Expected (2,1), got {preds.shape}"
    print(f"  HPG_frac_archAware output: {preds.shape}")


def test_archAware_zero_init_matches_frac():
    """At init (W=0) frac_arch_aware must produce identical output to frac_weighted.

    When W=0: h_tilde_i = h_i + W(m - f_i h_i) = h_i + 0 = h_i
    So h_poly = sum_i f_i h_i   (identical to HPG_frac)
    """
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    ffs  = [
        np.array([0.6, 0.4], dtype=np.float32),
        np.array([0.3, 0.7], dtype=np.float32),
    ]
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=ff) for ff in ffs]

    # Build models and copy all shared weights from frac → arch so only W=0 differs.
    # (Using the same random seed is insufficient because arch_interact adds a parameter
    #  that shifts the RNG state for all subsequent layers.)
    torch.manual_seed(0)
    model_frac = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    model_arch = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_arch_aware",
    )
    # Copy all shared parameters; arch_interact stays at its zero init
    shared_keys = set(model_frac.state_dict()) & set(model_arch.state_dict())
    with torch.no_grad():
        for k in shared_keys:
            model_arch.state_dict()[k].copy_(model_frac.state_dict()[k])
    # Verify W is zero at init
    assert torch.all(model_arch.arch_interact.weight == 0), "arch_interact.weight should be zero at init"

    model_frac.eval()
    model_arch.eval()
    bmg = BatchHPGMolGraph(graphs)
    with torch.no_grad():
        preds_frac = model_frac(bmg)
        preds_arch = model_arch(bmg)

    assert torch.allclose(preds_frac, preds_arch, atol=1e-5), (
        f"With W=0, frac_arch_aware should equal frac_weighted.\n"
        f"  frac: {preds_frac.tolist()}\n  arch: {preds_arch.tolist()}"
    )
    print(f"  W=0 → frac_arch_aware == frac_weighted: confirmed")


def test_archAware_diverges_after_nonzero_W():
    """With non-zero W, frac_arch_aware output differs from frac_weighted."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    ffs  = [
        np.array([0.6, 0.4], dtype=np.float32),
        np.array([0.3, 0.7], dtype=np.float32),
    ]
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=ff) for ff in ffs]
    bmg = BatchHPGMolGraph(graphs)

    torch.manual_seed(0)
    model_frac = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    torch.manual_seed(0)
    model_arch = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_arch_aware",
    )
    # Manually set W to non-zero
    nn.init.eye_(model_arch.arch_interact.weight)

    with torch.no_grad():
        preds_frac = model_frac(bmg)
        preds_arch = model_arch(bmg)

    assert not torch.allclose(preds_frac, preds_arch, atol=1e-5), (
        "With W=I, frac_arch_aware should differ from frac_weighted"
    )
    print(f"  Non-zero W → predictions differ: confirmed")
    print(f"    frac: {preds_frac.tolist()}")
    print(f"    arch: {preds_arch.tolist()}")


def test_archAware_batching_isolation():
    """Interaction is isolated per polymer — two identical samples must give identical preds."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    ff   = np.array([0.6, 0.4], dtype=np.float32)
    mg   = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)], frag_fracs=ff)

    bmg_single = BatchHPGMolGraph([mg])
    bmg_double = BatchHPGMolGraph([mg, mg])

    torch.manual_seed(0)
    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_arch_aware",
    )
    nn.init.eye_(model.arch_interact.weight)  # non-zero W to exercise interaction
    model.eval()  # disable dropout so identical samples give identical predictions

    with torch.no_grad():
        pred_single = model(bmg_single)
        pred_double = model(bmg_double)

    assert torch.allclose(pred_single[0], pred_double[0], atol=1e-5), (
        "Batching isolation failed: same polymer gives different prediction in batch"
    )
    assert torch.allclose(pred_double[0], pred_double[1], atol=1e-5), (
        "Two identical polymers in batch give different predictions"
    )
    print(f"  Batching isolation: confirmed (single={pred_single[0].item():.4f}, "
          f"batch[0]={pred_double[0].item():.4f}, batch[1]={pred_double[1].item():.4f})")


def test_archAware_backward_compat():
    """Adding frac_arch_aware to VALID_POOLING_TYPES does not break sum/frac_weighted."""
    from chemprop.models.hpg import VALID_POOLING_TYPES
    assert "frac_arch_aware" in VALID_POOLING_TYPES
    assert "sum" in VALID_POOLING_TYPES
    assert "frac_weighted" in VALID_POOLING_TYPES
    # arch_interact is None for non-arch-aware variants
    feat  = HPGMolGraphFeaturizer()
    model = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    assert model.arch_interact is None, "arch_interact should be None for frac_weighted"
    model2 = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                     n_tasks=1, d_xd=0, pooling_type="sum")
    assert model2.arch_interact is None, "arch_interact should be None for sum"
    print("  arch_interact is None for sum/frac_weighted: confirmed")
    print(f"  VALID_POOLING_TYPES: {VALID_POOLING_TYPES}")


# ──────────────────────────────────────────────────────────────────
# Phase 2A — HPG_relMsg tests
# ──────────────────────────────────────────────────────────────────

def test_relMsg_forward():
    """HPG_relMsg forward pass produces correct output shape [B, n_tasks]."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
               frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
               frag_fracs=np.array([0.3, 0.7], dtype=np.float32))
    bmg = BatchHPGMolGraph([mg1, mg2])
    torch.manual_seed(0)
    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, mp_type="rel_msg", pooling_type="frac_weighted",
    )
    with torch.no_grad():
        out = model(bmg)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  HPG_relMsg output: {out.shape}")


def test_relMsg_differs_from_frac():
    """HPG_relMsg gives different predictions than HPG_frac (different MP architecture)."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1 = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
               frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    bmg = BatchHPGMolGraph([mg1])
    torch.manual_seed(42)
    model_frac = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, mp_type="gat", pooling_type="frac_weighted",
    )
    torch.manual_seed(42)
    model_rel = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, mp_type="rel_msg", pooling_type="frac_weighted",
    )
    model_frac.eval()
    model_rel.eval()
    with torch.no_grad():
        pred_frac = model_frac(bmg)
        pred_rel  = model_rel(bmg)
    assert not torch.allclose(pred_frac, pred_rel, atol=1e-4), (
        "HPG_relMsg and HPG_frac should differ (different MP architecture)"
    )
    print(f"  frac={pred_frac.item():.6f}, relMsg={pred_rel.item():.6f} — differ: confirmed")


def test_relMsg_edge_content_sensitivity():
    """Perturbing edge features changes HPG_relMsg output.

    Verifies that edge features enter message CONTENT in relMsg:
    a large edge perturbation flows through W_msg and changes predictions.
    """
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
              frag_fracs=np.array([0.5, 0.5], dtype=np.float32))
    bmg_orig = BatchHPGMolGraph([mg])
    bmg_pert = BatchHPGMolGraph([mg])
    bmg_pert.E = bmg_orig.E.clone()
    bmg_pert.E[:] = 100.0  # extreme perturbation to edge scalars
    torch.manual_seed(7)
    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, mp_type="rel_msg", pooling_type="frac_weighted",
    )
    model.eval()
    with torch.no_grad():
        delta = (model(bmg_pert) - model(bmg_orig)).abs().item()
    assert delta > 0, "relMsg output must change when edge features are perturbed"
    print(f"  |Δ| with extreme edge perturbation: {delta:.4f} — edge enters message content: confirmed")


def test_relMsg_batching_isolation():
    """Same polymer gives identical prediction in singleton vs. batch of two."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
              frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    bmg_single = BatchHPGMolGraph([mg])
    bmg_double = BatchHPGMolGraph([mg, mg])
    torch.manual_seed(0)
    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, mp_type="rel_msg", pooling_type="frac_weighted",
    )
    model.eval()
    with torch.no_grad():
        pred_single = model(bmg_single)
        pred_double = model(bmg_double)
    assert torch.allclose(pred_single[0], pred_double[0], atol=1e-5), (
        "Batching isolation failed: same polymer gives different prediction in batch"
    )
    assert torch.allclose(pred_double[0], pred_double[1], atol=1e-5), (
        "Two identical polymers in batch give different predictions"
    )
    print(f"  Batching isolation: single={pred_single[0].item():.4f}, "
          f"batch[0]={pred_double[0].item():.4f}, batch[1]={pred_double[1].item():.4f} — confirmed")


def test_relMsg_backward_compat():
    """relMsg does not affect Phase 1 variants; mp_type defaults to 'gat'."""
    feat = HPGMolGraphFeaturizer()
    model_default = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                            n_tasks=1, d_xd=0, pooling_type="sum")
    assert model_default.hparams["mp_type"] == "gat", "Default mp_type should be 'gat'"
    model_frac = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                         n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    assert model_frac.hparams["mp_type"] == "gat"
    model_arch = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                         n_tasks=1, d_xd=0, pooling_type="frac_arch_aware")
    assert model_arch.hparams["mp_type"] == "gat"
    try:
        HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                n_tasks=1, d_xd=0, mp_type="invalid")
        raise AssertionError("Should have raised ValueError for invalid mp_type")
    except ValueError:
        pass
    print("  mp_type defaults to 'gat', Phase 1 variants unaffected, invalid mp_type raises: confirmed")


# ──────────────────────────────────────────────────────────────────
# Phase 2B — HPG_fragGraph tests
# ──────────────────────────────────────────────────────────────────

def test_fragGraph_forward():
    """HPG_fragGraph produces correct output shape."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.3, 0.7], dtype=np.float32))
    bmg  = BatchHPGMolGraph([mg1, mg2])

    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_graph",
    )
    with torch.no_grad():
        preds = model(bmg)
    assert preds.shape == (2, 1), f"Expected (2,1), got {preds.shape}"
    print(f"  HPG_fragGraph output: {preds.shape}")


def test_fragGraph_zero_init_matches_frac():
    """At init (W=0) frac_graph must produce identical output to frac_weighted.

    When W=0: m_i = sum W(u_j) = 0, so z_i = u_i.
    Fraction-weighted pooling is then identical to HPG_frac.
    """
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    ffs  = [
        np.array([0.6, 0.4], dtype=np.float32),
        np.array([0.3, 0.7], dtype=np.float32),
    ]
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=ff) for ff in ffs]
    bmg = BatchHPGMolGraph(graphs)

    # Build both models, copy shared weights so only frag_graph_layer.W (=0) differs.
    torch.manual_seed(42)
    model_frac = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    model_fg = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_graph",
    )
    # Copy all shared parameters; frag_graph_layer.W stays at its zero init
    shared_keys = set(model_frac.state_dict()) & set(model_fg.state_dict())
    with torch.no_grad():
        for k in shared_keys:
            model_fg.state_dict()[k].copy_(model_frac.state_dict()[k])
    # Verify W is zero at init
    assert torch.all(model_fg.frag_graph_layer.W.weight == 0), (
        "frag_graph_layer.W.weight should be zero at init"
    )

    model_frac.eval()
    model_fg.eval()
    with torch.no_grad():
        preds_frac = model_frac(bmg)
        preds_fg   = model_fg(bmg)

    assert torch.allclose(preds_frac, preds_fg, atol=1e-5), (
        f"With W=0, frac_graph should equal frac_weighted.\n"
        f"  frac: {preds_frac.tolist()}\n  fragGraph: {preds_fg.tolist()}"
    )
    print(f"  W=0 -> frac_graph == frac_weighted: confirmed")


def test_fragGraph_diverges_after_nonzero_W():
    """With non-zero W, frac_graph output differs from frac_weighted."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.6, 0.4], dtype=np.float32))]
    bmg = BatchHPGMolGraph(graphs)

    torch.manual_seed(0)
    model_frac = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_weighted",
    )
    torch.manual_seed(0)
    model_fg = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_graph",
    )
    # Set W to identity so messages carry full neighbour embedding
    nn.init.eye_(model_fg.frag_graph_layer.W.weight)

    with torch.no_grad():
        preds_frac = model_frac(bmg)
        preds_fg   = model_fg(bmg)
    assert not torch.allclose(preds_frac, preds_fg, atol=1e-4), (
        "Non-zero W should make frac_graph differ from frac_weighted"
    )
    print(f"  Non-zero W -> diverges from frac_weighted: confirmed")


def test_fragGraph_batching_isolation():
    """Same polymer gives identical prediction in singleton vs. batch of two.

    Also verifies that fragment message passing is within-polymer only:
    if adjacency crossed polymer boundaries the two identical polymers in a
    batch would still give the same result, but different from singleton.
    """
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
              frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    bmg_single = BatchHPGMolGraph([mg])
    bmg_double = BatchHPGMolGraph([mg, mg])

    torch.manual_seed(0)
    model = HPGMPNN(
        d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
        n_tasks=1, d_xd=0, pooling_type="frac_graph",
    )
    # Activate W so messages are non-zero
    nn.init.eye_(model.frag_graph_layer.W.weight)
    model.eval()
    with torch.no_grad():
        pred_single = model(bmg_single)
        pred_double = model(bmg_double)
    assert torch.allclose(pred_single[0], pred_double[0], atol=1e-5), (
        "Batching isolation failed: singleton and batch[0] differ"
    )
    assert torch.allclose(pred_double[0], pred_double[1], atol=1e-5), (
        "Two identical polymers in batch give different predictions"
    )
    print(f"  Batching isolation: single={pred_single[0].item():.4f}, "
          f"batch[0]={pred_double[0].item():.4f}, batch[1]={pred_double[1].item():.4f} — confirmed")


def test_fragGraph_backward_compat():
    """fragGraph uses mp_type='gat' (default); all prior variants unaffected."""
    feat = HPGMolGraphFeaturizer()
    # fragGraph allocates frag_graph_layer; other variants must have it None
    model_fg = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                       n_tasks=1, d_xd=0, pooling_type="frac_graph")
    assert model_fg.frag_graph_layer is not None
    assert model_fg.arch_interact is None
    assert model_fg.hparams["mp_type"] == "gat", "fragGraph must use standard gat mp_type"

    # Prior variants must NOT have frag_graph_layer
    for pt in ("sum", "frac_weighted", "frac_arch_aware"):
        m = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type=pt)
        assert m.frag_graph_layer is None, (
            f"pooling_type='{pt}' should not have frag_graph_layer"
        )
    print("  fragGraph backward compat: frag_graph_layer isolated, prior variants unaffected")


# ──────────────────────────────────────────────────────────────────
# Phase 3A — HPG_attnPool tests
# ──────────────────────────────────────────────────────────────────

def test_attnPool_forward():
    """HPG_attnPool produces correct output shape."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.3, 0.7], dtype=np.float32))
    bmg  = BatchHPGMolGraph([mg1, mg2])
    model = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="attn_pool")
    with torch.no_grad():
        out = model(bmg)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  HPG_attnPool output: {out.shape}")


def test_attnPool_zero_init_matches_frac():
    """At init (scorer=0) attn_pool must produce identical output to frac_weighted."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.6, 0.4], dtype=np.float32)),
              feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.3, 0.7], dtype=np.float32))]
    bmg = BatchHPGMolGraph(graphs)

    torch.manual_seed(0)
    mf = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    ma = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="attn_pool")
    shared = set(mf.state_dict()) & set(ma.state_dict())
    with torch.no_grad():
        for k in shared:
            ma.state_dict()[k].copy_(mf.state_dict()[k])
    assert torch.all(ma.attn_scorer.weight == 0), "attn_scorer.weight should be zero at init"
    assert torch.all(ma.attn_scorer.bias   == 0), "attn_scorer.bias should be zero at init"
    mf.eval(); ma.eval()
    with torch.no_grad():
        pf, pa = mf(bmg), ma(bmg)
    assert torch.allclose(pf, pa, atol=1e-5), (
        f"W=0 → attn_pool should equal frac_weighted.\n  frac: {pf.tolist()}\n  attn: {pa.tolist()}"
    )
    print(f"  attnPool zero-init == frac_weighted: confirmed")


def test_attnPool_diverges_nonzero():
    """With non-zero scorer, attn_pool deviates from frac_weighted."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.6, 0.4], dtype=np.float32))]
    bmg = BatchHPGMolGraph(graphs)
    torch.manual_seed(0)
    mf = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    torch.manual_seed(0)
    ma = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="attn_pool")
    nn.init.ones_(ma.attn_scorer.weight)  # push scores away from 0
    with torch.no_grad():
        pf, pa = mf(bmg), ma(bmg)
    assert not torch.allclose(pf, pa, atol=1e-4), "Non-zero scorer should differ from frac_weighted"
    print(f"  Non-zero scorer → diverges from frac_weighted: confirmed")


def test_attnPool_batching_isolation():
    """Same polymer gives identical prediction singleton vs batch of two."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
              frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    bmg_s = BatchHPGMolGraph([mg])
    bmg_d = BatchHPGMolGraph([mg, mg])
    torch.manual_seed(0)
    model = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="attn_pool")
    nn.init.ones_(model.attn_scorer.weight)
    model.eval()
    with torch.no_grad():
        ps, pd = model(bmg_s), model(bmg_d)
    assert torch.allclose(ps[0], pd[0], atol=1e-5), "Batching isolation failed"
    assert torch.allclose(pd[0], pd[1], atol=1e-5), "Identical polymers gave different results"
    print(f"  attnPool batching isolation: confirmed")


# ──────────────────────────────────────────────────────────────────
# Phase 3B — HPG_pairInteract tests
# ──────────────────────────────────────────────────────────────────

def test_pairInteract_forward():
    """HPG_pairInteract produces correct output shape."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.3, 0.7], dtype=np.float32))
    bmg  = BatchHPGMolGraph([mg1, mg2])
    model = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="pair_interact")
    with torch.no_grad():
        out = model(bmg)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  HPG_pairInteract output: {out.shape}")


def test_pairInteract_zero_init_matches_frac():
    """At init (MLP output layer=0) pair_interact must equal frac_weighted."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.6, 0.4], dtype=np.float32)),
              feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.3, 0.7], dtype=np.float32))]
    bmg = BatchHPGMolGraph(graphs)
    torch.manual_seed(0)
    mf = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    mp = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="pair_interact")
    shared = set(mf.state_dict()) & set(mp.state_dict())
    with torch.no_grad():
        for k in shared:
            mp.state_dict()[k].copy_(mf.state_dict()[k])
    assert torch.all(mp.pair_interact_layer.mlp[-1].weight == 0), "output layer should be zero"
    mf.eval(); mp.eval()
    with torch.no_grad():
        pf, pp = mf(bmg), mp(bmg)
    assert torch.allclose(pf, pp, atol=1e-5), (
        f"W=0 → pair_interact should equal frac_weighted.\n  {pf.tolist()} vs {pp.tolist()}"
    )
    print(f"  pairInteract zero-init == frac_weighted: confirmed")


def test_pairInteract_diverges_nonzero():
    """With non-zero MLP, pair_interact deviates from frac_weighted."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.6, 0.4], dtype=np.float32))]
    bmg = BatchHPGMolGraph(graphs)
    torch.manual_seed(0)
    mf = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    torch.manual_seed(0)
    mp = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                 n_tasks=1, d_xd=0, pooling_type="pair_interact")
    nn.init.eye_(mp.pair_interact_layer.mlp[-1].weight)
    with torch.no_grad():
        pf, pp = mf(bmg), mp(bmg)
    assert not torch.allclose(pf, pp, atol=1e-4), "Non-zero MLP should differ from frac_weighted"
    print(f"  pairInteract non-zero → diverges from frac_weighted: confirmed")


def test_pairInteract_batching_isolation():
    """Pairwise interaction does not cross polymer boundaries."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
              frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    bmg_s = BatchHPGMolGraph([mg])
    bmg_d = BatchHPGMolGraph([mg, mg])
    torch.manual_seed(0)
    model = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="pair_interact")
    nn.init.eye_(model.pair_interact_layer.mlp[-1].weight)
    model.eval()
    with torch.no_grad():
        ps, pd = model(bmg_s), model(bmg_d)
    assert torch.allclose(ps[0], pd[0], atol=1e-5), "Batching isolation failed"
    assert torch.allclose(pd[0], pd[1], atol=1e-5), "Identical polymers gave different results"
    print(f"  pairInteract batching isolation: confirmed")


# ──────────────────────────────────────────────────────────────────
# Phase 3C — HPG_pairInteractAttn tests
# ──────────────────────────────────────────────────────────────────

def test_pairInteractAttn_forward():
    """HPG_pairInteractAttn produces correct output shape."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    mg1  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.6, 0.4], dtype=np.float32))
    mg2  = feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                frag_fracs=np.array([0.3, 0.7], dtype=np.float32))
    bmg  = BatchHPGMolGraph([mg1, mg2])
    model = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="pair_interact_attn")
    with torch.no_grad():
        out = model(bmg)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  HPG_pairInteractAttn output: {out.shape}")


def test_pairInteractAttn_zero_init_matches_frac():
    """At init pair_interact_attn must equal frac_weighted (phi output layer = 0)."""
    import numpy as np
    feat = HPGMolGraphFeaturizer()
    graphs = [feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.6, 0.4], dtype=np.float32)),
              feat(["[*]CC[*]", "[*]OCC[*]"], connections=[(0, 1, 1.0)],
                   frag_fracs=np.array([0.3, 0.7], dtype=np.float32))]
    bmg = BatchHPGMolGraph(graphs)
    torch.manual_seed(0)
    mf  = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                  n_tasks=1, d_xd=0, pooling_type="frac_weighted")
    mpa = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                  n_tasks=1, d_xd=0, pooling_type="pair_interact_attn")
    shared = set(mf.state_dict()) & set(mpa.state_dict())
    with torch.no_grad():
        for k in shared:
            mpa.state_dict()[k].copy_(mf.state_dict()[k])
    assert torch.all(mpa.pair_interact_attn_layer.phi[-1].weight   == 0)
    assert torch.all(mpa.pair_interact_attn_layer.score.weight == 0)
    mf.eval(); mpa.eval()
    with torch.no_grad():
        pf, ppa = mf(bmg), mpa(bmg)
    assert torch.allclose(pf, ppa, atol=1e-5), (
        f"pairInteractAttn zero-init should equal frac_weighted.\n  {pf.tolist()} vs {ppa.tolist()}"
    )
    print(f"  pairInteractAttn zero-init == frac_weighted: confirmed")


def test_pairInteractAttn_separate_from_pairInteract():
    """pairInteractAttn and pairInteract are independent model branches."""
    feat = HPGMolGraphFeaturizer()
    m_pi  = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="pair_interact")
    m_pia = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type="pair_interact_attn")
    # Verify module isolation
    assert m_pi.pair_interact_layer       is not None
    assert m_pi.pair_interact_attn_layer  is None
    assert m_pia.pair_interact_attn_layer is not None
    assert m_pia.pair_interact_layer      is None
    print("  pairInteractAttn/pairInteract are separate module branches: confirmed")


def test_phase3_backward_compat():
    """All prior pooling types still instantiate cleanly with no Phase 3 modules."""
    feat = HPGMolGraphFeaturizer()
    for pt in ("sum", "frac_weighted", "frac_arch_aware", "frac_graph"):
        m = HPGMPNN(d_v=feat.d_v, d_h=64, d_ffn=32, depth=2, num_heads=4,
                    n_tasks=1, d_xd=0, pooling_type=pt)
        assert m.attn_scorer            is None, f"{pt} should have attn_scorer=None"
        assert m.pair_interact_layer    is None, f"{pt} should have pair_interact_layer=None"
        assert m.pair_interact_attn_layer is None, f"{pt} should have pair_interact_attn_layer=None"
    print("  Phase 3 backward compat: prior variants unaffected")


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
        # frac_edgeTyped tests
        ("18. encode_hpg_edge helper", test_encode_hpg_edge),
        ("19. encode_hpg_edge invalid type", test_encode_hpg_edge_invalid),
        ("20. Typed featurizer edge shape", test_typed_featurizer_edge_shape),
        ("21. Typed vs scalar featurizer", test_typed_vs_scalar_featurizer),
        ("22. HPG_frac_edgeTyped forward", test_hpg_frac_edgeTyped_forward),
        ("23. d_e mismatch raises", test_d_e_mismatch_raises),
        ("24. Existing variants unaffected", test_existing_variants_unaffected),
        # frac_archAware tests
        ("25. HPG_frac_archAware forward", test_hpg_frac_archAware_forward),
        ("26. W=0 → matches frac_weighted", test_archAware_zero_init_matches_frac),
        ("27. Non-zero W → diverges from frac", test_archAware_diverges_after_nonzero_W),
        ("28. Batching isolation", test_archAware_batching_isolation),
        ("29. Backward compat (arch_interact=None)", test_archAware_backward_compat),
        # relMsg (Phase 2A) tests
        ("30. HPG_relMsg forward", test_relMsg_forward),
        ("31. relMsg differs from HPG_frac", test_relMsg_differs_from_frac),
        ("32. Edge content sensitivity", test_relMsg_edge_content_sensitivity),
        ("33. Batching isolation", test_relMsg_batching_isolation),
        ("34. Backward compat (mp_type defaults to 'gat')", test_relMsg_backward_compat),
        # fragGraph (Phase 2B) tests
        ("35. HPG_fragGraph forward", test_fragGraph_forward),
        ("36. W=0 -> matches frac_weighted", test_fragGraph_zero_init_matches_frac),
        ("37. Non-zero W -> diverges from frac", test_fragGraph_diverges_after_nonzero_W),
        ("38. Batching isolation", test_fragGraph_batching_isolation),
        ("39. Backward compat (frag_graph_layer isolated)", test_fragGraph_backward_compat),
        # attnPool (Phase 3A) tests
        ("40. HPG_attnPool forward", test_attnPool_forward),
        ("41. scorer=0 -> matches frac_weighted", test_attnPool_zero_init_matches_frac),
        ("42. Non-zero scorer -> diverges from frac", test_attnPool_diverges_nonzero),
        ("43. Batching isolation", test_attnPool_batching_isolation),
        # pairInteract (Phase 3B) tests
        ("44. HPG_pairInteract forward", test_pairInteract_forward),
        ("45. MLP output=0 -> matches frac_weighted", test_pairInteract_zero_init_matches_frac),
        ("46. Non-zero MLP -> diverges from frac", test_pairInteract_diverges_nonzero),
        ("47. Batching isolation (within-polymer only)", test_pairInteract_batching_isolation),
        # pairInteractAttn (Phase 3C) tests
        ("48. HPG_pairInteractAttn forward", test_pairInteractAttn_forward),
        ("49. phi=0 -> matches frac_weighted", test_pairInteractAttn_zero_init_matches_frac),
        ("50. Separate branch from pairInteract", test_pairInteractAttn_separate_from_pairInteract),
        # Phase 3 shared
        ("51. Phase 3 backward compat (prior variants unaffected)", test_phase3_backward_compat),
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
