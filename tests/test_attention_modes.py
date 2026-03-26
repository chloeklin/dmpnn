"""Sanity checks for attention, frac_attn, and pairwise copolymer fusion modes.

Verifies:
1. Attention weights sum to 1 across monomers.
2. frac_attn changes weights when fractions change.
3. Outputs have the same embedding dimension as other fusion strategies.
4. Permutation invariance: swapping (z_A, fA) ↔ (z_B, fB) yields same output.
5. Pairwise pair-attention beta_ij sums to 1 across valid pairs.
6. Pairwise h_int changes when pair fractions change.
7. Pairwise modes produce same output dim as other d_mp modes.
8. Pairwise modes are permutation-invariant.
"""
import sys
from pathlib import Path

import torch

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemprop.models.copolymer import (
    CopolymerMPNN, MonomerScorer, PairInteractionMLP, PairScorer,
    SelfAttentionPooling,
    VALID_FUSION_TYPES, SumFusion, ConcatFusion, GatedFusion, ScalarResidualFusion,
)
from chemprop import nn as cpnn


def _make_model(mode: str, d_mp: int = 64, descriptor_dim: int = 0, fusion_type: str = "sum_fusion"):
    """Build a minimal CopolymerMPNN for testing."""
    mp = cpnn.BondMessagePassing(d_h=d_mp)
    agg = cpnn.MeanAggregation()
    d_mp_actual = mp.output_dim

    # Compute FFN input dim
    if mode.startswith("interact"):
        ffn_dim = 4 * d_mp_actual + 2
    elif mode.startswith("mix_frac"):
        ffn_dim = d_mp_actual + 2
    else:
        # mean, mix, mix_pair, mix_pair_attn, attention, frac_attn,
        # frac_attn_pair, frac_attn_pair_attn, self_attn all output d_mp
        ffn_dim = d_mp_actual

    if mode.endswith("_meta"):
        ffn_dim += descriptor_dim

    ffn = cpnn.RegressionFFN(input_dim=ffn_dim, n_tasks=1)
    model = CopolymerMPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        copolymer_mode=mode,
        fusion_type=fusion_type,
        batch_norm=False,
    )
    model.eval()
    return model, d_mp_actual


def test_attention_weights_sum_to_one():
    """attention: softmax weights must sum to 1."""
    model, d = _make_model("attention")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)

    H = torch.stack([z_A, z_B], dim=1)  # [B, 2, d]
    scores = model.monomer_scorer(H).squeeze(-1)  # [B, 2]
    alpha = torch.softmax(scores, dim=1)

    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-6), \
        f"Attention weights don't sum to 1: {sums}"
    print("PASS: attention weights sum to 1")


def test_frac_attn_weights_sum_to_one():
    """frac_attn: softmax weights must sum to 1."""
    model, d = _make_model("frac_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B, 1), 0.3)
    fB = torch.full((B, 1), 0.7)

    H = torch.stack([z_A, z_B], dim=1)
    scores = model.monomer_scorer(H).squeeze(-1)
    F = torch.cat([fA, fB], dim=1)
    logits = scores + torch.log(F + 1e-8)
    alpha = torch.softmax(logits, dim=1)

    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-6), \
        f"frac_attn weights don't sum to 1: {sums}"
    print("PASS: frac_attn weights sum to 1")


def test_frac_attn_changes_with_fractions():
    """frac_attn weights should change when fractions change."""
    model, d = _make_model("frac_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)

    # Run with fracA=0.3
    fA_1 = torch.full((B,), 0.3)
    fB_1 = torch.full((B,), 0.7)
    with torch.no_grad():
        out1 = model._apply_mode(z_A, z_B, fA_1, fB_1)

    # Run with fracA=0.9
    fA_2 = torch.full((B,), 0.9)
    fB_2 = torch.full((B,), 0.1)
    with torch.no_grad():
        out2 = model._apply_mode(z_A, z_B, fA_2, fB_2)

    assert not torch.allclose(out1, out2, atol=1e-6), \
        "frac_attn output should change when fractions change"
    print("PASS: frac_attn changes weights when fractions change")


def test_output_dim_consistency():
    """All fusion modes that map to d_mp should produce the same embedding width."""
    d_mp_target = 64
    modes_same_dim = ["mean", "mix", "mix_pair", "mix_pair_attn",
                      "attention", "frac_attn", "frac_attn_pair", "frac_attn_pair_attn",
                      "self_attn"]
    B = 2
    dims = {}

    for mode in modes_same_dim:
        model, d = _make_model(mode, d_mp=d_mp_target)
        z_A = torch.randn(B, d)
        z_B = torch.randn(B, d)
        fA = torch.full((B,), 0.5)
        fB = torch.full((B,), 0.5)
        with torch.no_grad():
            out = model._apply_mode(z_A, z_B, fA, fB)
        dims[mode] = out.shape[1]

    values = list(dims.values())
    assert all(v == values[0] for v in values), \
        f"Output dims differ across modes: {dims}"
    print(f"PASS: all base modes produce same embedding dim = {values[0]}: {dims}")


def test_permutation_invariance_attention():
    """Swapping (z_A, fA) ↔ (z_B, fB) should give the same fused output."""
    model, d = _make_model("frac_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)

    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)

    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"frac_attn is not permutation-invariant.\nmax diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: frac_attn is permutation-invariant")


def test_permutation_invariance_pure_attention():
    """Pure attention should also be permutation-invariant."""
    model, d = _make_model("attention")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.5)
    fB = torch.full((B,), 0.5)

    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)

    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"attention is not permutation-invariant.\nmax diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: attention is permutation-invariant")


def test_pair_attn_beta_sums_to_one():
    """frac_attn_pair_attn: pair attention weights beta must sum to 1."""
    model, d = _make_model("frac_attn_pair_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B, 1), 0.3)
    fB = torch.full((B, 1), 0.7)

    H = torch.stack([z_A, z_B], dim=1)  # [B, 2, d]
    F = torch.cat([fA, fB], dim=1)       # [B, 2]

    pair_logits = []
    for i in range(2):
        for j in range(i + 1, 2):
            pair_feat = torch.cat([
                H[:, i] + H[:, j],
                H[:, i] * H[:, j],
                (H[:, i] - H[:, j]).abs(),
            ], dim=1)
            t_ij = model.pair_scorer(pair_feat).squeeze(-1)
            f_ij = F[:, i] * F[:, j]
            pair_logits.append(t_ij + torch.log(f_ij + 1e-8))

    logits = torch.stack(pair_logits, dim=1)  # [B, P]
    beta = torch.softmax(logits, dim=1)
    sums = beta.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-6), \
        f"Pair attention weights don't sum to 1: {sums}"
    print("PASS: frac_attn_pair_attn beta sums to 1")


def test_pairwise_fixed_changes_with_fractions():
    """frac_attn_pair h_int should change when pair fractions change."""
    model, d = _make_model("frac_attn_pair")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)

    fA_1 = torch.full((B,), 0.3)
    fB_1 = torch.full((B,), 0.7)
    with torch.no_grad():
        out1 = model._apply_mode(z_A, z_B, fA_1, fB_1)

    fA_2 = torch.full((B,), 0.9)
    fB_2 = torch.full((B,), 0.1)
    with torch.no_grad():
        out2 = model._apply_mode(z_A, z_B, fA_2, fB_2)

    assert not torch.allclose(out1, out2, atol=1e-6), \
        "frac_attn_pair output should change when fractions change"
    print("PASS: frac_attn_pair changes with fractions")


def test_pairwise_attn_changes_with_fractions():
    """frac_attn_pair_attn h_int should change when pair fractions change."""
    model, d = _make_model("frac_attn_pair_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)

    fA_1 = torch.full((B,), 0.3)
    fB_1 = torch.full((B,), 0.7)
    with torch.no_grad():
        out1 = model._apply_mode(z_A, z_B, fA_1, fB_1)

    fA_2 = torch.full((B,), 0.9)
    fB_2 = torch.full((B,), 0.1)
    with torch.no_grad():
        out2 = model._apply_mode(z_A, z_B, fA_2, fB_2)

    assert not torch.allclose(out1, out2, atol=1e-6), \
        "frac_attn_pair_attn output should change when fractions change"
    print("PASS: frac_attn_pair_attn changes with fractions")


def test_permutation_invariance_pairwise_fixed():
    """frac_attn_pair: swapping (z_A, fA) <-> (z_B, fB) should give same output."""
    model, d = _make_model("frac_attn_pair")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)

    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)

    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"frac_attn_pair not permutation-invariant. max diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: frac_attn_pair is permutation-invariant")


def test_permutation_invariance_pairwise_attn():
    """frac_attn_pair_attn: swapping (z_A, fA) <-> (z_B, fB) should give same output."""
    model, d = _make_model("frac_attn_pair_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)

    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)

    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"frac_attn_pair_attn not permutation-invariant. max diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: frac_attn_pair_attn is permutation-invariant")


def test_mix_pair_changes_with_fractions():
    """mix_pair h_int should change when pair fractions change."""
    model, d = _make_model("mix_pair")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)

    fA_1 = torch.full((B,), 0.3)
    fB_1 = torch.full((B,), 0.7)
    with torch.no_grad():
        out1 = model._apply_mode(z_A, z_B, fA_1, fB_1)

    fA_2 = torch.full((B,), 0.9)
    fB_2 = torch.full((B,), 0.1)
    with torch.no_grad():
        out2 = model._apply_mode(z_A, z_B, fA_2, fB_2)

    assert not torch.allclose(out1, out2, atol=1e-6), \
        "mix_pair output should change when fractions change"
    print("PASS: mix_pair changes with fractions")


def test_mix_pair_attn_changes_with_fractions():
    """mix_pair_attn h_int should change when pair fractions change."""
    model, d = _make_model("mix_pair_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)

    fA_1 = torch.full((B,), 0.3)
    fB_1 = torch.full((B,), 0.7)
    with torch.no_grad():
        out1 = model._apply_mode(z_A, z_B, fA_1, fB_1)

    fA_2 = torch.full((B,), 0.9)
    fB_2 = torch.full((B,), 0.1)
    with torch.no_grad():
        out2 = model._apply_mode(z_A, z_B, fA_2, fB_2)

    assert not torch.allclose(out1, out2, atol=1e-6), \
        "mix_pair_attn output should change when fractions change"
    print("PASS: mix_pair_attn changes with fractions")


def test_permutation_invariance_mix_pair():
    """mix_pair: swapping (z_A, fA) <-> (z_B, fB) should give same output."""
    model, d = _make_model("mix_pair")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)

    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)

    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"mix_pair not permutation-invariant. max diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: mix_pair is permutation-invariant")


def test_permutation_invariance_mix_pair_attn():
    """mix_pair_attn: swapping (z_A, fA) <-> (z_B, fB) should give same output."""
    model, d = _make_model("mix_pair_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)

    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)

    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"mix_pair_attn not permutation-invariant. max diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: mix_pair_attn is permutation-invariant")


# ===================== FUSION STRATEGY TESTS =====================


def test_fusion_output_dim_all_types():
    """All fusion types must produce the same output dim (d_mp) for mix_pair_attn."""
    for ft in VALID_FUSION_TYPES:
        model, d = _make_model("mix_pair_attn", fusion_type=ft)
        B = 4
        z_A = torch.randn(B, d)
        z_B = torch.randn(B, d)
        fA = torch.full((B,), 0.3)
        fB = torch.full((B,), 0.7)
        with torch.no_grad():
            out = model._apply_mode(z_A, z_B, fA, fB)
        assert out.shape == (B, d), \
            f"{ft}: expected shape ({B}, {d}), got {out.shape}"
        print(f"PASS: {ft} output dim = {d}")


def test_sum_fusion_matches_original():
    """sum_fusion must exactly reproduce the old h_mix + h_int behavior."""
    torch.manual_seed(42)
    model, d = _make_model("mix_pair_attn", fusion_type="sum_fusion")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)

    with torch.no_grad():
        out = model._apply_mode(z_A, z_B, fA, fB)
        # Manually compute h_mix + h_int
        fA_col = fA.unsqueeze(1)
        fB_col = fB.unsqueeze(1)
        H = torch.stack([z_A, z_B], dim=1)
        F = torch.cat([fA_col, fB_col], dim=1)
        h_mix = (F.unsqueeze(-1) * H).sum(dim=1)
        pair_feat = torch.cat([z_A + z_B, z_A * z_B, (z_A - z_B).abs()], dim=1)
        phi = model.pair_mlp(pair_feat)
        t = model.pair_scorer(pair_feat).squeeze(-1)
        f_ij = fA * fB
        logit = t + torch.log(f_ij + 1e-8)
        beta = torch.softmax(logit.unsqueeze(1), dim=1)
        h_int = (beta.unsqueeze(-1) * phi.unsqueeze(1)).sum(dim=1)
        expected = h_mix + h_int

    assert torch.allclose(out, expected, atol=1e-5), \
        f"sum_fusion should match h_mix + h_int. max diff={torch.max(torch.abs(out - expected))}"
    print("PASS: sum_fusion reproduces original behavior")


def test_scalar_residual_fusion_lambda_stored():
    """scalar_residual_fusion: lambda should be a learnable parameter saved in state_dict."""
    model, d = _make_model("mix_pair_attn", fusion_type="scalar_residual_fusion")
    assert hasattr(model.fuse, 'lam'), "ScalarResidualFusion should have a 'lam' parameter"
    assert model.fuse.lam.requires_grad, "lambda must be learnable"
    # Check lambda appears in state_dict
    sd = model.state_dict()
    lam_keys = [k for k in sd if 'lam' in k]
    assert len(lam_keys) > 0, "lambda should appear in state_dict for checkpoint saving"
    print(f"PASS: scalar_residual_fusion lambda stored (init={model.fuse.lam.item():.3f}), keys={lam_keys}")


def test_concat_fusion_no_shape_mismatch():
    """concat_fusion: [h_mix || h_int] -> MLP -> d should run without error."""
    model, d = _make_model("mix_pair", fusion_type="concat_fusion")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.5)
    fB = torch.full((B,), 0.5)
    with torch.no_grad():
        out = model._apply_mode(z_A, z_B, fA, fB)
    assert out.shape == (B, d)
    print("PASS: concat_fusion runs without shape mismatch")


def test_gated_fusion_no_shape_mismatch():
    """gated_fusion: g=sigma(MLP), h=(1-g)*h_mix + g*h_int should run without error."""
    model, d = _make_model("mix_pair", fusion_type="gated_fusion")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.5)
    fB = torch.full((B,), 0.5)
    with torch.no_grad():
        out = model._apply_mode(z_A, z_B, fA, fB)
    assert out.shape == (B, d)
    print("PASS: gated_fusion runs without shape mismatch")


def test_fusion_permutation_invariance_all_types():
    """All fusion types should preserve permutation invariance of pairwise modes."""
    for ft in VALID_FUSION_TYPES:
        model, d = _make_model("mix_pair_attn", fusion_type=ft)
        B = 4
        z_A = torch.randn(B, d)
        z_B = torch.randn(B, d)
        fA = torch.full((B,), 0.3)
        fB = torch.full((B,), 0.7)
        with torch.no_grad():
            out_AB = model._apply_mode(z_A, z_B, fA, fB)
            out_BA = model._apply_mode(z_B, z_A, fB, fA)
        assert torch.allclose(out_AB, out_BA, atol=1e-5), \
            f"{ft} not permutation-invariant. max diff={torch.max(torch.abs(out_AB - out_BA))}"
        print(f"PASS: {ft} is permutation-invariant")


def test_fusion_type_in_hparams():
    """fusion_type should be stored in model hparams for checkpoint reproducibility."""
    for ft in VALID_FUSION_TYPES:
        model, _ = _make_model("mix_pair_attn", fusion_type=ft)
        assert model.fusion_type == ft, f"Expected fusion_type={ft}, got {model.fusion_type}"
    print("PASS: fusion_type correctly stored on model")


# ================================================================
# Self-attention tests
# ================================================================

def test_self_attn_output_dim():
    """self_attn should output d_mp (same as mix, frac_attn, etc.)."""
    model, d = _make_model("self_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.6)
    fB = torch.full((B,), 0.4)
    with torch.no_grad():
        out = model._apply_mode(z_A, z_B, fA, fB)
    assert out.shape == (B, d), f"Expected ({B}, {d}), got {out.shape}"
    print(f"PASS: self_attn output dim = {d}")


def test_self_attn_meta_output_dim():
    """self_attn_meta should output d_mp + descriptor_dim."""
    desc_dim = 5
    model, d = _make_model("self_attn_meta", descriptor_dim=desc_dim)
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.6)
    fB = torch.full((B,), 0.4)
    X_d = torch.randn(B, desc_dim)
    with torch.no_grad():
        out = model._apply_mode(z_A, z_B, fA, fB, X_d)
    expected = d + desc_dim
    assert out.shape == (B, expected), f"Expected ({B}, {expected}), got {out.shape}"
    print(f"PASS: self_attn_meta output dim = {expected}")


def test_self_attn_attention_weights_sum_to_one():
    """Self-attention softmax rows should sum to 1."""
    d = 64
    pool = SelfAttentionPooling(d)
    pool.eval()
    B, N = 4, 2
    H = torch.randn(B, N, d)
    F = torch.tensor([[0.6, 0.4]] * B)
    with torch.no_grad():
        Q = pool.W_Q(H)
        K = pool.W_K(H)
        scores = torch.bmm(Q, K.transpose(1, 2)) / pool.scale
        frac_bias = torch.log(F + 1e-8).unsqueeze(1)
        scores = scores + frac_bias
        alpha = torch.softmax(scores, dim=-1)
    row_sums = alpha.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"Attention rows don't sum to 1: {row_sums}"
    print("PASS: self_attn attention weights sum to 1")


def test_self_attn_changes_with_fractions():
    """Changing fractions should change self_attn output."""
    model, d = _make_model("self_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    with torch.no_grad():
        out1 = model._apply_mode(z_A, z_B, torch.full((B,), 0.5), torch.full((B,), 0.5))
        out2 = model._apply_mode(z_A, z_B, torch.full((B,), 0.8), torch.full((B,), 0.2))
    assert not torch.allclose(out1, out2, atol=1e-5), \
        "self_attn output did not change when fractions changed"
    print("PASS: self_attn changes with fractions")


def test_self_attn_permutation_invariance():
    """Swapping (z_A, fA) <-> (z_B, fB) should yield the same output."""
    model, d = _make_model("self_attn")
    B = 4
    z_A = torch.randn(B, d)
    z_B = torch.randn(B, d)
    fA = torch.full((B,), 0.3)
    fB = torch.full((B,), 0.7)
    with torch.no_grad():
        out_AB = model._apply_mode(z_A, z_B, fA, fB)
        out_BA = model._apply_mode(z_B, z_A, fB, fA)
    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"self_attn not permutation-invariant. max diff={torch.max(torch.abs(out_AB - out_BA))}"
    print("PASS: self_attn is permutation-invariant")


def test_self_attn_has_qkv_params():
    """self_attn model should have W_Q, W_K, W_V parameters."""
    model, d = _make_model("self_attn")
    param_names = [n for n, _ in model.named_parameters()]
    for proj in ["self_attn_pool.W_Q.weight", "self_attn_pool.W_K.weight", "self_attn_pool.W_V.weight"]:
        assert proj in param_names, f"Missing parameter: {proj}"
    print("PASS: self_attn has W_Q, W_K, W_V parameters")


def test_self_attn_no_fusion_module():
    """self_attn should NOT build a fusion module (no h_mix/h_int split)."""
    model, _ = _make_model("self_attn")
    assert model.fuse is None, f"Expected fuse=None, got {model.fuse}"
    assert model.pair_mlp is None, f"Expected pair_mlp=None, got {model.pair_mlp}"
    assert model.pair_scorer is None, f"Expected pair_scorer=None, got {model.pair_scorer}"
    assert model.monomer_scorer is None, f"Expected monomer_scorer=None, got {model.monomer_scorer}"
    print("PASS: self_attn has no fusion/pairwise/scorer modules")


if __name__ == "__main__":
    test_attention_weights_sum_to_one()
    test_frac_attn_weights_sum_to_one()
    test_frac_attn_changes_with_fractions()
    test_output_dim_consistency()
    test_permutation_invariance_attention()
    test_permutation_invariance_pure_attention()
    test_pair_attn_beta_sums_to_one()
    test_pairwise_fixed_changes_with_fractions()
    test_pairwise_attn_changes_with_fractions()
    test_permutation_invariance_pairwise_fixed()
    test_permutation_invariance_pairwise_attn()
    test_mix_pair_changes_with_fractions()
    test_mix_pair_attn_changes_with_fractions()
    test_permutation_invariance_mix_pair()
    test_permutation_invariance_mix_pair_attn()
    # Fusion strategy tests
    test_fusion_output_dim_all_types()
    test_sum_fusion_matches_original()
    test_scalar_residual_fusion_lambda_stored()
    test_concat_fusion_no_shape_mismatch()
    test_gated_fusion_no_shape_mismatch()
    test_fusion_permutation_invariance_all_types()
    test_fusion_type_in_hparams()
    # Self-attention tests
    test_self_attn_output_dim()
    test_self_attn_meta_output_dim()
    test_self_attn_attention_weights_sum_to_one()
    test_self_attn_changes_with_fractions()
    test_self_attn_permutation_invariance()
    test_self_attn_has_qkv_params()
    test_self_attn_no_fusion_module()
    print("\n=== All sanity checks passed ===")
