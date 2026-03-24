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

from chemprop.models.copolymer import CopolymerMPNN, MonomerScorer, PairInteractionMLP, PairScorer
from chemprop import nn as cpnn


def _make_model(mode: str, d_mp: int = 64, descriptor_dim: int = 0):
    """Build a minimal CopolymerMPNN for testing."""
    mp = cpnn.BondMessagePassing(d_h=d_mp)
    agg = cpnn.MeanAggregation()
    d_mp_actual = mp.output_dim

    # Compute FFN input dim
    if mode.startswith("mean") or mode.startswith("mix") and "frac" not in mode:
        ffn_dim = d_mp_actual
    elif mode.startswith("attention") or mode.startswith("frac_attn"):
        ffn_dim = d_mp_actual
    elif mode.startswith("interact"):
        ffn_dim = 4 * d_mp_actual + 2
    elif "frac" in mode:
        ffn_dim = d_mp_actual + 2
    else:
        ffn_dim = d_mp_actual

    if mode.endswith("_meta"):
        ffn_dim += descriptor_dim

    ffn = cpnn.RegressionFFN(input_dim=ffn_dim, n_tasks=1)
    model = CopolymerMPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        copolymer_mode=mode,
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
    modes_same_dim = ["mean", "mix", "attention", "frac_attn", "frac_attn_pair", "frac_attn_pair_attn"]
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
    print("\n=== All sanity checks passed ===")
