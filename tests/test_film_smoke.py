"""Smoke test for FiLM (Feature-wise Linear Modulation) conditioning.

Runs a single forward pass with fusion_mode=film on a small batch for each
supported model type (DMPNN, GIN, GAT, GATv2) and confirms:
  1. No crash
  2. Output shape matches baseline (fusion_mode=late_concat)

Usage:
    python tests/test_film_smoke.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.nn.film import FilmConditioner
from chemprop import nn, models


def make_tiny_batch(n_mols: int = 4):
    """Create a tiny BatchMolGraph from dummy SMILES-like data."""
    mgs = []
    for _ in range(n_mols):
        n_atoms = np.random.randint(3, 6)
        n_bonds = n_atoms - 1  # tree-like
        d_v, d_e = 72, 14  # must match chemprop DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM

        V = np.random.randn(n_atoms, d_v).astype(np.float32)
        # Directed edges: for each bond (i, i+1), create both directions
        src = []
        dst = []
        for b in range(n_bonds):
            src.extend([b, b + 1])
            dst.extend([b + 1, b])
        edge_index = np.array([src, dst], dtype=np.int64)
        n_directed = edge_index.shape[1]
        E = np.random.randn(n_directed, d_e).astype(np.float32)
        # rev_edge_index: for each directed edge pair (fwd, rev), map to its reverse
        rev = np.zeros(n_directed, dtype=np.int64)
        for b in range(n_bonds):
            rev[2 * b] = 2 * b + 1
            rev[2 * b + 1] = 2 * b

        mg = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev)
        mgs.append(mg)

    return BatchMolGraph(mgs)


def test_dmpnn_film():
    """Test FiLM with BondMessagePassing (DMPNN)."""
    print("Testing DMPNN + FiLM ... ", end="", flush=True)
    B, D, d_h = 4, 10, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)

    # Baseline: late_concat
    mp_base = nn.BondMessagePassing(d_h=d_h)
    agg = nn.MeanAggregation()
    ffn_base = nn.RegressionFFN(input_dim=d_h + D, n_tasks=1)
    mpnn_base = models.MPNN(mp_base, agg, ffn_base, fusion_mode="late_concat")
    mpnn_base.eval()
    with torch.no_grad():
        out_base = mpnn_base(bmg, X_d=X_d)

    # FiLM mode
    film = FilmConditioner(d_descriptor=D, d_hidden=d_h, n_layers=2, film_layers_mode="all")
    mp_film = nn.BondMessagePassing(d_h=d_h)
    mp_film.film_conditioner = film
    ffn_film = nn.RegressionFFN(input_dim=d_h, n_tasks=1)  # no +D
    mpnn_film = models.MPNN(mp_film, nn.MeanAggregation(), ffn_film, fusion_mode="film")
    mpnn_film.eval()
    with torch.no_grad():
        out_film = mpnn_film(bmg, X_d=X_d)

    assert out_base.shape == out_film.shape == (B, 1), \
        f"Shape mismatch: base={out_base.shape}, film={out_film.shape}"
    print(f"OK  (output shape: {out_film.shape})")


def test_gin_film():
    """Test FiLM with GINMessagePassing."""
    print("Testing GIN + FiLM ... ", end="", flush=True)
    B, D, d_h = 4, 8, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)

    film = FilmConditioner(d_descriptor=D, d_hidden=d_h, n_layers=3, film_layers_mode="all")
    mp = nn.GINMessagePassing(d_h=d_h, film_conditioner=film)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="film")
    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg, X_d=X_d)
    assert out.shape == (B, 1), f"Shape mismatch: {out.shape}"
    print(f"OK  (output shape: {out.shape})")


def test_gat_film():
    """Test FiLM with GATMessagePassing."""
    print("Testing GAT + FiLM ... ", end="", flush=True)
    B, D, d_h = 4, 6, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)

    film = FilmConditioner(d_descriptor=D, d_hidden=d_h, n_layers=3, film_layers_mode="all")
    mp = nn.GATMessagePassing(d_h=d_h, film_conditioner=film)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="film")
    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg, X_d=X_d)
    assert out.shape == (B, 1), f"Shape mismatch: {out.shape}"
    print(f"OK  (output shape: {out.shape})")


def test_gatv2_film():
    """Test FiLM with GATv2MessagePassing."""
    print("Testing GATv2 + FiLM ... ", end="", flush=True)
    B, D, d_h = 4, 6, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)

    film = FilmConditioner(d_descriptor=D, d_hidden=d_h, n_layers=3, film_layers_mode="all")
    mp = nn.GATv2MessagePassing(d_h=d_h, film_conditioner=film)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="film")
    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg, X_d=X_d)
    assert out.shape == (B, 1), f"Shape mismatch: {out.shape}"
    print(f"OK  (output shape: {out.shape})")


def test_film_last_only():
    """Test FiLM with film_layers='last' (only last MP layer modulated)."""
    print("Testing DMPNN + FiLM (last only) ... ", end="", flush=True)
    B, D, d_h = 4, 10, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)

    film = FilmConditioner(d_descriptor=D, d_hidden=d_h, n_layers=2, film_layers_mode="last")
    mp = nn.BondMessagePassing(d_h=d_h)
    mp.film_conditioner = film
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="film")
    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg, X_d=X_d)
    assert out.shape == (B, 1), f"Shape mismatch: {out.shape}"
    print(f"OK  (output shape: {out.shape})")


def test_backward_compat_no_film():
    """Test that fusion_mode='late_concat' (default) still works identically."""
    print("Testing backward compat (no FiLM) ... ", end="", flush=True)
    B, D, d_h = 4, 10, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)

    mp = nn.BondMessagePassing(d_h=d_h)
    assert mp.film_conditioner is None, "film_conditioner should be None by default"
    ffn = nn.RegressionFFN(input_dim=d_h + D, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="late_concat")
    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg, X_d=X_d)
    assert out.shape == (B, 1), f"Shape mismatch: {out.shape}"
    print(f"OK  (output shape: {out.shape})")


def test_no_descriptors():
    """Test fusion_mode='none' with no descriptors."""
    print("Testing no descriptors (fusion_mode=none) ... ", end="", flush=True)
    B, d_h = 4, 300
    bmg = make_tiny_batch(B)

    mp = nn.BondMessagePassing(d_h=d_h)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="none")
    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg)
    assert out.shape == (B, 1), f"Shape mismatch: {out.shape}"
    print(f"OK  (output shape: {out.shape})")


def test_film_gradient_flow():
    """Test that gradients flow through FiLM parameters."""
    print("Testing FiLM gradient flow ... ", end="", flush=True)
    B, D, d_h = 4, 10, 300
    bmg = make_tiny_batch(B)
    X_d = torch.randn(B, D)
    targets = torch.randn(B, 1)

    film = FilmConditioner(d_descriptor=D, d_hidden=d_h, n_layers=2, film_layers_mode="all")
    mp = nn.BondMessagePassing(d_h=d_h)
    mp.film_conditioner = film
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, fusion_mode="film")
    mpnn.train()

    out = mpnn(bmg, X_d=X_d)
    loss = (out - targets).pow(2).mean()
    loss.backward()

    # Check that FiLM parameters have gradients
    has_grad = False
    for name, param in film.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "FiLM parameters should have non-zero gradients"
    print("OK  (gradients flow through FiLM)")


if __name__ == "__main__":
    print("=" * 60)
    print("FiLM Conditioning Smoke Tests")
    print("=" * 60)

    tests = [
        test_dmpnn_film,
        test_gin_film,
        test_gat_film,
        test_gatv2_film,
        test_film_last_only,
        test_backward_compat_no_film,
        test_no_descriptors,
        test_film_gradient_flow,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")
