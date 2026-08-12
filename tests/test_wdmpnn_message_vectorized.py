"""Regression tests for the vectorised _WeightedBondMessagePassingMixin.message.

These tests verify that the new vectorised/cached atom-to-incoming-bond mapping
produces numerically identical output to the original loop-based reference
implementation, including cases with:
  - atoms that have differing numbers of incoming bonds, and
  - an isolated atom with zero incoming bonds.
"""

import numpy as np
import pytest
import torch

from chemprop.data import BatchPolymerMolGraph, PolymerMolGraph
from chemprop.nn.message_passing import WeightedBondMessagePassing


def _make_test_batch(device: torch.device) -> BatchPolymerMolGraph:
    """Build a small batched polymer graph with non-trivial degree structure."""
    # Graph 1: 4 atoms, 4 directed edges over 2 undirected bonds.
    #   Directed edges: 0->1, 1->0, 2->1, 1->2
    #   Atom 3 is isolated (no incoming and no outgoing bonds).
    V1 = np.zeros((4, 5), dtype=np.float32)
    E1 = np.zeros((4, 6), dtype=np.float32)
    atom_weights1 = np.ones(4, dtype=np.float32)
    monomer_index1 = np.array([0, 0, 1, 1], dtype=np.int64)
    edge_weights1 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    edge_index1 = np.array([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=np.int64)
    rev_edge_index1 = np.array([1, 0, 3, 2], dtype=np.int64)
    mg1 = PolymerMolGraph(
        V1, E1, atom_weights1, monomer_index1, edge_weights1, edge_index1, rev_edge_index1, np.float64(1.0)
    )

    # Graph 2: 2 atoms, single undirected bond -> 2 directed edges.
    V2 = np.zeros((2, 5), dtype=np.float32)
    E2 = np.zeros((2, 6), dtype=np.float32)
    atom_weights2 = np.ones(2, dtype=np.float32)
    monomer_index2 = np.array([0, 1], dtype=np.int64)
    edge_weights2 = np.array([0.9, 1.0], dtype=np.float32)
    edge_index2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    rev_edge_index2 = np.array([1, 0], dtype=np.int64)
    mg2 = PolymerMolGraph(
        V2, E2, atom_weights2, monomer_index2, edge_weights2, edge_index2, rev_edge_index2, np.float64(1.0)
    )

    bmg = BatchPolymerMolGraph([mg1, mg2])
    bmg.to(device)
    return bmg


@pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def test_message_vectorized_matches_reference(device):
    """Vectorised message() must match the loop-based reference exactly."""
    device = torch.device(device)
    bmg = _make_test_batch(device)

    d_h = 8
    mp = WeightedBondMessagePassing(d_v=5, d_e=6, d_h=d_h).to(device)
    mp.eval()

    torch.manual_seed(123)
    H = torch.randn(bmg.edge_index.shape[1], d_h, device=device)

    msg_ref = mp._message_reference(H, bmg)
    msg_new = mp.message(H, bmg)

    assert torch.allclose(msg_ref, msg_new, atol=1e-6)


def test_message_cache_reuse_is_consistent():
    """Calling message() repeatedly on the same batch must reuse the cached map."""
    device = torch.device("cpu")
    bmg = _make_test_batch(device)

    d_h = 8
    mp = WeightedBondMessagePassing(d_v=5, d_e=6, d_h=d_h)

    torch.manual_seed(42)
    H1 = torch.randn(bmg.edge_index.shape[1], d_h)
    H2 = torch.randn(bmg.edge_index.shape[1], d_h)

    msg1 = mp.message(H1, bmg)
    msg2 = mp.message(H2, bmg)

    # Cache was populated on the first call.
    assert bmg._a2b_padded is not None
    assert bmg._a2b_mask is not None
    # Both calls are deterministic.
    assert torch.allclose(mp.message(H1, bmg), msg1, atol=1e-6)
    assert torch.allclose(mp.message(H2, bmg), msg2, atol=1e-6)


def test_empty_batch():
    """A batch with zero directed edges must not crash."""
    V = np.zeros((2, 5), dtype=np.float32)
    E = np.zeros((0, 6), dtype=np.float32)
    atom_weights = np.ones(2, dtype=np.float32)
    monomer_index = np.array([0, 1], dtype=np.int64)
    edge_weights = np.zeros(0, dtype=np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    rev_edge_index = np.zeros(0, dtype=np.int64)
    mg = PolymerMolGraph(
        V, E, atom_weights, monomer_index, edge_weights, edge_index, rev_edge_index, np.float64(1.0)
    )

    bmg = BatchPolymerMolGraph([mg])
    mp = WeightedBondMessagePassing(d_v=5, d_e=6, d_h=8)

    H = torch.randn(0, 8)
    msg_ref = mp._message_reference(H, bmg)
    msg_new = mp.message(H, bmg)

    assert torch.allclose(msg_ref, msg_new, atol=1e-6)
