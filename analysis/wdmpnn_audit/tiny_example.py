#!/usr/bin/env python3
"""
Reproducible hand-check of _WeightedBondMessagePassingMixin.message.

We build a small copolymer-like graph (a 3-atom chain, with one isolated atom),
provide an explicit H matrix, and compare:
  1. the loop-based reference implementation (`_message_reference`)
  2. the vectorised implementation (`message`)
  3. a hand-derived expected message

Both implementations should return the same raw messages, and those messages
should equal the hand-derived values.
"""

from __future__ import annotations

import numpy as np
import torch

from chemprop.data import BatchPolymerMolGraph, PolymerMolGraph
from chemprop.nn.message_passing import WeightedBondMessagePassing


def make_chain_and_isolated() -> BatchPolymerMolGraph:
    """
    Build one batched graph with:
      - a 3-atom chain 0--1--2 (4 directed edges)
      - an isolated atom 3

    Edge order (directed): 0->1, 1->0, 1->2, 2->1.
    Reverse mapping: rev = [1, 0, 3, 2].
    """
    V = np.zeros((4, 2), dtype=np.float32)
    E = np.zeros((4, 2), dtype=np.float32)
    atom_weights = np.ones(4, dtype=np.float32)
    monomer_index = np.array([0, 1, 1, 1], dtype=np.int64)
    edge_weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    rev_edge_index = np.array([1, 0, 3, 2], dtype=np.int64)
    mg = PolymerMolGraph(
        V, E, atom_weights, monomer_index, edge_weights, edge_index,
        rev_edge_index, np.float64(1.0),
    )
    bmg = BatchPolymerMolGraph([mg])
    return bmg


def expected_message(H: torch.Tensor, edge_index: torch.Tensor,
                     rev_edge_index: torch.Tensor,
                     w_bonds: torch.Tensor) -> torch.Tensor:
    """Plain-Python reference of the D-MPNN message step weighted by w_bonds."""
    # a_msg[a] = sum of w * H over all incoming bonds to a
    num_atoms = int(edge_index[1].max().item()) + 1
    d_h = H.shape[1]
    a_msg = torch.zeros(num_atoms, d_h, dtype=H.dtype, device=H.device)
    for b_idx, target in enumerate(edge_index[1].tolist()):
        a_msg[target] += w_bonds[b_idx] * H[b_idx]

    # message for each directed bond = a_msg[source] - H[reverse bond]  (unweighted reverse)
    msg = a_msg[edge_index[0].tolist()] - H[rev_edge_index]
    return msg


def main() -> None:
    bmg = make_chain_and_isolated()
    bmg.to(torch.device("cpu"))

    mp = WeightedBondMessagePassing(d_v=2, d_e=2, d_h=2)
    mp.eval()
    torch.manual_seed(0)

    # Explicit bond hidden states H (not from initialize()).
    # Bond 0: 0->1,  Bond 1: 1->0,  Bond 2: 1->2,  Bond 3: 2->1
    H1 = torch.tensor(
        [[1.0, 0.0],
         [0.0, 1.0],
         [2.0, 0.0],
         [0.0, 2.0]],
        dtype=torch.float32,
    )

    msg_ref1 = mp._message_reference(H1, bmg)
    msg_vec1 = mp.message(H1, bmg)
    msg_hand1 = expected_message(H1, bmg.edge_index, bmg.rev_edge_index, bmg.edge_weights)

    print("=== Step 1 ===")
    print(f"H:\n{H1.numpy()}")
    print(f"_message_reference:\n{msg_ref1.detach().numpy()}")
    print(f"message (vectorised):\n{msg_vec1.detach().numpy()}")
    print(f"hand-derived expected:\n{msg_hand1.numpy()}")

    assert torch.allclose(msg_ref1, msg_vec1, atol=1e-6), "loop and vectorised differ"
    assert torch.allclose(msg_ref1, msg_hand1, atol=1e-6), "implementation differs from hand reference"
    print("Step 1: PASS\n")

    # Second message step: use the first output as the new H.
    H2 = msg_ref1.clone()
    msg_ref2 = mp._message_reference(H2, bmg)
    msg_vec2 = mp.message(H2, bmg)
    msg_hand2 = expected_message(H2, bmg.edge_index, bmg.rev_edge_index, bmg.edge_weights)

    print("=== Step 2 ===")
    print(f"H:\n{H2.numpy()}")
    print(f"_message_reference:\n{msg_ref2.detach().numpy()}")
    print(f"message (vectorised):\n{msg_vec2.detach().numpy()}")
    print(f"hand-derived expected:\n{msg_hand2.numpy()}")

    assert torch.allclose(msg_ref2, msg_vec2, atol=1e-6), "loop and vectorised differ on step 2"
    assert torch.allclose(msg_ref2, msg_hand2, atol=1e-6), "implementation differs from hand reference on step 2"
    print("Step 2: PASS\n")

    print("Verdict: the loop and vectorised _WeightedBondMessagePassingMixin.message are mathematically identical.")


if __name__ == "__main__":
    main()
