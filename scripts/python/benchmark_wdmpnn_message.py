"""
Micro-benchmark for the wDMPNN message() bottleneck.

Runs a short synthetic training loop first with the old loop-based message
implementation (``mp._message_reference``) and then with the new vectorised,
cached implementation (``mp.message``).  Prints per-epoch wall time and the
resulting speedup.

This isolates the exact host-device synchronisation loop that dominated
wDMPNN wall time, so it is relevant even on CPU (where the Python loop is
still slow).  On CUDA the difference is typically much larger.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow running from scripts/python/ or repo root.
ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from chemprop.data import BatchPolymerMolGraph, PolymerMolGraph
from chemprop.nn.message_passing import WeightedBondMessagePassing


def _random_polymer_molgraph(
    num_atoms: int, num_bonds: int, d_v: int, d_e: int, rng: np.random.Generator
) -> PolymerMolGraph:
    """Create a random PolymerMolGraph with ``num_bonds`` directed edges."""
    V = rng.random((num_atoms, d_v)).astype(np.float32)
    E = rng.random((num_bonds, d_e)).astype(np.float32)
    atom_weights = rng.random(num_atoms).astype(np.float32)
    edge_weights = (0.5 + 0.5 * rng.random(num_bonds)).astype(np.float32)

    # Random source/target with self-loops disallowed for realism.
    src = rng.integers(0, num_atoms, size=num_bonds)
    dst = rng.integers(0, num_atoms, size=num_bonds)
    mask = src == dst
    # Simple fix: replace self-loops with an edge to a different atom.
    for i in np.nonzero(mask)[0]:
        dst[i] = (dst[i] + 1) % num_atoms
        if dst[i] == src[i]:
            dst[i] = (dst[i] + 1) % num_atoms
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    # Simplistic rev_edge_index (not chemically valid, but sufficient for the
    # message-passing benchmark because the reverse message is just indexed).
    rev_edge_index = np.arange(num_bonds - 1, -1, -1, dtype=np.int64)

    return PolymerMolGraph(
        V=V,
        E=E,
        atom_weights=atom_weights,
        edge_weights=edge_weights,
        edge_index=edge_index,
        rev_edge_index=rev_edge_index,
        degree_of_polym=np.float64(1.0),
    )


def _build_batches(
    n_batches: int,
    batch_size: int,
    num_atoms: int,
    num_bonds: int,
    d_v: int,
    d_e: int,
    device: torch.device,
    seed: int = 42,
) -> list[BatchPolymerMolGraph]:
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        mgs = [
            _random_polymer_molgraph(num_atoms, num_bonds, d_v, d_e, rng)
            for _ in range(batch_size)
        ]
        bmg = BatchPolymerMolGraph(mgs)
        bmg.to(device)
        batches.append(bmg)
    return batches


class _TinyReadout(nn.Module):
    """Sum node embeddings per graph and project to a scalar target."""

    def __init__(self, d_h: int, n_tasks: int = 1):
        super().__init__()
        self.head = nn.Linear(d_h, n_tasks)

    def forward(self, H_v: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        d = H_v.shape[1]
        index = batch.unsqueeze(1).expand(-1, d)
        pooled = torch.zeros(
            n_graphs, d, dtype=H_v.dtype, device=H_v.device
        ).scatter_reduce_(0, index, H_v, reduce="sum", include_self=False)
        return self.head(pooled)


def _make_model(d_v: int, d_e: int, d_h: int, device: torch.device):
    mp = WeightedBondMessagePassing(d_v=d_v, d_e=d_e, d_h=d_h).to(device)
    readout = _TinyReadout(d_h).to(device)
    return mp, readout


def _run_epoch(
    mp: WeightedBondMessagePassing,
    readout: _TinyReadout,
    batches: list[BatchPolymerMolGraph],
    optimizer: torch.optim.Optimizer,
) -> float:
    mp.train()
    readout.train()
    start = time.perf_counter()
    for bmg in batches:
        optimizer.zero_grad()
        H_v = mp(bmg)
        preds = readout(H_v, bmg.batch)
        targets = torch.randn_like(preds)
        loss = nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_atoms", type=int, default=40)
    parser.add_argument("--num_bonds", type=int, default=80)
    parser.add_argument("--d_v", type=int, default=72)
    parser.add_argument("--d_e", type=int, default=86)
    parser.add_argument("--d_h", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Benchmarking on device: {device}")
    print(
        f"Config: {args.epochs} epochs, {args.n_batches} batches/epoch, "
        f"batch_size={args.batch_size}, atoms/graph={args.num_atoms}, "
        f"directed bonds/graph={args.num_bonds}, d_h={args.d_h}"
    )

    batches = _build_batches(
        args.n_batches,
        args.batch_size,
        args.num_atoms,
        args.num_bonds,
        args.d_v,
        args.d_e,
        device,
    )

    # --- Old (loop-based) implementation ---
    mp_old, readout_old = _make_model(args.d_v, args.d_e, args.d_h, device)
    # Monkey-patch the public message() to the reference loop implementation.
    import types

    mp_old.message = types.MethodType(
        mp_old.__class__._message_reference, mp_old
    )
    optimizer_old = torch.optim.Adam(
        list(mp_old.parameters()) + list(readout_old.parameters()), lr=1e-3
    )

    # Warm-up
    _run_epoch(mp_old, readout_old, batches[:1], optimizer_old)
    old_times = [
        _run_epoch(mp_old, readout_old, batches, optimizer_old)
        for _ in range(args.epochs)
    ]
    old_per_epoch = sum(old_times) / len(old_times)

    # --- New (vectorised + cached) implementation ---
    mp_new, readout_new = _make_model(args.d_v, args.d_e, args.d_h, device)
    optimizer_new = torch.optim.Adam(
        list(mp_new.parameters()) + list(readout_new.parameters()), lr=1e-3
    )

    # Warm-up
    _run_epoch(mp_new, readout_new, batches[:1], optimizer_new)
    new_times = [
        _run_epoch(mp_new, readout_new, batches, optimizer_new)
        for _ in range(args.epochs)
    ]
    new_per_epoch = sum(new_times) / len(new_times)

    print("\nPer-epoch wall time:")
    print(f"  Old (loop) : {old_per_epoch:.4f} s")
    print(f"  New (vector): {new_per_epoch:.4f} s")
    print(f"  Speedup    : {old_per_epoch / new_per_epoch:.2f}x")
    print(f"  Epoch times old: {[f'{t:.4f}' for t in old_times]}")
    print(f"  Epoch times new: {[f'{t:.4f}' for t in new_times]}")


if __name__ == "__main__":
    main()
