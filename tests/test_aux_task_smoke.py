"""Smoke test for auxiliary descriptor prediction task.

Runs a single forward + backward pass with aux_task=predict_descriptors
to confirm:
  1. No crash
  2. Auxiliary head produces correct output shape
  3. Combined loss computes and gradients flow through both heads
  4. Backward compatibility: n_aux_targets=0 works identically to before

Usage:
    python tests/test_aux_task_smoke.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop import nn, models


def make_tiny_batch(n_mols: int = 4):
    """Create a tiny BatchMolGraph from dummy data."""
    mgs = []
    for _ in range(n_mols):
        n_atoms = np.random.randint(3, 6)
        n_bonds = n_atoms - 1
        d_v, d_e = 72, 14  # chemprop defaults

        V = np.random.randn(n_atoms, d_v).astype(np.float32)
        src, dst = [], []
        for b in range(n_bonds):
            src.extend([b, b + 1])
            dst.extend([b + 1, b])
        edge_index = np.array([src, dst], dtype=np.int64)
        n_directed = edge_index.shape[1]
        E = np.random.randn(n_directed, d_e).astype(np.float32)
        rev = np.zeros(n_directed, dtype=np.int64)
        for b in range(n_bonds):
            rev[2 * b] = 2 * b + 1
            rev[2 * b + 1] = 2 * b

        mg = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev)
        mgs.append(mg)
    return BatchMolGraph(mgs)


def test_aux_head_forward():
    """Test that aux head produces correct output shape."""
    print("Testing aux head forward ... ", end="", flush=True)
    B, d_h, T_aux = 4, 300, 3
    bmg = make_tiny_batch(B)

    mp = nn.BondMessagePassing(d_h=d_h)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn,
                       n_aux_targets=T_aux, lambda_aux=0.1)
    mpnn.eval()

    assert mpnn.aux_head is not None, "aux_head should exist when n_aux_targets > 0"
    assert mpnn.n_aux_targets == T_aux

    with torch.no_grad():
        Z = mpnn.fingerprint(bmg)
        # Main prediction
        y_hat = mpnn.predictor(Z)
        assert y_hat.shape == (B, 1), f"Main pred shape: {y_hat.shape}"
        # Aux prediction
        embed_dim = mp.output_dim
        d_hat = mpnn.aux_head(Z[:, :embed_dim])
        assert d_hat.shape == (B, T_aux), f"Aux pred shape: {d_hat.shape}"
    print(f"OK  (main={y_hat.shape}, aux={d_hat.shape})")


def test_aux_training_step():
    """Test that combined loss computes and gradients flow through both heads."""
    print("Testing aux training_step ... ", end="", flush=True)
    B, d_h, T_main, T_aux = 4, 300, 1, 2
    lambda_aux = 0.5
    bmg = make_tiny_batch(B)

    mp = nn.BondMessagePassing(d_h=d_h)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=T_main)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn,
                       n_aux_targets=T_aux, lambda_aux=lambda_aux)
    mpnn.train()

    # Simulate targets: [main_target, aux_target1, aux_target2]
    targets = torch.randn(B, T_main + T_aux)
    mask = targets.isfinite()
    targets_clean = targets.nan_to_num(nan=0.0)

    # Manually replicate training_step logic (avoids needing full Lightning Trainer)
    Z = mpnn.fingerprint(bmg)
    main_targets, aux_targets, main_mask, aux_mask, _, _ = \
        mpnn._split_targets(targets_clean, mask, None, None)

    preds = mpnn.predictor.train_step(Z)
    main_loss = mpnn.criterion(preds, main_targets, main_mask, torch.ones(B, 1), None, None)

    embed_dim = mp.output_dim
    d_hat = mpnn.aux_head(Z[:, :embed_dim])
    aux_loss = (((d_hat - aux_targets) ** 2) * aux_mask.float()).sum() / aux_mask.float().sum().clamp(min=1)

    loss = main_loss + lambda_aux * aux_loss

    assert loss.requires_grad, "Loss should require grad"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"

    loss.backward()

    # Check gradients flow through aux head
    aux_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in mpnn.aux_head.parameters()
    )
    assert aux_has_grad, "Aux head parameters should have non-zero gradients"

    # Check gradients flow through main predictor
    main_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in mpnn.predictor.parameters()
    )
    assert main_has_grad, "Main predictor parameters should have non-zero gradients"

    # Check gradients flow through message passing (shared encoder)
    mp_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in mpnn.message_passing.parameters()
    )
    assert mp_has_grad, "Message passing parameters should have non-zero gradients"

    print(f"OK  (loss={loss.item():.4f}, grads flow through all components)")


def test_aux_target_splitting():
    """Test that _split_targets correctly separates main and aux targets."""
    print("Testing target splitting ... ", end="", flush=True)
    B, T_main, T_aux = 4, 1, 3

    mp = nn.BondMessagePassing()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, n_tasks=T_main)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, n_aux_targets=T_aux)

    targets = torch.randn(B, T_main + T_aux)
    mask = torch.ones(B, T_main + T_aux, dtype=torch.bool)
    lt_mask = torch.zeros(B, T_main + T_aux, dtype=torch.bool)
    gt_mask = torch.zeros(B, T_main + T_aux, dtype=torch.bool)

    main_t, aux_t, main_m, aux_m, main_lt, main_gt = mpnn._split_targets(
        targets, mask, lt_mask, gt_mask
    )

    assert main_t.shape == (B, T_main), f"Main targets shape: {main_t.shape}"
    assert aux_t.shape == (B, T_aux), f"Aux targets shape: {aux_t.shape}"
    assert main_m.shape == (B, T_main), f"Main mask shape: {main_m.shape}"
    assert aux_m.shape == (B, T_aux), f"Aux mask shape: {aux_m.shape}"
    assert torch.allclose(targets[:, :T_main], main_t)
    assert torch.allclose(targets[:, T_main:], aux_t)
    print("OK")


def test_no_aux_backward_compat():
    """Test that n_aux_targets=0 works identically to before."""
    print("Testing backward compat (no aux) ... ", end="", flush=True)
    B, d_h = 4, 300
    bmg = make_tiny_batch(B)

    mp = nn.BondMessagePassing(d_h=d_h)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, n_aux_targets=0)

    assert mpnn.aux_head is None, "aux_head should be None when n_aux_targets=0"
    assert mpnn.n_aux_targets == 0

    mpnn.eval()
    with torch.no_grad():
        out = mpnn(bmg)
    assert out.shape == (B, 1), f"Output shape: {out.shape}"

    # Test _split_targets with no aux
    targets = torch.randn(B, 1)
    mask = torch.ones(B, 1, dtype=torch.bool)
    main_t, aux_t, main_m, aux_m, _, _ = mpnn._split_targets(
        targets, mask, None, None
    )
    assert main_t.shape == (B, 1)
    assert aux_t is None
    assert aux_m is None
    print(f"OK  (output shape: {out.shape})")


def test_aux_with_nan_targets():
    """Test that aux loss handles NaN (masked) targets correctly."""
    print("Testing aux with NaN targets ... ", end="", flush=True)
    B, d_h, T_main, T_aux = 4, 300, 1, 2
    bmg = make_tiny_batch(B)

    mp = nn.BondMessagePassing(d_h=d_h)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=T_main)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn,
                       n_aux_targets=T_aux, lambda_aux=0.1)
    mpnn.train()

    # Create targets with some NaN in aux columns
    targets = torch.randn(B, T_main + T_aux)
    targets[0, T_main] = float('nan')  # NaN in first aux target of first sample
    targets[2, T_main + 1] = float('nan')  # NaN in second aux target of third sample

    weights = torch.ones(B, 1)
    lt_mask = torch.zeros(B, T_main + T_aux, dtype=torch.bool)
    gt_mask = torch.zeros(B, T_main + T_aux, dtype=torch.bool)

    from chemprop.data.collate import TrainingBatch
    batch = TrainingBatch(bmg, None, None, targets, weights, lt_mask, gt_mask)

    loss = mpnn.training_step(batch, 0)
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    loss.backward()
    print(f"OK  (loss={loss.item():.4f}, finite with NaN targets)")


def test_forward_returns_main_only():
    """Test that forward() returns only main predictions, not aux."""
    print("Testing forward returns main only ... ", end="", flush=True)
    B, d_h, T_aux = 4, 300, 3
    bmg = make_tiny_batch(B)

    mp = nn.BondMessagePassing(d_h=d_h)
    ffn = nn.RegressionFFN(input_dim=d_h, n_tasks=1)
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn,
                       n_aux_targets=T_aux, lambda_aux=0.1)
    mpnn.eval()

    with torch.no_grad():
        out = mpnn(bmg)
    # forward() should return only main predictions (1 task), not aux
    assert out.shape == (B, 1), f"Expected (B, 1), got {out.shape}"
    print(f"OK  (output shape: {out.shape})")


if __name__ == "__main__":
    print("=" * 60)
    print("Auxiliary Task Smoke Tests")
    print("=" * 60)

    tests = [
        test_aux_head_forward,
        test_aux_training_step,
        test_aux_target_splitting,
        test_no_aux_backward_compat,
        test_aux_with_nan_targets,
        test_forward_returns_main_only,
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
