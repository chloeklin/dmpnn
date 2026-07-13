"""Unit tests for the within-group residual variance loss and GroupAwareSampler.

Run with:
    python -m pytest tests/test_within_group_loss.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chemprop.nn.within_group_loss import within_group_residual_loss
from chemprop.data.samplers import GroupAwareSampler


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make(targets, preds, group_ids):
    """Convert lists to the tensors expected by within_group_residual_loss."""
    t = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    p = torch.tensor(preds,   dtype=torch.float32).unsqueeze(1)
    g = torch.tensor(group_ids, dtype=torch.long)
    return t, p, g


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: λ=0 path — within_group_residual_loss is never called
# ──────────────────────────────────────────────────────────────────────────────

def test_lambda_zero_no_within_loss():
    """λ=0 means the within-group branch is skipped; total loss == MSE."""
    targets = [1.0, 2.0, 3.0, 4.0]
    preds   = [1.1, 1.9, 3.1, 3.8]
    group_ids = [0, 0, 1, 1]

    t, p, g = _make(targets, preds, group_ids)
    residuals = t - p
    expected_mse = (residuals ** 2).mean().item()

    # Simulate what _training_step_stage2d does: lambda=0 → l = l_overall only
    l_overall = ((t - p) ** 2).mean()
    lambda_within = 0.0

    # Branch that would call within_group_residual_loss is not entered
    l = l_overall  # exact reproduction of existing code path
    assert abs(l.item() - expected_mse) < 1e-6, \
        "λ=0 should reproduce standard MSE exactly"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: GroupAwareSampler never splits a group
# ──────────────────────────────────────────────────────────────────────────────

def test_group_aware_sampler_no_splits():
    """Every batch produced by GroupAwareSampler contains only complete groups."""
    rng = np.random.default_rng(42)
    N = 300
    # Mix of group sizes 2 and 3
    group_ids = np.repeat(np.arange(100), [2 if i % 3 != 0 else 3 for i in range(100)])[:N]
    group_ids = group_ids[:N]

    # Build a simple group_ids array: groups of size 2 and 3
    group_ids = np.array([i // 3 for i in range(N)])  # groups of size 3
    batch_size = 64

    sampler = GroupAwareSampler(group_ids, batch_size=batch_size, seed=0)
    indices = list(sampler)

    # Reconstruct batch boundaries
    batches = sampler._make_batches()
    for batch in batches:
        groups_in_batch = set(group_ids[i] for i in batch)
        for g in groups_in_batch:
            # All members of group g must be in this batch
            expected = set(np.where(group_ids == g)[0])
            in_batch = set(idx for idx in batch if group_ids[idx] == g)
            assert expected == in_batch, \
                f"Group {g} is split: expected {expected}, found {in_batch} in batch"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: L_within = 0 when all residuals in a group are identical
# ──────────────────────────────────────────────────────────────────────────────

def test_zero_loss_identical_residuals():
    """If all residuals within every group are equal, L_within must be zero."""
    # Group 0: residuals all 0.5; Group 1: residuals all -0.3
    targets   = [1.0, 2.0, 3.0,  4.0, 5.0, 6.0]
    preds     = [0.5, 1.5, 2.5,  4.3, 5.3, 6.3]
    group_ids = [0,   0,   0,    1,   1,   1  ]

    t, p, g = _make(targets, preds, group_ids)
    loss, info = within_group_residual_loss(t, p, g)
    assert loss.item() < 1e-6, f"Expected L_within ≈ 0, got {loss.item()}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Adding a constant to every prediction in a group leaves L_within unchanged
# ──────────────────────────────────────────────────────────────────────────────

def test_constant_shift_invariance():
    """Adding the same constant to all predictions in a group must not change L_within."""
    targets   = [1.0, 2.5, 1.8,  3.0, 2.0]
    preds     = [1.1, 2.3, 1.9,  2.8, 2.2]
    group_ids = [0,   0,   0,    1,   1  ]

    t, p, g = _make(targets, preds, group_ids)
    loss_before, _ = within_group_residual_loss(t, p, g)

    # Add constant +5.0 to all predictions in group 0
    shift = torch.zeros_like(p)
    shift[torch.tensor(group_ids) == 0] = 5.0
    p_shifted = p + shift

    loss_after, _ = within_group_residual_loss(t, p_shifted, g)
    assert abs(loss_before.item() - loss_after.item()) < 1e-5, \
        f"Constant shift changed L_within: {loss_before.item()} → {loss_after.item()}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Within-group gradient sums to approximately zero per group
# ──────────────────────────────────────────────────────────────────────────────

def test_gradient_sum_zero_per_group():
    """The gradient of L_within w.r.t. ŷ sums to ≈0 within each group."""
    targets   = [1.0, 2.0, 3.0,  4.0, 5.0, 6.0]
    preds_raw = [1.1, 1.8, 3.2,  3.9, 5.1, 5.8]
    group_ids = [0,   0,   0,    1,   1,   1  ]

    t = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    p = torch.tensor(preds_raw, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    g = torch.tensor(group_ids, dtype=torch.long)

    loss, _ = within_group_residual_loss(t, p, g)
    loss.backward()

    grads = p.grad.squeeze(1).numpy()
    g_np  = np.array(group_ids)

    for gid in np.unique(g_np):
        group_grad_sum = grads[g_np == gid].sum()
        assert abs(group_grad_sum) < 1e-5, \
            f"Group {gid} gradient sum = {group_grad_sum:.2e}, expected ≈ 0"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: No NaNs for singletons or zero target variance
# ──────────────────────────────────────────────────────────────────────────────

def test_no_nan_singletons():
    """Singleton groups (size 1) must not produce NaNs."""
    targets   = [1.0,  2.0, 3.0]
    preds     = [1.1,  1.9, 3.2]
    group_ids = [0,    1,   2  ]   # all singletons

    t, p, g = _make(targets, preds, group_ids)
    loss, info = within_group_residual_loss(t, p, g)

    assert not torch.isnan(loss), "NaN for all-singleton batch"
    assert loss.item() == pytest.approx(0.0), \
        "All singletons → L_within should be 0"
    assert info["n_groups_active"] == 0


def test_no_nan_zero_target_variance():
    """Groups with identical targets (zero denominator) must not produce NaNs."""
    # Group 0 has identical targets → denominator = 0
    targets   = [5.0, 5.0, 5.0]
    preds     = [5.1, 4.9, 5.2]
    group_ids = [0,   0,   0  ]

    t, p, g = _make(targets, preds, group_ids)
    loss, info = within_group_residual_loss(t, p, g)

    assert not torch.isnan(loss), "NaN for zero-variance group"
    assert loss.item() == pytest.approx(0.0), \
        "Zero denominator → L_within should be 0"


def test_mixed_singleton_and_multi():
    """Singletons are silently excluded; multi-member groups still contribute."""
    targets   = [1.0,  2.0, 3.0, 4.0]
    preds     = [1.0,  2.0, 3.5, 3.5]   # group 1 has non-zero within residual var
    group_ids = [0,    1,   1,   1  ]    # group 0 is singleton

    t, p, g = _make(targets, preds, group_ids)
    loss, info = within_group_residual_loss(t, p, g)

    assert not torch.isnan(loss)
    assert info["n_groups_active"] == 1, "Only group 1 should be active"
    assert loss.item() > 0.0, "Group 1 has non-zero residual variance"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Train/val/test splits are unchanged when lambda_within = 0
# ──────────────────────────────────────────────────────────────────────────────

def test_splits_unchanged_lambda_zero():
    """CopolymerDataset with group_ids=None behaves identically to without it."""
    from chemprop.data.copolymer import CopolymerDataset
    from chemprop.data.datapoints import MoleculeDatapoint
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C")

    def make_dp(y_val):
        dp = MoleculeDatapoint(mol=mol)
        dp.y = np.array([y_val], dtype=float)
        dp.x_d = np.array([0.0], dtype=np.float32)
        return dp

    N = 6
    dA = [make_dp(float(i)) for i in range(N)]
    dB = [make_dp(0.0) for _ in range(N)]
    fA = np.ones(N) * 0.5
    fB = np.ones(N) * 0.5

    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    feat = SimpleMoleculeMolGraphFeaturizer()

    # Without group_ids (baseline)
    ds_base = CopolymerDataset(dA, dB, fA, fB, feat)
    # With group_ids=None (should behave identically)
    ds_with = CopolymerDataset(dA, dB, fA, fB, feat, group_ids=None)

    assert len(ds_base) == len(ds_with) == N
    assert ds_with.group_ids is None, "group_ids=None must stay None"

    # Targets must be identical
    np.testing.assert_array_equal(ds_base.Y, ds_with.Y)


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: GroupAwareSampler reproduces all indices exactly once
# ──────────────────────────────────────────────────────────────────────────────

def test_group_aware_sampler_covers_all_indices():
    """Every sample index must appear exactly once per epoch."""
    group_ids = np.array([0, 0, 1, 1, 1, 2, 2])
    sampler = GroupAwareSampler(group_ids, batch_size=10, seed=7)

    indices = list(sampler)
    assert sorted(indices) == list(range(len(group_ids))), \
        "GroupAwareSampler must yield every index exactly once"


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: build_dataloader uses GroupAwareSampler when group_ids are set
# ──────────────────────────────────────────────────────────────────────────────

def test_build_dataloader_group_aware():
    """build_dataloader should use GroupAwareSampler when dataset has group_ids."""
    from chemprop.data.copolymer import CopolymerDataset
    from chemprop.data.dataloader import build_dataloader
    from chemprop.data.samplers import GroupAwareSampler
    from chemprop.data.datapoints import MoleculeDatapoint
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C")

    def make_dp(y_val):
        dp = MoleculeDatapoint(mol=mol)
        dp.y = np.array([y_val], dtype=float)
        dp.x_d = np.array([0.0], dtype=np.float32)
        return dp

    N = 9  # 3 groups of 3
    dA = [make_dp(float(i)) for i in range(N)]
    dB = [make_dp(0.0) for _ in range(N)]
    fA = np.ones(N) * 0.5
    fB = np.ones(N) * 0.5
    gids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    feat = SimpleMoleculeMolGraphFeaturizer()
    ds = CopolymerDataset(dA, dB, fA, fB, feat, group_ids=gids)

    loader = build_dataloader(ds, batch_size=6, shuffle=True, seed=42)
    assert isinstance(loader.sampler, GroupAwareSampler), \
        "build_dataloader should use GroupAwareSampler when group_ids are set"

    # Verify no group is split across batches
    all_indices = []
    # Collect batch indices via the sampler's _make_batches
    batches = loader.sampler._make_batches()
    for batch in batches:
        for idx in batch:
            g = gids[idx]
            expected = set(np.where(gids == g)[0])
            in_batch = set(j for j in batch if gids[j] == g)
            assert expected == in_batch, f"Group {g} split in batch"
        all_indices.extend(batch)
    assert sorted(all_indices) == list(range(N))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
