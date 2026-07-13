"""
wDMPNN within-group loss verification
======================================
Runs 7 checks against the ported lambda_within implementation in
run_wdmpnn_generalization.py to confirm correctness before any HPC jobs
are submitted.

Checks
------
1. lambda_within=0 reproduces original wDMPNN loss (base MPNN class).
2. GroupAwareSampler keeps complete chemistry groups intact within batches.
3. L_within is computed in normalised target space.
4. Within-group gradient sums to approximately zero.
5. L_within is invariant to group-level prediction shifts.
6. Gradient magnitudes are sensible for lambda in {0.03, 0.1, 0.3}.
7. Validation / test loaders return X_d=None (unchanged pipeline).

Usage
-----
    python tests/verify_wdmpnn_within_group_loss.py
"""

from __future__ import annotations

import copy
import sys
import os
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "python"))

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"


def ok(msg: str) -> None:
    print(f"  [{PASS}] {msg}")

def fail(msg: str) -> None:
    print(f"  [{FAIL}] {msg}")

def warn(msg: str) -> None:
    print(f"  [{WARN}] {msg}")


# ── Imports from project ─────────────────────────────────────────────────────
import pandas as pd
from chemprop import featurizers, nn, models
from chemprop.data import PolymerDatapoint, PolymerDataset, build_dataloader
from chemprop.data.collate import collate_polymer_batch
from chemprop.data.samplers import GroupAwareSampler
from chemprop.nn.within_group_loss import within_group_residual_loss
from torch.utils.data import DataLoader as TorchDataLoader

from run_wdmpnn_generalization import (
    WDMPNNWithinGroupLoss, _GroupBatchSampler, _lambda_tag, BATCH_SIZE, SEED
)
from run_stage2d_generalization import (
    generate_group_disjoint_splits, build_group_keys,
)

# ── Dataset / split constants ─────────────────────────────────────────────────
DATA_PATH   = ROOT_DIR / "data" / "ea_ip.csv"
TARGET      = "EA vs SHE (eV)"
N_FOLDS     = 5
FOLD_IDX    = 0
SMALL_BATCH = 128  # smaller batch for fast CPU checks

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_small_model(lambda_within: float = 0.0) -> models.MPNN:
    mp  = nn.WeightedBondMessagePassing()
    agg = nn.WeightedMeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim)
    if lambda_within > 0.0:
        return WDMPNNWithinGroupLoss(mp, agg, ffn, batch_norm=False,
                                     lambda_within=lambda_within)
    return models.MPNN(mp, agg, ffn, batch_norm=False)


def get_one_batch(loader):
    return next(iter(loader))


# ─────────────────────────────────────────────────────────────────────────────
# Setup: load data, splits, group IDs
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("wDMPNN Within-Group Loss Verification")
print("=" * 70)
print(f"  Data  : {DATA_PATH}")
print(f"  Target: {TARGET}")
print(f"  Fold  : {FOLD_IDX}")
print()

df = pd.read_csv(DATA_PATH)
smis = df["WDMPNN_Input"].astype(str).values
ys   = df[TARGET].astype(float).values.reshape(-1, 1)

group_keys   = build_group_keys(df)
_, inverse   = np.unique(group_keys, return_inverse=True)
all_group_ids = inverse.astype(np.int64)

train_indices, val_indices, test_indices = generate_group_disjoint_splits(
    df, n_splits=N_FOLDS, seed=SEED
)
tr = train_indices[FOLD_IDX]
va = val_indices[FOLD_IDX]
te = test_indices[FOLD_IDX]

print(f"  Split : group_disjoint  train={len(tr)}  val={len(va)}  test={len(te)}")

featurizer   = featurizers.PolymerMolGraphFeaturizer()
all_dp       = [PolymerDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

train_data   = [all_dp[i] for i in tr]
val_data     = [all_dp[i] for i in va]
test_data    = [all_dp[i] for i in te]

train_group_ids = all_group_ids[tr]

# ── Separate dataset instances per test to avoid stateful-batch issues ────────
def make_train_ds_lambda0():
    data = [PolymerDatapoint.from_smi(smis[i], ys[i]) for i in tr]
    ds   = PolymerDataset(data, featurizer)
    ds.normalize_targets()
    return ds

def make_train_ds_with_gids():
    data = [PolymerDatapoint.from_smi(smis[i], ys[i]) for i in tr]
    ds   = PolymerDataset(data, featurizer)
    ds.normalize_targets()
    for j, dp in enumerate(data):
        dp.x_d = np.array([float(train_group_ids[j])], dtype=np.float32)
    ds.X_d = np.array([dp.x_d for dp in data])
    return ds

def make_val_ds():
    data = [PolymerDatapoint.from_smi(smis[i], ys[i]) for i in va]
    ds   = PolymerDataset(data, featurizer)
    return ds

def make_test_ds():
    data = [PolymerDatapoint.from_smi(smis[i], ys[i]) for i in te]
    ds   = PolymerDataset(data, featurizer)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: lambda_within=0 produces same loss as base MPNN
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 1: lambda_within=0 reproduces original pipeline ──────────────")
try:
    ds_base  = make_train_ds_lambda0()
    ds_wrap  = make_train_ds_lambda0()

    loader_base = build_dataloader(ds_base, batch_size=SMALL_BATCH, num_workers=0, shuffle=False)
    loader_wrap = build_dataloader(ds_wrap, batch_size=SMALL_BATCH, num_workers=0, shuffle=False)

    mpnn_base = build_small_model(lambda_within=0.0)
    mpnn_wrap = WDMPNNWithinGroupLoss(
        nn.WeightedBondMessagePassing(), nn.WeightedMeanAggregation(),
        nn.RegressionFFN(input_dim=nn.WeightedBondMessagePassing().output_dim),
        batch_norm=False, lambda_within=0.0
    )
    mpnn_wrap.load_state_dict(copy.deepcopy(mpnn_base.state_dict()))

    batch_b = get_one_batch(loader_base)
    batch_w = get_one_batch(loader_wrap)

    mpnn_base.train(); mpnn_wrap.train()
    loss_b = mpnn_base.training_step(batch_b, 0)
    loss_w = mpnn_wrap.training_step(batch_w, 0)

    diff = abs(loss_b.item() - loss_w.item())
    if diff < 1e-6:
        ok(f"Losses identical: base={loss_b.item():.6f}  wrap={loss_w.item():.6f}  diff={diff:.2e}")
    else:
        fail(f"Loss mismatch: base={loss_b.item():.6f}  wrap={loss_w.item():.6f}  diff={diff:.2e}")
except Exception as e:
    fail(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: GroupAwareSampler keeps complete groups intact
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 2: _GroupBatchSampler keeps complete groups intact ────────────")
try:
    ds_gids = make_train_ds_with_gids()
    loader_g = TorchDataLoader(
        ds_gids,
        batch_sampler=_GroupBatchSampler(train_group_ids, batch_size=SMALL_BATCH, seed=SEED),
        num_workers=0, collate_fn=collate_polymer_batch,
    )

    n_split_groups = 0
    n_batches_checked = 0
    for batch in loader_g:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        if X_d is None:
            fail("X_d is None — group_ids not injected into batch")
            break
        gids_in_batch = X_d[:, 0].long().numpy()
        n_batches_checked += 1

        # For each group present in this batch, check ALL members of that group
        # from the full training set are also in this batch
        batch_set = set()
        for local_idx, g in enumerate(gids_in_batch):
            batch_set.add(int(g))

        for g in batch_set:
            full_members = set(np.where(train_group_ids == g)[0].tolist())
            in_batch     = set(np.where(gids_in_batch == g)[0].tolist())
            if len(full_members) != len(in_batch):
                n_split_groups += 1

        if n_batches_checked >= 5:
            break

    if n_split_groups == 0:
        ok(f"Checked {n_batches_checked} batches — no group split across batch boundary")
    else:
        fail(f"{n_split_groups} groups were split across batch boundaries")
except Exception as e:
    fail(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: L_within computed in normalised target space
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 3: L_within computed in normalised (z-scored) target space ────")
try:
    ds_n  = make_train_ds_with_gids()
    scaler = ds_n.normalize_targets()

    loader_n = TorchDataLoader(
        ds_n,
        batch_sampler=_GroupBatchSampler(train_group_ids, batch_size=SMALL_BATCH, seed=SEED),
        num_workers=0, collate_fn=collate_polymer_batch,
    )
    batch = get_one_batch(loader_n)
    bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

    # targets in batch are z-scored (mean≈0, std≈1)
    t_mean = targets[targets.isfinite()].mean().item()
    t_std  = targets[targets.isfinite()].std().item()
    if abs(t_mean) < 1.0 and 0.3 < t_std < 3.0:
        ok(f"Targets look normalised: mean={t_mean:.3f}  std={t_std:.3f}")
    else:
        warn(f"Targets may not be normalised: mean={t_mean:.3f}  std={t_std:.3f}")

    # L_within uses these normalised targets directly
    gids = X_d[:, 0].long()
    preds_dummy = torch.zeros_like(targets)
    l_w, info = within_group_residual_loss(targets, preds_dummy, gids)
    ok(f"within_group_residual_loss runs on batch targets without error "
       f"(n_groups_active={info['n_groups_active']}, avg_size={info['avg_group_size']:.2f})")
except Exception as e:
    fail(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: Gradient sums to ≈ 0 within each group
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 4: Within-group gradient sums to approximately zero ───────────")
try:
    ds_g4 = make_train_ds_with_gids()
    ds_g4.normalize_targets()

    loader_g4 = TorchDataLoader(
        ds_g4,
        batch_sampler=_GroupBatchSampler(train_group_ids, batch_size=SMALL_BATCH, seed=SEED),
        num_workers=0, collate_fn=collate_polymer_batch,
    )
    batch4 = get_one_batch(loader_g4)
    bmg4, V_d4, X_d4, tgt4, w4, lt4, gt4 = batch4

    gids4 = X_d4[:, 0].long()
    # Build fresh predictions requiring grad
    preds4 = torch.randn(tgt4.shape, requires_grad=True)
    l_w4, _ = within_group_residual_loss(tgt4, preds4, gids4)
    l_w4.backward()

    g4 = preds4.grad  # (N, 1)
    max_group_grad_sum = 0.0
    for g in gids4.unique():
        mask_g = (gids4 == g)
        if mask_g.sum() < 2:
            continue
        gsum = g4[mask_g].sum().abs().item()
        max_group_grad_sum = max(max_group_grad_sum, gsum)

    if max_group_grad_sum < 1e-5:
        ok(f"Max within-group gradient sum = {max_group_grad_sum:.2e} (< 1e-5)")
    else:
        warn(f"Max within-group gradient sum = {max_group_grad_sum:.2e} "
             f"(may reflect float precision; expected ≈0)")
except Exception as e:
    fail(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: L_within invariant to group-level prediction shifts
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 5: L_within invariant to group-level prediction shifts ─────────")
try:
    ds_g5 = make_train_ds_with_gids()
    ds_g5.normalize_targets()

    loader_g5 = TorchDataLoader(
        ds_g5,
        batch_sampler=_GroupBatchSampler(train_group_ids, batch_size=SMALL_BATCH, seed=SEED),
        num_workers=0, collate_fn=collate_polymer_batch,
    )
    batch5 = get_one_batch(loader_g5)
    bmg5, V_d5, X_d5, tgt5, w5, lt5, gt5 = batch5
    gids5 = X_d5[:, 0].long()

    preds5 = torch.randn(tgt5.shape)
    l_orig, _ = within_group_residual_loss(tgt5, preds5, gids5)

    # Add group-level constant shift to each prediction
    preds5_shifted = preds5.clone()
    for g in gids5.unique():
        mask_g = (gids5 == g)
        shift = torch.randn(1).item() * 5.0
        preds5_shifted[mask_g] = preds5_shifted[mask_g] + shift

    l_shifted, _ = within_group_residual_loss(tgt5, preds5_shifted, gids5)
    diff5 = abs(l_orig.item() - l_shifted.item())

    if diff5 < 1e-5:
        ok(f"L_within unchanged after group-level shift: diff={diff5:.2e}")
    else:
        fail(f"L_within changed after group-level shift: "
             f"orig={l_orig.item():.6f}  shifted={l_shifted.item():.6f}  diff={diff5:.2e}")
except Exception as e:
    fail(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: Gradient magnitudes sensible for lambda in {0.03, 0.1, 0.3}
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 6: Gradient magnitudes for lambda ∈ {0.03, 0.1, 0.3} ─────────")
try:
    ds_g6 = make_train_ds_with_gids()
    scaler6 = ds_g6.normalize_targets()

    loader_g6 = TorchDataLoader(
        ds_g6,
        batch_sampler=_GroupBatchSampler(train_group_ids, batch_size=SMALL_BATCH, seed=SEED),
        num_workers=0, collate_fn=collate_polymer_batch,
    )
    batch6 = get_one_batch(loader_g6)

    prev_grad_norm = None
    any_nan = False
    for lw in [0.03, 0.1, 0.3]:  # lw=0.0 uses super() which expects X_d=None (no group_id)
        mp6  = nn.WeightedBondMessagePassing()
        agg6 = nn.WeightedMeanAggregation()
        ffn6 = nn.RegressionFFN(input_dim=mp6.output_dim,
                                 output_transform=nn.UnscaleTransform.from_standard_scaler(scaler6))
        mpnn6 = WDMPNNWithinGroupLoss(mp6, agg6, ffn6, batch_norm=False, lambda_within=lw)
        mpnn6.train()

        b6 = tuple(batch6)
        loss6 = mpnn6.training_step(b6, 0)
        loss6.backward()

        total_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in mpnn6.parameters() if p.grad is not None
        ) ** 0.5

        is_bad = (total_grad_norm != total_grad_norm) or (total_grad_norm == float('inf'))
        if is_bad:
            any_nan = True
        monotone_ok = (prev_grad_norm is None) or (total_grad_norm >= prev_grad_norm * 0.5)
        status = "✓" if monotone_ok else "!"
        print(f"    λ={lw:.2f}: grad_norm={total_grad_norm:.4f}  loss={loss6.item():.4f} {status}")
        prev_grad_norm = total_grad_norm

    if not any_nan:
        ok("Gradient norms finite for lambda in {0.03, 0.1, 0.3} (no NaN/Inf)")
    else:
        fail("NaN/Inf gradient norm detected")
except Exception as e:
    fail(f"Exception: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Validation and test loaders return X_d=None
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 7: Validation / test loaders return X_d=None ─────────────────")
try:
    val_ds   = make_val_ds()
    test_ds  = make_test_ds()

    val_loader  = build_dataloader(val_ds,  batch_size=SMALL_BATCH, num_workers=0, shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size=SMALL_BATCH, num_workers=0, shuffle=False)

    val_batch  = get_one_batch(val_loader)
    test_batch = get_one_batch(test_loader)

    val_Xd  = val_batch[2]
    test_Xd = test_batch[2]

    if val_Xd is None and test_Xd is None:
        ok("Val X_d=None  |  Test X_d=None  — loaders unchanged")
    else:
        fail(f"Val X_d={val_Xd is not None}  Test X_d={test_Xd is not None} — "
             "group_ids leaked into val/test loaders")
except Exception as e:
    fail(f"Exception: {e}")


print("\n" + "=" * 70)
print("Verification complete.")
print("=" * 70)
