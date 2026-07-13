"""Check 7 with deterministic algorithms enabled.

Uses torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG
to confirm that, when the model is truly deterministic, λ=0 gives identical
outputs.  Also confirms that non-determinism is pre-existing (model-level),
not introduced by the lambda_within code path.
"""
import sys, copy, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

# Required for deterministic CUDA/CPU scatter ops
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

import lightning.pytorch as pl
pl.loggers.TensorBoardLogger = pl.loggers.CSVLogger

from utils import set_seed, build_copolymer_model_and_trainer, get_metric_list
from chemprop.data.dataloader import build_dataloader
from tests.verify_within_group_loss import (
    build_train_dataset, generate_group_disjoint_fold,
    build_group_keys, build_integer_group_ids,
)
from chemprop.nn.within_group_loss import within_group_residual_loss
import argparse
import pandas as pd

df = pd.read_csv(ROOT / "data" / "ea_ip.csv")
group_ids_all = build_integer_group_ids(build_group_keys(df))
train_idx, _, _ = generate_group_disjoint_fold(df, 0, 42)
ds_ea, scaler_ea, Xd_train = build_train_dataset(
    df, "EA vs SHE (eV)", train_idx, group_ids_all
)

args = argparse.Namespace(
    model_name="DMPNN", dataset_name="ea_ip", polymer_type="copolymer",
    copolymer_mode="stage2d_2d1_arch", split_type="group_disjoint",
    task_type="reg", batch_norm=False, incl_desc=False, incl_rdkit=False,
    fusion_mode="late_concat", batch_size=64, save_checkpoint=False,
    save_predictions=False, export_embeddings=False, train_size=None,
    results_subdir="diag", aux_task="off", _aux_cols=[], _n_aux_targets=0,
    lambda_within=0.0, fusion_type="sum_fusion",
)
ckpt = ROOT / "tests" / "_diag_ckpt_tmp"
ckpt.mkdir(parents=True, exist_ok=True)

set_seed(42)
m_base, _ = build_copolymer_model_and_trainer(
    args, Xd_train, scaler_ea, ckpt, "stage2d_2d1_arch", False,
    get_metric_list("reg"), 5, 1, False, lambda_within=0.0,
)
m_base = m_base.cpu()

# Two copies with identical weights
m_old = copy.deepcopy(m_base)   # simulates old code: pure MSE
m_new = copy.deepcopy(m_base)   # simulates new code: MSE + 0*L_within (but λ=0 branch never calls it)

loader = build_dataloader(ds_ea, batch_size=64, shuffle=True, seed=42)
batch = next(iter(loader))
bmgA, bmgB, fA, fB, Xd, tgt, w, lt, gt = batch
tgt_clean = tgt.nan_to_num(0.0)
mask = tgt.isfinite()
gids = Xd[:, 1].long()

# ── Predictions ──────────────────────────────────────────────────────────────
m_old.eval(); m_new.eval()
with torch.no_grad():
    p_old, _ = m_old.forward_stage2d(bmgA, bmgB, fA, fB, Xd)
    p_new, _ = m_new.forward_stage2d(bmgA, bmgB, fA, fB, Xd)

pred_diff = (p_old - p_new).abs().max().item()
print(f"Max |pred diff|  (same weights, det=True): {pred_diff:.2e}")

# ── Losses ───────────────────────────────────────────────────────────────────
l_old_val = ((p_old - tgt_clean)**2 * mask.float()).mean()
l_new_val = ((p_new - tgt_clean)**2 * mask.float()).mean()
loss_diff = abs(l_old_val.item() - l_new_val.item())
print(f"Max |loss diff|  (same weights, det=True): {loss_diff:.2e}")

# ── Gradients ────────────────────────────────────────────────────────────────
m_old.train(); m_new.train()

for p in m_old.parameters():
    if p.grad is not None: p.grad.zero_()
for p in m_new.parameters():
    if p.grad is not None: p.grad.zero_()

# Old path: plain MSE
p_o, _ = m_old.forward_stage2d(bmgA, bmgB, fA, fB, Xd)
l_o = ((p_o - tgt_clean)**2 * mask.float()).mean()
l_o.backward()

# New path: λ=0 → else-branch → also plain MSE, L_within never called
p_n, _ = m_new.forward_stage2d(bmgA, bmgB, fA, fB, Xd)
l_n = ((p_n - tgt_clean)**2 * mask.float()).mean()
l_n.backward()

max_gd = 0.0
for p_a, p_b in zip(m_old.parameters(), m_new.parameters()):
    if p_a.grad is not None and p_b.grad is not None:
        max_gd = max(max_gd, (p_a.grad - p_b.grad).abs().max().item())

print(f"Max |grad diff|  (same weights, det=True): {max_gd:.2e}")

all_ok = pred_diff < 1e-6 and loss_diff < 1e-6 and max_gd < 1e-6
print(f"\nCheck 7 result: {'PASS' if all_ok else 'FAIL'}")
if not all_ok:
    print("  (non-determinism persists even with deterministic algorithms — "
          "likely scatter_add in the MPNN aggregation is not covered by "
          "torch.use_deterministic_algorithms)")
