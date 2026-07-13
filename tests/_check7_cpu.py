"""Check whether the deepcopy non-determinism is MPS-specific.

Runs the same forward+backward pass twice from the same deepcopy on CPU only.
"""
import sys, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

import torch
import numpy as np
import pandas as pd

# Patch out the TensorBoardLogger requirement
import lightning.pytorch as pl
pl.loggers.TensorBoardLogger = pl.loggers.CSVLogger

from utils import set_seed, build_copolymer_model_and_trainer, get_metric_list
from chemprop.data.dataloader import build_dataloader
from tests.verify_within_group_loss import (
    build_train_dataset, generate_group_disjoint_fold,
    build_group_keys, build_integer_group_ids,
)
import argparse

# Force CPU in the trainer kwargs by monkey-patching defaults
_orig_build = build_copolymer_model_and_trainer

def build_cpu(*a, **kw):
    kw["accelerator"] = "cpu"
    kw["devices"] = 1
    kw["precision"] = 32
    return _orig_build(*a, **kw)

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
m1, _ = build_cpu(
    args, Xd_train, scaler_ea, ckpt, "stage2d_2d1_arch", False,
    get_metric_list("reg"), 5, 1, False, lambda_within=0.0,
)
m1 = m1.cpu()

m2 = copy.deepcopy(m1)

loader = build_dataloader(ds_ea, batch_size=64, shuffle=True, seed=42)
batch = next(iter(loader))
bmgA, bmgB, fA, fB, Xd, tgt, w, lt, gt = batch
tgt_clean = tgt.nan_to_num(0.0)
mask = tgt.isfinite()

m1.train(); m2.train()
for p in m1.parameters():
    if p.grad is not None: p.grad.zero_()
for p in m2.parameters():
    if p.grad is not None: p.grad.zero_()

p1, _ = m1.forward_stage2d(bmgA, bmgB, fA, fB, Xd)
l1 = ((p1 - tgt_clean)**2 * mask.float()).mean()
l1.backward()

p2, _ = m2.forward_stage2d(bmgA, bmgB, fA, fB, Xd)
l2 = ((p2 - tgt_clean)**2 * mask.float()).mean()
l2.backward()

max_pred_diff = (p1 - p2).abs().max().item()
max_loss_diff = abs(l1.item() - l2.item())
max_gd = 0.0
for p_a, p_b in zip(m1.parameters(), m2.parameters()):
    if p_a.grad is not None and p_b.grad is not None:
        max_gd = max(max_gd, (p_a.grad - p_b.grad).abs().max().item())

print(f"Device: CPU")
print(f"deepcopy pred diff    : {max_pred_diff:.2e}")
print(f"deepcopy loss diff    : {max_loss_diff:.2e}")
print(f"deepcopy grad diff    : {max_gd:.2e}")
if max_pred_diff > 1e-5:
    print("NON-DETERMINISM on CPU — scatter/scatter_add is non-deterministic")
elif max_gd > 1e-5:
    print("PRED OK but grad differs — autograd non-determinism")
else:
    print("PASS — CPU is fully deterministic; MPS non-determinism confirmed")
