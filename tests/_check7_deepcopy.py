"""Isolate check 7: use deepcopy of one model instance, same batch, same loss."""
import sys, copy, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

import torch
import pandas as pd

import lightning.pytorch as pl
pl.loggers.TensorBoardLogger = pl.loggers.CSVLogger

from utils import set_seed, build_copolymer_model_and_trainer, get_metric_list
from chemprop.data.dataloader import build_dataloader
from tests.verify_within_group_loss import (
    build_train_dataset, generate_group_disjoint_fold,
    build_group_keys, build_integer_group_ids,
)

df = pd.read_csv(ROOT / "data" / "ea_ip.csv")
group_ids_all = build_integer_group_ids(build_group_keys(df))
train_idx, _, _ = generate_group_disjoint_fold(df, 0, 42)
ds_ea, scaler_ea, Xd_train = build_train_dataset(df, "EA vs SHE (eV)", train_idx, group_ids_all)

args = argparse.Namespace(
    model_name="DMPNN", dataset_name="ea_ip", polymer_type="copolymer",
    copolymer_mode="stage2d_2d1_arch", split_type="group_disjoint",
    task_type="reg", batch_norm=False, incl_desc=False, incl_rdkit=False,
    fusion_mode="late_concat", batch_size=64, save_checkpoint=False,
    save_predictions=False, export_embeddings=False, train_size=None,
    results_subdir="diag", aux_task="off", _aux_cols=[], _n_aux_targets=0,
    lambda_within=0.0, fusion_type="sum_fusion",
)

# Force CPU to check whether the non-determinism is MPS-specific
import lightning.pytorch as _pl_inner
_pl_inner.Trainer.__init__.__defaults__  # just import

ckpt = ROOT / "tests" / "_diag_ckpt_tmp"
ckpt.mkdir(parents=True, exist_ok=True)

set_seed(42)
m1, _ = build_copolymer_model_and_trainer(
    args, Xd_train, scaler_ea, ckpt, "stage2d_2d1_arch", False,
    get_metric_list("reg"), 5, 1, False, lambda_within=0.0,
)

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

max_gd = 0.0
for p_a, p_b in zip(m1.parameters(), m2.parameters()):
    if p_a.grad is not None and p_b.grad is not None:
        max_gd = max(max_gd, (p_a.grad - p_b.grad).abs().max().item())

print(f"deepcopy same-model grad diff : {max_gd:.2e}")
print(f"pred diff                     : {(p1-p2).abs().max().item():.2e}")
print(f"loss diff                     : {abs(l1.item()-l2.item()):.2e}")
print("PASS" if max_gd < 1e-5 else "FAIL — non-determinism in graph ops")
