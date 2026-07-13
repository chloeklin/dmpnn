"""Identify which parameters produce the gradient discrepancy in Check 7."""
import sys, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

import torch
import lightning.pytorch as pl
pl.loggers.TensorBoardLogger = pl.loggers.CSVLogger

from utils import set_seed, build_copolymer_model_and_trainer, get_metric_list
from chemprop.data.dataloader import build_dataloader
from tests.verify_within_group_loss import (
    build_train_dataset, generate_group_disjoint_fold,
    build_group_keys, build_integer_group_ids,
)
import argparse, pandas as pd

df = pd.read_csv(ROOT / "data" / "ea_ip.csv")
gids_all = build_integer_group_ids(build_group_keys(df))
tidx, _, _ = generate_group_disjoint_fold(df, 0, 42)
ds, sc, Xd = build_train_dataset(df, "EA vs SHE (eV)", tidx, gids_all)

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
mb, _ = build_copolymer_model_and_trainer(
    args, Xd, sc, ckpt, "stage2d_2d1_arch", False,
    get_metric_list("reg"), 5, 1, False, lambda_within=0.0,
)
mb = mb.cpu()
m1 = copy.deepcopy(mb)
m2 = copy.deepcopy(mb)

# Two fresh loaders
l1r = build_dataloader(ds, batch_size=64, shuffle=True, seed=42)
l2r = build_dataloader(ds, batch_size=64, shuffle=True, seed=42)
b1 = next(iter(l1r))
b2 = next(iter(l2r))

tgt = b1[5]; mask = tgt.isfinite(); tc = tgt.nan_to_num(0.0)
print(f"tgt match: {(b1[5]-b2[5]).abs().max().item():.2e}")

m1.train(); m1.zero_grad()
m2.train(); m2.zero_grad()

p1, _ = m1.forward_stage2d(*b1[:5])
l1 = ((p1-tc)**2*mask.float()).mean(); l1.backward()

p2, _ = m2.forward_stage2d(*b2[:5])
l2 = ((p2-tc)**2*mask.float()).mean(); l2.backward()

print(f"pred diff: {(p1-p2).abs().max().item():.2e}")
print(f"loss diff: {abs(l1.item()-l2.item()):.2e}")

worst_params = []
for (n, pa), pb in zip(m1.named_parameters(), m2.parameters()):
    if pa.grad is None or pb.grad is None:
        continue
    d = (pa.grad - pb.grad).abs().max().item()
    if d > 1e-6:
        worst_params.append((d, n))

worst_params.sort(reverse=True)
print("\nTop parameters with grad discrepancy:")
for d, n in worst_params[:10]:
    print(f"  {d:.4e}  {n}")
