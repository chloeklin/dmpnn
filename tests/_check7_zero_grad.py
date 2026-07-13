"""Check 7 — use model.zero_grad() instead of manual loop."""
import sys, copy, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "python"))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)

import lightning.pytorch as pl
pl.loggers.TensorBoardLogger = pl.loggers.CSVLogger

from utils import set_seed, build_copolymer_model_and_trainer, get_metric_list
from chemprop.data.dataloader import build_dataloader
from tests.verify_within_group_loss import (
    build_train_dataset, generate_group_disjoint_fold,
    build_group_keys, build_integer_group_ids,
)
import argparse
import pandas as pd

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

loader = build_dataloader(ds, batch_size=64, shuffle=True, seed=42)
batch = next(iter(loader))
bmgA, bmgB, fA, fB, Xd_b, tgt, w, lt, gt = batch
tc = tgt.nan_to_num(0.0)
mk = tgt.isfinite()

m1.train(); m1.zero_grad()
m2.train(); m2.zero_grad()

p1, _ = m1.forward_stage2d(bmgA, bmgB, fA, fB, Xd_b)
l1 = ((p1 - tc)**2 * mk.float()).mean()
l1.backward()

p2, _ = m2.forward_stage2d(bmgA, bmgB, fA, fB, Xd_b)
l2 = ((p2 - tc)**2 * mk.float()).mean()
l2.backward()

max_gd = 0.0
for pa, pb in zip(m1.parameters(), m2.parameters()):
    if pa.grad is not None and pb.grad is not None:
        max_gd = max(max_gd, (pa.grad - pb.grad).abs().max().item())

print(f"pred diff : {(p1-p2).abs().max().item():.2e}")
print(f"loss diff : {abs(l1.item()-l2.item()):.2e}")
print(f"grad diff : {max_gd:.2e}")
print("PASS" if max_gd < 1e-6 else "FAIL")
