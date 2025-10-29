import argparse, os, random
import torch as th
import torch.nn as nn
from accelerate import Accelerator
from dgl.data import download
from dgl.dataloading import GraphDataLoader
from transformers.optimization import AdamW, get_polynomial_decay_schedule_with_warmup
import yaml

from datasets.base import BaseGraphDataset
from datasets.base import GraphBatch
from datasets.mol_from_csv import CSVPolymerDataset
from model import Graphormer

accelerator = Accelerator()

def build_dataset(cfg):
    # split
    if cfg["dataset"]["split"]["type"] == "scaffold":
        # optional: implement scaffold split here; for now, do random
        pass

    ds = CSVPolymerDataset(
        path=cfg["dataset"]["path"],
        smiles_col=cfg["dataset"]["smiles_col"],
        num_global_desc=cfg["dataset"]["num_global_desc"],
        target_cols=cfg["dataset"]["target_cols"],
        task_type=cfg["dataset"]["task_type"],
        normalize_targets=cfg["dataset"]["normalize_targets"],
        standardize_globals=cfg["dataset"]["standardize_globals"],
        seed=cfg["train"]["seed"],
    )

    # random split if not provided
    N = len(ds)
    idx = list(range(N))
    random.Random(cfg["train"]["seed"]).shuffle(idx)
    valn = int(N * cfg["dataset"]["split"]["val_ratio"])
    testn = int(N * cfg["dataset"]["split"]["test_ratio"])
    val_idx = idx[:valn]
    test_idx = idx[valn:valn + testn]
    train_idx = idx[valn + testn:]

    class _Subset(th.utils.data.Dataset):
        def __init__(self, base, ids): self.base, self.ids = base, ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.base[self.ids[i]]

    return (_Subset(ds, train_idx), _Subset(ds, val_idx), _Subset(ds, test_idx)), ds

def build_loaders(splits, batch_size, num_workers, multi_hop_max_dist):
    def _collate(samples):
        return BaseGraphDataset.collate(samples, multi_hop_max_dist)

    train, val, test = splits
    return (
        GraphDataLoader(train, batch_size=batch_size, shuffle=True,  collate_fn=_collate, pin_memory=True, num_workers=num_workers),
        GraphDataLoader(val,   batch_size=batch_size, shuffle=False, collate_fn=_collate, pin_memory=True, num_workers=num_workers),
        GraphDataLoader(test,  batch_size=batch_size, shuffle=False, collate_fn=_collate, pin_memory=True, num_workers=num_workers),
    )

def loss_and_metric(task_type: str):
    if task_type == "binary" or task_type == "multilabel":
        return nn.BCEWithLogitsLoss(), "binary"
    elif task_type == "regression":
        return nn.MSELoss(), "regression"
    else:
        raise ValueError(f"Unknown task_type={task_type}")

def evaluate(model, data_loader, task_mode):
    model.eval()
    loss_fn, _ = (nn.BCEWithLogitsLoss(), "binary") if task_mode=="binary" else (nn.MSELoss(), "regression")
    losses, preds, trues = [], [], []
    with th.no_grad():
        for b in data_loader:
            device = accelerator.device
            out = model(
                b.node_feat.to(device),
                b.in_degree.to(device),
                b.out_degree.to(device),
                b.path_data.to(device),
                b.dist.to(device),
                attn_mask=b.attn_mask.to(device),
                global_desc=(b.global_desc.to(device) if b.global_desc is not None else None),
            )
            # gather for metrics
            p, y = accelerator.gather_for_metrics((out, b.labels.to(device)))
            loss = loss_fn(p, y) if task_mode!="binary" else loss_fn(p, y.float())
            losses.append(loss.item())
            preds.append(p.cpu())
            trues.append(y.cpu())
    import torchmetrics.functional as tmf
    import torch
    P = torch.cat(preds); Y = torch.cat(trues)
    if task_mode == "binary":
        auc = tmf.auroc(P.sigmoid(), Y.int(), task="binary") if Y.shape[1]==1 else tmf.auroc(P.sigmoid(), Y.int(), task="multilabel", num_labels=Y.shape[1], average="macro")
        return sum(losses)/max(1,len(losses)), auc.item()
    else:
        # report RMSE for readability
        rmse = ( (P - Y).pow(2).mean() ).sqrt().item()
        return sum(losses)/max(1,len(losses)), rmse

def main(cfg):
    # seeds
    random.seed(cfg["train"]["seed"])
    th.manual_seed(cfg["train"]["seed"])
    if th.cuda.is_available(): th.cuda.manual_seed(cfg["train"]["seed"])

    splits, full_ds = build_dataset(cfg)
    train_loader, val_loader, test_loader = build_loaders(
        splits,
        cfg["train"]["batch_size"],
        cfg["train"]["num_workers"],
        cfg["model"]["multi_hop_max_dist"],
    )

    # model
    mcfg = cfg["model"]
    model = Graphormer(
        num_classes=mcfg["num_classes"],
        edge_dim=mcfg["edge_dim"],
        num_atoms=mcfg["num_atoms"],
        max_degree=mcfg["max_degree"],
        num_spatial=mcfg["num_spatial"],
        multi_hop_max_dist=mcfg["multi_hop_max_dist"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        embedding_dim=mcfg["embedding_dim"],
        ffn_embedding_dim=mcfg["ffn_embedding_dim"],
        num_attention_heads=mcfg["num_attention_heads"],
        dropout=mcfg["dropout"],
        pre_layernorm=mcfg["pre_layernorm"],
        use_global_desc=mcfg["use_global_desc"],
        global_desc_dim=full_ds.df.iloc[:, full_ds.global_slice].shape[1] if full_ds.global_slice and mcfg["use_global_desc"] else 0,
    )

    if cfg["train"]["load_pcqm_pretrained"]:
        download(url=cfg["train"]["pcqm_ckpt_url"])
        state_dict = th.load(os.path.basename(cfg["train"]["pcqm_ckpt_url"]), map_location="cpu")
        # allow missing keys if head/global_proj differ
        model.load_state_dict(state_dict, strict=False)
        model.reset_output_layer_parameters()

    # opt/sched
    epochs = cfg["train"]["epochs"]
    total_updates = max(1, int(len(train_loader) * epochs))
    warmup_updates = int(total_updates * cfg["train"]["warmup_ratio"])

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], eps=1e-8, weight_decay=cfg["train"]["weight_decay"])
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates, lr_end=1e-9, power=1.0
    )

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    task_mode = "binary" if cfg["dataset"]["task_type"] in ["binary","multilabel"] else "regression"
    loss_fn, _ = (nn.BCEWithLogitsLoss(), "binary") if task_mode=="binary" else (nn.MSELoss(), "regression")

    best_val_metric, best_idx = float("-inf") if task_mode=="binary" else float("+inf"), -1
    train_metrics, val_metrics, test_metrics = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            out = model(
                b.node_feat.to(accelerator.device),
                b.in_degree.to(accelerator.device),
                b.out_degree.to(accelerator.device),
                b.path_data.to(accelerator.device),
                b.dist.to(accelerator.device),
                attn_mask=b.attn_mask.to(accelerator.device),
                global_desc=(b.global_desc.to(accelerator.device) if b.global_desc is not None else None),
            )
            y = b.labels.to(accelerator.device)
            loss = loss_fn(out, y if task_mode!="binary" else y.float())
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        val_loss, val_metric = evaluate(model, val_loader, task_mode)
        test_loss, test_metric = evaluate(model, test_loader, task_mode)

        train_metrics.append(epoch_loss / max(1, len(train_loader)))
        val_metrics.append(val_metric)
        test_metrics.append(test_metric)

        if task_mode == "binary":
            is_better = val_metric > best_val_metric
        else:
            is_better = val_metric < best_val_metric

        if is_better:
            best_val_metric = val_metric
            best_idx = epoch

        accelerator.print(f"Epoch {epoch+1:02d} | train_loss={train_metrics[-1]:.4f} | "
                          f"val_{'AUC' if task_mode=='binary' else 'RMSE'}={val_metric:.4f} | "
                          f"test_{'AUC' if task_mode=='binary' else 'RMSE'}={test_metric:.4f}")

    accelerator.print(f"Best epoch: {best_idx+1}")
    accelerator.print(f"Best val metric: {best_val_metric:.4f} | test metric @best: {test_metrics[best_idx]:.4f}")
    # (you can save model here if needed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/example.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
