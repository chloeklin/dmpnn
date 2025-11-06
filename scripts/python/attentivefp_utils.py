"""
AttentiveFP-specific utilities and data processing functions.

This module provides clean, dedicated functions for AttentiveFP training and evaluation,
separate from the DMPNN/Lightning infrastructure.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondStereo, BondType


# AttentiveFP paper features (Table 1) - exact implementation from original
ATOM_SYMBOLS_16 = ["B","C","N","O","F","Si","P","S","Cl","As","Se","Br","Te","I","At","metal"]

def _one_hot_idx(i, K):
    v = [0]*K
    if 0 <= i < K: v[i] = 1
    return v

def _hyb6(atom):
    hyb = atom.GetHybridization()
    base = [int(hyb == h) for h in [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
                                    HybridizationType.SP3D, HybridizationType.SP3D2]]
    other = int(not any(base))
    return base + [other]  # 6

def _atom_symbol_bucket(atom):
    sym = atom.GetSymbol()
    return ATOM_SYMBOLS_16.index(sym) if sym in ATOM_SYMBOLS_16[:-1] else len(ATOM_SYMBOLS_16)-1

def atom_features_attfp(atom: Chem.Atom) -> np.ndarray:
    f_sym = _one_hot_idx(_atom_symbol_bucket(atom), len(ATOM_SYMBOLS_16))            # 16
    f_deg = _one_hot_idx(min(atom.GetTotalDegree(), 5), 6)                           # 6
    f_charge = [int(atom.GetFormalCharge())]                                         # 1
    f_rad = [int(atom.GetNumRadicalElectrons())]                                     # 1
    f_hyb = _hyb6(atom)                                                              # 6
    f_arom = [int(atom.GetIsAromatic())]                                             # 1
    f_h = _one_hot_idx(min(atom.GetTotalNumHs(includeNeighbors=True), 4), 5)         # 5
    f_chicenter = [int(atom.HasProp('_ChiralityPossible'))]                          # 1
    cip = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else ''
    f_chitype = [int(cip == 'R'), int(cip == 'S')]                                   # 2
    v = np.asarray(f_sym + f_deg + f_charge + f_rad + f_hyb + f_arom + f_h + f_chicenter + f_chitype, dtype=np.float32)
    assert v.shape[0] == 39
    return v

def bond_features_attfp(bond: Chem.Bond) -> np.ndarray:
    bt = bond.GetBondType()
    f_type = [int(bt == BondType.SINGLE), int(bt == BondType.DOUBLE), int(bt == BondType.TRIPLE), int(bt == BondType.AROMATIC)]
    f_conj = [int(bond.GetIsConjugated())]
    f_ring = [int(bond.IsInRing())]
    stereo_cats = [BondStereo.STEREONONE, BondStereo.STEREOANY, BondStereo.STEREOZ, BondStereo.STEREOE]
    stereo = bond.GetStereo()
    f_stereo = [int(stereo == s) for s in stereo_cats]
    v = np.asarray(f_type + f_conj + f_ring + f_stereo, dtype=np.float32)
    assert v.shape[0] == 10
    return v


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data object - matches original mol_to_pyg_attfp."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    x = [atom_features_attfp(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.stack(x, 0), dtype=torch.float)

    ei, ea = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        e = torch.tensor(bond_features_attfp(b), dtype=torch.float)
        ei.append([i, j]); ea.append(e)
        ei.append([j, i]); ea.append(e.clone())

    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.empty((2,0), dtype=torch.long)
    edge_attr  = torch.stack(ea, 0) if ea else torch.empty((0,10), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

def compute_reg_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(math.sqrt(np.mean((y_pred - y_true)**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2) + 1e-12)
    r2 = 1 - ss_res/ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}

class GraphCSV(torch.utils.data.Dataset):
    """Dataset class for AttentiveFP that converts SMILES to graphs - matches original name."""
    
    def __init__(self, df: pd.DataFrame, smiles_col: str, target_col: str, task_type='reg'):
        self.smiles = df[smiles_col].astype(str).tolist()
        self.task_type = task_type
        
        # Handle targets based on task type
        if task_type == 'reg':
            self.y = df[target_col].to_numpy(dtype=np.float32).reshape(-1,1)
        else:
            # Classification: use integer targets
            self.y = df[target_col].astype(int).to_numpy().reshape(-1,1)
            
        self.graphs = [smiles_to_graph(s) for s in self.smiles]
        self.target_col = target_col
    def __len__(self): return len(self.graphs)
    def __getitem__(self, i):
        d = self.graphs[i].clone()
        if self.task_type == 'reg':
            d.y = torch.tensor(self.y[i], dtype=torch.float)
        else:
            # Classification: use long tensors for CrossEntropyLoss
            d.y = torch.tensor(self.y[i], dtype=torch.long)
        return d



class EdgeGuard(nn.Module):
    """Wrapper to handle molecules with no edges - matches original implementation."""
    
    def __init__(self, core, edge_dim=10):
        super().__init__()
        self.core = core
        self.edge_dim = edge_dim
    
    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index.numel() == 0:
            edge_attr = x.new_zeros((0, self.edge_dim))
        elif edge_attr is None or edge_attr.size(-1) == 0:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))
        return self.core(x, edge_index, edge_attr, batch)


def build_attentivefp_loaders(args, df_tr, df_va, df_te, smiles_col, target, eval=False):
    from attentivefp_utils import GraphCSV
    from torch_geometric.loader import DataLoader
    from sklearn.preprocessing import StandardScaler

    ds_tr = GraphCSV(df_tr, smiles_col, target, task_type=args.task_type)
    ds_va = GraphCSV(df_va, smiles_col, target, task_type=args.task_type)
    ds_te = GraphCSV(df_te, smiles_col, target, task_type=args.task_type)

    scaler = None
    if args.task_type == 'reg':
        scaler = StandardScaler().fit(ds_tr.y)
        ds_tr.y = scaler.transform(ds_tr.y)
        ds_va.y = scaler.transform(ds_va.y)  # test left unscaled
    if eval:
        train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    else:
        train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader, scaler

def create_attentivefp_model(task_type: str, n_classes: Optional[int] = None, 
                           hidden_channels: int = 200, num_layers: int = 2, 
                           num_timesteps: int = 2, dropout: float = 0.0) -> nn.Module:
    """Create AttentiveFP model with appropriate output layer - matches original dimensions."""
    from torch_geometric.nn import models
    
    if task_type == 'reg':
        out_channels = 1
    elif task_type == 'binary':
        out_channels = 1
    elif task_type == 'multi':
        if n_classes is None:
            raise ValueError("n_classes must be provided for multi-class classification")
        out_channels = n_classes
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Create AttentiveFP core model with correct dimensions
    core = models.AttentiveFP(
        in_channels=39,  # Number of atom features (matches original)
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=10,  # Number of bond features (matches original)
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout
    )
    
    # Wrap with EdgeGuard (using edge_dim=10 to match bond features)
    model = EdgeGuard(core, edge_dim=10)
    return model


def eval_loss(model, loader, device, task):
    """Evaluate loss on a dataset - matches original eval_loss function."""
    model.eval()
    tot = 0
    n = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            if task == "reg":
                loss = F.mse_loss(pred.view(-1,1), batch.y.view(-1,1), reduction='sum')
            elif task == "binary":
                # Binary classification: use BCEWithLogitsLoss for stability
                loss = F.binary_cross_entropy_with_logits(pred.view(-1), batch.y.view(-1), reduction='sum')
            elif task == "multi":
                # Multi-class classification: use CrossEntropyLoss
                loss = F.cross_entropy(pred, batch.y.view(-1).long(), reduction='sum')
            else:
                raise ValueError(f"Unknown task type: {task}")
            
            tot += loss.item()
            n += batch.num_graphs
    
    return tot / max(1, n)


def train_epoch(model, loader, optimizer, device, task_type: str):
    """Train AttentiveFP for one epoch - matches original training logic exactly."""
    model.train()
    tr_loss = 0
    ntr = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Use original forward pass style
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Use same loss functions as original
        if task_type == 'reg':
            loss = F.mse_loss(pred.view(-1,1), batch.y.view(-1,1))
        elif task_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(pred.view(-1), batch.y.view(-1))
        elif task_type == 'multi':
            loss = F.cross_entropy(pred, batch.y.view(-1).long())
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * batch.num_graphs
        ntr += batch.num_graphs
    
    return tr_loss / max(1, ntr)


def evaluate_model(model, loader, device, task_type: str, scaler=None):
    """
    AttentiveFP evaluation matching Version 2 semantics:

    - Regression: inverse-transform predictions only (test targets are NOT scaled).
    - Binary: return hard classes; compute metrics with sigmoid probs via eval_binary.
    - Multi-class: return hard classes; compute metrics with softmax probs via eval_multi.
    - Single pass over the loader (we collect probs during the same pass).

    Returns:
        metrics: dict
        y_pred: np.ndarray (continuous for reg; class labels for cls)
        y_true: np.ndarray (gold labels, unscaled)
    """
    import numpy as np
    import torch
    from tabular_utils import eval_binary, eval_multi

    # These must be available in the scope (as in your script)
    # from tabular_utils import eval_binary, eval_multi
    # from your module: compute_reg_metrics

    model.eval()

    y_true_list = []
    y_pred_list = []
    prob_list   = []  # y_probs (binary) or y_proba (multi)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            if task_type == 'reg':
                # raw preds
                pred = out.view(-1, 1).cpu().numpy().reshape(-1)
                # inverse-transform predictions ONLY (test targets were not scaled)
                if scaler is not None:
                    pred = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
                y_pred_list.append(pred)
                # ground-truth (already unscaled)
                y_true_list.append(batch.y.view(-1, 1).cpu().numpy().reshape(-1))

            elif task_type == 'binary':
                # probs for metrics; hard classes for y_pred
                probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                pred_class = (probs >= 0.5).astype(int)
                y_pred_list.append(pred_class)
                prob_list.append(probs)
                y_true_list.append(batch.y.view(-1).cpu().numpy().astype(int))

            elif task_type == 'multi':
                # class-probabilities and class labels
                proba = torch.softmax(out, dim=1).cpu().numpy()
                pred_class = np.argmax(proba, axis=1)
                y_pred_list.append(pred_class)
                prob_list.append(proba)
                y_true_list.append(batch.y.view(-1).cpu().numpy().astype(int))

            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Compute metrics to match Version 2
    if task_type == 'reg':
        metrics = compute_reg_metrics(y_true, y_pred)

    elif task_type == 'binary':
        y_probs = np.concatenate(prob_list) if prob_list else None
        metrics = eval_binary(y_true, y_pred, y_probs)

    else:  # 'multi'
        y_proba = np.concatenate(prob_list) if prob_list else None
        metrics = eval_multi(y_true, y_pred, y_proba)

    return metrics, y_pred, y_true



from pathlib import Path
import os, tempfile, torch

def save_attentivefp_checkpoint(
    best_state: dict,
    checkpoint_file: Path,   # full path to .../best.pt
    *,
    epoch: int,
    best_val: float,
    metrics: dict | None = None,
    optimizer=None,
    target_scaler=None,
    complete: bool = False
):
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": best_state,
        "best_val": float(best_val),
        "best_epoch": int(epoch),
        "epochs_seen": int(epoch),
        "complete": bool(complete),
        "metrics": metrics or {}
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if target_scaler is not None:
        payload["target_scaler"] = target_scaler

    import tempfile, os, torch
    with tempfile.NamedTemporaryFile(dir=str(checkpoint_file.parent), delete=False) as tmp:
        tmp_name = tmp.name
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, checkpoint_file)  # atomic on POSIX
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass



def load_attentivefp_checkpoint(checkpoint_path: Path, model, optimizer=None, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
    model.load_state_dict(ckpt[state_key])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('target_scaler'), ckpt.get('best_epoch', 0), ckpt.get('metrics', {})



# class GraphRepExtractor(nn.Module):
#     def __init__(self, attentivefp_core):
#         super().__init__()
#         self.core = attentivefp_core  # your AttentiveFP model (possibly wrapped)
#     def forward(self, x, edge_index, edge_attr, batch):
#         # NOTE: core must have .gnn; this returns the graph embedding tensor [num_graphs, hidden]
#         return self.core.gnn(x, edge_index, edge_attr, batch)


# @torch.no_grad()
# def extract_attentivefp_embeddings(model, loader, device):
#     model.eval()
#     # If you wrapped the core with EdgeGuard(... core ...), pass model.core to the extractor
#     core = model.core if hasattr(model, "core") else model
#     assert hasattr(core, "gnn"), "This AttentiveFP build has no `.gnn` attribute; use the hook-based fallback."
#     rep_model = GraphRepExtractor(core).to(device).eval()

#     embs = []
#     for batch in loader:
#         batch = batch.to(device)
#         h = rep_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B, hidden]
#         embs.append(h.detach().cpu())
#     return torch.cat(embs, dim=0).numpy()   # (N_graphs, hidden)



@torch.no_grad()
def extract_attentivefp_embeddings(model, loader, device):
    """
    Returns molecule-level embeddings from AttentiveFP by hooking into the
    final linear layer (lin2). The input to lin2 is the pooled graph embedding.
    """
    model.eval()
    core = model.core if hasattr(model, "core") else model

    if not hasattr(core, "lin2"):
        raise RuntimeError("Expected AttentiveFP core to have .lin2")

    captured = []

    def _grab_embedding(mod, inp):
        # inp is a 1-tuple; inp[0] is the tensor just before lin2
        captured.append(inp[0].detach().cpu())

    hook = core.lin2.register_forward_pre_hook(_grab_embedding)

    try:
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # forward pass triggers hook
    finally:
        hook.remove()

    return torch.cat(captured, dim=0).numpy()

