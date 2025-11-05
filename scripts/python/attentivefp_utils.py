"""
AttentiveFP-specific utilities and data processing functions.

This module provides clean, dedicated functions for AttentiveFP training and evaluation,
separate from the DMPNN/Lightning infrastructure.
"""

import json
import logging
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


# Remove old atom_features and bond_features functions
def atom_features(atom):
    """DEPRECATED: Use atom_features_attfp instead."""
    return atom_features_attfp(atom)


def bond_features(bond):
    """DEPRECATED: Use bond_features_attfp instead."""
    return bond_features_attfp(bond)


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
    """Evaluate AttentiveFP model - matches original forward pass."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Use original forward pass style
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            if task_type == 'reg':
                pred = out.cpu().numpy()
                target = batch.y.view(-1, 1).cpu().numpy()
            elif task_type == 'binary':
                pred = torch.sigmoid(out).cpu().numpy()
                target = batch.y.cpu().numpy()
            else:  # multi-class
                pred = torch.softmax(out, dim=1).cpu().numpy()
                target = batch.y.cpu().numpy()
            
            predictions.append(pred)
            targets.append(target)
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Unscale predictions for regression
    if task_type == 'reg' and scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        targets = scaler.inverse_transform(targets.reshape(-1, 1))
    
    # Calculate metrics
    metrics = {}
    if task_type == 'reg':
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['r2'] = r2_score(targets, predictions)
    elif task_type == 'binary':
        pred_binary = (predictions > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(targets, pred_binary)
        metrics['f1'] = f1_score(targets, pred_binary)
        try:
            metrics['roc_auc'] = roc_auc_score(targets, predictions)
        except ValueError:
            metrics['roc_auc'] = np.nan
    else:  # multi-class
        pred_classes = np.argmax(predictions, axis=1)
        metrics['accuracy'] = accuracy_score(targets, pred_classes)
        metrics['f1'] = f1_score(targets, pred_classes, average='weighted')
    
    return metrics, predictions, targets


def save_attentivefp_checkpoint(model, optimizer, scaler, epoch, metrics, checkpoint_path: Path):
    """Save AttentiveFP checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)


def load_attentivefp_checkpoint(checkpoint_path: Path, model, optimizer=None, device='cpu'):
    """Load AttentiveFP checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('scaler'), checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def extract_embeddings(model, loader, device):
    """Extract embeddings from AttentiveFP model - matches original forward pass."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Get embeddings from the core model before final prediction layer
            # Use original forward pass style
            if hasattr(model.core, 'gnn'):
                # If the model has a separate GNN component
                emb = model.core.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                # Extract from intermediate layers (this might need adjustment based on model structure)
                emb = model.core(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            embeddings.append(emb.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


def create_data_splits(df: pd.DataFrame, target_col: str, n_splits: int = 5, 
                      test_size: float = 0.2, random_state: int = 42):
    """Create train/val/test splits for AttentiveFP."""
    from sklearn.model_selection import train_test_split
    
    # Remove rows with NaN targets
    valid_df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    splits = []
    for i in range(n_splits):
        # Create train/test split
        train_val_idx, test_idx = train_test_split(
            range(len(valid_df)), 
            test_size=test_size, 
            random_state=random_state + i
        )
        
        # Create train/val split from train_val
        train_idx, val_idx = train_test_split(
            train_val_idx, 
            test_size=test_size, 
            random_state=random_state + i + 1000
        )
        
        splits.append((train_idx, val_idx, test_idx))
    
    return splits, valid_df
