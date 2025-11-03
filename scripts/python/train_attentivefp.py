#!/usr/bin/env python
import argparse, json, math, os
from pathlib import Path
from typing import List, Tuple
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import models
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondStereo, BondType

# ===== import YOUR helpers (exact names from your module) =====
from utils import (
    set_seed,
    setup_training_environment,
    load_and_preprocess_data,
    process_data,
    determine_split_strategy,
    generate_data_splits,
    build_experiment_paths,
    save_aggregate_results,
    save_predictions,
)
from tabular_utils import eval_binary, eval_multi

from joblib import load as joblib_load




def load_dmpnn_preproc(preproc_dir: Path, split_i: int):
    import json, numpy as np
    meta_f = preproc_dir / f"preprocessing_metadata_split_{split_i}.json"
    mask_f = preproc_dir / "correlation_mask.npy"
    const_f = preproc_dir / "constant_features_removed.npy"
    scaler_f = preproc_dir / "descriptor_scaler.pkl"

    with open(meta_f, "r") as f:
        meta = json.load(f)
    corr_mask = np.load(mask_f).astype(bool)
    const_idx = np.load(const_f)
    scaler = joblib_load(scaler_f) if scaler_f.exists() else None

    stats = meta["cleaning"].get("imputer_statistics", None)
    f32_min = np.float32(meta["cleaning"]["float32_min"])
    f32_max = np.float32(meta["cleaning"]["float32_max"])
    return {
        "corr_mask": corr_mask,
        "const_idx": const_idx,
        "imputer_stats": (np.array(stats, dtype=np.float64) if stats is not None else None),
        "float32_min": f32_min,
        "float32_max": f32_max,
        "scaler": scaler,
    }

def apply_dmpnn_preproc(X_raw: np.ndarray, arts):
    import numpy as np
    X = np.asarray(X_raw, dtype=np.float64)
    # drop constants (global)
    if arts["const_idx"].size:
        keep = np.ones(X.shape[1], dtype=bool); keep[arts["const_idx"].astype(int)] = False
        X = X[:, keep]
    # impute with train-fitted medians
    if arts["imputer_stats"] is not None:
        med = arts["imputer_stats"].reshape(1, -1)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(med, np.where(nan_mask)[1], axis=1)
    # clip + cast
    X = np.clip(X, arts["float32_min"], arts["float32_max"]).astype(np.float32)
    # correlation mask
    X = X[:, arts["corr_mask"]]
    # standardize
    if arts["scaler"] is not None:
        X = arts["scaler"].transform(X)
    return X


# -----------------------
# AttentiveFP paper features (Table 1)
# -----------------------
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

def mol_to_pyg_attfp(smiles: str) -> Data:
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

# -----------------------
# Dataset
# -----------------------
class GraphCSV(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, smiles_col: str, target_col: str, task_type='reg'):
        self.smiles = df[smiles_col].astype(str).tolist()
        self.task_type = task_type
        
        # Handle targets based on task type
        if task_type == 'reg':
            self.y = df[target_col].to_numpy(dtype=np.float32).reshape(-1,1)
        else:
            # Classification: use integer targets
            self.y = df[target_col].astype(int).to_numpy().reshape(-1,1)
            
        self.graphs = [mol_to_pyg_attfp(s) for s in self.smiles]
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

# -----------------------
# Edge guard (robust on single-atom graphs)
# -----------------------
class EdgeGuard(nn.Module):
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

# -----------------------
# Train/Eval
# -----------------------
@torch.no_grad()
def eval_loss(model, loader, device, task):
    model.eval(); tot=0; n=0
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
        tot += loss.item(); n += batch.num_graphs
    return tot / max(1,n)

def compute_reg_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(math.sqrt(np.mean((y_pred - y_true)**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2) + 1e-12)
    r2 = 1 - ss_res/ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# -----------------------
# Splits JSON helpers (to mirror D-MPNN)
# -----------------------
def save_splits_json(out_dir: Path, dataset_name: str, target: str,
                     train_indices: List[np.ndarray], val_indices: List[np.ndarray], test_indices: List[np.ndarray]):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for tr, va, te in zip(train_indices, val_indices, test_indices):
        payload.append({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()})
    with open(out_dir / f"{dataset_name}__{target}.json", "w") as f:
        json.dump(payload, f, indent=2)

# -----------------------
# Main
# -----------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)    
    ap = argparse.ArgumentParser(description="AttentiveFP training aligned with Chemprop/D-MPNN pipeline")
    ap.add_argument('--dataset_name', type=str, required=True)
    ap.add_argument('--target', type=str, default="")
    ap.add_argument("--polymer_type", type=str, choices=["homo", "copolymer"], default="homo",
                        help='Type of polymer: "homo" for homopolymer or "copolymer" for copolymer')
    ap.add_argument('--incl_desc', action='store_true',
                    help='Use dataset-specific descriptors')
    ap.add_argument('--incl_rdkit', action='store_true',
                    help='Include RDKit descriptors')
    ap.add_argument('--train_size', type=str, default=None,
                    help='Number of training samples to use (e.g., "500", "5000", "full"). If not specified, uses full training set.')
    ap.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default='reg')  # support classification
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--export_embeddings', action='store_true')
    ap.add_argument('--save_predictions', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--model_name', type=str, default="AttentiveFP")
    
    args = ap.parse_args()
    
    # Validate train_size argument
    if args.train_size is not None:
        if args.train_size.lower() == "full":
            # "full" is a valid option, no further validation needed
            pass
        else:
            try:
                train_size_int = int(args.train_size)
                if train_size_int <= 0:
                    ap.error("--train_size must be a positive integer or 'full' (e.g., 500, 5000, full)")
            except ValueError:
                ap.error("--train_size must be a valid integer or 'full' (e.g., 500, 5000, full)")
    
    device = torch.device(args.device)

    # ===== setup (same as your D-MPNN script) =====
    setup_info = setup_training_environment(args, model_type="graph")

    # Extract commonly used variables for backward compatibility
    config = setup_info['config']
    chemprop_dir = setup_info['chemprop_dir']
    checkpoint_dir = setup_info['checkpoint_dir']
    results_dir = setup_info['results_dir']

    # Create predictions directory if saving predictions
    if args.save_predictions:
        predictions_dir = chemprop_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Predictions will be saved to: {predictions_dir}")
    else:
        predictions_dir = None
    smiles_column = setup_info['smiles_column']
    ignore_columns = setup_info['ignore_columns']
    descriptor_columns = setup_info['descriptor_columns']
    SEED = setup_info['SEED']
    REPLICATES = setup_info['REPLICATES']
    EPOCHS = setup_info['EPOCHS']
    PATIENCE = setup_info['PATIENCE']
    num_workers = setup_info['num_workers']
    DATASET_DESCRIPTORS = setup_info['DATASET_DESCRIPTORS']
    MODEL_NAME = args.model_name

    # Check model configuration
    model_config = config['MODELS'].get(args.model_name, {})
    if not model_config:
        logger.warning(f"No configuration found for model '{args.model_name}'. Using defaults.")

    # === Set Random Seed ===
    set_seed(SEED)

    # === Load and Preprocess Data ===
    df_input, target_columns = load_and_preprocess_data(args, setup_info)

    # Debug: Show what columns we're working with
    logger.info(f"DataFrame columns after preprocessing: {list(df_input.columns)}")
    logger.info(f"Detected target columns: {target_columns}")
    logger.info(f"DataFrame shape: {df_input.shape}")

    # Check for any remaining string columns
    for col in df_input.columns:
        if df_input[col].dtype == 'object':
            logger.warning(f"Column '{col}' still has object dtype after preprocessing")

    # Filter to specific target if specified
    if args.target:
        if args.target not in target_columns:
            logger.error(f"Specified target '{args.target}' not found in dataset. Available targets: {target_columns}")
            exit(1)
        target_columns = [args.target]
        logger.info(f"Training on single target: {args.target}")



    logger.info("\n=== Training Configuration ===")
    logger.info(f"Dataset          : {args.dataset_name}")
    logger.info(f"Task type        : {args.task_type}")
    logger.info(f"Model            : {args.model_name}")
    logger.info(f"SMILES column    : {smiles_column}")
    logger.info(f"Descriptor cols  : {descriptor_columns}")
    logger.info(f"Target columns   : {target_columns}")
    logger.info(f"Descriptors      : {'Enabled' if args.incl_desc else 'Disabled'}")
    logger.info(f"RDKit desc.      : {'Enabled' if args.incl_rdkit else 'Disabled'}")
    logger.info(f"Epochs           : {EPOCHS}")
    logger.info(f"Replicates       : {REPLICATES}")
    logger.info(f"Workers          : {num_workers}")
    logger.info(f"Random seed      : {SEED}")
    if args.target:
        logger.info(f"Single target    : {args.target}")
    if args.train_size is not None:
        if args.train_size.lower() == "full":
            logger.info(f"Training size    : full (no subsampling)")
        else:
            logger.info(f"Training size    : {args.train_size} samples")
    logger.info("================================\n")


    smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)
    # Store all results for aggregate saving
    all_results = []
    
    # Initialize suffix variables (will be set in first iteration)
    desc_suffix = ""
    rdkit_suffix = ""
    bn_suffix = ""
    size_suffix = ""

    for target in target_columns:
        # Extract target values
        ys = df_input.loc[:, target].astype(float).values
        if args.task_type != 'reg':
            ys = ys.astype(int)
        ys = ys.reshape(-1, 1) # reshaping target to be 2D

        # Determine split strategy and generate splits
        n_splits, local_reps = determine_split_strategy(len(ys), REPLICATES)
        
        if n_splits > 1:
            logger.info(f"Using {n_splits}-fold cross-validation with {local_reps} replicate(s)")
        else:
            logger.info(f"Using holdout validation with {local_reps} replicate(s)")

        # Generate data splits
        train_indices, val_indices, test_indices = generate_data_splits(args, ys, n_splits, local_reps, SEED)
        
        # Apply train_size subsampling if specified
        if args.train_size is not None and args.train_size.lower() != "full":
            target_train_size = int(args.train_size)
            logger.info(f"Subsampling training data to {target_train_size} samples")
            
            for i in range(len(train_indices)):
                original_train_size = len(train_indices[i])
                new_train_size = min(target_train_size, original_train_size)
                
                if new_train_size < original_train_size:
                    # Use per-split RNG for reproducible but distinct subsampling
                    rng = np.random.default_rng(SEED + i)  # stable, split-specific
                    subsampled_indices = rng.choice(
                        train_indices[i], 
                        size=new_train_size, 
                        replace=False
                    )
                    train_indices[i] = subsampled_indices
                    logger.info(f"Split {i}: Training set reduced from {original_train_size} to {new_train_size} samples")
                else:
                    logger.info(f"Split {i}: Training set size ({original_train_size}) is already <= target size ({target_train_size}), keeping all samples")
        elif args.train_size is not None and args.train_size.lower() == "full":
            logger.info("Using full training set (no subsampling)")

        for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
            # per-split bookkeeping (mirrors your naming)
            ckpt_path, preprocessing_path, desc_suf, rdkit_suf, bn_suf, size_suf = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            ckpt_path.mkdir(parents=True, exist_ok=True)
            
            # Store suffixes from first iteration for use in aggregate results
            if i == 0:
                desc_suffix = desc_suf
                rdkit_suffix = rdkit_suf
                bn_suffix = bn_suf
                size_suffix = size_suf
            
            # Check if checkpoint exists for this split
            checkpoint_file = ckpt_path / "best.pt"
            skip_training = checkpoint_file.exists()
            
            if skip_training:
                logger.info(f"[{target}] split {i}: Found existing checkpoint, skipping training and loading for evaluation")

            # if combined_descriptor_data is not None:
            #     arts = load_dmpnn_preproc(preprocessing_path, i)
            #     X_all_proc = apply_dmpnn_preproc(combined_descriptor_data, arts)

            #     # slice rows for each fold piece
            #     X_tr = X_all_proc[tr]
            #     X_va = X_all_proc[va]
            #     X_te = X_all_proc[te]
            # else:
            #     X_tr = X_va = X_te = None

            # (ckpt_path / "logs").mkdir(parents=True, exist_ok=True)

            df_tr, df_va, df_te = df_input.iloc[tr].reset_index(drop=True), df_input.iloc[va].reset_index(drop=True), df_input.iloc[te].reset_index(drop=True)

            # datasets
            ds_tr = GraphCSV(df_tr, smiles_column, target, task_type=args.task_type)
            ds_va = GraphCSV(df_va, smiles_column, target, task_type=args.task_type)
            ds_te = GraphCSV(df_te, smiles_column, target, task_type=args.task_type)

            # scaler (fit on train, apply to train+val; test left raw) - only for regression
            if args.task_type == 'reg':
                scaler = StandardScaler().fit(ds_tr.y)
                for d, df_part in [(ds_tr, df_tr), (ds_va, df_va)]:
                    d.y = scaler.transform(d.y)
            else:
                scaler = None  # No scaling for classification

            train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

            # model - determine output channels based on task type
            if args.task_type == 'reg':
                out_channels = 1
            elif args.task_type == 'binary':
                out_channels = 1  # Binary classification uses single logit
            elif args.task_type == 'multi':
                n_classes = n_classes_per_target.get(target, None)
                if n_classes is None:
                    raise ValueError(f"Number of classes not found for target {target}")
                out_channels = n_classes
            else:
                raise ValueError(f"Unsupported task_type: {args.task_type}")
            
            core = models.AttentiveFP(
                in_channels=39, hidden_channels=args.hidden, out_channels=out_channels, edge_dim=10,
                num_layers=2, num_timesteps=2, dropout=0.0
            ).to(device)
            model = EdgeGuard(core, edge_dim=10).to(device)
            
            if not skip_training:
                # Train from scratch
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)

                # train w/ early stopping on val loss
                best_val = float("inf"); best_state = None; no_improve = 0
                E = args.epochs if args.epochs is not None else EPOCHS
                P = args.patience if args.patience is not None else PATIENCE

                for ep in range(1, E+1):
                    model.train(); tr_loss=0; ntr=0
                    for batch in train_loader:
                        batch = batch.to(device)
                        opt.zero_grad()
                        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        
                        # Use appropriate loss function based on task type
                        if args.task_type == 'reg':
                            loss = F.mse_loss(pred.view(-1,1), batch.y.view(-1,1))
                        elif args.task_type == 'binary':
                            loss = F.binary_cross_entropy_with_logits(pred.view(-1), batch.y.view(-1))
                        elif args.task_type == 'multi':
                            loss = F.cross_entropy(pred, batch.y.view(-1).long())
                        else:
                            raise ValueError(f"Unsupported task_type: {args.task_type}")
                        
                        loss.backward(); opt.step()
                        tr_loss += loss.item()*batch.num_graphs; ntr += batch.num_graphs
                    tr_loss /= max(1,ntr)

                    va_loss = eval_loss(model, val_loader, device, task=args.task_type)
                    print(f"[{target}] split {i} | epoch {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

                    if va_loss + 1e-12 < best_val:
                        best_val = va_loss
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        no_improve = 0
                        torch.save({"state_dict": best_state}, ckpt_path / "best.pt")
                    else:
                        no_improve += 1
                        if no_improve >= P:
                            break

                # load best
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            else:
                # Load existing checkpoint
                logger.info(f"[{target}] split {i}: Loading checkpoint from {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location=device)
                model.load_state_dict({k: v.to(device) for k, v in checkpoint["state_dict"].items()})

            # test evaluation with appropriate metrics
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    
                    # Handle different prediction formats for different tasks
                    if args.task_type == 'reg':
                        # Regression: inverse transform scaling
                        pred = scaler.inverse_transform(pred.view(-1,1).cpu().numpy()).reshape(-1)
                        y_pred.append(pred)
                        y_true.append(batch.y.view(-1,1).cpu().numpy().reshape(-1))  # test was NOT scaled
                    elif args.task_type == 'binary':
                        # Binary classification: apply sigmoid to get probabilities
                        pred_prob = torch.sigmoid(pred.view(-1)).cpu().numpy()
                        pred_class = (pred_prob >= 0.5).astype(int)
                        y_pred.append(pred_class)
                        y_true.append(batch.y.view(-1).cpu().numpy().astype(int))
                    elif args.task_type == 'multi':
                        # Multi-class classification: use argmax
                        pred_class = torch.argmax(pred, dim=1).cpu().numpy()
                        y_pred.append(pred_class)
                        y_true.append(batch.y.view(-1).cpu().numpy().astype(int))
                    else:
                        raise ValueError(f"Unsupported task_type: {args.task_type}")
                        
            y_pred = np.concatenate(y_pred); y_true = np.concatenate(y_true)

            # Compute appropriate metrics
            if args.task_type == 'reg':
                m = compute_reg_metrics(y_true, y_pred)
            elif args.task_type == 'binary':
                # For binary, we need probabilities for some metrics
                # Re-compute probabilities for evaluation
                model.eval()
                y_probs = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        pred_prob = torch.sigmoid(pred.view(-1)).cpu().numpy()
                        y_probs.append(pred_prob)
                y_probs = np.concatenate(y_probs)
                m = eval_binary(y_true, y_pred, y_probs)
            elif args.task_type == 'multi':
                # For multi-class, we need probabilities for some metrics
                model.eval()
                y_proba = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        pred_proba = torch.softmax(pred, dim=1).cpu().numpy()
                        y_proba.append(pred_proba)
                y_proba = np.concatenate(y_proba)
                m = eval_multi(y_true, y_pred, y_proba)
            else:
                raise ValueError(f"Unsupported task_type: {args.task_type}")
                
            m['split'] = i
            m['target'] = target
            all_results.append(pd.DataFrame([m]))

            # optional predictions save (same helper you use)
            if args.save_predictions:
                # align to your helper’s filename scheme & IDs
                test_ids = df_te[smiles_column].tolist()
                save_predictions(
                    y_true, y_pred, predictions_dir, args.dataset_name, target,
                    "AttentiveFP", desc_suf, rdkit_suf, "", size_suf, i, logger=logger,
                    test_ids=test_ids
                )

            # optional embedding export (pooled graph reps)
            if args.export_embeddings:
                emb_dir = (results_dir / "embeddings")
                emb_dir.mkdir(parents=True, exist_ok=True)
                # Some PyG versions don’t expose .gnn on AttentiveFP; avoid crashing.
                if not hasattr(model.core, "gnn"):
                    logger.warning("Embeddings export skipped: this AttentiveFP version does not expose `.gnn`.")
                else:
                    class RepExtractor(nn.Module):
                        def __init__(self, core): super().__init__(); self.core = core
                        def forward(self, x, edge_index, edge_attr, batch):
                            return self.core.gnn(x, edge_index, edge_attr, batch)  # [B, hidden]
                    rep = RepExtractor(model.core).to(device).eval()

                if hasattr(model.core, "gnn"):
                    dump(train_loader, df_tr, "train")
                    dump(val_loader,   df_va, "val")
                    dump(test_loader,  df_te, "test")
    # aggregate + save like your script
    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    save_aggregate_results([results_df], results_dir, "AttentiveFP", args.dataset_name, desc_suffix, rdkit_suffix, bn_suffix, size_suffix, logger)





if __name__ == "__main__":
    main()
