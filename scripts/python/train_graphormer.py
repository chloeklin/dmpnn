#!/usr/bin/env python3
"""
Graphormer Training Script - Integrated with Existing Infrastructure

This script trains Graphormer models following the same patterns as train_graph.py
and train_tabular.py, using the official DGL Graphormer implementation.

Usage:
    python train_graphormer.py --dataset_name insulator --model_name Graphormer
    python train_graphormer.py --dataset_name opv_camb3lyp --model_name Graphormer --train_size 500
"""

import argparse
import os
import sys
import random
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

import torch as th
import torch.nn as nn
from accelerate import Accelerator
from dgl import shortest_dist
from dgl.dataloading import GraphDataLoader
from transformers.optimization import AdamW, get_polynomial_decay_schedule_with_warmup

# Import DGL Graphormer components
sys.path.insert(0, str(Path(__file__).parent / "Graphormer" / "dgl"))
from model import Graphormer

# Import your existing utilities
from chemprop.utils import load_train_config, setup_info_from_config
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.utils import make_split_indices

accelerator = Accelerator()

def load_config():
    """Load training configuration"""
    config_path = Path("scripts/python/train_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_graphormer_dataset(csv_path, smiles_col, target_cols, config):
    """
    Create dataset compatible with Graphormer using your existing data format.
    
    Args:
        csv_path: Path to CSV file
        smiles_col: Column name for SMILES
        target_cols: List of target column names
        config: Training configuration
        
    Returns:
        List of (graph, label) tuples with preprocessed graphs
    """
    df = pd.read_csv(csv_path)
    
    # Initialize featurizer (same as your train_graph.py)
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    
    dataset = []
    failed = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        
        # Extract targets
        targets = []
        for tcol in target_cols:
            if tcol in df.columns:
                val = row[tcol]
                targets.append(val if pd.notna(val) else np.nan)
        
        # Skip if all targets are NaN
        if all(np.isnan(t) if isinstance(t, float) else False for t in targets):
            continue
        
        try:
            # Create molecule datapoint
            mol_datapoint = MoleculeDatapoint.from_smi(smiles)
            
            # Featurize to get DGL graph
            graph = featurizer(mol_datapoint)
            
            # Precompute shortest path distances and paths (required by Graphormer)
            spd, path = shortest_dist(graph, root=None, return_paths=True)
            graph.ndata["spd"] = spd
            graph.ndata["path"] = path
            
            # Convert targets to tensor
            label = th.tensor(targets, dtype=th.float32)
            
            dataset.append((graph, label))
            
        except Exception as e:
            failed.append((idx, smiles, str(e)))
            continue
    
    if failed:
        accelerator.print(f"⚠️  Failed to process {len(failed)}/{len(df)} molecules")
        if len(failed) <= 10:
            for idx, smi, err in failed:
                accelerator.print(f"  Row {idx}: {smi[:50]}... - {err}")
    
    accelerator.print(f"✓ Created dataset with {len(dataset)} molecules")
    return dataset

def collate_graphormer(samples):
    """
    Collate function for Graphormer (adapted from DGL official example).
    Matches the format expected by Graphormer model.
    """
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    
    graphs, labels = map(list, zip(*samples))
    labels = th.stack(labels)
    
    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g in graphs]
    max_num_nodes = max(num_nodes)
    
    # Attention mask for padding (True = invalid position)
    attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1, dtype=th.bool)
    
    node_feat = []
    in_degree, out_degree = [], []
    path_data = []
    
    # Distance matrix (-1 for unreachable/padded)
    dist = -th.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=th.long)
    
    for i in range(num_graphs):
        # Mask padded positions
        attn_mask[i, :, num_nodes[i] + 1:] = True
        
        # Node features (+1 to distinguish padding from real nodes)
        node_feat.append(graphs[i].ndata["feat"] + 1)
        
        # Degree features (clamped to max 512)
        in_degree.append(th.clamp(graphs[i].in_degrees() + 1, min=0, max=512))
        out_degree.append(th.clamp(graphs[i].out_degrees() + 1, min=0, max=512))
        
        # Path encoding
        path = graphs[i].ndata["path"]
        path_len = path.size(dim=2)
        max_len = 5  # multi_hop_max_dist
        
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            shortest_path = F.pad(path, p1d, "constant", -1)
        
        # Pad to max_num_nodes
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
        shortest_path = F.pad(shortest_path, p3d, "constant", -1)
        
        # Edge features (+1 to distinguish padding)
        edata = graphs[i].edata["feat"] + 1
        edata = th.cat((edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0)
        path_data.append(edata[shortest_path])
        
        # Shortest path distances
        dist[i, :num_nodes[i], :num_nodes[i]] = graphs[i].ndata["spd"]
    
    # Pad sequences
    node_feat = pad_sequence(node_feat, batch_first=True)
    in_degree = pad_sequence(in_degree, batch_first=True)
    out_degree = pad_sequence(out_degree, batch_first=True)
    
    return (
        labels.reshape(num_graphs, -1),
        attn_mask,
        node_feat,
        in_degree,
        out_degree,
        th.stack(path_data),
        dist,
    )

def train_epoch(model, optimizer, data_loader, lr_scheduler, task_type):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    if task_type == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
    for batch_labels, attn_mask, node_feat, in_degree, out_degree, path_data, dist in data_loader:
        optimizer.zero_grad()
        device = accelerator.device
        
        batch_scores = model(
            node_feat.to(device),
            in_degree.to(device),
            out_degree.to(device),
            path_data.to(device),
            dist.to(device),
            attn_mask=attn_mask.to(device),
        )
        
        batch_labels = batch_labels.to(device)
        
        if task_type == "regression":
            loss = loss_fn(batch_scores, batch_labels)
        else:
            loss = loss_fn(batch_scores, batch_labels.float())
        
        accelerator.backward(loss)
        
        # Gradient clipping (as in paper)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, task_type):
    """Evaluate model"""
    model.eval()
    epoch_loss = 0
    
    if task_type == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
    all_preds = []
    all_labels = []
    
    with th.no_grad():
        for batch_labels, attn_mask, node_feat, in_degree, out_degree, path_data, dist in data_loader:
            device = accelerator.device
            
            batch_scores = model(
                node_feat.to(device),
                in_degree.to(device),
                out_degree.to(device),
                path_data.to(device),
                dist.to(device),
                attn_mask=attn_mask.to(device),
            )
            
            batch_labels = batch_labels.to(device)
            
            # Gather for metrics
            preds, labels = accelerator.gather_for_metrics((batch_scores, batch_labels))
            
            if task_type == "regression":
                loss = loss_fn(preds, labels)
            else:
                loss = loss_fn(preds, labels.float())
            
            epoch_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = th.cat(all_preds)
    all_labels = th.cat(all_labels)
    
    # Calculate metrics
    if task_type == "regression":
        mse = ((all_preds - all_labels) ** 2).mean().item()
        rmse = np.sqrt(mse)
        mae = (all_preds - all_labels).abs().mean().item()
        
        # R2 score
        ss_res = ((all_labels - all_preds) ** 2).sum()
        ss_tot = ((all_labels - all_labels.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot).item() if ss_tot > 0 else 0.0
        
        return epoch_loss / len(data_loader), {"mae": mae, "rmse": rmse, "r2": r2}
    else:
        # For classification, compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels.numpy(), th.sigmoid(all_preds).numpy())
            return epoch_loss / len(data_loader), {"auc": auc}
        except:
            return epoch_loss / len(data_loader), {"auc": 0.0}

def main(args):
    """Main training function"""
    # Load config
    config = load_config()
    SEED = config['global']['seed']
    REPLICATES = config['global']['replicates']
    
    # Set seeds
    random.seed(SEED)
    th.manual_seed(SEED)
    if th.cuda.is_available():
        th.cuda.manual_seed(SEED)
    
    # Get dataset setup info
    setup_info = setup_info_from_config(args.dataset_name, args.model_name, config)
    csv_path = Path(config['paths']['data_dir']) / f"{args.dataset_name}.csv"
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Training Graphormer on {args.dataset_name}")
    accelerator.print(f"{'='*70}\n")
    
    # Load data
    df = pd.read_csv(csv_path)
    smiles_col = setup_info['smiles_column']
    
    # Get target columns
    ignore_cols = setup_info.get('ignore_columns', []) + [smiles_col]
    if 'dataset_ignore' in config and args.dataset_name in config['dataset_ignore']:
        ignore_cols.extend(config['dataset_ignore'][args.dataset_name])
    
    target_cols = [col for col in df.columns if col not in ignore_cols and df[col].dtype in [np.float64, np.int64]]
    
    accelerator.print(f"Targets: {target_cols}")
    accelerator.print(f"Task type: {args.task_type}")
    
    # Create full dataset
    full_dataset = create_graphormer_dataset(csv_path, smiles_col, target_cols, config)
    
    # Create splits (using your existing split logic)
    indices = list(range(len(full_dataset)))
    train_indices, val_indices, test_indices = make_split_indices(
        indices, 
        split_type='random',
        sizes=(0.8, 0.1, 0.1),
        seed=SEED,
        num_folds=REPLICATES
    )
    
    # Train on each split
    all_results = []
    
    for split_idx in range(REPLICATES):
        accelerator.print(f"\n{'='*70}")
        accelerator.print(f"Split {split_idx + 1}/{REPLICATES}")
        accelerator.print(f"{'='*70}\n")
        
        # Create split datasets
        train_data = [full_dataset[i] for i in train_indices[split_idx]]
        val_data = [full_dataset[i] for i in val_indices[split_idx]]
        test_data = [full_dataset[i] for i in test_indices[split_idx]]
        
        # Apply train_size subsampling if specified
        if args.train_size != "full":
            train_size = int(args.train_size)
            if train_size < len(train_data):
                rng = np.random.default_rng(SEED + split_idx * 9973)
                subsample_idx = rng.choice(len(train_data), size=train_size, replace=False)
                train_data = [train_data[i] for i in subsample_idx]
                accelerator.print(f"Subsampled training data to {len(train_data)} samples")
        
        # Create dataloaders
        train_loader = GraphDataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_graphormer,
            pin_memory=True,
            num_workers=config['global']['num_workers']
        )
        val_loader = GraphDataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_graphormer,
            pin_memory=True,
            num_workers=config['global']['num_workers']
        )
        test_loader = GraphDataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_graphormer,
            pin_memory=True,
            num_workers=config['global']['num_workers']
        )
        
        # Create model
        model = Graphormer(
            num_classes=len(target_cols),
            edge_dim=3,
            num_atoms=4608,
            max_degree=512,
            num_spatial=511,
            multi_hop_max_dist=5,
            num_encoder_layers=args.num_layers,
            embedding_dim=args.hidden_dim,
            ffn_embedding_dim=args.hidden_dim,
            num_attention_heads=args.num_heads,
            dropout=args.dropout,
            pre_layernorm=True,
        )
        
        # Optimizer and scheduler
        total_updates = len(train_loader) * args.epochs
        warmup_updates = int(total_updates * 0.06)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_updates,
            num_training_steps=total_updates,
            lr_end=1e-9,
            power=1.0
        )
        
        # Prepare with accelerator
        model, optimizer, train_loader, val_loader, test_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
        )
        
        # Training loop
        best_val_metric = float('inf') if args.task_type == "regression" else float('-inf')
        best_test_metrics = None
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, optimizer, train_loader, lr_scheduler, args.task_type)
            val_loss, val_metrics = evaluate(model, val_loader, args.task_type)
            test_loss, test_metrics = evaluate(model, test_loader, args.task_type)
            
            # Check for improvement
            if args.task_type == "regression":
                val_metric = val_metrics['mae']
                is_better = val_metric < best_val_metric
            else:
                val_metric = val_metrics['auc']
                is_better = val_metric > best_val_metric
            
            if is_better:
                best_val_metric = val_metric
                best_test_metrics = test_metrics
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or is_better:
                if args.task_type == "regression":
                    accelerator.print(
                        f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                        f"val_mae={val_metrics['mae']:.4f} | test_mae={test_metrics['mae']:.4f} | "
                        f"test_r2={test_metrics['r2']:.4f}"
                    )
                else:
                    accelerator.print(
                        f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                        f"val_auc={val_metrics['auc']:.4f} | test_auc={test_metrics['auc']:.4f}"
                    )
            
            # Early stopping
            if patience_counter >= config['global']['patience']:
                accelerator.print(f"Early stopping at epoch {epoch}")
                break
        
        # Store results
        result = {'split': split_idx}
        result.update({f'test/{k}': v for k, v in best_test_metrics.items()})
        all_results.append(result)
        
        accelerator.print(f"\nSplit {split_idx + 1} best test metrics: {best_test_metrics}")
    
    # Save results using modular function
    results_df = pd.DataFrame(all_results)
    
    # Use modular results saving
    from utils import save_model_results
    import logging
    logger = logging.getLogger(__name__)
    
    results_dir = Path(config['paths']['results_dir'])
    save_model_results(results_df, args, "Graphormer", results_dir, logger)

if __name__ == "__main__":
    # Use modular argument parser
    import sys
    sys.path.append('.')
    from utils import (create_base_argument_parser, add_model_specific_args, 
                      validate_train_size_argument, setup_model_environment, save_model_results)
    
    parser = create_base_argument_parser("Train Graphormer on custom datasets")
    parser = add_model_specific_args(parser, "graphormer")
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_train_size_argument(args, parser)
    
    main(args)
