"""
This script finetunes and tests a Graphormer model (pretrained on PCQM4Mv2)
for graph classification on ogbg-molhiv dataset.

Paper: [Do Transformers Really Perform Bad for Graph Representation?]
(https://arxiv.org/abs/2106.05234)

This flowchart describes the main functional sequence of the provided example.
main
│
└───> train_val_pipeline
      │
      ├───> Load and preprocess dataset
      │
      ├───> Download pretrained model
      │
      ├───> train_epoch
      │     │
      │     └───> Graphormer.forward
      │
      └───> evaluate_network
            │
            └───> Graphormer.inference
"""
import argparse
import random
import math
import sys
import os
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
from accelerate import Accelerator
from datasets import CustomMolDataset
from torch.utils.data import DataLoader, Subset
from model import Graphormer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# Import splitting and metric utilities from your existing codebase
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))
from utils import generate_data_splits, get_metric_list

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Instantiate an accelerator object to support distributed
# training and inference.
accelerator = Accelerator()

def _parse_cols(arg):
    if arg is None: return None
    parts = [p.strip() for p in arg.split(",")]
    # cast any pure ints to int
    out = []
    for p in parts:
        if p.isdigit(): out.append(int(p))
        else: out.append(p)
    return out


def train_epoch(model, optimizer, data_loader, lr_scheduler, loss_fn, metrics):
    """Train for one epoch and compute metrics."""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    
    for (
        batch_labels,
        attn_mask,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
        global_feat,
    ) in data_loader:
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

        loss = loss_fn(batch_scores, batch_labels.to(device))

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
        
        all_preds.append(batch_scores.detach().cpu())
        all_targets.append(batch_labels.cpu())

        # Release GPU memory
        del batch_labels, batch_scores, loss, attn_mask, node_feat
        del in_degree, out_degree, path_data, dist
        th.cuda.empty_cache()

    epoch_loss /= len(data_loader)
    
    # Compute metrics
    preds = th.cat(all_preds)
    targets = th.cat(all_targets)
    
    metric_results = {}
    for metric in metrics:
        metric_results[metric.__class__.__name__] = metric(preds, targets).item()

    return epoch_loss, metric_results


def evaluate_network(model, data_loader, loss_fn, metrics):
    """Evaluate model and compute metrics."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    
    with th.no_grad():
        for (
            batch_labels,
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            global_feat,
        ) in data_loader:
            device = accelerator.device

            batch_scores = model(
                node_feat.to(device),
                in_degree.to(device),
                out_degree.to(device),
                path_data.to(device),
                dist.to(device),
                attn_mask=attn_mask.to(device),
            )

            # Gather all predictions and targets
            all_predictions, all_targets = accelerator.gather_for_metrics(
                (batch_scores, batch_labels.to(device))
            )
            loss = loss_fn(all_predictions, all_targets)

            epoch_loss += loss.item()
            all_preds.append(all_predictions.cpu())
            all_targets.append(all_targets.cpu())

        epoch_loss /= len(data_loader)
        
        # Compute metrics
        preds = th.cat(all_preds)
        targets = th.cat(all_targets)
        
        metric_results = {}
        for metric in metrics:
            metric_results[metric.__class__.__name__] = metric(preds, targets).item()

    return epoch_loss, metric_results


def get_loss_fn(task_type, num_classes=None):
    """Get loss function based on task type."""
    if task_type == 'reg':
        return nn.L1Loss()  # MAE loss (paper uses MAE for PCQM4M)
    elif task_type == 'binary':
        return nn.BCEWithLogitsLoss()
    elif task_type == 'multi':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def train_single_split(params, full_ds, train_idx, val_idx, test_idx, split_num, total_splits):
    """Train and evaluate a single train/val/test split."""
    
    # Create split datasets
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Split {split_num}/{total_splits}")
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"{'='*80}")

    # Create data loaders
    collate_fn = lambda batch: CustomMolDataset.collate(batch, path_max_len=5)
    train_loader = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=params.num_workers)
    val_loader = DataLoader(val_ds, batch_size=params.batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True, num_workers=params.num_workers)
    test_loader = DataLoader(test_ds, batch_size=params.batch_size, shuffle=False,
                             collate_fn=collate_fn, pin_memory=True, num_workers=params.num_workers)

    # Get number of output classes/targets
    num_classes = len(full_ds.target_cols) if hasattr(full_ds, "target_cols") else 1

    # Create model
    model = Graphormer(
        num_classes=num_classes,
        edge_dim=4,
        num_atoms=101,
        num_encoder_layers=params.num_layers,
        embedding_dim=params.hidden_dim,
        ffn_embedding_dim=params.hidden_dim,
        num_attention_heads=params.num_heads,
        dropout=params.dropout,
    )
    model.reset_output_layer_parameters()
    
    # Setup loss and metrics
    loss_fn = get_loss_fn(params.task_type, num_classes)
    metrics = get_metric_list(params.task_type, n_classes=num_classes if params.task_type == 'multi' else None)
    
    # Setup optimizer and scheduler
    steps_per_epoch = math.ceil(len(train_ds) / params.batch_size)
    total_updates = steps_per_epoch * params.num_epochs
    warmup_updates = max(int(0.06 * total_updates), 1)

    optimizer = AdamW(model.parameters(), lr=params.lr, betas=(0.99, 0.999), 
                     eps=1e-8, weight_decay=params.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
    )

    # Training loop
    best_val_metric = float('-inf') if params.task_type != 'reg' else float('inf')
    best_epoch = 0
    best_results = {}
    
    for epoch in range(params.num_epochs):
        train_loss, train_metrics = train_epoch(model, optimizer, train_loader, lr_scheduler, loss_fn, metrics)
        val_loss, val_metrics = evaluate_network(model, val_loader, loss_fn, metrics)
        test_loss, test_metrics = evaluate_network(model, test_loader, loss_fn, metrics)
        
        # Determine primary metric for model selection
        primary_metric_name = list(val_metrics.keys())[0]
        val_primary = val_metrics[primary_metric_name]
        
        # Check if this is the best epoch (lower is better for MAE/RMSE, higher for others)
        is_better = False
        if params.task_type == 'reg':
            is_better = val_primary < best_val_metric
        else:
            is_better = val_primary > best_val_metric
            
        if is_better:
            best_val_metric = val_primary
            best_epoch = epoch
            best_results = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_metrics': train_metrics.copy(),
                'val_metrics': val_metrics.copy(),
                'test_metrics': test_metrics.copy(),
            }
        
        if accelerator.is_main_process and epoch % params.log_frequency == 0:
            logger.info(f"Epoch {epoch+1}/{params.num_epochs} | "
                       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                       f"Val {primary_metric_name}: {val_primary:.4f}")
    
    logger.info(f"\nBest epoch: {best_epoch+1} | Val {primary_metric_name}: {best_val_metric:.4f}")
    logger.info(f"Test metrics at best epoch: {best_results['test_metrics']}")
    
    return best_results, split_num

def train_val_pipeline(params):
    # --- parse CLI values ---
    smiles_col = int(params.smiles_col) if params.smiles_col.isdigit() else params.smiles_col
    descriptor_cols = _parse_cols(params.descriptor_cols)
    target_cols = _parse_cols(params.target_cols)

    # --- Load full dataset once ---
    logger.info(f"Loading dataset from {params.csv_path}")
    full_ds = CustomMolDataset(
        csv_path=params.csv_path,
        smiles_col=smiles_col,
        descriptor_cols=descriptor_cols,
        target_cols=target_cols,
        descriptor_count=params.descriptor_count,
        cache_dir="./graph_cache",
        normalize_descriptors=True,
    )
    
    logger.info(f"Loaded full dataset: {len(full_ds)} samples")
    logger.info(f"Target columns: {full_ds.target_cols}")
    
    # --- Generate splits using same logic as train_graph.py/train_tabular.py ---
    class SplitArgs:
        def __init__(self, task_type):
            self.task_type = task_type
    
    split_args = SplitArgs(task_type=params.task_type)
    
    # Get target values for splitting
    ys = full_ds.targets.numpy()  # shape: (N, num_targets)
    
    # For multi-target, use first target for stratification if classification
    if params.task_type in ['binary', 'multi']:
        y_for_split = ys[:, 0] if ys.ndim > 1 else ys
    else:
        y_for_split = ys
    
    n_splits = params.n_splits if params.n_splits > 1 else 1
    local_reps = params.replicates if params.n_splits == 1 else 1
    
    logger.info(f"Generating splits: n_splits={n_splits}, replicates={local_reps}, seed={params.seed}")
    
    train_indices, val_indices, test_indices = generate_data_splits(
        split_args, y_for_split, n_splits, local_reps, params.seed
    )
    
    total_splits = len(train_indices)
    logger.info(f"Total splits to train: {total_splits}")
    
    # --- Train all splits and collect results ---
    all_results = []
    
    for split_idx in range(total_splits):
        results, _ = train_single_split(
            params, full_ds,
            train_indices[split_idx],
            val_indices[split_idx],
            test_indices[split_idx],
            split_idx + 1,
            total_splits
        )
        
        # Format results for saving
        row = {'split': split_idx}
        
        # Add all metrics
        for metric_name, value in results['test_metrics'].items():
            row[f'test_{metric_name}'] = value
        for metric_name, value in results['val_metrics'].items():
            row[f'val_{metric_name}'] = value
        for metric_name, value in results['train_metrics'].items():
            row[f'train_{metric_name}'] = value
            
        all_results.append(row)
    
    # --- Save results ---
    if accelerator.is_main_process:
        results_df = pd.DataFrame(all_results)
        
        # Create results directory
        results_dir = Path(params.results_dir) / "Graphormer"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        dataset_name = Path(params.csv_path).stem
        filename = f"{dataset_name}_results.csv"
        output_path = results_dir / filename
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*80}")
        
        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(results_df.describe())
        
        # Print mean ± std for each metric
        logger.info("\nMean ± Std across splits:")
        for col in results_df.columns:
            if col != 'split':
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                logger.info(f"{col}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Please give a value for random seed",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please give a value for batch_size",
    )

    # --- Dataset and splitting arguments ---
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with SMILES and targets")
    parser.add_argument("--smiles_col", type=str, default="0",
                        help="SMILES column index or name")
    parser.add_argument("--descriptor_count", type=int, default=None,
                        help="If set, takes the next K columns after SMILES as descriptors.")
    parser.add_argument("--descriptor_cols", type=str, default=None,
                        help="Comma-separated names or indices. Overrides descriptor_count.")
    parser.add_argument("--target_cols", type=str, default=None,
                        help="Comma-separated names or indices. If None, all remaining columns are targets.")
    
    # Splitting arguments (matching train_graph.py/train_tabular.py)
    parser.add_argument("--task_type", type=str, default="reg", choices=["reg", "binary", "multi"],
                        help="Task type: reg (regression), binary, or multi (classification)")
    parser.add_argument("--n_splits", type=int, default=1,
                        help="Number of CV splits (>1 for CV, 1 for holdout)")
    parser.add_argument("--replicates", type=int, default=5,
                        help="Number of replicates for holdout splitting (only used when n_splits=1)")
    
    # Model architecture arguments
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of Graphormer layers")
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="Hidden dimension size")
    parser.add_argument("--num_heads", type=int, default=32,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=16,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--log_frequency", type=int, default=1,
                        help="Log every N epochs")
    
    # Output arguments
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Set manual seed to bind the order of training data to the random seed.
    random.seed(args.seed)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)
    
    # Log configuration
    logger.info("\n" + "="*80)
    logger.info("Graphormer Training Configuration")
    logger.info("="*80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("="*80 + "\n")

    train_val_pipeline(args)