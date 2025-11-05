#!/usr/bin/env python
"""
Clean AttentiveFP Training Script

This script provides a streamlined AttentiveFP training pipeline without
the DMPNN/Lightning dependencies that don't apply to AttentiveFP.
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

# AttentiveFP-specific utilities
from attentivefp_utils import (
    GraphCSV,
    create_attentivefp_model,
    train_epoch,
    eval_loss,
    evaluate_model,
    save_attentivefp_checkpoint,
    load_attentivefp_checkpoint,
    extract_embeddings,
    create_data_splits
)

# Minimal imports from existing utils
from utils import set_seed, load_and_preprocess_data


def setup_attentivefp_environment(args):
    """Setup environment specifically for AttentiveFP using existing infrastructure."""
    # Use existing setup function from utils.py
    from utils import setup_model_environment
    
    setup_info = setup_model_environment(args, "attentivefp")
    
    # Create AttentiveFP-specific checkpoint directory
    checkpoint_dir = setup_info['checkpoint_dir'] / "AttentiveFP"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Update checkpoint_dir in setup_info
    setup_info['checkpoint_dir'] = checkpoint_dir
    
    return setup_info


def create_checkpoint_name(args, target, split_idx):
    """Create checkpoint filename for AttentiveFP."""
    parts = [args.dataset_name, target]
    
    if args.train_size and args.train_size != "full":
        parts.append(f"size{args.train_size}")
    
    parts.append(f"rep{split_idx}")
    
    return "__".join(parts) + ".pt"


def train_attentivefp_split(args, df, target, train_idx, val_idx, test_idx, 
                           checkpoint_path, device, logger):
    """Train AttentiveFP for a single data split."""
    
    # Create datasets
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # Get number of classes for multi-class tasks
    n_classes = None
    if args.task_type == 'multi':
        n_classes = df[target].dropna().nunique()
        logger.info(f"Multi-class task detected: {n_classes} classes")
    
    # Create AttentiveFP datasets
    train_dataset = GraphCSV(train_df, args.smiles_column, target, args.task_type)
    val_dataset = GraphCSV(val_df, args.smiles_column, target, args.task_type)
    test_dataset = GraphCSV(test_df, args.smiles_column, target, args.task_type)
    
    if len(train_dataset) == 0:
        logger.warning("No valid training data after processing")
        return None
    
    # Setup target scaling for regression
    scaler = None
    if args.task_type == 'reg':
        scaler = StandardScaler()
        train_dataset.y = scaler.fit_transform(train_dataset.y)
        val_dataset.y = scaler.transform(val_dataset.y)
        # Keep test targets unscaled for evaluation
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model - match original parameters exactly
    model = create_attentivefp_model(
        task_type=args.task_type,
        n_classes=n_classes,
        hidden_channels=args.hidden,
        num_layers=2,  # Match original
        num_timesteps=2,  # Match original
        dropout=0.0  # Match original (no dropout)
    ).to(device)
    
    # Setup optimizer - match original
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop - match original logic exactly
    best_val = float("inf")
    best_state = None
    no_improve = 0
    E = args.epochs
    P = args.patience
    
    logger.info(f"Starting training for {E} epochs with patience {P}...")
    
    for ep in range(1, E + 1):
        # Train epoch - match original
        tr_loss = train_epoch(model, train_loader, optimizer, device, args.task_type)
        
        # Validate - match original (use eval_loss, not metrics)
        va_loss = eval_loss(model, val_loader, device, args.task_type)
        
        logger.info(f"[{target}] split {checkpoint_path.stem} | epoch {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        
        # Early stopping - match original logic exactly
        if va_loss + 1e-12 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
            # Save checkpoint in original format
            torch.save({"state_dict": best_state}, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= P:
                logger.info(f"Early stopping at epoch {ep}")
                break
    
    # Load best model - match original
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test evaluation - match original exactly
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Handle different prediction formats for different tasks - match original
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
    
    # Concatenate results
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # Calculate metrics - match original
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
    
    test_metrics = {}
    if args.task_type == 'reg':
        test_metrics['mae'] = mean_absolute_error(y_true, y_pred)
        test_metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        test_metrics['r2'] = r2_score(y_true, y_pred)
    else:  # binary or multi-class
        test_metrics['accuracy'] = accuracy_score(y_true, y_pred)
        test_metrics['f1'] = f1_score(y_true, y_pred, average='weighted' if args.task_type == 'multi' else 'binary')
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Export embeddings if requested
    if args.export_embeddings:
        embeddings_dir = checkpoint_path.parent / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        
        # Extract embeddings
        train_emb = extract_embeddings(model, train_loader, device)
        val_emb = extract_embeddings(model, val_loader, device)
        test_emb = extract_embeddings(model, test_loader, device)
        
        # Save embeddings
        split_suffix = f"_split_{checkpoint_path.stem.split('rep')[-1].replace('.pt', '')}"
        np.save(embeddings_dir / f"X_train{split_suffix}.npy", train_emb)
        np.save(embeddings_dir / f"X_val{split_suffix}.npy", val_emb)
        np.save(embeddings_dir / f"X_test{split_suffix}.npy", test_emb)
        
        logger.info(f"Embeddings saved to {embeddings_dir}")
    
    return test_metrics


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Parse arguments using standardized parser
    from utils import create_base_argument_parser, add_model_specific_args
    
    parser = create_base_argument_parser(description="Clean AttentiveFP Training")
    add_model_specific_args(parser, "attentivefp")
    
    # Add AttentiveFP-specific arguments
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set model_name for compatibility with setup_training_environment
    args.model_name = "AttentiveFP"
    
    logger.info("=== AttentiveFP Training ===")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Task type: {args.task_type}")
    logger.info(f"Device: {args.device}")
    
    # Setup environment
    setup_info = setup_attentivefp_environment(args)
    args.smiles_column = setup_info['smiles_column']
    
    # Set seed
    set_seed(setup_info['SEED'])
    device = torch.device(args.device)
    
    # Load data (minimal processing - just CSV loading)
    df_input, target_columns = load_and_preprocess_data(args, setup_info)
    
    # Filter targets if specified
    if args.target:
        if args.target not in target_columns:
            logger.error(f"Target '{args.target}' not found. Available: {target_columns}")
            return
        target_columns = [args.target]
    
    # Results storage
    all_results = []
    
    # Process each target
    for target in target_columns:
        logger.info(f"\n=== Training target: {target} ===")
        
        # Create data splits
        splits, valid_df = create_data_splits(
            df_input, target, n_splits=setup_info['REPLICATES'], random_state=setup_info['SEED']
        )
        
        # Train each split
        for split_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Split {split_idx + 1}/{len(splits)} ---")
            
            # Create checkpoint path
            checkpoint_name = create_checkpoint_name(args, target, split_idx)
            checkpoint_path = setup_info['checkpoint_dir'] / checkpoint_name
            
            # Check if checkpoint exists
            if checkpoint_path.exists():
                logger.info(f"Checkpoint exists: {checkpoint_path}")
                logger.info("Skipping training (remove checkpoint to retrain)")
                continue
            
            # Train split
            test_metrics = train_attentivefp_split(
                args, valid_df, target, train_idx, val_idx, test_idx,
                checkpoint_path, device, logger
            )
            
            if test_metrics:
                # Store results
                result = {'target': target, 'split': split_idx}
                for metric, value in test_metrics.items():
                    result[f'test/{metric}'] = value
                all_results.append(result)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Create results directory
        results_dir = setup_info['results_dir'] / "AttentiveFP"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        filename_parts = [args.dataset_name]
        if args.train_size and args.train_size != "full":
            filename_parts.append(f"size{args.train_size}")
        if args.target:
            filename_parts.append(f"target_{args.target}")
        
        results_file = results_dir / ("__".join(filename_parts) + "_results.csv")
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"\n✓ Results saved to: {results_file}")
        
        # Print summary
        logger.info("\n=== Summary Statistics ===")
        for col in results_df.columns:
            if col.startswith('test/'):
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                logger.info(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    
    logger.info("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
