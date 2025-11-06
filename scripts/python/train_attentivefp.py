#!/usr/bin/env python
import json
from pathlib import Path
from typing import List
import logging
import numpy as np
import pandas as pd
import torch

# ===== import YOUR helpers (exact names from your module) =====
from utils import (
    set_seed,
    setup_training_environment,
    load_and_preprocess_data,
    process_data,
    determine_split_strategy,
    generate_data_splits,
    build_experiment_paths,
    save_predictions,
)

from attentivefp_utils import (
    create_attentivefp_model,
    build_attentivefp_loaders,
    evaluate_model,
    extract_attentivefp_embeddings,
    train_epoch,
    eval_loss,
    save_attentivefp_checkpoint
)


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
    
    # Use modular argument parser
    from utils import (create_base_argument_parser, add_model_specific_args, 
                      validate_train_size_argument, setup_model_environment, save_model_results)
    
    parser = create_base_argument_parser("AttentiveFP training aligned with Chemprop/D-MPNN pipeline")
    parser = add_model_specific_args(parser, "attentivefp")
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_train_size_argument(args, parser)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    logger.info(f"Ignore cols      : {ignore_columns}")
    logger.info(f"Target columns   : {target_columns}")
    logger.info(f"Descriptors      : {'Enabled' if args.incl_desc else 'Disabled'}")
    logger.info(f"RDKit desc.      : {'Enabled' if args.incl_rdkit else 'Disabled'}")
    if args.target:
        logger.info(f"Single target    : {args.target}")
    if args.train_size is not None:
        if args.train_size.lower() == "full":
            logger.info(f"Training size    : full (no subsampling)")
        else:
            logger.info(f"Training size    : {args.train_size} samples")
    logger.info("================================\n")


    _, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)
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
            done_flag = ckpt_path / "TRAINING_COMPLETE"
            skip_training = checkpoint_file.exists() and done_flag.exists()
            
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

            train_loader, val_loader, test_loader, scaler = build_attentivefp_loaders(args, df_tr, df_va, df_te, smiles_column, target, eval=False)

            model = create_attentivefp_model(
                task_type=args.task_type, n_classes=n_classes_per_target.get(target, None), hidden_channels=args.hidden,
                num_layers=2, num_timesteps=2, dropout=0.0
            ).to(device)
            
            
            
            if not skip_training:
                # Train from scratch
                inprog_flag = ckpt_path / "TRAINING_IN_PROGRESS"
                done_flag   = ckpt_path / "TRAINING_COMPLETE"
                inprog_flag.touch(exist_ok=True)
                try:
                    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

                    # train w/ early stopping on val loss
                    best_val = float("inf"); best_state = None; no_improve = 0
                    E = args.epochs if args.epochs is not None else EPOCHS
                    P = args.patience if args.patience is not None else PATIENCE

                    for ep in range(1, E+1):
                        tr_loss = train_epoch(model, train_loader, opt, device, task_type=args.task_type)
                        va_loss = eval_loss(model, val_loader, device, task=args.task_type)
                        print(f"[{target}] split {i} | epoch {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

                        if va_loss + 1e-12 < best_val:
                            best_val = va_loss
                            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                            no_improve = 0
                            save_attentivefp_checkpoint(
                                best_state=best_state,
                                checkpoint_file=checkpoint_file,
                                epoch=ep,
                                best_val=best_val,
                                metrics={"val_loss": best_val},
                                optimizer=opt,
                                target_scaler=scaler,     # your sklearn StandardScaler for regression, or None
                                complete=False
                            )

                        else:
                            no_improve += 1
                            if no_improve >= P:
                                break


                    # load best
                    if best_state is not None:
                        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                    # Load, flip to complete=True, and resave, OR just call again with complete=True
                    save_attentivefp_checkpoint(
                        best_state=best_state,
                        checkpoint_file=checkpoint_file,
                        epoch=ep,
                        best_val=best_val,
                        metrics={"val_loss": best_val},
                        optimizer=opt,
                        target_scaler=scaler,
                        complete=True
                    )
                    done_flag.touch()
                finally:
                    # remove stale “done” if it exists (fresh run)
                    if inprog_flag.exists():
                        inprog_flag.unlink()

            else:
                # Load existing checkpoint
                logger.info(f"[{target}] split {i}: Loading checkpoint from {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
                model.load_state_dict({k: v.to(device) for k, v in checkpoint["state_dict"].items()})

            # test evaluation with appropriate metrics
            m, y_pred, y_true = evaluate_model(model, test_loader, device, args.task_type, scaler)
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
                logger.info(f"Exporting embeddings for split {i}, target {target}")
                
                # Create embeddings directory with target/model/size specificity (same as train_graph.py)
                emb_dir = (results_dir / "embeddings")
                emb_dir.mkdir(parents=True, exist_ok=True)
                
                # Some PyG versions don't expose .gnn on AttentiveFP; avoid crashing.
                
                def dump(loader, split_name):
                    """Extract and save embeddings using the same format as train_graph.py"""
                    X = extract_attentivefp_embeddings(model, loader, device)
                    
                    # Apply same filtering as train_graph.py and evaluate_model.py
                    eps = 1e-8
                    if split_name == "train":
                        std_train = X.std(axis=0)
                        keep = std_train > eps
                        # Save feature mask for reproducibility
                        embedding_prefix = f"{args.dataset_name}__{args.model_name}__{target}{desc_suffix}{rdkit_suffix}{bn_suffix}{size_suffix}"
                        np.save(emb_dir / f"{embedding_prefix}__feature_mask_split_{i}.npy", keep)
                        logger.info(f"Split {i}: Kept {int(keep.sum())} / {len(keep)} embedding dimensions")
                    else:
                        # Use the feature mask from train split
                        embedding_prefix = f"{args.dataset_name}__{args.model_name}__{target}{desc_suffix}{rdkit_suffix}{bn_suffix}{size_suffix}"
                        feature_mask_file = emb_dir / f"{embedding_prefix}__feature_mask_split_{i}.npy"
                        if feature_mask_file.exists():
                            keep = np.load(feature_mask_file)
                        else:
                            # Fallback: keep all features if mask not found
                            keep = np.ones(X.shape[1], dtype=bool)
                            logger.warning(f"Feature mask not found for split {i}, keeping all features")
                    
                    # Apply filtering
                    X_filtered = X[:, keep]
                    
                    # Save embeddings with consistent naming
                    np.save(emb_dir / f"{embedding_prefix}__X_{split_name}_split_{i}.npy", X_filtered)
                    logger.info(f"Split {i}: Saved {split_name} embeddings: {X_filtered.shape}")
                    
                # Extract embeddings for train/val/test
                dump(train_loader, "train")
                dump(val_loader,   "val") 
                dump(test_loader,  "test")
    # aggregate + save using modular function
    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    save_model_results(results_df, args, "AttentiveFP", results_dir, logger)





if __name__ == "__main__":
    main()
