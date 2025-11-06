import argparse
import logging
import numpy as np
import pandas as pd

import joblib
from sklearn.preprocessing import StandardScaler

# Local imports
from tabular_utils import (
    build_features,
    eval_regression,
    eval_binary,
    eval_multi,
    preprocess_descriptor_data,
    save_preprocessing_objects,
    prepare_target_data,
    group_splits
)
from utils import (
    set_seed,
    load_existing_results,
    save_combined_results,
    build_sklearn_models,
    setup_training_environment,
    load_and_preprocess_data,
    determine_split_strategy,
    generate_data_splits,
    
)


# ---------------------------- Training Loop ----------------------------

def train(df, y, target_name, descriptor_columns, replicates, seed, out_dir, args, existing_results=None, smiles_column="smiles"):
    """Train and evaluate models for a single target variable.
    
    Args:
        df: Input DataFrame containing features and target
        y: Target values
        target_name: Name of the target variable
        descriptor_columns: List of descriptor columns to use
        replicates: Number of train/val/test splits
        seed: Random seed
        out_dir: Output directory for results
        args: Command line arguments
        existing_results: Dict of existing results to skip completed experiments
        
    Returns:
        List of dictionaries containing evaluation metrics
    """
    logger = logging.getLogger(__name__)

    # Filter out NaN values for splitting (especially important for copolymers)
    if args.task_type == "reg":
        valid_mask = ~np.isnan(y)
    else:
        valid_mask = ~pd.isna(y)
    
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.info(f"Filtering out {n_invalid} samples with NaN target values for {target_name}")
        df_valid = df.iloc[valid_mask].reset_index(drop=True)
        y_valid = y[valid_mask]
    else:
        df_valid = df
        y_valid = y

    # Determine split strategy and generate splits
    n_splits, local_reps = determine_split_strategy(len(y_valid), replicates)

    if args.polymer_type == "copolymer":
        # produce repeats by reseeding group splits
        train_indices, val_indices, test_indices = [], [], []
        for r in range(local_reps):
            tr, va, te = group_splits(df_valid, y_valid, args.task_type, n_splits, seed + r)
            train_indices.extend(tr)
            val_indices.extend(va)
            test_indices.extend(te)
    else:
        train_indices, val_indices, test_indices = generate_data_splits(
            args, y_valid, n_splits, local_reps, seed
        )
    
    if args.train_size is not None and args.train_size.lower() != "full":
        target_train_size = int(args.train_size)
        logger.info(f"Subsampling training data to {target_train_size} samples")
        
        for i in range(len(train_indices)):
            original_train_size = len(train_indices[i])
            new_train_size = min(target_train_size, original_train_size)
            
            if new_train_size < original_train_size:
                # Use per-split RNG for reproducible but distinct subsampling
                rng = np.random.default_rng(seed + i)  # stable, split-specific
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


    detailed_rows = []
    for i, (train_idx, val_idx, test_idx) in enumerate(zip(train_indices, val_indices, test_indices)):
        # Extract and process features
        ab_block, descriptor_block, feat_names = build_features(df_valid, train_idx, descriptor_columns, args.polymer_type, use_rdkit=args.incl_rdkit, use_ab=args.incl_ab, smiles_column=smiles_column)
        
        orig_desc_names = [n for n in feat_names if not n.startswith('AB_')]

        # Process AB block (no cleaning/selection needed) - only if AB features are included
        if ab_block is not None:
            ab_tr, ab_val, ab_te = ab_block[train_idx], ab_block[val_idx], ab_block[test_idx]
            ab_names = [name for name in feat_names if name.startswith('AB_')]

        
        # Process descriptor block separately (clean and select features)
        if descriptor_block is not None:
            # Use modularized preprocessing function
            (desc_tr_selected, desc_val_selected, desc_te_selected, selected_desc_names, 
             preprocessing_metadata, imputer, constant_mask, corr_mask) = preprocess_descriptor_data(
                descriptor_block, train_idx, val_idx, test_idx, orig_desc_names, logger
            )    
            
            
            # Combine AB block with selected descriptor block
            if ab_block is not None:
                X_tr = np.concatenate([ab_tr, desc_tr_selected], axis=1)
                X_val = np.concatenate([ab_val, desc_val_selected], axis=1)
                X_te = np.concatenate([ab_te, desc_te_selected], axis=1)
                feat_names = ab_names + selected_desc_names
            else:
                # Only descriptor block available
                X_tr, X_val, X_te = desc_tr_selected, desc_val_selected, desc_te_selected
                feat_names = selected_desc_names
        
        else:
            # Only AB block available (or no features if AB disabled)
            X_tr, X_val, X_te = ab_tr, ab_val, ab_te
            feat_names = ab_names

        # Save preprocessing metadata and objects
        if descriptor_block is not None:
            # Update preprocessing metadata with feature names and AB count
            preprocessing_metadata['feat_names'] = feat_names
            preprocessing_metadata['ab_feature_count'] = len(ab_names) if ab_block is not None else 0
            preprocessing_metadata['use_ab'] = ab_block is not None
            
            save_preprocessing_objects(out_dir, i, preprocessing_metadata, imputer, 
                                     constant_mask, corr_mask, selected_desc_names)
        elif ab_block is not None:
            # AB-only case: create minimal preprocessing metadata
            ab_metadata = {
                'feat_names': feat_names,
                'ab_feature_count': len(ab_names),
                'use_ab': True,
                'n_desc_before_any_selection': 0,
                'n_desc_after_constant_removal': 0,
                'n_desc_after_corr_removal': 0,
                'n_desc_after_final_zero_var_removal': 0,
                'constant_features_removed': [],
                'correlated_features_removed': [],
                'zero_var_after_impute_removed': []
            }
            save_preprocessing_objects(out_dir, i, ab_metadata, None, 
                                     np.array([]), np.array([]), [])


        if X_tr.shape[1] == 0:
            logger.warning("Feature selection yielded 0 columns; reverting to AB features only for this split.")
            X_tr, X_val, X_te = ab_tr, ab_val, ab_te
            feat_names = ab_names

        # Initialize target scaler for regression tasks
        target_scaler = None
        if args.task_type == 'reg':
            target_scaler = StandardScaler()
            y_tr = target_scaler.fit_transform(y_valid[train_idx].reshape(-1, 1)).flatten()
            y_val = target_scaler.transform(y_valid[val_idx].reshape(-1, 1)).flatten()
            y_te = y_valid[test_idx]  # Keep original for final evaluation
        else:
            y_tr, y_val, y_te = y_valid[train_idx], y_valid[val_idx], y_valid[test_idx]

        # models
        num_classes = len(np.unique(y_tr)) if args.task_type != "reg" else None
        model_specs = build_sklearn_models(args.task_type, num_classes, scaler_flag=True)

        for name, (model, needs_scaler) in model_specs.items():
            # Check if this target-split-model combination already exists
            if (existing_results and 
                target_name in existing_results and 
                i in existing_results[target_name] and 
                name in existing_results[target_name][i]):
                logger.info(f"Skipping {target_name} split {i} model {name} (already completed)")
                continue
            
            # Apply scaling for models that require it (linear/logistic)
            scaler = StandardScaler() if needs_scaler else None

            if scaler is not None:
                Xtr_fit = scaler.fit_transform(X_tr)
                Xval_fit = scaler.transform(X_val)
                Xte_fit = scaler.transform(X_te)
                
                # save per split **and** per model
                joblib.dump(scaler, out_dir / f"feature_scaler_split_{i}_{name}.pkl")
            else:
                Xtr_fit = X_tr
                Xval_fit = X_val
                Xte_fit = X_te

            if args.task_type == "reg":
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="rmse")
                    model.fit(Xtr_fit, y_tr, eval_set=[(Xval_fit, y_val)], verbose=False)
                else:
                    model.fit(Xtr_fit, y_tr)
                
                # Get predictions and inverse transform if this is regression
                y_pred = model.predict(Xte_fit)
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                metrics = eval_regression(y_te, y_pred)
            elif args.task_type == "binary":
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="logloss")
                    model.fit(Xtr_fit, y_tr, eval_set=[(Xval_fit, y_val)], verbose=False)
                else:
                    model.fit(Xtr_fit, y_tr)
                y_prob = getattr(model, "predict_proba", None)
                prob1 = y_prob(Xte_fit)[:,1] if y_prob is not None else None
                y_pred = model.predict(Xte_fit)
                metrics = eval_binary(y_te, y_pred, prob1)
            else:  # multi
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="mlogloss")
                    model.fit(Xtr_fit, y_tr, eval_set=[(Xval_fit, y_val)], verbose=False)
                else:
                    model.fit(Xtr_fit, y_tr)
                y_proba = getattr(model, "predict_proba", None)
                proba = y_proba(Xte_fit) if y_proba is not None else None
                y_pred = model.predict(Xte_fit)
                metrics = eval_multi(y_te, y_pred, proba)

            row = {"target": target_name, "split": i, "model": name}
            row.update({k: float(v) for k, v in metrics.items()})
            detailed_rows.append(row)
            
            # Save trained model for feature importance analysis
            model_file = out_dir / f"{name}_split_{i}.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved trained model: {model_file}")

    return detailed_rows


# ------------------------------- Main --------------------------------
"""
Train tabular baselines using RDKit and/or graph-pooled features.
Supports regression, binary, and multi-class classification tasks.
"""

def main():
    parser = argparse.ArgumentParser(description='Train tabular baselines (RDKit + Atom/Bond pooled) for regression or classification.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset file (without .csv extension), expected at data/{name}.csv')
    parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                        help='Task type: "reg" (regression), "binary", or "multi" (multi-class)')
    parser.add_argument('--incl_desc', action='store_true',
                    help='Use dataset-specific descriptors')
    parser.add_argument('--incl_rdkit', action='store_true',
                        help='Include RDKit 2D descriptors')
    parser.add_argument('--incl_ab', action='store_true',
                        help='Include atom/bond pooled features')
    parser.add_argument("--polymer_type", type=str, choices=["homo", "copolymer"], default="homo",
                        help='Type of polymer: "homo" for homopolymer or "copolymer" for copolymer')
    parser.add_argument('--train_size', type=str, default=None,
                    help='Number of training samples to use (e.g., "500", "5000", "full"). If not specified, uses full training set.')
    
    args = parser.parse_args()
    
    # Validate train_size argument
    if args.train_size is not None:
        if args.train_size.lower() == "full":
            # "full" is a valid option, no further validation needed
            pass
        else:
            try:
                train_size_int = int(args.train_size)
                if train_size_int <= 0:
                    parser.error("--train_size must be a positive integer or 'full' (e.g., 500, 5000, full)")
            except ValueError:
                parser.error("--train_size must be a valid integer or 'full' (e.g., 500, 5000, full)")


    logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # ----------------------------- Setup -----------------------------
    
    # Setup training environment with common configuration
    setup_info = setup_training_environment(args, model_type="tabular")
    
    # Extract commonly used variables
    chemprop_dir = setup_info['chemprop_dir']
    results_dir = setup_info['results_dir']
    feat_select_dir = setup_info['feat_select_dir']
    descriptor_columns = setup_info['descriptor_columns']
    SEED = setup_info['SEED']
    REPLICATES = setup_info['REPLICATES']

    
    # === Set Random Seed ===
    set_seed(SEED)
    
    # === Load and Preprocess Data ===
    df_input, target_columns = load_and_preprocess_data(args, setup_info)


    # Check for existing results and determine what needs to be run
    suffix = ("_descriptors" if args.incl_desc else "") + ("_rdkit" if args.incl_rdkit else "") + ("_ab" if args.incl_ab else "")
    train_size = getattr(args, 'train_size', None)
    if train_size is not None and train_size.lower() != "full":
        size_suffix = f"__size{train_size}"
    else:
        size_suffix = ""
    suffix += size_suffix
    
    tabular_results_dir = results_dir / "tabular"
    tabular_results_dir.mkdir(exist_ok=True)
    detailed_csv = tabular_results_dir / f"{args.dataset_name}{suffix}.csv"
    
    # Load existing results
    existing_results = load_existing_results(detailed_csv, logger)

    # Process each target variable independently
    all_rows = []
    for tcol in target_columns:
        # Get raw target values
        y_raw = df_input[tcol].values
        
        # Handle different task types using modularized function
        try:
            y_vec = prepare_target_data(y_raw, args.task_type, tcol, logger)
        except ValueError as e:
            logger.warning(f"Skipping {tcol}: {e}")
            continue
        # Create output directory for this target
        out_dir = feat_select_dir / tcol
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Train models and get evaluation results
        target_existing = existing_results.get(tcol, {})
        rows = train(df_input, y_vec, tcol, descriptor_columns, REPLICATES, SEED, out_dir, args, target_existing, setup_info['smiles_column'])

        # Aggregate results across all targets
        all_rows.extend(rows)

    # Save combined results
    existing_df = None
    if detailed_csv.exists() and existing_results:
        try:
            existing_df = pd.read_csv(detailed_csv)
        except Exception as e:
            logger.warning(f"Could not reload existing results: {e}")
    
    save_combined_results(detailed_csv, existing_df, all_rows, logger)    

if __name__ == "__main__":
    main()
