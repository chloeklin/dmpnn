import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from chemprop import data, featurizers
from utils import (set_seed, process_data, 
                  create_all_data, build_model_and_trainer, get_metric_list,
                  build_experiment_paths, validate_checkpoint_compatibility, manage_preprocessing_cache,
                  setup_training_environment, load_and_preprocess_data, determine_split_strategy, 
                  generate_data_splits, save_aggregate_results)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')
parser.add_argument('--incl_desc', action='store_true',
                    help='Use dataset-specific descriptors')
parser.add_argument('--incl_rdkit', action='store_true',
                    help='Include RDKit descriptors')
parser.add_argument('--model_name', type=str, default="DMPNN", choices=["DMPNN", "wDMPNN", "PPG"],
                    help='Name of the model to use')
parser.add_argument('--target', type=str, default=None,
                    help='Specific target column to train on (if not specified, trains on all targets)')
parser.add_argument('--batch_norm', action='store_true',
                    help='Enable batch normalization in the model')
args = parser.parse_args()


# Setup training environment with common configuration
setup_info = setup_training_environment(args, model_type="graph")

# Extract commonly used variables for backward compatibility
config = setup_info['config']
chemprop_dir = setup_info['chemprop_dir']
checkpoint_dir = setup_info['checkpoint_dir']
results_dir = setup_info['results_dir']
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
logger.info("================================\n")


smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() if args.model_name == "DMPNN" else featurizers.PolymerMolGraphFeaturizer()
      

# Store all results for aggregate saving
all_results = []

for target in target_columns:
    # Extract target values
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1) # reshaping target to be 2D
    all_data = create_all_data(smis, ys, combined_descriptor_data, MODEL_NAME)

    # Determine split strategy and generate splits
    n_splits, local_reps = determine_split_strategy(len(ys), REPLICATES)
    
    if n_splits > 1:
        logger.info(f"Using {n_splits}-fold cross-validation with {local_reps} replicate(s)")
    else:
        logger.info(f"Using holdout validation with {local_reps} replicate(s)")

    # Generate data splits
    train_indices, val_indices, test_indices = generate_data_splits(args, ys, n_splits, local_reps, SEED)
    


    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    if combined_descriptor_data is not None:
        # Initial data preparation (no imputation yet to avoid leakage)
        orig_Xd = np.asarray(combined_descriptor_data, dtype=np.float64)  # Use float64 first
        
        # Replace inf with NaN
        inf_mask = np.isinf(orig_Xd)
        if np.any(inf_mask):
            logger.warning(f"Found {np.sum(inf_mask)} infinite values, replacing with NaN")
            orig_Xd[inf_mask] = np.nan
        
        
        # Store float32 limits for later use
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        
        logger.debug(f"Original descriptor data shape: {orig_Xd.shape}")
        logger.debug(f"Original descriptor data - Inf count: {np.sum(inf_mask)}")
        
        # Create temporary DataFrame for constant feature detection (with NaNs)
        Xd_temp_df = pd.DataFrame(orig_Xd)
        
        # 1) Remove constants on FULL dataset (consistent across splits)
        non_na_uniques = Xd_temp_df.nunique(dropna=True)
        constant_features = non_na_uniques[non_na_uniques < 2].index.tolist()
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features from full dataset")
            # Remove constant features from original data
            orig_Xd = np.delete(orig_Xd, constant_features, axis=1)
        
        # Check for NaN values but don't impute yet
        nan_mask = np.isnan(orig_Xd)
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} NaN values, will use per-split median imputation")
        

        # Store base cleaning metadata (imputer stats will be added per split)
        base_cleaning_metadata = {
            "imputation_strategy": "median",
            "float32_max": float(float32_max),
            "float32_min": float(float32_min),
            "inf_values_found": bool(np.any(inf_mask)),
            "nan_values_found": bool(np.any(nan_mask))
        }
        
        data_metadata = {
            "original_data_shape": list(orig_Xd.shape),
            "descriptor_columns": descriptor_columns if descriptor_columns else [],
            "rdkit_included": args.incl_rdkit,
            "constant_features_removed": constant_features,
            "post_constant_shape": list(orig_Xd.shape)  # After constant removal
        }
        
        split_metadata = {
            "train_indices": [idx.tolist() for idx in train_indices],
            "val_indices": [idx.tolist() for idx in val_indices], 
            "test_indices": [idx.tolist() for idx in test_indices],
            "random_seed": SEED,
            "correlation_threshold": 0.90,
            "correlation_method": "pearson" if args.task_type == "reg" else "spearman"
        }
    
    # Initialize preprocessing metadata and object storage (outside descriptor block)
    split_preprocessing_metadata = {}
    split_imputers = {}
    
    if combined_descriptor_data is not None:
            
        for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
            # Initialize default values
            correlated_features = []
            # Per-split data cleaning: fit imputer on training data only
            
            # Get training data for this split (after constant removal)
            train_data_split = orig_Xd[tr]
            
            # Fit imputer on training data only
            imputer = None
            if np.any(nan_mask):
                imputer = SimpleImputer(strategy='median')
                train_data_clean = imputer.fit_transform(train_data_split)
            else:
                train_data_clean = train_data_split.copy()
            
            # Apply imputation to all splits using training-fitted imputer
            if imputer is not None:
                all_data_clean = imputer.transform(orig_Xd)
            else:
                all_data_clean = orig_Xd.copy()
            
            # Clip and convert to float32
            all_data_clean = np.clip(all_data_clean, float32_min, float32_max)
            all_data_clean = all_data_clean.astype(np.float32)
            
            # Create DataFrame for correlation analysis
            train_df = pd.DataFrame(all_data_clean[tr])
            
            # Find highly correlated features in training set
            if train_df.shape[1] > 1:
                corr_matrix = train_df.corr(method="pearson" if args.task_type == "reg" else "spearman").abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                correlated_features = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.90)]
                
                if correlated_features:
                    logger.info(f"Split {i}: Removing {len(correlated_features)} correlated features based on training set")
                    keep_features = [col for col in range(train_df.shape[1]) if col not in correlated_features]
                else:
                    keep_features = list(range(train_df.shape[1]))
            else:
                keep_features = list(range(train_df.shape[1]))
            
            # Create mask for features to keep after correlation removal
            mask = np.zeros(all_data_clean.shape[1], dtype=bool)
            mask[keep_features] = True
            
            # Update cleaning metadata with per-split imputer statistics
            cleaning_metadata = base_cleaning_metadata.copy()
            cleaning_metadata["imputer_statistics"] = imputer.statistics_.tolist() if imputer is not None else None

            # Store split-specific preprocessing metadata
            preprocessing_metadata = {
                "cleaning": cleaning_metadata,
                "data_info": data_metadata,
                "splits": split_metadata,
                "split_specific": {
                    "split_id": i,
                    "correlated_features": correlated_features,
                    "keep_features": keep_features,
                    "correlation_mask": mask.tolist()
                },
                "target": target,
                "task_type": args.task_type
            }
            
            # Store metadata and imputer for later saving (after checkpoint_path is created)
            split_preprocessing_metadata[i] = preprocessing_metadata
            split_imputers[i] = imputer

            # Apply preprocessing and masking to datapoints
            def _apply_preprocessing_and_mask(datapoints, row_indices):
                for dp, ridx in zip(datapoints, row_indices):
                    # Get cleaned data for this row
                    row_clean = all_data_clean[ridx]  # Already imputed, clipped, and converted
                    # Apply correlation mask (keep only non-correlated features)
                    dp.x_d = row_clean[mask].astype(np.float32)
            
            _apply_preprocessing_and_mask(train_data[i], tr)
            _apply_preprocessing_and_mask(val_data[i], va)
            _apply_preprocessing_and_mask(test_data[i], te)

            # Enhanced sanity check with more detailed debugging
            logger.debug(f"Split {i}: Applied preprocessing - shape: {all_data_clean.shape}")
            logger.debug(f"Split {i}: Features kept: {np.sum(mask)} out of {len(mask)}")
            logger.debug(f"Split {i}: Imputer fitted on {len(tr)} training samples")
            def _check(dps):
                arrs = []
                for dp in dps:
                    x = np.asarray(dp.x_d, dtype=np.float32)
                    if not np.isfinite(x).all():
                        nan_mask = ~np.isfinite(x)
                        logger.debug(f"Found {nan_mask.sum()} non-finite values in a datapoint")
                        logger.debug(f"Non-finite indices: {np.where(nan_mask)[0]}")
                        logger.debug(f"Non-finite values: {x[nan_mask]}")
                    arrs.append(x)
                
                X = np.stack(arrs, axis=0)   # will fail if lengths differ
                logger.debug(f"Final tabular data - shape: {X.shape}, dtype: {X.dtype}")
                logger.debug(f"Final tabular data - finite values: {np.isfinite(X).all()}")
                logger.debug(f"Final tabular data - NaN count: {np.isnan(X).sum()}")
                logger.debug(f"Final tabular data - Inf count: {np.isinf(X).sum()}")
                
                if not np.isfinite(X).all():
                    # Print some statistics about the non-finite values
                    nan_mask = ~np.isfinite(X)
                    logger.debug("\nNon-finite value statistics:")
                    logger.debug(f"Total non-finite values: {nan_mask.sum()}")
                    logger.debug(f"Non-finite values per feature: {np.sum(nan_mask, axis=0)}")
                    logger.debug(f"Samples with non-finite values: {np.any(nan_mask, axis=1).sum()}")
                
                return X
                
            logger.debug("\n=== Training Data Check ===")
            _check(train_data[i])
            logger.debug("\n=== Validation Data Check ===")
            _check(val_data[i])
            logger.debug("\n=== Test Data Check ===")
            _check(test_data[i])




    # === Train ===
    results_all = []
    num_splits = len(train_data)  # robust for both CV and holdout
    for i in range(num_splits):

        if combined_descriptor_data is not None:
            imputer = split_imputers[i]
            if imputer is not None:
                all_data_clean_i = imputer.transform(orig_Xd)
            else:
                all_data_clean_i = orig_Xd.copy()
            all_data_clean_i = np.clip(all_data_clean_i, float32_min, float32_max).astype(np.float32)

            mask_i = np.array(
                split_preprocessing_metadata[i]['split_specific']['correlation_mask'],
                dtype=bool
            )

            # overwrite x_d for ONLY this split right before dataset construction
            for dp, ridx in zip(train_data[i], train_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
            for dp, ridx in zip(val_data[i],   val_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
            for dp, ridx in zip(test_data[i],  test_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]

        # Create datasets after cleaning
        DS = data.MoleculeDataset if MODEL_NAME == "DMPNN" else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)

        # Chemprop convention:
        # - Fit scaler on train, apply to train/val targets (for training stability)
        # - DO NOT scale test targets; predictions are unscaled by output_transform
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
            # test targets intentionally left unscaled
        

        # Modular metric selection
        n_classes_arg = n_classes_per_target[target] if args.task_type == 'multi' else None
        metric_list = get_metric_list(
            args.task_type,
            target=target,
            n_classes=n_classes_arg,
            df_input=df_input
        )
        batch_norm = args.batch_norm
        
        # Build experiment paths
        checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix = build_experiment_paths(
            args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
        )
        
        # 1) Try to load cached scaler & masks BEFORE normalization
        preprocessing_reused, cached_scaler, correlation_mask, constant_features = manage_preprocessing_cache(
            preprocessing_path, i, combined_descriptor_data, split_preprocessing_metadata, None, logger
        )
        
        # Safety check: ensure cached correlation_mask matches local computation (only if descriptors exist)
        if combined_descriptor_data is not None and i in split_preprocessing_metadata:
            m_local = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'], dtype=bool)
            cm = np.array(correlation_mask, dtype=bool)
            assert m_local.shape == cm.shape and np.all(m_local == cm), \
                "Local correlation mask != cached correlation mask (split consistency issue)"

        # 2) Normalize using the decided scaler
        if combined_descriptor_data is not None:
            if cached_scaler is not None:
                descriptor_scaler = cached_scaler
                train.normalize_inputs("X_d", descriptor_scaler)
                val.normalize_inputs("X_d", descriptor_scaler)
                test.normalize_inputs("X_d", descriptor_scaler)
            else:
                descriptor_scaler = train.normalize_inputs("X_d")
                val.normalize_inputs("X_d", descriptor_scaler)
                test.normalize_inputs("X_d", descriptor_scaler)
                # persist the fitted scaler
                _ = manage_preprocessing_cache(
                    preprocessing_path, i, combined_descriptor_data, split_preprocessing_metadata, descriptor_scaler, logger
                )
            
            # No need to re-apply zero mask since dp.x_d is already sliced to kept features only
            logger.debug(f"Descriptor features already filtered to kept features only for split {i}")
        else:
            descriptor_scaler = None
        
        # Create dataloaders
        train_loader = data.build_dataloader(train, num_workers=num_workers)
        val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)
        
        # Clean up incompatible checkpoints if preprocessing changed
        # Only delete checkpoints if descriptors are used and preprocessing actually changed
        if not preprocessing_reused and checkpoint_path.exists() and combined_descriptor_data is not None:
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed incompatible checkpoint directory: {checkpoint_path}")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Create processed descriptor data for model building (with constant features dropped)
        if combined_descriptor_data is not None:
            # Use orig_Xd (constants already removed)
            processed_descriptor_data = orig_Xd.copy()
            processed_descriptor_data = processed_descriptor_data[:, mask_i]
            logger.info(f"Model input descriptor shape: {processed_descriptor_data.shape} (after constant removal and correlation filtering)")
        else:
            processed_descriptor_data = None
        
        # Assertion to prevent descriptor dimension regression
        desc_len_seen = len(train[0].x_d) if getattr(train[0], "x_d", None) is not None else 0
        if processed_descriptor_data is not None:
            logger.info(f"[split {i}] datapoint descriptor dim: {desc_len_seen}, processed dim: {processed_descriptor_data.shape[1]}")
            assert processed_descriptor_data.shape[1] == desc_len_seen, \
                "Descriptor dim mismatch: datapoints vs processed_descriptor_data."
        
        scaler_arg = scaler if args.task_type == 'reg' else None
        mpnn, trainer = build_model_and_trainer(
            args=args,
            combined_descriptor_data=processed_descriptor_data,
            n_classes=n_classes_arg,
            scaler=scaler_arg,
            checkpoint_path=checkpoint_path,
            batch_norm=batch_norm,
            metric_list=metric_list,
            early_stopping_patience=PATIENCE,
            max_epochs=EPOCHS,
            
        )
        # Validate checkpoint compatibility and get resume path
        descriptor_dim = processed_descriptor_data.shape[1] if processed_descriptor_data is not None else 0
        last_ckpt = validate_checkpoint_compatibility(
            checkpoint_path, preprocessing_path, i, descriptor_dim, logger
        )

        # Train
        
        trainer.fit(mpnn, train_loader, val_loader, ckpt_path=last_ckpt)
        results = trainer.test(dataloaders=test_loader)
        test_metrics = results[0]
        test_metrics['split'] = i  # Add split index to metrics
        results_all.append(test_metrics)
    

    # Convert to DataFrame
    results_df = pd.DataFrame(results_all)
    # Calculate mean/std only for numeric metric columns (exclude 'split')
    numeric_cols = [col for col in results_df.columns if col != 'split']
    mean_metrics = results_df[numeric_cols].mean()
    std_metrics = results_df[numeric_cols].std()

    n_evals = len(results_all)
    logger.info(f"\n[{target}] Mean across {n_evals} splits:\n{mean_metrics}")
    logger.info(f"\n[{target}] Std across {n_evals} splits:\n{std_metrics}")


    # Add target column to results and store for aggregation
    results_df['target'] = target
    all_results.append(results_df)
    
    # Save progressive aggregate results after each target
    save_aggregate_results(all_results, results_dir, MODEL_NAME, args.dataset_name, desc_suffix, rdkit_suffix, batch_norm_suffix, logger)