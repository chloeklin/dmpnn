import argparse
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.impute import SimpleImputer
from joblib import dump

from chemprop import data, featurizers
from utils import (set_seed, process_data, make_repeated_splits, 
                  load_drop_indices, 
                  create_all_data, build_model_and_trainer, get_metric_list, load_config, filter_insulator_data)

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
args = parser.parse_args()


# Load configuration
config = load_config()

# Set up paths and parameters
chemprop_dir = Path.cwd()

# Get paths from config with defaults
paths = config.get('PATHS', {})
data_dir = chemprop_dir / paths.get('data_dir', 'data')
checkpoint_dir = chemprop_dir / paths.get('checkpoint_dir', 'checkpoints') / args.model_name
results_dir = chemprop_dir / paths.get('results_dir', 'results') / args.model_name
feat_select_dir = chemprop_dir / paths.get('feat_select_dir', 'out') / args.model_name / args.dataset_name

# Create necessary directories
checkpoint_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
feat_select_dir.mkdir(parents=True, exist_ok=True)

# Set up file paths
input_path = chemprop_dir / data_dir / f"{args.dataset_name}.csv"

# Get model-specific configuration
model_config = config['MODELS'].get(args.model_name, {})
if not model_config:
    logger.warning(f"No configuration found for model '{args.model_name}'. Using defaults.")

# Set parameters from config with defaults
GLOBAL_CONFIG = config.get('GLOBAL', {})
SEED = GLOBAL_CONFIG.get('SEED', 42)

EPOCHS = GLOBAL_CONFIG.get('EPOCHS', 300)
PATIENCE = GLOBAL_CONFIG.get('PATIENCE', 30)
num_workers = min(
    GLOBAL_CONFIG.get('NUM_WORKERS', 8),
    os.cpu_count() or 1
)

# Model-specific parameters
smiles_column = model_config.get('smiles_column', 'smiles')
ignore_columns = model_config.get('ignore_columns', [])
MODEL_NAME = args.model_name
REPLICATES = GLOBAL_CONFIG.get('REPLICATES', 5)
# Get dataset descriptors from config
DATASET_DESCRIPTORS = config.get('DATASET_DESCRIPTORS', {}).get(args.dataset_name, [])
descriptor_columns = DATASET_DESCRIPTORS if args.incl_desc else []

# === Set Random Seed ===
set_seed(SEED)

# === Load Data ===
df_input = pd.read_csv(input_path)

# Apply insulator dataset filtering if needed
if args.dataset_name == "insulator" and args.model_name == "wDMPNN":
    df_input = filter_insulator_data(args, df_input, smiles_column)

# Read the saved exclusions from the wDMPNN preprocessing step
if args.model_name != "wDMPNN":
    drop_idx, excluded_smis = load_drop_indices(chemprop_dir, args.dataset_name)
    if drop_idx:
        logger.info(f"Dropping {len(drop_idx)} rows from {args.dataset_name} due to exclusions.")
        df_input = df_input.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


# Automatically detect target columns (all columns except ignored ones)
target_columns = [c for c in df_input.columns
                 if c not in ([smiles_column] + 
                            DATASET_DESCRIPTORS + 
                            ignore_columns)]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")



logger.info("\n=== Training Configuration ===")
logger.info(f"Dataset          : {args.dataset_name}")
logger.info(f"Task type        : {args.task_type}")
logger.info(f"Model            : {args.model_name}")
logger.info(f"SMILES column    : {smiles_column}")
logger.info(f"Descriptor cols  : {descriptor_columns}")
logger.info(f"Ignore columns   : {ignore_columns}")
logger.info(f"Descriptors      : {'Enabled' if args.incl_desc else 'Disabled'}")
logger.info(f"RDKit desc.      : {'Enabled' if args.incl_rdkit else 'Disabled'}")
logger.info(f"Epochs           : {EPOCHS}")
logger.info(f"Replicates       : {REPLICATES}")
logger.info(f"Workers          : {num_workers}")
logger.info(f"Random seed      : {SEED}")
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

    # Decide CV vs holdout
    # For small datasets (<2000 samples), use 5-fold CV with 1 replicate
    # For larger datasets, use a single train/val/test split with multiple replicates
    n_splits = 5 if len(ys) < 2000 else 1
    local_reps = 1 if n_splits > 1 else REPLICATES
    
    if n_splits > 1:
        logger.info(f"Using {n_splits}-fold cross-validation with {local_reps} replicate(s)")
    else:
        logger.info(f"Using holdout validation with {local_reps} replicate(s)")

    # === Split via Random/Stratified Split with 5 Repetitions ===
    if args.task_type in ['binary', 'multi']:
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=local_reps,
            seed=SEED,
            y_class=ys,
            n_splits=n_splits
            
        )
    else:
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=local_reps,
            seed=SEED,
            y_reg=ys,
            n_splits=n_splits
        )
    


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
        
        # Initialize preprocessing metadata and object storage
        split_preprocessing_metadata = {}
        split_imputers = {}
            
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
                    # Apply correlation mask (zero out dropped features)
                    out = np.zeros_like(row_clean, dtype=np.float32)
                    out[mask] = row_clean[mask]
                    dp.x_d = out
            
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
        # Create datasets after cleaning
        DS = data.MoleculeDataset if MODEL_NAME == "DMPNN" else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)

        # Chemprop convention:
        # - Fit scaler on train, apply to train/val targets (for training stability)
        # - DO NOT scale test targets; predictions are unscaled by output_transform
        if args.task_type == 'reg':
            scaler  = train.normalize_targets()
            val.normalize_targets(scaler)
            # test targets intentionally left unscaled
        

        if combined_descriptor_data is not None:
            # normalise descriptors
            descriptor_scaler = train.normalize_inputs("X_d")
            val.normalize_inputs("X_d", descriptor_scaler)
            test.normalize_inputs("X_d", descriptor_scaler)
            
            # Re-apply zero mask after scaling to prevent mean-subtraction from unzeroing features
            correlation_mask = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'])
            
            def _reapply_zero_mask(datapoints):
                for dp in datapoints:
                    if hasattr(dp, 'x_d') and dp.x_d is not None:
                        # Re-zero the dropped features after scaling
                        dp.x_d[~correlation_mask] = 0.0
            
            _reapply_zero_mask(train)
            _reapply_zero_mask(val)
            _reapply_zero_mask(test)
            
            logger.debug(f"Re-applied zero mask after scaling for split {i}")
        
        # Create dataloaders
        train_loader = data.build_dataloader(train, num_workers=num_workers)
        val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)

        # Modular metric selection
        n_classes_arg = n_classes_per_target[target] if args.task_type == 'multi' else None
        metric_list = get_metric_list(
            args.task_type,
            target=target,
            n_classes=n_classes_arg,
            df_input=df_input
        )
        batch_norm = False
        # Create checkpoint directory structure with feature configuration
        desc_suffix = "__desc" if descriptor_columns else ""
        rdkit_suffix = "__rdkit" if args.incl_rdkit else ""
        
        checkpoint_path = (
            checkpoint_dir / 
            f"{args.dataset_name}__{target}{desc_suffix}{rdkit_suffix}__rep{i}"
        )
        
        # Create separate preprocessing directory to avoid Lightning conflicts
        preprocessing_path = (
            chemprop_dir / "preprocessing" / "DMPNN" /
            f"{args.dataset_name}__{target}{desc_suffix}{rdkit_suffix}__rep{i}"
        )
        preprocessing_path.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing checkpoints to prevent Lightning auto-resume with incompatible models
        import glob
        existing_ckpts = glob.glob(str(checkpoint_path / "*.ckpt"))
        if existing_ckpts:
            for ckpt in existing_ckpts:
                os.remove(ckpt)
                logger.info(f"Removed incompatible checkpoint: {ckpt}")
        
        # Save preprocessing metadata and objects
            
            # Save descriptor scaler
            dump(descriptor_scaler, preprocessing_path / "descriptor_scaler.pkl")
            logger.info(f"Saved descriptor scaler to {preprocessing_path / 'descriptor_scaler.pkl'}")
            
            # Save correlation mask as numpy array
            correlation_mask = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'])
            np.save(preprocessing_path / "correlation_mask.npy", correlation_mask)
            logger.info(f"Saved correlation mask to {preprocessing_path / 'correlation_mask.npy'}")
            
            # Save constant features removed as numpy array
            constant_features = split_preprocessing_metadata[i]['data_info']['constant_features_removed']
            np.save(preprocessing_path / "constant_features_removed.npy", np.array(constant_features, dtype=np.int64))
            logger.info(f"Saved constant features to {preprocessing_path / 'constant_features_removed.npy'}")
        
        scaler_arg = scaler if args.task_type == 'reg' else None
        mpnn, trainer = build_model_and_trainer(
            args=args,
            combined_descriptor_data=combined_descriptor_data,
            n_classes=n_classes_arg,
            scaler=scaler_arg,
            checkpoint_path=checkpoint_path,
            batch_norm=batch_norm,
            metric_list=metric_list,
            early_stopping_patience=PATIENCE,
            max_epochs=EPOCHS,
            
        )
        last_ckpt = None
        if os.path.exists(checkpoint_path):
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
            if ckpt_files:
                # Check if preprocessing metadata exists and matches current preprocessing
                metadata_file = preprocessing_path / f"preprocessing_metadata_split_{i}.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            saved_metadata = json.load(f)
                        
                        # Compare key preprocessing parameters
                        current_n_features = combined_descriptor_data.shape[1] if combined_descriptor_data is not None else 0
                        saved_n_features = saved_metadata.get('data_info', {}).get('n_features_after_preprocessing', 0)
                        
                        if current_n_features == saved_n_features:
                            last_ckpt = str(Path(checkpoint_path) / sorted(ckpt_files)[-1])
                            logger.info(f"Loading checkpoint: {last_ckpt} (features match: {current_n_features})")
                        else:
                            logger.warning(f"Skipping checkpoint due to feature mismatch: current={current_n_features}, saved={saved_n_features}")
                            logger.warning("Starting training from scratch")
                    except Exception as e:
                        logger.warning(f"Could not validate checkpoint compatibility: {e}")
                        logger.warning("Starting training from scratch")
                else:
                    # No metadata file, assume incompatible
                    logger.warning("No preprocessing metadata found for checkpoint validation")
                    logger.warning("Starting training from scratch")

        # Train - force fresh start if no compatible checkpoint
        
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
    
    # Save progressive aggregate results after each target (same filename, updated each time)
    suffix_desc  = "_descriptors" if args.incl_desc else ""
    suffix_rdkit = "_rdkit"       if args.incl_rdkit else ""
    model_results_dir = results_dir / MODEL_NAME
    model_results_dir.mkdir(exist_ok=True)
    aggregate_csv = model_results_dir / f"{args.dataset_name}{suffix_desc}{suffix_rdkit}_results.csv"
    
    # Combine all completed targets so far
    current_aggregate_df = pd.concat(all_results, ignore_index=True)
    
    # Organize columns: target, split, then metrics
    base_cols = ["target", "split"]
    metric_cols = sorted([c for c in current_aggregate_df.columns if c not in base_cols])
    current_aggregate_df = current_aggregate_df[base_cols + metric_cols]
    
    # Overwrite the same file with updated results
    current_aggregate_df.to_csv(aggregate_csv, index=False)
    logger.info(f"Updated aggregate results with {target} -> {aggregate_csv}")