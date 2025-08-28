import argparse
import logging
import re
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd


from chemprop import data, featurizers, nn
from utils import (set_seed, process_data, make_repeated_splits, 
                  load_drop_indices, 
                  create_all_data, build_model_and_trainer, get_metric_list, load_config, filter_insulator_data, select_features_remove_constant_and_correlated)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')
parser.add_argument('--descriptor', action='store_true',
                    help='Use dataset-specific descriptors')
parser.add_argument('--incl_rdkit', action='store_true',
                    help='Include RDKit descriptors')
parser.add_argument('--model_name', type=str, default="DMPNN",choices=["DMPNN", "wDMPNN","periodic"],
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
descriptor_columns = DATASET_DESCRIPTORS if args.descriptor else []

# === Set Random Seed ===
set_seed(SEED)

# === Load Data ===
df_input = pd.read_csv(input_path)

# Apply insulator dataset filtering if needed
if args.dataset_name == "insulator" and args.model_name == "wDMPNN":
    df_input = filter_insulator_data(args, df_input, smiles_column)

# Read the saved exclusions from the wDMPNN preprocessing step
if args.model_name == "DMPNN":
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
logger.info(f"Descriptors      : {'Enabled' if args.descriptor else 'Disabled'}")
logger.info(f"RDKit desc.      : {'Enabled' if args.incl_rdkit else 'Disabled'}")
logger.info(f"Epochs           : {EPOCHS}")
logger.info(f"Replicates       : {REPLICATES}")
logger.info(f"Workers          : {num_workers}")
logger.info(f"Random seed      : {SEED}")
logger.info("================================\n")


smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() if args.model_name == "DMPNN" else featurizers.PolymerMolGraphFeaturizer()
      

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
        # Debug: Check for non-finite values in the original data
        orig_Xd = np.asarray(combined_descriptor_data, dtype=np.float32)
        logger.debug(f"Original descriptor data shape: {orig_Xd.shape}")
        logger.debug(f"Original descriptor data - finite values: {np.isfinite(orig_Xd).all()}")
        logger.debug(f"Original descriptor data - NaN count: {np.isnan(orig_Xd).sum()}")
        logger.debug(f"Original descriptor data - Inf count: {np.isinf(orig_Xd).sum()}")
        
        Xd_df = pd.DataFrame(orig_Xd)
            
        for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
        # 1) fit selector on TRAIN ONLY
            sel = select_features_remove_constant_and_correlated(
                X_train=Xd_df.iloc[tr],
                y_train=pd.Series(ys.squeeze()[tr]) if args.task_type == "reg" else pd.Series(ys.squeeze()[tr]),
                corr_threshold=0.90,
                method="pearson" if args.task_type == "reg" else "spearman",
                min_unique=2,
                verbose=True
            )
            keep = sel["kept"]
            keep_idx = Xd_df.columns.get_indexer(keep)

            # 2) apply same mask row-wise from the ORIGINAL matrix
            def _apply_mask_from_source(datapoints, row_indices, col_indices):
                for dp, ridx in zip(datapoints, row_indices):
                    row = orig_Xd[ridx]                  # pick THIS datapoint's row
                    dp.x_d = row[col_indices].astype(np.float32, copy=False)

            _apply_mask_from_source(train_data[i], tr, keep_idx)
            _apply_mask_from_source(val_data[i],   va, keep_idx)
            _apply_mask_from_source(test_data[i],  te, keep_idx)

            # Enhanced sanity check with more detailed debugging
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
        # Function to clean and process tabular data
        def clean_tabular_data(datapoints):
            if not datapoints or not hasattr(datapoints[0], 'x_d'):
                return None
                
            # Convert to numpy array with float64 dtype first for better precision
            X = np.vstack([np.asarray(dp.x_d, dtype=np.float64) for dp in datapoints])
            
            # Replace inf with NaN
            inf_mask = np.isinf(X)
            if np.any(inf_mask):
                logger.warning(f"Found {np.sum(inf_mask)} infinite values, replacing with NaN")
                X[inf_mask] = np.nan
            
            # Handle NaN values with median imputation
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                logger.warning(f"Found {np.sum(nan_mask)} NaN values, using median imputation")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Clip extreme values to prevent float32 overflow
            float32_max = np.finfo(np.float32).max
            float32_min = np.finfo(np.float32).min
            X = np.clip(X, float32_min, float32_max)
            
            # Update the data points with cleaned features (now safely convert to float32)
            for idx, dp in enumerate(datapoints):
                dp.x_d = X[idx].astype(np.float32)
            
            return X
        
        # Clean data before creating datasets
        logger.debug("\n=== Cleaning tabular data ===")
        Xtr = clean_tabular_data(train_data[i])
        logger.debug("Training data processed - shape:", Xtr.shape if Xtr is not None else "No tabular data")
        
        Xval = clean_tabular_data(val_data[i])
        logger.debug("Validation data processed - shape:", Xval.shape if Xval is not None else "No tabular data")
        
        Xtest = clean_tabular_data(test_data[i])
        logger.debug("Test data processed - shape:", Xtest.shape if Xtest is not None else "No tabular data")
        
        # Create datasets after cleaning
        DS = data.MoleculeDataset if MODEL_NAME == "DMPNN" else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)

        # Normalize targets
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
        # Normalize descriptors (if any)
        X_d_transform = None
        # Verify data is clean
        Xtr = np.vstack([np.asarray(dp.x_d, dtype=np.float32) for dp in train_data[i]])
        logger.debug("\n=== Data Verification ===")
        logger.debug("Tabular data shape:", Xtr.shape)
        logger.debug("Finite values:", np.isfinite(Xtr).all())
        logger.debug("Max absolute values (first 10 features):", np.nanmax(np.abs(Xtr), axis=0)[:10])
        
        if combined_descriptor_data is not None:
            descriptor_scaler = train.normalize_inputs("X_d")
            val.normalize_inputs("X_d", descriptor_scaler)
            X_d_transform = nn.ScaleTransform.from_standard_scaler(descriptor_scaler)
        
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
        # Create checkpoint directory structure
        checkpoint_path = (
            checkpoint_dir / 
            f"{args.dataset_name}__{target}"
            f"{'__desc' if descriptor_columns else ''}"
            f"{'__rdkit' if args.incl_rdkit else ''}"
            f"__rep{i}"
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        scaler_arg = scaler if args.task_type == 'reg' else None
        mpnn, trainer = build_model_and_trainer(
            args=args,
            combined_descriptor_data=combined_descriptor_data,
            n_classes=n_classes_arg,
            scaler=scaler_arg,
            X_d_transform=X_d_transform,
            checkpoint_path=checkpoint_path,
            batch_norm=batch_norm,
            metric_list=metric_list
        )
        last_ckpt = None
        if os.path.exists(checkpoint_path):
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
            if ckpt_files:
                last_ckpt = str(Path(checkpoint_path) / sorted(ckpt_files)[-1])

        # Train
        trainer.fit(mpnn, train_loader, val_loader, ckpt_path=last_ckpt)
        results = trainer.test(dataloaders=test_loader)
        test_metrics = results[0]
        results_all.append(test_metrics)
    

    # Convert to DataFrame
    results_df = pd.DataFrame(results_all)
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()

    logger.info(f"\n[{target}] Mean across {REPLICATES} splits:\n{mean_metrics}")
    logger.info(f"\n[{target}] Std across {REPLICATES} splits:\n{std_metrics}")


    # Optional: save to file
    results_df.to_csv(results_dir / f"{args.dataset_name}_{target}{"_descriptors" if descriptor_columns is not None else ""}{"_rdkit" if args.incl_rdkit else ""}_{MODEL_NAME}_results.csv", index=False)
