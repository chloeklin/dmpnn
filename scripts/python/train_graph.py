import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import os, json
import torch

from chemprop import data, featurizers
from utils import (set_seed, process_data, 
                  create_all_data, build_model_and_trainer, get_metric_list,
                  build_experiment_paths, validate_checkpoint_compatibility, manage_preprocessing_cache,
                  setup_training_environment, load_and_preprocess_data, determine_split_strategy, 
                  generate_data_splits, save_aggregate_results, get_encodings_from_loader, save_predictions,
                  create_base_argument_parser, add_model_specific_args, validate_train_size_argument,
                  setup_model_environment, save_model_results, pick_best_checkpoint)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments using modular parser
parser = create_base_argument_parser('Train a Chemprop model for regression or classification')
parser = add_model_specific_args(parser, "dmpnn")

args = parser.parse_args()

# Validate arguments
validate_train_size_argument(args, parser)

# Setup training environment with common configuration
setup_info = setup_model_environment(args, "dmpnn")

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
import lightning.pytorch as pl
pl.seed_everything(SEED, workers=True)

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


if args.pretrain_monomer:
    assert args.model_name == "DMPNN", "Monomer pretraining uses the small-molecule D-MPNN."
    # Parse multiclass specs
    mc_map = {}
    if args.multiclass_targets:
        for tok in args.multiclass_targets.split(","):
            name, k = tok.split(":")
            mc_map[name.strip()] = int(k)

    # Build task_specs aligned with target_columns
    task_specs = []
    for t in target_columns:
        if t in mc_map:
            task_specs.append(("multi", mc_map[t]))
        else:
            task_specs.append(("reg", None))
    args.task_specs = task_specs

    # Select the mixed head
    args.task_type = 'mixed-reg-multi'

    # Use all numeric target columns at once (masked multitask)
    ys_df = df_input[target_columns].copy()  
    
    # Debug: Check for non-numeric columns
    for col in target_columns:
        if ys_df[col].dtype == 'object':
            logger.warning(f"Target column '{col}' has object dtype. Sample values: {ys_df[col].dropna().head().tolist()}")
            # Try to convert to numeric
            try:
                ys_df[col] = pd.to_numeric(ys_df[col], errors='coerce')
                logger.info(f"Successfully converted '{col}' to numeric")
            except Exception as e:
                logger.error(f"Failed to convert '{col}' to numeric: {e}")
    
    # Factorize multiclass columns (0..K-1), keep NaNs
    for tname in target_columns:
        if tname in mc_map:
            col = ys_df[tname]
            not_nan = col.notna()
            codes, classes = pd.factorize(col[not_nan].astype(str), sort=True)
            tmp = pd.Series(np.nan, index=col.index, dtype=float)
            tmp.loc[not_nan] = codes.astype(float)
            ys_df[tname] = tmp

    # Build datapoints with ALL targets
    smis, df_input, combined_descriptor_data, _ = process_data(
        df_input, smiles_column, descriptor_columns, target_columns, args
    )
    

    # Generate splits FIRST (we'll need train indices for stats)
    ys_full = ys_df.values  # shape [N, T] with NaNs
    first_mc = next((i for i,(k,_) in enumerate(task_specs) if k == "multi"), None)

    # Choose a stratification column if we have a multiclass task
    y_strat = ys_df.iloc[:, first_mc].values if first_mc is not None else None

    # --- PATCH: if stratification labels contain NaNs, fallback to unstratified ---
    use_strat = (first_mc is not None)
    if use_strat:
        y_strat_np = y_strat.astype(float)
        if np.isnan(y_strat_np).any():
            logger.warning("NaNs found in stratification labels for pretrain_monomer; "
                        "falling back to unstratified splits.")
            use_strat = False

    n_splits, local_reps = determine_split_strategy(len(ys_full), REPLICATES)

    from copy import deepcopy
    split_args = deepcopy(args)
    # Only stratify if we have a valid multiclass column WITHOUT NaNs
    split_args.task_type = 'multi' if use_strat else 'reg'
    ys_for_split = (y_strat if use_strat else ys_full)

    train_indices, val_indices, test_indices = generate_data_splits(
        split_args, ys_for_split, n_splits, local_reps, SEED
    )
    # Identify regression target indices
    reg_idx = [i for i,(k,_) in enumerate(args.task_specs) if k == 'reg']
    
    # For pretrain_monomer (single split path in your code), pick split 0.
    tr = train_indices[0]

    # Train-only stats (μ, σ) per regression column
    if reg_idx:
        reg_data = ys_full[tr][:, reg_idx]
        mu = np.nanmean(reg_data, axis=0)
        sd = np.nanstd(reg_data, axis=0)
    else:
        mu = np.array([])
        sd = np.array([])
    # handle all-NaN columns
    nan_cols = np.isnan(mu) | np.isnan(sd)
    mu[nan_cols] = 0.0
    sd[nan_cols] = 1.0
    sd[sd < 1e-8] = 1.0

    # Map μ,σ to per-task lists aligned with task_specs
    reg_mu_per_task = [None] * len(task_specs)
    reg_sd_per_task = [None] * len(task_specs)
    rj = 0
    for t, (kind, _) in enumerate(task_specs):
        if kind == "reg":
            reg_mu_per_task[t] = float(mu[rj])
            reg_sd_per_task[t] = float(sd[rj])
            rj += 1

    args.reg_mu_per_task = reg_mu_per_task
    args.reg_sd_per_task = reg_sd_per_task

    # Create normalized targets for datapoints
    ys_norm = ys_full.copy()
    if reg_idx:  # Only normalize if there are regression columns
        ys_norm[:, reg_idx] = (ys_norm[:, reg_idx] - mu) / sd

    all_data = create_all_data(smis, ys_norm, combined_descriptor_data, args.model_name)

    train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

    # Build datasets (small-molecule featurizer)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train = data.MoleculeDataset(train_data[0], featurizer)
    val   = data.MoleculeDataset(val_data[0],   featurizer)
    test  = data.MoleculeDataset(test_data[0],  featurizer)


    # Metrics list: pick a default (RMSE for reg, etc.)
    n_classes_arg = None
    metric_list = []

    # Paths and model
    checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix = \
        build_experiment_paths(args, chemprop_dir, checkpoint_dir, "__multitask__", descriptor_columns, 0)

    processed_descriptor_data = None
    mpnn, trainer = build_model_and_trainer(
        args=args,
        combined_descriptor_data=processed_descriptor_data,
        n_classes=n_classes_arg,
        scaler=None,
        checkpoint_path=checkpoint_path,
        batch_norm=args.batch_norm,
        metric_list=metric_list,
        early_stopping_patience=PATIENCE,
        max_epochs=EPOCHS,
        save_checkpoint=args.save_checkpoint,
    )

    # Train one model (no per-target loop)
    trainer.fit(mpnn, data.build_dataloader(train, batch_size=args.batch_size, num_workers=num_workers,pin_memory=True),
                      data.build_dataloader(val, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True))
    _ = trainer.test(dataloaders=data.build_dataloader(test, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True))

    # Optionally export embeddings for all monomers now
    if args.export_embeddings:
        mpnn.eval()
        # Re-embed ALL datapoints (train+val+test order) for convenience
        full_loader = data.build_dataloader(data.MoleculeDataset(all_data, featurizer), batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        X_full = get_encodings_from_loader(mpnn, full_loader)
        # Save as .npy; map to smiles with df_input[smiles_column]
        emb_dir = checkpoint_dir / "embeddings"; emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / f"{args.dataset_name}__{args.model_name}__monomer_encoder.npy", X_full)
        pd.DataFrame({
            "smiles": [dp.smiles for dp in all_data],
        }).assign(idx=np.arange(len(all_data))).to_csv(emb_dir / f"{args.dataset_name}__{args.model_name}__monomer_index.csv", index=False)

    # Exit after pretraining (we don’t do per-target loops in this mode)
    save_aggregate_results([], results_dir, args.model_name, args.dataset_name, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, logger)
    raise SystemExit(0)


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


smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

# Choose featurizer based on model type
# PPG uses PPGMolGraphFeaturizer for periodic polymer graph construction
# Small molecule models use SimpleMoleculeMolGraphFeaturizer
# Polymer models (wDMPNN) use PolymerMolGraphFeaturizer
small_molecule_models = ["DMPNN", "DMPNN_DiffPool"]
if args.model_name == "PPG":
    featurizer = featurizers.PPGMolGraphFeaturizer()
elif args.model_name in small_molecule_models:
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
else:
    featurizer = featurizers.PolymerMolGraphFeaturizer()
      

# Store all results for aggregate saving
all_results = []

for target in target_columns:
    # Extract target values
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1) # reshaping target to be 2D
    all_data = create_all_data(smis, ys, combined_descriptor_data, args.model_name)

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


    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # Descriptor cleaning (if incl_desc or incl_rdkit is enabled)
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
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            # Try to load cached preprocessing BEFORE doing any heavy work
            preprocessing_reused, cached_scaler, cached_mask, cache_meta = manage_preprocessing_cache(
                preprocessing_path, i, orig_Xd, None, None, logger
            )

            if preprocessing_reused and cached_mask is not None:
                imputer_stats = None
                if cache_meta is not None:
                    imputer_stats = ((cache_meta.get("cleaning") or {}).get("imputer_statistics"))
                imputer = None
                if imputer_stats is not None:
                    stats = np.asarray(imputer_stats, dtype=float)
                    imputer = SimpleImputer(strategy="median")
                    imputer.statistics_ = stats
                    imputer.n_features_in_ = stats.shape[0]
                    imputer._fit_dtype = np.asarray(orig_Xd, dtype=np.float64).dtype

                base = orig_Xd.copy()
                if imputer is not None:
                    base = imputer.transform(base)
                elif np.isnan(base).any():
                    # cache didn’t have stats => fit on TRAIN ONLY for this split
                    tmp_imputer = SimpleImputer(strategy="median")
                    tmp_imputer.fit(orig_Xd[tr])
                    base = tmp_imputer.transform(base)
                    imputer = tmp_imputer
                base = np.clip(base, float32_min, float32_max).astype(np.float32)
                mask = np.array(cached_mask, dtype=bool)

                def _apply(datapoints, row_indices):
                    for dp, ridx in zip(datapoints, row_indices):
                        dp.x_d = base[ridx][mask]
                _apply(train_data[i], tr); _apply(val_data[i], va); _apply(test_data[i], te)

                split_preprocessing_metadata[i] = cache_meta or {}
                # Ensure correlation_mask reflects what we actually used
                split_preprocessing_metadata[i].setdefault("split_specific", {})
                split_preprocessing_metadata[i]["split_specific"].update({
                    "split_id": i,
                    "correlation_mask": mask.tolist(),
                })
                split_imputers[i] = imputer
                logger.info(f"Split {i}: reused cached preprocessing (imputer+mask).")
            else:    
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
        # Build experiment paths
        checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix = build_experiment_paths(
            args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
        )

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

            # ---- cache/scaler handling (descriptor-only) ----
            preprocessing_reused, cached_scaler, correlation_mask, constant_features = manage_preprocessing_cache(
                preprocessing_path, i, orig_Xd, split_preprocessing_metadata, None, logger
            )
            m_local = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'], dtype=bool)
            cm = np.array(correlation_mask, dtype=bool)
            assert m_local.shape == cm.shape and np.all(m_local == cm), \
                "Local correlation mask != cached correlation mask (split consistency issue)"

            processed_descriptor_data = orig_Xd[:, mask_i]
            descriptor_scaler = cached_scaler  # Will be used after dataset creation
        else:
            preprocessing_reused = False
            descriptor_scaler = None
            processed_descriptor_data = None

        # Create datasets after cleaning
        # PPG uses MoleculeDataset like other small molecule models
        DS = data.MoleculeDataset if args.model_name in ["DMPNN", "DMPNN_DiffPool", "PPG"] else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)
        
        # Now normalize inputs if we have descriptors
        if combined_descriptor_data is not None:
            if descriptor_scaler is not None:
                # Use cached scaler
                train.normalize_inputs("X_d", descriptor_scaler)
                val.normalize_inputs("X_d", descriptor_scaler)
                test.normalize_inputs("X_d", descriptor_scaler)
            else:
                # Fit new scaler on training data
                descriptor_scaler = train.normalize_inputs("X_d")
                val.normalize_inputs("X_d", descriptor_scaler)
                test.normalize_inputs("X_d", descriptor_scaler)
                # persist the fitted scaler
                _ = manage_preprocessing_cache(
                    preprocessing_path, i, orig_Xd, split_preprocessing_metadata, descriptor_scaler, logger
                )

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
        
        
        # Create dataloaders
        train_loader = data.build_dataloader(train, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
        val_loader = data.build_dataloader(val, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        test_loader = data.build_dataloader(test, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        
        # Clean up incompatible checkpoints if preprocessing changed
        # Only delete checkpoints if descriptors are used and preprocessing actually changed
        if not preprocessing_reused and checkpoint_path.exists() and combined_descriptor_data is not None:
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed incompatible checkpoint directory: {checkpoint_path}")
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        
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
            save_checkpoint=args.save_checkpoint,
            
        )
        # Validate checkpoint compatibility and get resume path
        descriptor_dim = processed_descriptor_data.shape[1] if processed_descriptor_data is not None else 0
        last_ckpt = validate_checkpoint_compatibility(
            checkpoint_path, preprocessing_path, i, descriptor_dim, logger
        )
        # ---- Skip training logic (align with AttentiveFP semantics) ----
        inprog_flag = checkpoint_path / "TRAINING_IN_PROGRESS"
        done_flag   = checkpoint_path / "TRAINING_COMPLETE"

        best_ckpt_path, best_val_loss = None, None
        skip_training = False

        if done_flag.exists():
            best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
            if best_ckpt_path is not None:
                skip_training = True
                logger.info(f"[{target}] split {i}: Found TRAINING_COMPLETE; skipping training.\n"
                            f"  -> best_ckpt: {best_ckpt_path}"
                            + (f" (val_loss={best_val_loss:.6f})" if best_val_loss is not None else ""))
        else:
            logger.info(f"[{target}] split {i}: No TRAINING_COMPLETE flag; will (re)train.")


        # Train or skip
        if skip_training and best_ckpt_path:
            logger.info(f"Loading checkpoint for evaluation: {best_ckpt_path}")
            from chemprop import models
            use_cuda = torch.cuda.is_available()
            map_location = None if use_cuda else torch.device("cpu")
            mpnn = models.MPNN.load_from_checkpoint(best_ckpt_path, map_location=map_location)
            if use_cuda:
                mpnn = mpnn.to(torch.device("cuda"))
            mpnn.eval()
        else:
            inprog_flag.touch(exist_ok=True)
            try:
                trainer.fit(mpnn, train_loader, val_loader, ckpt_path=last_ckpt)
                best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
                if best_ckpt_path is None:
                    logger.warning(f"[{target}] split {i}: training finished but no checkpoint found.")
                else:
                    with open(checkpoint_path / "best.json", "w") as f:
                        json.dump({"best_ckpt": best_ckpt_path, "best_val_loss": best_val_loss}, f, indent=2)
                    done_flag.touch()
            finally:
                if inprog_flag.exists():
                    inprog_flag.unlink(missing_ok=True)


        results = trainer.test(model=mpnn, dataloaders=test_loader)
        # results = trainer.test(dataloaders=test_loader)
        test_metrics = results[0]
        test_metrics['split'] = i  # Add split index to metrics
        results_all.append(test_metrics)
        
        # Save predictions if requested
        if args.save_predictions:
            logger.info(f"Extracting predictions for split {i}, target {target}")
            
            # Use trainer.predict for unscaled outputs (applies same transform as trainer.test)
            # y_pred = trainer.predict(dataloaders=test_loader)
            y_pred = trainer.predict(model=mpnn, dataloaders=test_loader)
            
            # Extract y_true and IDs directly from test dataset to match loader order
            y_true = np.array([dp.y[0] if isinstance(dp.y, (list, np.ndarray)) else dp.y for dp in test], dtype=float)
            
            # Extract IDs/indices for order verification
            test_ids = []
            for dp in test:
                if hasattr(dp, 'id') and dp.id is not None:
                    test_ids.append(dp.id)
                elif hasattr(dp, 'smiles'):
                    test_ids.append(dp.smiles)  # Use SMILES as fallback ID
                else:
                    test_ids.append(f"idx_{len(test_ids)}")  # Fallback to index
            
            # Convert predictions to numpy - handle list of tensors properly
            if isinstance(y_pred, list):
                # Concatenate tensors from different batches
                import torch
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
            elif hasattr(y_pred, 'cpu'):
                y_pred = y_pred.cpu().numpy()
            
            # Save predictions with IDs
            save_predictions(
                y_true, y_pred, predictions_dir, args.dataset_name, target, args.model_name,
                desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, i, logger,
                test_ids=test_ids
            )
        
        # Export embeddings if requested
        if args.export_embeddings:
            logger.info(f"Exporting embeddings for split {i}, target {target}")
            
            # Set model to evaluation mode and extract embeddings
            mpnn.eval()
            X_train = get_encodings_from_loader(mpnn, train_loader)
            X_val = get_encodings_from_loader(mpnn, val_loader)
            X_test = get_encodings_from_loader(mpnn, test_loader)
            
            # Apply same filtering as in evaluate_model.py (remove low-variance features)
            eps = 1e-8
            std_train = X_train.std(axis=0)
            keep = std_train > eps
            
            X_train = X_train[:, keep]
            X_val = X_val[:, keep]
            X_test = X_test[:, keep]
            
            logger.info(f"Split {i}: Kept {int(keep.sum())} / {len(keep)} embedding dimensions")
            
            # Create embeddings directory with target/model/size specificity
            embeddings_dir = results_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Build embedding filename prefix with all identifiers including model name
            embedding_prefix = f"{args.dataset_name}__{args.model_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}"
            
            # Save embeddings as numpy arrays with full identifiers
            np.save(embeddings_dir / f"{embedding_prefix}__X_train_split_{i}.npy", X_train)
            np.save(embeddings_dir / f"{embedding_prefix}__X_val_split_{i}.npy", X_val)
            np.save(embeddings_dir / f"{embedding_prefix}__X_test_split_{i}.npy", X_test)
            
            # Save feature mask for reproducibility
            np.save(embeddings_dir / f"{embedding_prefix}__feature_mask_split_{i}.npy", keep)
            
            logger.info(f"Split {i}: Saved embeddings to {embeddings_dir}")
            logger.info(f"  - X_train: {X_train.shape}")
            logger.info(f"  - X_val: {X_val.shape}")
            logger.info(f"  - X_test: {X_test.shape}")
    

    # Convert to DataFrame
    results_df = pd.DataFrame(results_all)
    # Calculate mean/std only for numeric metric columns (exclude 'split')
    numeric_cols = [col for col in results_df.columns if col != 'split']
    mean_metrics = results_df[numeric_cols].mean()
    std_metrics = results_df[numeric_cols].std()

    n_evals = len(results_all)
    logger.info(f"\n[{target}] Mean across {n_evals} splits:\n{mean_metrics}")
    logger.info(f"\n[{target}] Std across {n_evals} splits:\n{std_metrics}")


    # Always add target column for proper result organization
    results_df['target'] = target
    all_results.append(results_df)
    
# Save final results using modular function
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    save_model_results(combined_results, args, args.model_name, results_dir, logger)