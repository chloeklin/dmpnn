import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging

from utils import (
    set_seed, process_data, determine_split_strategy, generate_data_splits, 
    create_all_data,
    build_sklearn_models,
    build_experiment_paths,
    setup_training_environment,
    load_and_preprocess_data,
    load_best_checkpoint,
    get_encodings_from_loader,
)


from chemprop import data, featurizers, models, nn

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)

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
                    help='Include RDKit 2D descriptors')
parser.add_argument('--model_name', type=str, choices=['DMPNN', 'wDMPNN'], default="DMPNN",
                    help='Name of the model to use')


args = parser.parse_args()

logger.info("\n=== Evaluation Configuration ===")
logger.info(f"Dataset       : {args.dataset_name}")
logger.info(f"Task type     : {args.task_type}")
logger.info(f"Model         : {args.model_name}")
logger.info(f"Descriptors   : {'Enabled' if args.incl_desc else 'Disabled'}")
logger.info(f"RDKit desc.   : {'Enabled' if args.incl_rdkit else 'Disabled'}")
logger.info("===============================\n")

# Setup evaluation environment with common configuration
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
    logger.info(f"Warning: No configuration found for model '{args.model_name}'. Using defaults.")

# === Set Random Seed ===
set_seed(SEED)

# === Load and Preprocess Data ===
df_input, target_columns = load_and_preprocess_data(args, setup_info)

# Which variant are we evaluating?
use_desc = bool(args.incl_desc)
use_rdkit = bool(args.incl_rdkit)

variant_tokens = []
if use_desc:
    variant_tokens.append("desc")
if use_rdkit:
    variant_tokens.append("rdkit")

variant_label = "original" if not variant_tokens else "+".join(variant_tokens)
variant_qstattag = "" if variant_label == "original" else "_" + variant_label.replace("+", "_")

smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() if args.model_name == "DMPNN" else featurizers.PolymerMolGraphFeaturizer()

# Prepare results list for tabular format (same as train_tabular.py)
all_results = []

# Iterate per target
for target in target_columns:
    # Prepare data
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
    
    
    # Split to datasets for each replicate
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    num_splits = len(train_data)  # robust for both CV and holdout

    # Apply same preprocessing as train_graph.py
    if combined_descriptor_data is not None:
        # Initial data preparation (same as train_graph.py)
        orig_Xd = np.asarray(combined_descriptor_data, dtype=np.float64)
        
        # Replace inf with NaN
        inf_mask = np.isinf(orig_Xd)
        if np.any(inf_mask):
            logger.info(f"Found {np.sum(inf_mask)} infinite values, replacing with NaN")
            orig_Xd[inf_mask] = np.nan
        
        # Load preprocessing metadata directly (same as train_graph.py)
        split_preprocessing_metadata = {}
        for i in range(REPLICATES):
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            
            # Load the metadata file directly
            metadata_path = preprocessing_path / f"preprocessing_metadata_split_{i}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    split_preprocessing_metadata[i] = json.load(f)
                logger.info(f"Loaded preprocessing metadata for split {i}")
            else:
                logger.info(f"Warning: No preprocessing metadata found for split {i}")
                split_preprocessing_metadata[i] = None
        
        # Remove constant features using saved metadata (same as train_graph.py)
        if split_preprocessing_metadata[0] is not None:
            constant_features = split_preprocessing_metadata[0]['data_info']['constant_features_removed']
            if constant_features:
                logger.info(f"Removing {len(constant_features)} constant features from full dataset")
                orig_Xd = np.delete(orig_Xd, constant_features, axis=1)
        
    # === Evaluation Loop ===
    rep_model_to_row = {}
    for i in range(num_splits):
        logger.info(f"\n=== Replicate {i+1}/{num_splits} ===")
        
        # Apply per-split preprocessing (same as train_graph.py)
        if combined_descriptor_data is not None:
            if split_preprocessing_metadata[i] is None:
                logger.info(f"Warning: Skipping split {i} due to missing metadata")
                continue
                
            # Get preprocessing components for this split
            cleaning_meta = split_preprocessing_metadata[i]['cleaning']
            
            # Apply imputation using saved imputer (same as train_graph.py)
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            
            from joblib import load
            imputer_path = preprocessing_path / "descriptor_imputer.pkl"
            if imputer_path.exists():
                saved_imputer = load(imputer_path)
                all_data_clean_i = saved_imputer.transform(orig_Xd)
            else:
                all_data_clean_i = orig_Xd.copy()
            
            # Apply clipping and conversion
            float32_min = cleaning_meta['float32_min']
            float32_max = cleaning_meta['float32_max']
            all_data_clean_i = np.clip(all_data_clean_i, float32_min, float32_max)
            all_data_clean_i = all_data_clean_i.astype(np.float32)
            
            # Apply correlation mask (same as train_graph.py)
            mask_i = np.array(
                split_preprocessing_metadata[i]['split_specific']['correlation_mask'],
                dtype=bool
            )
            
            # Apply preprocessing to datapoints (same as train_graph.py)
            for dp, ridx in zip(train_data[i], train_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
            for dp, ridx in zip(val_data[i], val_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
            for dp, ridx in zip(test_data[i], test_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
        
        # Create datasets after cleaning (same as train_graph.py)
        DS = data.MoleculeDataset if MODEL_NAME == "DMPNN" else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)
        
        # Apply descriptor scaling using saved scaler (same as train_graph.py)
        if combined_descriptor_data is not None:
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            
            # Load saved descriptor scaler
            from joblib import load
            scaler_path = preprocessing_path / "descriptor_scaler.pkl"
            if scaler_path.exists():
                saved_descriptor_scaler = load(scaler_path)
                logger.info(f"Loaded descriptor scaler for split {i}")
                
                # Apply saved scaler to datasets (same as train_graph.py)
                train.normalize_inputs("X_d", saved_descriptor_scaler)
                val.normalize_inputs("X_d", saved_descriptor_scaler)
                test.normalize_inputs("X_d", saved_descriptor_scaler)
            else:
                logger.info(f"Warning: No descriptor scaler found for split {i}")
        
        # Target scaling (same as train_graph.py)
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
            # test targets intentionally left unscaled

        # Use multiprocessing only if GPU is available to avoid spawn issues on CPU-only systems
        import torch
        eval_num_workers = num_workers if torch.cuda.is_available() else 0
        train_loader = data.build_dataloader(train, num_workers=eval_num_workers, shuffle=False)
        val_loader = data.build_dataloader(val, num_workers=eval_num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=eval_num_workers, shuffle=False)
        checkpoint_path, _, _, _ = build_experiment_paths(
            args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
        )
        last_ckpt = load_best_checkpoint(Path(checkpoint_path))
        if last_ckpt is None:
            # no checkpoint → skip this replicate (leave row without this target's metrics)
            logger.warning(f"No checkpoint found at {checkpoint_path}; skipping rep {i} for target {target}.")
            continue

        # Load encoder and make fingerlogger.infos (map to CPU if CUDA not available)
        import torch
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        mpnn = models.MPNN.load_from_checkpoint(str(last_ckpt), map_location=map_location)
        mpnn.eval()
        X_train = get_encodings_from_loader(mpnn, train_loader)
        X_val = get_encodings_from_loader(mpnn, val_loader)
        X_test = get_encodings_from_loader(mpnn, test_loader)

        eps = 1e-8  # or 1e-6 if you want to be stricter
        std_train = X_train.std(axis=0)
        keep = std_train > eps

        X_train = X_train[:, keep]
        X_val   = X_val[:, keep]
        X_test  = X_test[:, keep]

        logger.info(f"kept dims: {int(keep.sum())} / 300")

        # Get target data for each split
        y_train = df_input.loc[train_indices[i], target].to_numpy()
        y_val = df_input.loc[val_indices[i], target].to_numpy()
        y_test = df_input.loc[test_indices[i], target].to_numpy()
        
        # Ensure all target arrays match the number of samples actually processed by the model
        # (DataLoader may drop incomplete batches)
        if len(y_train) != len(X_train):
            logger.warning(f"Truncating y_train from {len(y_train)} to {len(X_train)} samples to match processed data")
            y_train = y_train[:len(X_train)]
        if len(y_val) != len(X_val):
            logger.warning(f"Truncating y_val from {len(y_val)} to {len(X_val)} samples to match processed data")
            y_val = y_val[:len(X_val)]
        if len(y_test) != len(X_test):
            logger.warning(f"Truncating y_test from {len(y_test)} to {len(X_test)} samples to match processed data")
            y_test = y_test[:len(X_test)]
        
        # Initialize target scaler for regression tasks (same as train_tabular.py)
        target_scaler = None
        if args.task_type == 'reg':
            from sklearn.preprocessing import StandardScaler
            target_scaler = StandardScaler()
            y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            # Keep y_test original for final evaluation
        else:
            y_train_scaled = y_train
            y_val_scaled = y_val

        # Build requested baselines with scaler info (same as train_tabular.py)
        num_classes = len(np.unique(y_train_scaled)) if args.task_type != "reg" else None
        model_specs = build_sklearn_models(args.task_type, num_classes, scaler_flag=True)

        # Initialize result rows for this replicate
        for name in model_specs.keys():
            if (i, name) not in rep_model_to_row:
                rep_model_to_row[(i, name)] = {
                    "dataset": args.dataset_name,
                    "encoder": args.model_name,
                    "variant": variant_label,
                    "replicate": i,
                    "model": name
                }

        # Train baselines following same logic as train_tabular.py
        for name, (model, needs_scaler) in model_specs.items():
            # Apply scaling for models that require it (linear/logistic)
            scaler = None
            if needs_scaler:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                X_test_scaled = X_test

            if args.task_type == "reg":
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="rmse")
                    model.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)], verbose=False)
                else:
                    model.fit(X_train_scaled, y_train_scaled)
                
                # Get predictions and inverse transform if this is regression
                y_pred = model.predict(X_test_scaled)
                if target_scaler is not None:
                    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                r2   = r2_score(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rep_model_to_row[(i, name)][f"{target}_R2"] = r2; rep_model_to_row[(i, name)][f"{target}_MAE"] = mae; rep_model_to_row[(i, name)][f"{target}_RMSE"] = rmse

            else:
                if name == "XGB":
                    model.set_params(early_stopping_rounds=30, eval_metric="logloss")
                    model.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)], verbose=False)
                else:
                    model.fit(X_train_scaled, y_train_scaled)
                
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                avg = "macro" if args.task_type == "multi" else "binary"
                f1  = f1_score(y_test, y_pred, average=avg)
                rep_model_to_row[(i, name)][f"{target}_ACC"] = acc; rep_model_to_row[(i, name)][f"{target}_F1"] = f1

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test_scaled)
                    try:
                        if args.task_type == "binary":
                            auc = roc_auc_score(y_test, proba[:, 1])
                        else:
                            from sklearn.preprocessing import label_binarize
                            y_bin = label_binarize(y_test, classes=list(range(n_classes_per_target[target])))
                            auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                        rep_model_to_row[(i, name)][f"{target}_ROC_AUC"] = auc
                    except Exception:
                        pass
        

    # Convert to train_graph.py format: target,split,test/mae,test/r2,test/rmse
    eval_results = []
    for (rep_idx, model_name), row_data in rep_model_to_row.items():
        for target in target_columns:
            if f"{target}_R2" in row_data:
                eval_results.append({
                    'target': target,
                    'split': rep_idx,
                    'test/mae': row_data[f"{target}_MAE"],
                    'test/r2': row_data[f"{target}_R2"],
                    'test/rmse': row_data[f"{target}_RMSE"],
                    'model': model_name  # Keep model info for baseline comparison
                })
    
    if eval_results:
        results_df = pd.DataFrame(eval_results)
        
        # Use train_graph.py naming convention and directory structure
        desc_suffix = "__desc" if descriptor_columns else ""
        rdkit_suffix = "__rdkit" if args.incl_rdkit else ""
        
        model_results_dir = results_dir / args.model_name
        model_results_dir.mkdir(exist_ok=True)
        out_csv = model_results_dir / f"{args.dataset_name}{desc_suffix}{rdkit_suffix}_baseline.csv"
        
        # Organize columns to match train_graph.py: target, split, then metrics, then model
        base_cols = ["target", "split"]
        metric_cols = ["test/mae", "test/r2", "test/rmse"]
        extra_cols = [c for c in results_df.columns if c not in base_cols + metric_cols]
        results_df = results_df[base_cols + metric_cols + extra_cols]
        
        results_df.to_csv(out_csv, index=False)
        logger.info(f"Wrote/updated: {out_csv}")