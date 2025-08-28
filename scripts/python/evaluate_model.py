import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

from utils import (
    process_data,
    create_all_data,
    make_repeated_splits,
    set_seed,
    load_best_checkpoint,
    get_encodings_from_loader,
    upsert_csv,
    load_drop_indices,
    load_config,
    filter_insulator_data,
    build_sklearn_models
)

from chemprop import data, featurizers, models, nn

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')
parser.add_argument('--descriptor', action='store_true',
                    help='Use dataset-specific descriptors')
parser.add_argument('--model_name', type=str, choices=['DMPNN', 'wDMPNN'], default="DMPNN",
                    help='Name of the model to use')
parser.add_argument('--incl_rdkit', action='store_true',
                    help='Include RDKit descriptors')

args = parser.parse_args()


print("\n=== Training Configuration ===")
print(f"Dataset       : {args.dataset_name}")
print(f"Task type     : {args.task_type}")
print(f"Model         : {args.model_name}")
print(f"Descriptors   : {'Enabled' if args.descriptor else 'Disabled'}")
print(f"RDKit desc.   : {'Enabled' if args.incl_rdkit else 'Disabled'}")
print("===============================\n")


# Load configuration
config = load_config()

# Set up paths and parameters
chemprop_dir = Path.cwd()

# Get paths from config with defaults
paths = config.get('PATHS', {})
data_dir = chemprop_dir / paths.get('data_dir', 'data')
checkpoint_dir = chemprop_dir / paths.get('checkpoint_dir', 'checkpoints') / args.model_name
results_dir = chemprop_dir / paths.get('results_dir', 'results') / args.model_name

# Create necessary directories
checkpoint_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Set up file paths
input_path = chemprop_dir / data_dir / f"{args.dataset_name}.csv"

# Get model-specific configuration
model_config = config['MODELS'].get(args.model_name, {})
if not model_config:
    print(f"Warning: No configuration found for model '{args.model_name}'. Using defaults.")

# Set parameters from config with defaults
GLOBAL_CONFIG = config.get('GLOBAL', {})
SEED = GLOBAL_CONFIG.get('SEED', 42)
REPLICATES = GLOBAL_CONFIG.get('REPLICATES', 5)
EPOCHS = GLOBAL_CONFIG.get('EPOCHS', 300)
num_workers = min(
    GLOBAL_CONFIG.get('NUM_WORKERS', 8),
    os.cpu_count() or 1
)

# Model-specific parameters
smiles_column = model_config.get('smiles_column', 'smiles')
ignore_columns = model_config.get('ignore_columns', [])
MODEL_NAME = args.model_name

# Get dataset descriptors from config
DATASET_DESCRIPTORS = config.get('DATASET_DESCRIPTORS', {}).get(args.dataset_name, [])
descriptor_columns = DATASET_DESCRIPTORS if args.descriptor else []

# === Set Random Seed ===
set_seed(SEED)

# === Load Data ===
df_input = pd.read_csv(input_path)

# Apply insulator dataset filtering if needed
if args.dataset_name == "insulator" and args.model_name == "wDMPNN":
    df_input = filter_insulator_data(df_input, smiles_column)

# Read the saved exclusions from the wDMPNN preprocessing step
if args.model_name == "DMPNN":
    drop_idx, excluded_smis = load_drop_indices(chemprop_dir, args.dataset_name)
    if drop_idx:
        print(f"Dropping {len(drop_idx)} rows from {args.dataset_name} due to exclusions.")
        df_input = df_input.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


# Automatically detect target columns (all columns except ignored ones)
target_columns = [c for c in df_input.columns
                 if c not in ([smiles_column] + 
                            DATASET_DESCRIPTORS + 
                            ignore_columns)]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")

# Which variant are we evaluating?
use_desc = bool(descriptor_columns)
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

 # Prepare wide-format rows: a dict per (replicate, model)
rep_model_to_row = {}
# ensure pre-create rows for all replicate×model combos
# (we'll fill target-specific metric keys as we compute them)
models_dict = build_sklearn_models(args.task_type)
model_names_example = list(models_dict.keys())

# Common metadata for this run
base_meta = {
    "dataset": args.dataset_name,
    "encoder": args.model_name,   # "DMPNN" or "wDMPNN"
    "variant": variant_label,     # "original", "desc", "rdkit", or "desc+rdkit"
}

    
# Iterate per target
for target in target_columns:
    # Prepare data
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1) # reshaping target to be 2D
    all_data = create_all_data(smis, ys, combined_descriptor_data, args.model_name)


    # Build replicates of splits
    if args.task_type in ['binary', 'multi']:
        y_class = df_input[target].to_numpy(dtype=int, copy=False)
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=REPLICATES,
            seed=SEED,
            y_class=y_class
            
        )
    else:
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=REPLICATES,
            seed=SEED,
            mols=[d.mol for d in all_data]
        )
    
    # Split to datasets for each replicate
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # Apply same preprocessing as train_graph.py
    if combined_descriptor_data is not None:
        # Load preprocessing metadata and objects for each split
        preprocessing_metadata = {}
        preprocessing_components = {}
        
        for i in range(REPLICATES):
            checkpoint_path = (
                Path("out") / args.model_name / args.dataset_name / 
                f"{args.dataset_name}__{target}"
                f"{'__desc' if descriptor_columns else ''}"
                f"{'__rdkit' if args.incl_rdkit else ''}"
                f"__rep{i}"
            )
            metadata_path = checkpoint_path / f"preprocessing_metadata_split_{i}.json"
            
            # Load JSON metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    preprocessing_metadata[i] = json.load(f)
                print(f"Loaded preprocessing metadata for split {i}")
            else:
                print(f"Warning: No preprocessing metadata found for split {i}")
                preprocessing_metadata[i] = None
                continue
            
            # Load preprocessing components separately
            from joblib import load
            components = {}
            
            # Load imputer if it exists
            imputer_path = checkpoint_path / "descriptor_imputer.pkl"
            if imputer_path.exists():
                components['imputer'] = load(imputer_path)
                print(f"Loaded descriptor imputer for split {i}")
            else:
                components['imputer'] = None
            
            # Load descriptor scaler
            scaler_path = checkpoint_path / "descriptor_scaler.pkl"
            if scaler_path.exists():
                components['descriptor_scaler'] = load(scaler_path)
                print(f"Loaded descriptor scaler for split {i}")
            else:
                components['descriptor_scaler'] = None
                print(f"Warning: No descriptor scaler found for split {i}")
            
            # Load correlation mask
            mask_path = checkpoint_path / "correlation_mask.npy"
            if mask_path.exists():
                components['correlation_mask'] = np.load(mask_path)
                print(f"Loaded correlation mask for split {i}")
            else:
                components['correlation_mask'] = None
                print(f"Warning: No correlation mask found for split {i}")
            
            # Load constant features
            const_path = checkpoint_path / "constant_features_removed.npy"
            if const_path.exists():
                components['constant_features_removed'] = np.load(const_path)
                print(f"Loaded constant features for split {i}")
            else:
                components['constant_features_removed'] = None
                print(f"Warning: No constant features found for split {i}")
            
            preprocessing_components[i] = components

        # Initial data preparation (same as train_graph.py)
        orig_Xd = np.asarray(combined_descriptor_data, dtype=np.float64)
        
        # Replace inf with NaN
        inf_mask = np.isinf(orig_Xd)
        if np.any(inf_mask):
            print(f"Found {np.sum(inf_mask)} infinite values, replacing with NaN")
            orig_Xd[inf_mask] = np.nan
        
        # Remove constant features using saved numpy array
        if preprocessing_components[0] is not None and preprocessing_components[0]['constant_features_removed'] is not None:
            constant_features = preprocessing_components[0]['constant_features_removed']
            if len(constant_features) > 0:
                print(f"Removing {len(constant_features)} constant features")
                orig_Xd = np.delete(orig_Xd, constant_features, axis=1)
        
        # Apply per-split preprocessing using saved components
        for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
            if preprocessing_metadata[i] is None or preprocessing_components[i] is None:
                print(f"Warning: Skipping split {i} due to missing metadata or components")
                continue
            
            # Get saved preprocessing components
            saved_imputer = preprocessing_components[i]['imputer']
            correlation_mask = preprocessing_components[i]['correlation_mask']
            
            # Apply imputation using saved imputer object
            if saved_imputer is not None:
                all_data_clean = saved_imputer.transform(orig_Xd)
            else:
                all_data_clean = orig_Xd.copy()
            
            # Apply clipping and conversion (get limits from metadata)
            cleaning_meta = preprocessing_metadata[i]['cleaning']
            float32_min = cleaning_meta['float32_min']
            float32_max = cleaning_meta['float32_max']
            all_data_clean = np.clip(all_data_clean, float32_min, float32_max)
            all_data_clean = all_data_clean.astype(np.float32)
            
            # Apply preprocessing to datapoints
            def _apply_saved_preprocessing(datapoints, row_indices):
                for dp, ridx in zip(datapoints, row_indices):
                    row_clean = all_data_clean[ridx]
                    # Apply mask (zero out dropped features)
                    out = np.zeros_like(row_clean, dtype=np.float32)
                    out[correlation_mask] = row_clean[correlation_mask]
                    dp.x_d = out
            
            _apply_saved_preprocessing(train_data[i], tr)
            _apply_saved_preprocessing(val_data[i], va)
            _apply_saved_preprocessing(test_data[i], te)
            
            print(f"Applied saved preprocessing for split {i} using joblib/numpy components")

            # Use saved descriptor scaler if available
            if combined_descriptor_data is not None and preprocessing_components[i] is not None:
                saved_descriptor_scaler = preprocessing_components[i]['descriptor_scaler']
                if saved_descriptor_scaler is not None:
                    # Apply the saved scaler directly
                    train_data[i].normalize_inputs("X_d", saved_descriptor_scaler)
                    val_data[i].normalize_inputs("X_d", saved_descriptor_scaler)
                    test_data[i].normalize_inputs("X_d", saved_descriptor_scaler)
                    
                    # Re-apply zero mask after scaling (same as training)
                    correlation_mask = preprocessing_components[i]['correlation_mask']
                    
                    def _reapply_zero_mask(dataset):
                        for dp in dataset:
                            if hasattr(dp, 'x_d') and dp.x_d is not None:
                                dp.x_d[~correlation_mask] = 0.0
                    
                    _reapply_zero_mask(train_data[i])
                    _reapply_zero_mask(val_data[i])
                    _reapply_zero_mask(test_data[i])
                    
                    print(f"Applied saved descriptor scaler and re-applied zero mask for split {i}")
                else:
                    print(f"Warning: No saved descriptor scaler found for split {i}")
                    preprocessing_metadata[i] = None
                    continue

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() if args.model_name == "DMPNN" else featurizers.PolymerMolGraphFeaturizer()
    for i in range(REPLICATES):
        # Ensure pre-create rows for all replicate×model combos
        for mname in model_names_example:
            rep_model_to_row[(i, mname)] = {
                **base_meta,
                "replicate": i,
                "model": mname,
            }

        # Prepare datasets
        if args.model_name == "DMPNN":
            train, val, test = data.MoleculeDataset(train_data[i], featurizer), data.MoleculeDataset(val_data[i], featurizer), data.MoleculeDataset(test_data[i], featurizer)
        else:
            train, val, test = data.PolymerDataset(train_data[i], featurizer), data.PolymerDataset(val_data[i], featurizer), data.PolymerDataset(test_data[i], featurizer)
        # Normalize targets
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
            test.normalize_targets(scaler)
        # Normalize input descriptors (to match training’s scaling, if used)
        X_d_transform = None
        if combined_descriptor_data is not None:
            descriptor_scaler = train.normalize_inputs("X_d")
            val.normalize_inputs("X_d", descriptor_scaler)
            test.normalize_inputs("X_d", descriptor_scaler)

        train_loader = data.build_dataloader(train, num_workers=num_workers)
        val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)
        
        # Load best ckpt
        checkpoint_path = (
            f"checkpoints/{args.model_name}/"
            f"{args.dataset_name}__{target}"
            f"{'__desc' if descriptor_columns else ''}"
            f"{'__rdkit' if args.incl_rdkit else ''}"
            f"__rep{i}/"
            )
        last_ckpt = load_best_checkpoint(Path(checkpoint_path))
        if last_ckpt is None:
            # no checkpoint → skip this replicate (leave row without this target's metrics)
            print(f"WARNING: No checkpoint found at {checkpoint_path}; skipping rep {i} for target {target}.", file=sys.stderr)
            continue

        # Load encoder and make fingerprints
        mpnn = models.MPNN.load_from_checkpoint(str(last_ckpt))
        mpnn.eval()
        X_train = get_encodings_from_loader(mpnn, train_loader)
        X_val = get_encodings_from_loader(mpnn, val_loader)
        X_test = get_encodings_from_loader(mpnn, test_loader)

        # Combine train+val to fit baselines
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([
            df_input.loc[train_indices[i], target].to_numpy(),
            df_input.loc[val_indices[i],   target].to_numpy()
        ], axis=0)

        y_test = df_input.loc[test_indices[i], target].to_numpy()

        # Build requested baselines
        models_dict = build_sklearn_models(args.task_type)

        # Fit baselines on train+val, compute metrics on test only
        for name, model in models_dict.items():
            model.fit(X_fit, y_fit)

            if args.task_type == "reg":
                y_pred = model.predict(X_test)
                r2   = r2_score(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rep_model_to_row[(i, name)][f"{target}_R2"] = r2; rep_model_to_row[(i, name)][f"{target}_MAE"] = mae; rep_model_to_row[(i, name)][f"{target}_RMSE"] = rmse

            else:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                avg = "macro" if args.task_type == "multi" else "binary"
                f1  = f1_score(y_test, y_pred, average=avg)
                rep_model_to_row[(i, name)][f"{target}_ACC"] = acc; rep_model_to_row[(i, name)][f"{target}_F1"] = f1

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
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
        

    # finalize wide dataframe
    wide_rows = list(rep_model_to_row.values())
    results_df = pd.DataFrame(wide_rows).sort_values(["replicate", "model"]).reset_index(drop=True)

    # write
    results_dir = Path(chemprop_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"{args.dataset_name}_{args.model_name}_baseline.csv"
    KEY_COLS = ["dataset", "encoder", "variant", "replicate", "model"]
    upsert_csv(out_csv, results_df, KEY_COLS)
    print(f"Wrote/updated: {out_csv}")