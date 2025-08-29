import argparse
import json
import logging
import numpy as np
import pandas as pd
# from rdkit import Chem
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Local imports
from tabular_utils import (
    build_features,
    eval_regression,
    eval_binary,
    eval_multi,
    
)
from utils import (
    set_seed,
    load_drop_indices,
    load_config,
    build_sklearn_models,
    make_repeated_splits,
)


# ---------------------------- Training Loop ----------------------------

def train(df, y, target_name, descriptor_columns, replicates, seed, out_dir, args):
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
        
    Returns:
        List of dictionaries containing evaluation metrics
    """
    logger = logging.getLogger(__name__)

    # Choose CV for small datasets; holdout for large (as per your plan)
    n_splits = 5 if len(y) < 2000 else 1
    local_reps = 1 if n_splits > 1 else replicates

    if args.task_type in ['binary', 'multi']:
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=local_reps,
            seed=seed,
            y_class=y,
            n_splits=n_splits
            
        )
    else:
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=local_reps,
            seed=seed,
            y_reg=y,
            n_splits=n_splits
        )
    

    detailed_rows = []
    for i, (train_idx, val_idx, test_idx) in enumerate(zip(train_indices, val_indices, test_indices)):
        # Extract and process features
        ab_block, descriptor_block, feat_names = build_features(df, train_idx, descriptor_columns, args.polymer_type, use_rdkit=args.incl_rdkit, use_ab=args.incl_ab)
        
        orig_desc_names = [n for n in feat_names if not n.startswith('AB_')]

        # Process AB block (no cleaning/selection needed) - only if AB features are included
        if ab_block is not None:
            ab_tr, ab_val, ab_te = ab_block[train_idx], ab_block[val_idx], ab_block[test_idx]
            ab_names = [name for name in feat_names if name.startswith('AB_')]

        
        # Process descriptor block separately (clean and select features)
        if descriptor_block is not None:
            # Clean descriptor data before converting to float32
            desc_X = np.asarray(descriptor_block, dtype=np.float64)  # Use float64 first
            
            # Replace inf with NaN
            inf_mask = np.isinf(desc_X)
            if np.any(inf_mask):
                logger.warning(f"Found {np.sum(inf_mask)} infinite values in descriptors, replacing with NaN")
                desc_X[inf_mask] = np.nan
            
            # Clip extreme values to prevent float32 overflow
            float32_max = np.finfo(np.float32).max
            float32_min = np.finfo(np.float32).min
            desc_X = np.clip(desc_X, float32_min, float32_max)
            
            # Now safely convert to float32
            desc_X = desc_X.astype(np.float32)
            logger.debug(f"Descriptor data shape: {desc_X.shape}")
            logger.debug(f"Descriptor data - finite values: {np.isfinite(desc_X).all()}")
            
            # 1) Remove constants on FULL descriptor dataset BEFORE imputation (like train_graph.py)
            desc_full_df = pd.DataFrame(desc_X, columns=orig_desc_names)
            non_na_uniques = desc_full_df.nunique(dropna=True)
            constant_mask = (non_na_uniques >= 2).values             # True = keep
            constant_features = [n for n, keep in zip(orig_desc_names, constant_mask) if not keep]
            const_kept_names = [n for n, keep in zip(orig_desc_names, constant_mask) if keep]

            # Remove constants from full dataset
            const_keep_idx = np.where(constant_mask)[0]
            desc_X_no_const = desc_X[:, const_keep_idx]
            
            # Split descriptor data AFTER constant removal but BEFORE imputation
            desc_tr, desc_val, desc_te = desc_X_no_const[train_idx], desc_X_no_const[val_idx], desc_X_no_const[test_idx]
            
            # Handle NaN values with median imputation fitted only on training data
            nan_mask = np.isnan(desc_tr)
            if np.any(nan_mask):
                logger.warning(f"Found {np.sum(nan_mask)} NaN values in training descriptors, using median imputation")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                desc_tr = imputer.fit_transform(desc_tr)
                desc_val = imputer.transform(desc_val)
                desc_te = imputer.transform(desc_te)
            else:
                # Create dummy imputer for consistency even if no NaNs
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                imputer.fit(desc_tr)  # Fit on training data for consistency
            
            # correlation on TRAIN ONLY, using constant-removed training set
            desc_tr_df = pd.DataFrame(desc_tr, columns=const_kept_names)
            if len(const_kept_names) > 1:
                corr = desc_tr_df.corr(method=("pearson" if args.task_type=="reg" else "spearman")).abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = {c for c in upper.columns if (upper[c] >= 0.90).any()}
                keep_names = [n for n in const_kept_names if n not in to_drop]
            else:
                to_drop = set()
                keep_names = const_kept_names[:]
    

            # final selection by name
            desc_tr_selected = desc_tr_df[keep_names].values
            desc_val_selected = pd.DataFrame(desc_val, columns=const_kept_names)[keep_names].values
            desc_te_selected  = pd.DataFrame(desc_te,  columns=const_kept_names)[keep_names].values

            # masks to SAVE
            # constant_mask: length == len(orig_desc_names)      (global, split-invariant)
            # corr_mask:     length == len(const_kept_names)     (per split)
            corr_mask = np.isin(const_kept_names, keep_names)

            # define the selected names (you use this below)
            selected_desc_names = keep_names    
            
            
            # Combine AB block with selected descriptor block
            if ab_block is not None:
                X_tr = np.concatenate([ab_tr, desc_tr_selected], axis=1)
                X_val = np.concatenate([ab_val, desc_val_selected], axis=1)
                X_te = np.concatenate([ab_te, desc_te_selected], axis=1)
                feat_names = ab_names + keep_names
            else:
                # Only descriptor block available
                X_tr, X_val, X_te = desc_tr_selected, desc_val_selected, desc_te_selected
                feat_names = keep_names
        
        else:
            # Only AB block available (or no features if AB disabled)
            X_tr, X_val, X_te = ab_tr, ab_val, ab_te
            feat_names = ab_names

        # Ensure output directory exists for saving preprocessing objects
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessing metadata and objects
        if descriptor_block is not None:

            preprocessing_metadata = {
                "n_desc_before_any_selection": len(orig_desc_names),
                "n_desc_after_constant_removal": len(const_kept_names),
                "n_desc_after_corr_removal": len(keep_names),
                "constant_features_removed": constant_features,         # list of names
                "correlated_features_removed": sorted(list(to_drop)),  # list of names
                "selected_features": selected_desc_names,              # list of names
                "imputation_strategy": "median",
                "correlation_threshold": 0.90,
                "orig_desc_names": orig_desc_names,                    # to make masks resolvable later
                "const_kept_names": const_kept_names
            }
            with open(out_dir / f"preprocessing_metadata_split_{i}.json", "w") as f:
                json.dump(preprocessing_metadata, f, indent=2)

            # objects + masks
            joblib.dump(imputer, out_dir / f"descriptor_imputer_{i}.pkl")

            # boolean masks for reproducibility
            # constant_mask aligns with orig_desc_names (True = keep)
            np.save(out_dir / f"constant_mask_{i}.npy", constant_mask)
            # corr_mask aligns with const_kept_names (True = keep)
            np.save(out_dir / f"corr_mask_{i}.npy", corr_mask)

            # for quick diff-friendly lists too
            with open(out_dir / f"split_{i}.txt", "w") as f:
                f.write("\n".join(selected_desc_names))


        if X_tr.shape[1] == 0:
            logger.warning("Feature selection yielded 0 columns; reverting to AB features only for this split.")
            X_tr, X_val, X_te = ab_tr, ab_val, ab_te
            feat_names = ab_names

        # Initialize target scaler for regression tasks
        target_scaler = None
        if args.task_type == 'reg':
            target_scaler = StandardScaler()
            y_tr = target_scaler.fit_transform(y[train_idx].reshape(-1, 1)).flatten()
            y_val = target_scaler.transform(y[val_idx].reshape(-1, 1)).flatten()
            y_te = y[test_idx]  # Keep original for final evaluation
        else:
            y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]


        # models
        num_classes = len(np.unique(y_tr)) if args.task_type != "reg" else None
        model_specs = build_sklearn_models(args.task_type, num_classes, scaler_flag=True)

        for name, (model, needs_scaler) in model_specs.items():
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
    parser.add_argument('--use_dataset_descriptors', dest='incl_desc', action='store_true',
                        help='Include dataset-specific descriptors from config (in addition to pooled atom/bond features)')
    parser.add_argument('--incl_desc', action='store_true',
                    help='Use dataset-specific descriptors')
    parser.add_argument('--incl_rdkit', action='store_true',
                        help='Include RDKit 2D descriptors')
    parser.add_argument('--incl_ab', action='store_true',
                        help='Include atom/bond pooled features')
    parser.add_argument("--polymer_type", type=str, choices=["homo", "copolymer"], default="homo",
                        help='Type of polymer: "homo" for homopolymer or "copolymer" for copolymer')
    args = parser.parse_args()


    logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # ----------------------------- Setup -----------------------------

    # Load configuration
    config = load_config()

    # Configure paths and parameters
    chemprop_dir = Path.cwd()

    # Get paths from config with defaults
    paths = config.get('PATHS', {})
    data_dir = chemprop_dir / paths.get('data_dir', 'data')
    results_dir = chemprop_dir / paths.get('results_dir', 'results') 
    feat_select_dir = chemprop_dir / paths.get('feat_select_dir', 'out') / "tabular" / args.dataset_name
    # Create necessary directories
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up file paths
    input_path = chemprop_dir / data_dir / f"{args.dataset_name}.csv"


    # Set parameters from config with defaults
    GLOBAL_CONFIG = config.get('GLOBAL', {})
    SEED = GLOBAL_CONFIG.get('SEED', 42)
    REPLICATES = GLOBAL_CONFIG.get('REPLICATES', 5)

    # Model-specific parameters
    smiles_column = 'smiles'
    ignore_columns = ['WDMPNN_Input']

    # Load dataset-specific descriptors from config
    DATASET_DESCRIPTORS = config.get('DATASET_DESCRIPTORS', {}).get(args.dataset_name, [])
    descriptor_columns = DATASET_DESCRIPTORS if args.incl_desc else []

    # === Set Random Seed ===
    set_seed(SEED)

    # Load input data from CSV
    df_input = pd.read_csv(input_path)
    
    # Load and apply any precomputed data exclusions
    drop_idx, excluded_smis = load_drop_indices(chemprop_dir, args.dataset_name)
    if drop_idx:
        logger.info(f"Dropping {len(drop_idx)} rows from {args.dataset_name} due to exclusions.")
        df_input = df_input.drop(index=drop_idx, errors="ignore").reset_index(drop=True)

    # Convert SMILES strings to RDKit molecule objects
    # smis = df_input[smiles_column].values
    # mols = [Chem.MolFromSmiles(smi) for smi in smis]

    # Identify target columns by excluding SMILES, descriptors, and ignored columns
    target_columns = [c for c in df_input.columns
                    if c not in ([smiles_column] + 
                                DATASET_DESCRIPTORS + 
                                ignore_columns)]
    # Validate that we have at least one target column
    if not target_columns:
        raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")


    # Process each target variable independently
    all_rows = []
    for tcol in target_columns:
        # Get raw target values
        y_raw = df_input[tcol].values
        
        # Handle different task types (regression, binary, or multi-class)
        if args.task_type == "reg":
            # For regression, ensure numeric type
            y_vec = y_raw.astype(float)
            
        elif args.task_type == "binary":
            # For binary classification, handle both string and numeric labels
            if y_raw.dtype.kind in "OUS":  # If string type
                y_vec = LabelEncoder().fit_transform(y_raw)
            else:
                uniq = np.unique(y_raw)
                # Convert to 0/1 if already binary, otherwise encode
                y_vec = y_raw.astype(int) if set(uniq) <= {0,1} else LabelEncoder().fit_transform(y_raw)
            # Verify binary encoding
            assert set(np.unique(y_vec)) <= {0,1}, f"{tcol} is not binary."
            
        else:  # multi-class classification
            # Encode string labels to integers
            y_vec = LabelEncoder().fit_transform(y_raw)
            # Skip if not enough classes for multi-class
            if len(np.unique(y_vec)) < 3:
                logger.warning(f"{tcol}: only {len(np.unique(y_vec))} classes; skipping for multi-class.")
                continue
        # Create output directory for this target
        out_dir = feat_select_dir / tcol
        
        # Train models and get evaluation results
        rows = train(df_input, y_vec, tcol, descriptor_columns, REPLICATES, SEED, out_dir, args)

        # Aggregate results across all targets
        all_rows.extend(rows)

    # Save aggregated results to CSV if we have any data
    if all_rows:
        # Convert results to DataFrame
        df_detailed = pd.DataFrame(all_rows)
        
        # Organize columns: target, split, model, then metrics
        base_cols = ["target", "split", "model"]
        metric_cols = sorted([c for c in df_detailed.columns if c not in base_cols])
        df_detailed = df_detailed[base_cols + metric_cols]
        
        # Write to CSV with timestamp
        suffix = ("_descriptors" if args.incl_desc else "") + ("_rdkit" if args.incl_rdkit else "") + ("_ab" if args.incl_ab else "")
        detailed_csv = results_dir / f"{args.dataset_name}_tabular{suffix}.csv"
        df_detailed.to_csv(detailed_csv, index=False)
        logger.info(f"Saved split-level results to {detailed_csv}")    

if __name__ == "__main__":
    main()
