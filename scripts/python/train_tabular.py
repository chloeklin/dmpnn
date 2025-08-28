import argparse
import logging
import numpy as np
import pandas as pd
# from rdkit import Chem
from pathlib import Path


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
    select_features_remove_constant_and_correlated
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
        X, feat_names = build_features(df, train_idx, descriptor_columns, args.polymer_type, use_rdkit=args.incl_rdkit)

        # Clean data before converting to float32
        X = np.asarray(X, dtype=np.float64)  # Use float64 first for better precision
        
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
        
        # Now safely convert to float32
        orig_Xd = X.astype(np.float32)
        logger.debug(f"Original descriptor data shape: {orig_Xd.shape}")
        logger.debug(f"Original descriptor data - finite values: {np.isfinite(orig_Xd).all()}")
        logger.debug(f"Original descriptor data - NaN count: {np.isnan(orig_Xd).sum()}")
        logger.debug(f"Original descriptor data - Inf count: {np.isinf(orig_Xd).sum()}")


        X_tr, X_val, X_te = X[train_idx], X[val_idx], X[test_idx]
        
        # 0) Convert to DataFrame so we can keep column names for selection
        X_tr_df = pd.DataFrame(X_tr, columns=feat_names)
        X_val_df = pd.DataFrame(X_val, columns=feat_names)
        X_te_df = pd.DataFrame(X_te, columns=feat_names)

        sel_info = select_features_remove_constant_and_correlated(
            X_train=X_tr_df,
            y_train=pd.Series(y[train_idx]),
            corr_threshold=0.90,       
            method="spearman" if args.task_type != "reg" else "pearson",
            min_unique=2,
            verbose=True
        )

        # 2) Apply the same mask to train/val/test
        X_tr = sel_info["transform"](X_tr_df).values
        X_val = sel_info["transform"](X_val_df).values
        X_te = sel_info["transform"](X_te_df).values
        feat_names = sel_info["kept"]

        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f"split_{i}.txt"
        with open(mask_path, "w") as f:
            f.write("\n".join(sel_info["kept"]))


        if X_tr.shape[1] == 0:
            logger.warning("Feature selection yielded 0 columns; reverting to full set for this split.")
            X_tr, X_val, X_te = X[train_idx], X[val_idx], X[test_idx]

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
            Xtr_fit = scaler.fit_transform(X_tr) if scaler is not None else X_tr
            Xval_fit = scaler.transform(X_val) if scaler is not None else X_val
            Xte_fit = scaler.transform(X_te) if scaler is not None else X_te

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
    parser.add_argument('--incl_desc', action='store_true',
                        help='Use atom+bond pooled features (graph-pooled tabular baseline)')
    parser.add_argument('--incl_rdkit', action='store_true',
                        help='Include RDKit 2D descriptors')
    parser.add_argument("--polymer_type", type=str, choices=["homo", "polymer"], default="homo",
                        help='Type of polymer: "homo" for homopolymer or "polymer" for copolymer')
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
        detailed_csv = results_dir / f"{args.dataset_name}_tabular{"_descriptors" if args.incl_desc else ""}{"_rdkit" if args.incl_rdkit else ""}.csv"
        df_detailed.to_csv(detailed_csv, index=False)
        logger.info(f"Saved split-level results to {detailed_csv}")    

if __name__ == "__main__":
    main()
