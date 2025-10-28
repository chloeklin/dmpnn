"""Utility functions for model training and data processing.

This module provides various utility functions for:
- Configuration management
- Data preprocessing and loading
- Model building and training
- Feature selection and processing
- File I/O operations
"""

# Standard library imports
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGKF = True
except Exception:
    StratifiedGroupKFold = None  # type: ignore
    HAVE_SGKF = False

def load_dmpnn_preproc(preprocessing_path: Path, split_id: int):
    """
    Load the exact preprocessing artifacts D-MPNN saved for this split.
    Files expected under preprocessing_path:
      - preprocessing_metadata_split_{i}.json
      - correlation_mask.npy
      - constant_features_removed.npy
      - descriptor_scaler.pkl (optional)
    Returns a dict with everything needed to reproduce X_d.
    """
    meta_fp = preprocessing_path / f"preprocessing_metadata_split_{split_id}.json"
    cm_fp   = preprocessing_path / "correlation_mask.npy"
    cf_fp   = preprocessing_path / "constant_features_removed.npy"
    sc_fp   = preprocessing_path / "descriptor_scaler.pkl"

    with meta_fp.open("r") as f:
        meta = json.load(f)
    corr_mask = np.load(cm_fp).astype(bool)
    const_idx = np.load(cf_fp)
    scaler = joblib.load(sc_fp) if sc_fp.exists() else None

    cleaning = meta.get("cleaning", {})
    imputer_stats = cleaning.get("imputer_statistics", None)
    f32_min = np.float32(cleaning.get("float32_min", np.finfo(np.float32).min))
    f32_max = np.float32(cleaning.get("float32_max", np.finfo(np.float32).max))

    return {
        "meta": meta,
        "const_idx": const_idx,
        "corr_mask": corr_mask,
        "scaler": scaler,
        "imputer_stats": imputer_stats,
        "f32_min": f32_min,
        "f32_max": f32_max,
    }

def apply_dmpnn_preproc(X_all: np.ndarray, artifacts: dict) -> np.ndarray:
    """
    Reproduce D-MPNN pipeline on the full descriptor matrix:
      1) drop constant columns (global)
      2) impute with train medians from JSON
      3) clip to float32 range & cast
      4) apply per-split correlation mask
      5) (optional) apply the saved StandardScaler
    """
    X = np.asarray(X_all, dtype=np.float64, copy=True)

    # 1) drop constants (indices are w.r.t original combined_descriptor_data)
    const_idx = artifacts["const_idx"]
    if const_idx.size > 0:
        X = np.delete(X, const_idx, axis=1)

    # 2) impute from saved train statistics (median)
    im_stats = artifacts["imputer_stats"]
    if im_stats is not None:
        imp = SimpleImputer(strategy="median")
        imp.statistics_ = np.asarray(im_stats, dtype=np.float64)
        X = imp.transform(X)

    # 3) clip & cast
    X = np.clip(X, artifacts["f32_min"], artifacts["f32_max"]).astype(np.float32, copy=False)

    # 4) correlation mask (boolean over post-constant space)
    X = X[:, artifacts["corr_mask"]]

    # 5) optional descriptor scaler
    if artifacts["scaler"] is not None:
        X = artifacts["scaler"].transform(X)

    return X


def _norm_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none", "") else s

def _row_group_from_two_sides(row, A_cols, B_cols):
    A = sorted([_norm_str(row[c]) for c in A_cols if c in row.index and _norm_str(row[c])])
    B = sorted([_norm_str(row[c]) for c in B_cols if c in row.index and _norm_str(row[c])])
    a_tag = "+".join(A) if A else "Ø"
    b_tag = "+".join(B) if B else "Ø"
    side1, side2 = sorted([a_tag, b_tag])  # unordered AB
    return f"{side1}||{side2}"

def compute_group_id(df: pd.DataFrame) -> pd.Series:
    """Return group IDs for unordered A/B monomer *sets* across several schemas."""
    # 0) If caller already provided group_id, just use it.
    if "group_id" in df.columns:
        return df["group_id"].astype(str)

    # 1) PAE-wide by names
    A_name = [c for c in ["a1","a2","a3","a4"] if c in df.columns]
    B_name = [c for c in ["b1","b2","b3","b4"] if c in df.columns]
    if A_name and B_name:
        return df.apply(lambda r: _row_group_from_two_sides(r, A_name, B_name), axis=1)

    # 2) PAE-wide by SMILES
    A_sm = [c for c in ["smilesA1","smilesA2","smilesA3","smilesA4"] if c in df.columns]
    B_sm = [c for c in ["smilesB1","smilesB2","smilesB3","smilesB4"] if c in df.columns]
    if A_sm and B_sm:
        return df.apply(lambda r: _row_group_from_two_sides(r, A_sm, B_sm), axis=1)

    # 3) Legacy simple AB
    if {"smiles_A","smiles_B"}.issubset(df.columns):
        return df.apply(lambda r: _row_group_from_two_sides(r, ["smiles_A"], ["smiles_B"]), axis=1)

    raise KeyError(
        "Could not infer A/B monomer columns to build groups "
        "(expected a1..a4/b1..b4 or smilesA*/smilesB* or smiles_A/smiles_B, "
        "or provide a 'group_id' column)."
    )

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and process configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, looks for 'train_config.yaml' 
                   in the same directory as this module.
                   
    Returns:
        Dict containing configuration with the following structure:
        {
            'GLOBAL': Dict of global configuration parameters,
            'MODELS': Dict of model configurations,
            'DATASET_DESCRIPTORS': Dict of dataset descriptors,
            'DATASET_IGNORE': Dict of dataset ignore-lists,
            'PATHS': Dict of path settings
        }
    """
    import yaml
    import os
    from pathlib import Path
    
    if config_path is None:
        config_path = Path(__file__).parent / 'train_config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # normalize keys and build result
    result = {
        'GLOBAL': {k.upper(): v for k, v in config.get('global', {}).items()},
        'MODELS': config.get('models', {}),
        'DATASET_DESCRIPTORS': config.get('dataset_descriptors', {}),
        'DATASET_IGNORE': config.get('dataset_ignore', {}),
        'PATHS': config.get('paths', {}),
    }
    
    # Handle dynamic values
    if 'NUM_WORKERS' not in result['GLOBAL'] and 'num_workers' in config.get('global', {}):
        num_workers = config['global']['num_workers']
        max_workers = os.cpu_count() or 1
        result['GLOBAL']['NUM_WORKERS'] = min(
            num_workers if isinstance(num_workers, int) else max_workers,
            max_workers
        )
    
    return result


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value. Defaults to 42.
    """
    import torch  # Local import for heavy dependency
    import random
    
    # Set Python built-in random
    random.seed(seed)
    
    # Set NumPy
    np.random.seed(seed)
    
    # Set PyTorch
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Additional libraries that might need seeding
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def combine_descriptors(
    rdkit_data: Optional[np.ndarray], 
    descriptor_data: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Combine RDKit descriptors with additional descriptor data.
    
    Args:
        rdkit_data: Array of RDKit descriptors (n_samples, n_rdkit_features)
        descriptor_data: Array of additional descriptors (n_samples, n_extra_features)
        
    Returns:
        Combined feature array or None if both inputs are None
        
    Raises:
        ValueError: If input arrays have incompatible shapes
    """
    if rdkit_data is not None and descriptor_data is not None:
        if len(rdkit_data) != len(descriptor_data):
            raise ValueError(
                f"Mismatched array lengths: rdkit_data ({len(rdkit_data)}) "
                f"and descriptor_data ({len(descriptor_data)})"
            )
        return np.asarray(
            [np.concatenate([r, d]) for r, d in zip(rdkit_data, descriptor_data)], 
            dtype=np.float32
        )
    elif rdkit_data is not None:
        return np.asarray(rdkit_data, dtype=np.float32)
    elif descriptor_data is not None:
        return np.asarray(descriptor_data, dtype=np.float32)
    return None

def preprocess_classification_labels(
    df_input: pd.DataFrame, 
    target_columns: List[str], 
    task_type: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Preprocess classification labels in a DataFrame.
    
    Args:
        df_input: Input DataFrame containing the target columns
        target_columns: List of column names containing target variables
        task_type: Type of task - 'binary' or 'multi'
        verbose: Whether to print information about the preprocessing
        
    Returns:
        DataFrame with preprocessed labels
        
    Raises:
        ValueError: If labels are invalid for the specified task type
        TypeError: If input types are incorrect
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df_input).__name__}")
    
    if task_type not in ('binary', 'multi'):
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'binary' or 'multi'")
    
    df = df_input.copy()
    
    for tcol in target_columns:
        if tcol not in df.columns:
            raise ValueError(f"Target column '{tcol}' not found in DataFrame")
        
        # Convert string labels to integers if needed
        if pd.api.types.is_string_dtype(df[tcol]) or pd.api.types.is_object_dtype(df[tcol]):
            classes = sorted(df[tcol].dropna().unique().tolist())
            class_to_idx = {c: i for i, c in enumerate(classes)}
            df[tcol] = df[tcol].map(class_to_idx)

        # Check for negative labels (use NaN for missing values)
        if (df[tcol].dropna() < 0).any():
            raise ValueError(
                f"Found negative class labels in {tcol}. "
                "Replace missing labels with NaN, not -1."
            )

        # Convert to integer type
        df[tcol] = pd.to_numeric(df[tcol], errors='coerce').astype('Int64')
        uniq = np.sort(df[tcol].dropna().unique())

        if task_type == 'multi':
            # Ensure labels are contiguous 0..C-1
            expected_max = len(uniq) - 1
            if uniq.size > 0 and (uniq.min() != 0 or uniq.max() != expected_max):
                remap = {c: i for i, c in enumerate(uniq)}
                df[tcol] = df[tcol].map(remap)
                uniq = np.sort(df[tcol].dropna().unique())
            
            if verbose:
                print(f"[multi] {tcol}: classes={uniq} (n={len(uniq)})")
        else:  # binary
            valid_binary = [np.array([0]), np.array([1]), np.array([0, 1])]
            if not any(np.array_equal(uniq, arr) for arr in valid_binary):
                raise ValueError(
                    f"Binary labels in {tcol} must be 0/1. Found {uniq}."
                )
    
    return df

def process_data(
    df_input: pd.DataFrame,
    smiles_column: str,
    descriptor_columns: Optional[List[str]],
    target_columns: List[str],
    args: Any,
    verbose: bool = True
) -> Tuple[np.ndarray, pd.DataFrame, Optional[np.ndarray], Dict[str, int]]:
    """Process input data for model training or inference.
    
    This function handles:
    - Extracting SMILES strings
    - Preprocessing classification labels if needed
    - Computing RDKit descriptors if requested
    - Combining with additional descriptor data
    
    Args:
        df_input: Input DataFrame containing SMILES and target values
        smiles_column: Name of the column containing SMILES strings
        descriptor_columns: List of column names containing additional descriptors
        target_columns: List of target column names
        args: Command line arguments or config object with model parameters
        verbose: Whether to print progress information
        
    Returns:
        Tuple containing:
        - Array of SMILES strings
        - Processed DataFrame
        - Combined descriptor data (or None if no descriptors)
        - Dictionary mapping target columns to number of classes (for classification)
        
    Raises:
        ValueError: If input data is invalid or processing fails
        KeyError: If required columns are missing from the input DataFrame
    """

    from tabular_utils import rdkit_block_from_smiles
    # Check required columns
    required_columns = {smiles_column, *target_columns}
    missing_columns = required_columns - set(df_input.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Extract SMILES strings
    smis = df_input[smiles_column].values
    
    # Validate SMILES strings
    invalid_smiles = [
        (i, smi, type(smi).__name__) 
        for i, smi in enumerate(smis) 
        if not isinstance(smi, str) or not smi.strip()
    ]
    
    if invalid_smiles and verbose:
        for i, smi, typ in invalid_smiles[:5]:  # Show first 5 invalid SMILES
            print(f"Warning: Invalid SMILES at index {i}: {smi!r} (type: {typ})")
        if len(invalid_smiles) > 5:
            print(f"... and {len(invalid_smiles) - 5} more invalid SMILES")
    
    # Preprocess classification labels if needed
    n_classes_per_target = {}
    if hasattr(args, 'task_type') and args.task_type in ['binary', 'multi']:
        df_input = preprocess_classification_labels(
            df_input, target_columns, args.task_type, verbose=verbose
        )
        
        if args.task_type == 'multi':
            for tcol in target_columns:
                n_classes = df_input[tcol].dropna().nunique()
                n_classes_per_target[tcol] = int(n_classes)
    
    # Process descriptor data
    descriptor_data = None
    if descriptor_columns:
        try:
            descriptor_data = df_input[descriptor_columns].values.astype(np.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert descriptor columns to float32: {e}"
            ) from e
    
    # Process RDKit descriptors if requested
    rdkit_data = None
    if getattr(args, 'incl_rdkit', False):
        # Use original SMILES for wDMPNN model if available
        use_smiles = df_input["smiles"].values if hasattr(args, 'model_name') and args.model_name == "wDMPNN" else smis
        rdkit_data = rdkit_block_from_smiles(use_smiles)

    # Combine RDKit and additional descriptors
    combined_descriptor_data = combine_descriptors(rdkit_data, descriptor_data)
    
    # Validate combined descriptor shape
    if combined_descriptor_data is not None:
        if combined_descriptor_data.ndim != 2:
            raise ValueError(
                f"Combined descriptor data must be 2D, got shape {combined_descriptor_data.shape}"
            )
        if len(combined_descriptor_data) != len(smis):
            raise ValueError(
                f"Mismatched number of samples: {len(smis)} SMILES vs "
                f"{len(combined_descriptor_data)} descriptor rows"
            )
    
    return smis, df_input, combined_descriptor_data, n_classes_per_target

def create_all_data(
    smis: List[str],
    ys: Union[List[float], np.ndarray],
    combined_descriptor_data: Optional[np.ndarray],
    model_name: str
) -> List[Any]:
    """Create a list of datapoints for model training or inference.
    
    Args:
        smis: List of SMILES strings
        ys: Target values (floats for regression, ints for classification)
        combined_descriptor_data: Optional array of combined RDKit and custom descriptors
        model_name: Name of the model type ("DMPNN" or other for polymer models)
        
    Returns:
        List of datapoint objects suitable for the specified model type
        
    Raises:
        ValueError: If inputs have incompatible shapes or types
        ImportError: If required chemprop modules are not available
    """
    try:
        from chemprop import data
    except ImportError as e:
        raise ImportError(
            "chemprop package is required for creating datapoints. "
            "Install with: pip install chemprop"
        ) from e
    
    # Validate input shapes
    if len(smis) != len(ys):
        raise ValueError(
            f"Mismatched number of SMILES ({len(smis)}) and targets ({len(ys)})"
        )
    
    if combined_descriptor_data is not None and len(combined_descriptor_data) != len(smis):
        raise ValueError(
            f"Mismatched number of SMILES ({len(smis)}) and descriptor rows "
            f"({len(combined_descriptor_data)})"
        )
    
    # Convert ys to list if it's a numpy array
    if isinstance(ys, np.ndarray):
        ys = ys.tolist()
    
    # Create datapoints based on model type
    # Use MoleculeDatapoint for small molecule models (DMPNN, DMPNN_DiffPool, AttentiveFP, PPG)
    # Use PolymerDatapoint for polymer models (wDMPNN)
    small_molecule_models = ["DMPNN", "DMPNN_DiffPool", "PPG"]
    datapoint_class = (
        data.MoleculeDatapoint 
        if model_name in small_molecule_models
        else data.PolymerDatapoint
    )
    
    if combined_descriptor_data is not None:
        return [
            datapoint_class.from_smi(smi, y, x_d=desc) 
            for smi, y, desc in zip(smis, ys, combined_descriptor_data)
            if smi and pd.notna(y).any()  # Skip invalid SMILES, but allow NaN targets (for multitask)
        ]
    else:
        return [
            datapoint_class.from_smi(smi, y)
            for smi, y in zip(smis, ys)
            if smi and pd.notna(y).any()  # Skip invalid SMILES, but allow NaN targets (for multitask)
        ]


        


def get_metric_list(
    task_type: str, 
    target: Optional[str] = None, 
    n_classes: Optional[int] = None, 
    df_input: Optional[pd.DataFrame] = None
) -> List[Any]:
    """Get a list of metrics appropriate for the given task type.

    Backward-compatible:
      - 'reg'  -> regression metrics
      - 'binary' -> binary metrics (AUROC only if both classes present in df_input[target])
      - 'multi'  -> multiclass metrics (requires n_classes)
      - 'mixed-reg-multi' -> returns a *combined* set of regression + multiclass metrics
        using a single num_classes for the multiclass metrics, inferred from `target`
        (encoded as 'name:k,name2:k2,...') or else from df_input as a safe maximum.
    """
    try:
        from chemprop import nn  # Local import for heavy dependency
    except ImportError as e:
        raise ImportError(
            "chemprop package is required for metrics. Install with: pip install chemprop"
        ) from e

    # ---------------- existing behavior ----------------
    if task_type == 'reg':
        return [nn.metrics.MAE(), nn.metrics.RMSE(), nn.metrics.R2Score()]

    elif task_type == 'binary':
        if df_input is not None and target is not None:
            unique_classes = df_input[target].dropna().unique()
            has_both_classes = len(unique_classes) > 1
        else:
            has_both_classes = False

        metrics = [nn.metrics.BinaryAccuracy(), nn.metrics.BinaryF1Score()]
        if has_both_classes:
            metrics.append(nn.metrics.BinaryAUROC())
        return metrics

    elif task_type == 'multi':
        if n_classes is None:
            raise ValueError("n_classes must be provided for multi-class tasks")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        return [
            nn.metrics.MulticlassAccuracy(num_classes=n_classes, average='macro'),
            nn.metrics.MulticlassF1Score(num_classes=n_classes, average='macro'),
            nn.metrics.MulticlassAUROC(num_classes=n_classes, average='macro')
        ]

    # ---------------- new mixed branch (keeps signature) ----------------
    elif task_type == 'mixed-reg-multi':
        # Determine a single num_classes to configure multiclass metrics.
        # Priority:
        #   1) parse from `target` if provided as "name:k,name2:k2,..."
        #   2) infer from df_input by scanning integer-like columns (safe upper bound)
        num_classes_for_metrics = None

        # Parse explicit "name:k" list passed via `target` (no API change)
        if isinstance(target, str) and (":" in target):
            try:
                parts = [p.strip() for p in target.split(",") if p.strip()]
                ks = []
                for p in parts:
                    name, k = p.split(":")
                    ks.append(int(k))
                if ks:
                    num_classes_for_metrics = max(ks)
            except Exception:
                # fall back to inference if parsing fails
                num_classes_for_metrics = None

        # Fallback: infer from df_input (take a safe max)
        if num_classes_for_metrics is None and df_input is not None:
            candidate_max = 0
            for col in df_input.columns:
                s = df_input[col].dropna()
                if s.empty:
                    continue
                # consider integer dtype or floats that look like integer labels
                is_intish = (pd.api.types.is_integer_dtype(s) or
                             (pd.api.types.is_float_dtype(s) and np.all(np.isclose(s, np.round(s)))))
                if is_intish:
                    vmax = int(s.max())
                    candidate_max = max(candidate_max, vmax)
            if candidate_max >= 1:
                num_classes_for_metrics = candidate_max + 1  # assume 0-based

        # Last resort default (keeps metrics constructible)
        if num_classes_for_metrics is None:
            num_classes_for_metrics = 2

        # Return a combined set: regression + multiclass metrics
        return [
            # regression metrics
            nn.metrics.MAE(),
            nn.metrics.RMSE(),
            nn.metrics.R2Score(),
            # multiclass metrics (macro)
            nn.metrics.MulticlassAccuracy(num_classes=num_classes_for_metrics, average='macro'),
            nn.metrics.MulticlassF1Score(num_classes=num_classes_for_metrics, average='macro'),
            nn.metrics.MulticlassAUROC(num_classes=num_classes_for_metrics, average='macro')
        ]

    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. Must be one of: 'reg', 'binary', 'multi', 'mixed-reg-multi'"
        )


def build_model_and_trainer(
    args: Any,
    combined_descriptor_data: Optional[np.ndarray],
    n_classes: Optional[int],
    scaler: Optional[Any],
    checkpoint_path: Union[str, Path],
    batch_norm: bool = True,
    metric_list: Optional[List[Any]] = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 0.0,
    max_epochs: int = 300,
    gradient_clip_val: float = 10.0,
    save_checkpoint: bool = True,
    **trainer_kwargs
) -> Tuple[Any, Any]:  # Returns (model, trainer)
    """Build and configure a chemprop model and PyTorch Lightning trainer.
    
    Args:
        args: Command line arguments or config object with model parameters
        combined_descriptor_data: Combined RDKit and custom descriptors
        n_classes: Number of classes (for classification tasks)
        scaler: Scaler object for regression tasks
        checkpoint_path: Directory to save model checkpoints
        batch_norm: Whether to use batch normalization
        metric_list: List of metrics to track during training
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        max_epochs: Maximum number of training epochs
        gradient_clip_val: Maximum gradient norm for gradient clipping
        save_checkpoint: Whether to save model checkpoints (default: True)
        **trainer_kwargs: Additional arguments for the PyTorch Lightning Trainer
        
    Returns:
        Tuple containing:
        - Configured chemprop model
        - PyTorch Lightning Trainer instance
        
    Raises:
        ValueError: If invalid arguments are provided
        ImportError: If required packages are not available
    """
    try:
        # Local imports for heavy dependencies
        import torch
        from chemprop import nn, models
        from chemprop.nn import MixedRegMultiFFN
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        import lightning.pytorch as pl
    except ImportError as e:
        raise ImportError(
            "Required packages (chemprop, pytorch_lightning) not found. "
            "Install with: pip install chemprop pytorch_lightning"
        ) from e

    # Validate inputs
    if not hasattr(args, 'task_type'):
        raise ValueError("args must have 'task_type' attribute")
    
    if args.task_type == 'multi' and n_classes is None:
        raise ValueError("n_classes must be provided for multi-class tasks")
    
    if not hasattr(args, 'model_name'):
        raise ValueError("args must have 'model_name' attribute")
    
    # Create aggregation and model
    if args.model_name == "wDMPNN":
        mp = nn.WeightedBondMessagePassing()
        agg = nn.WeightedMeanAggregation()        # ✅ no-op for graph-level outputs
    elif args.model_name == "DMPNN":
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
    elif args.model_name == "DMPNN_SumPool":
        mp = nn.BondMessagePassing()
        agg = nn.SumAggregation()
    elif args.model_name == "DMPNN_AttnPool":
        mp = nn.BondMessagePassing()
        agg = nn.AttentiveAggregation()
    elif args.model_name == "DMPNN_DiffPool":
        # ---- base D-MPNN used INSIDE the wrapper ----
        base_mp_cls = nn.BondMessagePassing
        
        mp = nn.BondMessagePassingWithDiffPool(
        base_mp_cls=base_mp_cls,
        depth=getattr(args, "diffpool_depth", 1)
        )

        agg = nn.IdentityAggregation()
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    
    # Calculate input dimension for FFN
    descriptor_dim = combined_descriptor_data.shape[1] if combined_descriptor_data is not None else 0
    input_dim = mp.output_dim + descriptor_dim
    
    # Predictions are unscaled to original units before metrics via output_transform
    output_transform = None
    if args.task_type == 'reg':
        output_transform = (
            nn.UnscaleTransform.from_standard_scaler(scaler) 
            if scaler is not None 
            else None
        )
    
    # Create Feed-Forward Network based on task type
    if args.task_type == 'reg':
        ffn = nn.RegressionFFN(
            output_transform=output_transform, 
            n_tasks=1, 
            input_dim=input_dim
        )
    elif args.task_type == 'binary':
        ffn = nn.BinaryClassificationFFN(input_dim=input_dim)
    elif args.task_type == 'multi':
        if n_classes is None:
            raise ValueError("n_classes must be provided for multi-class tasks")
        ffn = nn.MulticlassClassificationFFN(
            n_classes=n_classes, 
            input_dim=input_dim
        )
    elif args.task_type == 'mixed-reg-multi':
        if not hasattr(args, "task_specs"):
            raise ValueError("For task_type='mixed-reg-multi', provide args.task_specs aligned with target_columns.")
        task_weights_tensor = None
        if getattr(args, "task_weights", ""):
            tw = [float(x) for x in args.task_weights.split(",")]
            assert len(tw) == len(args.task_specs), "--task_weights must match number of tasks"
            task_weights_tensor = torch.tensor(tw, dtype=torch.float32)

        ffn = MixedRegMultiFFN(
            task_specs=args.task_specs,
            input_dim=input_dim,
            task_weights=task_weights_tensor,
            reg_mu_per_task=getattr(args, "reg_mu_per_task", None),
            reg_sd_per_task=getattr(args, "reg_sd_per_task", None),
        )
    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")
    
    
    mpnn = models.MPNN(
        message_passing=mp, 
        agg=agg, 
        predictor=ffn, 
        batch_norm=batch_norm, 
        metrics=metric_list or [],
    )
    
    # Convert to Path object but don't create directory yet - let Lightning handle it
    checkpoint_path = Path(checkpoint_path)
    
    # Configure callbacks
    callbacks = []
    
    # Add checkpointing only if save_checkpoint is True
    if save_checkpoint:
        checkpointing = ModelCheckpoint(
            dirpath=str(checkpoint_path),
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,  # Save the last model as well for resuming training
            save_weights_only=False,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpointing)
    
    # Configure early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode="min",
        verbose=True,
        check_finite=True,  # Stop if loss becomes NaN or infinite
        check_on_train_epoch_end=False,  # Only validate at the end of validation epoch
    )
    callbacks.append(early_stop)
    
    # Configure learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
        log_momentum=True
    )
    callbacks.append(lr_monitor)
    
    # Configure logging with explicit flush_logs_every_n_steps
    logger = pl.loggers.CSVLogger(
        save_dir=str(checkpoint_path),
        name="logs",
        version="",  # Use empty string to avoid creating versioned subdirectories
        flush_logs_every_n_steps=50  # Flush logs less frequently to avoid file access issues
    )
    
    # Ensure logs directory exists
    log_dir = checkpoint_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",  # Clip by norm
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,  # Log metrics every 10 steps
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,  # Disable validation sanity check for faster startup
        deterministic=True,  # For reproducibility
        enable_checkpointing=True,  # Explicitly enable checkpointing
        default_root_dir=str(checkpoint_path),  # Set default root dir for checkpoints
        **trainer_kwargs  # Allow overriding any trainer parameter
    )
    
    return mpnn, trainer






def make_repeated_splits(
    task_type: str,
    replicates: int,
    seed: int,
    *,
    y_class: Optional[Union[np.ndarray, List[float]]] = None,  # for classification
    y_reg:   Optional[Union[np.ndarray, List[float]]] = None,  # for regression (NEW)
    n_splits: int = 5,                # >1 => CV; 1 => holdout
    stratify: bool = True,            # only used for classification
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns (train_indices, val_indices, test_indices).

    Plan it implements:
      • CV (n_splits>1): test = 1/n_splits; val = 10% of the train chunk
                         (for 5-fold: overall 72/8/20).
      • Holdout (n_splits==1): 80/10/10 overall, repeated `replicates` times.

    Classification: stratified splits.
    Regression: unstratified, purely random (as per your plan).
    """
    from sklearn.model_selection import (
        KFold, StratifiedKFold,
        train_test_split,
        StratifiedShuffleSplit, ShuffleSplit
    )

    if task_type not in ("reg", "binary", "multi"):
        raise ValueError("task_type must be 'reg', 'binary', or 'multi'")
    is_clf = task_type in ("binary", "multi")

    if is_clf:
        if y_class is None:
            raise ValueError("y_class must be provided for classification")
        y_arr = np.asarray(y_class)
        n_samples = len(y_arr)
    else:
        if y_reg is None:
            raise ValueError("y_reg must be provided for regression")
        y_arr = None  # not used for stratification in regression
        n_samples = len(y_reg)

    idx_all = np.arange(n_samples)
    train_indices: List[np.ndarray] = []
    val_indices:   List[np.ndarray] = []
    test_indices:  List[np.ndarray] = []

    use_cv = (n_splits > 1)
    val_frac_within_train = 0.10  # 10% of train chunk

    if use_cv:
        # ----- CV path -----
        for rep in range(replicates):
            rep_seed = seed + rep
            if is_clf:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
                split_iter = splitter.split(idx_all, y_arr)
            else:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
                split_iter = splitter.split(idx_all)

            for train_val_idx, test_idx in split_iter:
                strat = (y_arr[train_val_idx] if (is_clf and stratify) else None)
                tr_idx, va_idx = train_test_split(
                    train_val_idx,
                    test_size=val_frac_within_train,
                    stratify=strat,
                    random_state=rep_seed
                )
                train_indices.append(np.asarray(tr_idx))
                val_indices.append(np.asarray(va_idx))
                test_indices.append(np.asarray(test_idx))
    else:
        # ----- Holdout 80/10/10 repeated -----
        for i in range(replicates):
            rs = seed + i
            if is_clf:
                sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=rs)
                (train_val_idx, test_idx), = sss_outer.split(idx_all, y_arr)
                sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=rs)  # 10% overall val
                (tr_idx, va_idx), = sss_inner.split(train_val_idx, y_arr[train_val_idx])
            else:
                ss_outer = ShuffleSplit(n_splits=1, test_size=0.10, random_state=rs)
                (train_val_idx, test_idx), = ss_outer.split(idx_all)
                ss_inner = ShuffleSplit(n_splits=1, test_size=1/9, random_state=rs)  # 10% overall val
                (tr_idx, va_idx), = ss_inner.split(train_val_idx)

            train_indices.append(train_val_idx[tr_idx])
            val_indices.append(train_val_idx[va_idx])
            test_indices.append(test_idx)

    # sanity checks
    for tr, va, te in zip(train_indices, val_indices, test_indices):
        s_tr, s_va, s_te = set(tr), set(va), set(te)
        if s_tr & s_va or s_tr & s_te or s_va & s_te:
            raise ValueError("Overlap detected among train/val/test indices")
        if len(s_tr | s_va | s_te) != n_samples:
            raise ValueError("Split does not cover all samples")

    return train_indices, val_indices, test_indices




    



def build_sklearn_models(task_type, n_classes=None, scaler_flag=False):
    # Local imports for scikit-learn and xgboost
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from xgboost import XGBClassifier, XGBRegressor
    
    models_dict = {}
    if task_type == "reg":
        baselines=["Linear", "RF", "XGB"]
        for baseline in baselines:
            if baseline == "Linear":
                models_dict["Linear"] = (LinearRegression(), True) if scaler_flag else LinearRegression()
            elif baseline == "RF":
                models_dict["RF"] = (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), False) if scaler_flag else RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
            elif baseline == "XGB":
                models_dict["XGB"] = (XGBRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1), False) if scaler_flag else XGBRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    else:
        multi = (task_type == "multi")
        baselines=["LogReg", "RF", "XGB"]
        for baseline in baselines:
            if baseline == "LogReg":
                models_dict["LogReg"] = (LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=-1 if hasattr(LogisticRegression(), "n_jobs") else None,
                    multi_class="multinomial" if multi else "auto"
                ), True) if scaler_flag else LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=-1 if hasattr(LogisticRegression(), "n_jobs") else None,
                    multi_class="multinomial" if multi else "auto"
                )
            elif baseline == "RF":
                models_dict["RF"] = (RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),False) if scaler_flag else RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
            elif baseline == "XGB":
                models_dict["XGB"] = (XGBClassifier(
                    n_estimators=500,
                    max_depth=10,
                    objective="multi:softprob" if multi else "binary:logistic",
                    num_class=n_classes if multi else None,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss" if multi else "logloss"
                ), False) if scaler_flag else XGBClassifier(
                    n_estimators=500,
                    max_depth=10,
                    objective="multi:softprob" if multi else "binary:logistic",
                    num_class=n_classes if multi else None,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss" if multi else "logloss"
                )
    return models_dict




def manage_preprocessing_cache(preprocessing_path, i, combined_descriptor_data, split_preprocessing_metadata, 
                             descriptor_scaler, logger):
    """Handle loading/saving of preprocessing cache."""
    import json
    from joblib import load, dump
    
    # If no descriptors, return defaults
    if combined_descriptor_data is None:
        return False, None, np.array([]), np.array([])
    
    # Define file paths
    metadata_file = preprocessing_path / f"preprocessing_metadata_split_{i}.json"
    scaler_file = preprocessing_path / "descriptor_scaler.pkl"
    correlation_mask_file = preprocessing_path / "correlation_mask.npy"
    constant_features_file = preprocessing_path / "constant_features_removed.npy"
    
    preprocessing_path.mkdir(parents=True, exist_ok=True)
    
    # If descriptor_scaler is provided, this is a save operation
    if descriptor_scaler is not None:
        # Save scaler and return (don't re-save metadata)
        dump(descriptor_scaler, scaler_file)
        logger.info(f"Saved descriptor scaler to {scaler_file}")
        return False, descriptor_scaler, np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'], dtype=bool), np.array(split_preprocessing_metadata[i]['data_info']['constant_features_removed'])
    
    # Check if preprocessing can be reused (load operation)
    preprocessing_exists = (
        metadata_file.exists() and 
        correlation_mask_file.exists() and 
        constant_features_file.exists()
    )
    
    if preprocessing_exists:
        try:
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            
            # Get current preprocessing parameters for comparison
            current_constant_features = split_preprocessing_metadata[i]['data_info']['constant_features_removed']
            current_correlation_mask = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'], dtype=bool)
            current_n_features_final = np.sum(current_correlation_mask)
            
            # Get saved preprocessing parameters
            saved_constant_features = existing_metadata.get('data_info', {}).get('constant_features_removed', [])
            saved_n_features_final = existing_metadata.get('data_info', {}).get('n_features_after_preprocessing', 0)
            
            # Load saved masks for detailed comparison
            saved_correlation_mask = np.load(correlation_mask_file).astype(bool)
            saved_constant_features_array = np.load(constant_features_file)
            
            # Deterministic reuse: require exact match of preprocessing parameters
            masks_match = np.array_equal(current_correlation_mask, saved_correlation_mask)
            constants_match = (current_constant_features == saved_constant_features or 
                             np.array_equal(np.array(current_constant_features), saved_constant_features_array))
            counts_match = (current_n_features_final == saved_n_features_final)
            
            if masks_match and constants_match and counts_match:
                # Load existing preprocessing objects with type safety
                correlation_mask = saved_correlation_mask
                constant_features = saved_constant_features
                
                # Only load scaler if we don't already have one
                if descriptor_scaler is None and scaler_file.exists():
                    descriptor_scaler = load(scaler_file)
                    logger.info(f"Loaded existing descriptor scaler from {scaler_file}")
                
                logger.info(f"Reusing existing preprocessing for split {i} with {current_n_features_final} final features")
                return True, descriptor_scaler, correlation_mask, constant_features
            else:
                logger.info(f"Preprocessing parameters changed (masks_match={masks_match}, constants_match={constants_match}, counts_match={counts_match}), recomputing...")
                
        except Exception as e:
            logger.warning(f"Could not load existing preprocessing: {e}, recomputing...")
    
    # Save new preprocessing - add n_features_after_preprocessing to metadata
    correlation_mask = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'], dtype=bool)
    n_features_final = np.sum(correlation_mask)
    
    # Add the missing field to metadata
    split_preprocessing_metadata[i]['data_info']['n_features_after_preprocessing'] = int(n_features_final)
    
    # Save preprocessing files
    np.save(correlation_mask_file, correlation_mask)
    np.save(constant_features_file, np.array(split_preprocessing_metadata[i]['data_info']['constant_features_removed']))
    
    with open(metadata_file, 'w') as f:
        json.dump(split_preprocessing_metadata[i], f, indent=2)
    
    logger.info(f"Saved preprocessing metadata for split {i} with {n_features_final} final features")
    return False, None, correlation_mask, np.array(split_preprocessing_metadata[i]['data_info']['constant_features_removed'])


def build_experiment_paths(args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i):
    """Build all paths needed for experiment tracking."""
    desc_suffix = "__desc" if descriptor_columns else ""
    rdkit_suffix = "__rdkit" if args.incl_rdkit else ""
    batch_norm_suffix = "__batch_norm" if getattr(args, 'batch_norm', False) else ""
    
    # Add train_size suffix if specified and not "full"
    size_suffix = ""
    train_size = getattr(args, 'train_size', None)
    if train_size is not None and train_size.lower() != "full":
        size_suffix = f"__size{train_size}"
    
    base_name = f"{args.dataset_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}__rep{i}"
    
    checkpoint_path = checkpoint_dir / base_name
    model_name = getattr(args, 'model_name', None) or getattr(args, 'model', 'DMPNN')
    if model_name == "AttentiveFP":
        model_name = "DMPNN"
    preprocessing_path = chemprop_dir / "preprocessing" / model_name / base_name
    
    return checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix


def validate_checkpoint_compatibility(checkpoint_path, preprocessing_path, i, descriptor_dim, logger):
    """Check if existing checkpoints are compatible with current preprocessing.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        preprocessing_path: Path to preprocessing metadata
        i: Split index
        descriptor_dim: Number of descriptor features after all preprocessing (int)
        logger: Logger instance
    """
    import os
    import json
    
    if not checkpoint_path.exists():
        return None
        
    ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
    if not ckpt_files:
        return None
    
    # Sort by modification time (newest first), not lexicographically
    ckpt_files_with_time = [(f, os.path.getmtime(checkpoint_path / f)) for f in ckpt_files]
    ckpt_files_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_ckpt = ckpt_files_with_time[0][0]
    
    # If no descriptors, skip metadata validation and allow resume
    if descriptor_dim == 0:
        last_ckpt = str(checkpoint_path / latest_ckpt)
        logger.info(f"Loading checkpoint: {last_ckpt} (no descriptors, skipping metadata check)")
        return last_ckpt
    
    # For descriptor-based models, validate preprocessing compatibility
    metadata_file = preprocessing_path / f"preprocessing_metadata_split_{i}.json"
    logger.info(f"Looking for preprocessing metadata at: {metadata_file}")
    if not metadata_file.exists():
        logger.warning(f"No preprocessing metadata found for checkpoint validation at: {metadata_file}")
        return None
        
    try:
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        # Use the processed descriptor dimension directly
        current_n_features = descriptor_dim
        saved_n_features = saved_metadata.get('data_info', {}).get('n_features_after_preprocessing', 0)
        
        if current_n_features == saved_n_features:
            last_ckpt = str(checkpoint_path / latest_ckpt)
            logger.info(f"Loading checkpoint: {last_ckpt} (features match: {current_n_features})")
            return last_ckpt
        else:
            logger.warning(f"Checkpoint feature mismatch: current={current_n_features}, saved={saved_n_features}")
            return None
            
    except Exception as e:
        logger.warning(f"Could not validate checkpoint compatibility: {e}")
        return None


def load_existing_results(detailed_csv: Path, logger: logging.Logger) -> Dict[str, Dict[int, set]]:
    """Load existing results from CSV file and return structured data.
    
    Args:
        detailed_csv: Path to detailed results CSV file
        logger: Logger instance
        
    Returns:
        Dictionary mapping target -> split -> set of completed models
    """
    existing_results = {}
    if detailed_csv.exists():
        try:
            existing_df = pd.read_csv(detailed_csv)
            logger.info(f"Found existing results file: {detailed_csv}")
            
            # Group by target and split to track completed experiments
            for _, row in existing_df.iterrows():
                target = row['target']
                split = row['split']
                model = row['model']
                
                if target not in existing_results:
                    existing_results[target] = {}
                if split not in existing_results[target]:
                    existing_results[target][split] = set()
                existing_results[target][split].add(model)
            
            logger.info(f"Loaded existing results for {len(existing_results)} targets")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            existing_results = {}
    
    return existing_results


def save_combined_results(detailed_csv: Path, existing_results_df: Optional[pd.DataFrame], 
                         new_results: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """Save combined existing and new results to CSV file.
    
    Args:
        detailed_csv: Path to output CSV file
        existing_results_df: DataFrame of existing results (can be None)
        new_results: List of new result dictionaries
        logger: Logger instance
    """
    final_rows = []
    
    # Add existing results that weren't recomputed
    if existing_results_df is not None:
        final_rows.extend(existing_results_df.to_dict('records'))
        logger.info(f"Loaded {len(existing_results_df)} existing result rows")
    
    # Add new results
    final_rows.extend(new_results)
    
    # Save combined results to CSV if we have any data
    if final_rows:
        # Convert results to DataFrame and remove duplicates
        df_detailed = pd.DataFrame(final_rows)
        
        # Remove duplicates based on target, split, model (keep last occurrence)
        df_detailed = df_detailed.drop_duplicates(subset=['target', 'split', 'model'], keep='last')
        
        # Organize columns: target, split, model, then metrics
        base_cols = ["target", "split", "model"]
        metric_cols = sorted([c for c in df_detailed.columns if c not in base_cols])
        df_detailed = df_detailed[base_cols + metric_cols]
        
        # Write to CSV
        df_detailed.to_csv(detailed_csv, index=False)
        logger.info(f"Saved {len(df_detailed)} total results to {detailed_csv}")
        
        # Log summary of what was done
        new_results_count = len(new_results)
        existing_count = len(final_rows) - new_results_count
        logger.info(f"Results summary: {existing_count} existing + {new_results_count} new = {len(df_detailed)} total")
    else:
        logger.info("No results to save")


def prepare_target_data(y_raw: np.ndarray, task_type: str, target_name: str, logger: logging.Logger) -> np.ndarray:
    """Prepare target data based on task type.
    
    Args:
        y_raw: Raw target values
        task_type: Task type ('reg', 'binary', 'multi')
        target_name: Name of target variable
        logger: Logger instance
        
    Returns:
        Processed target values
        
    Raises:
        ValueError: If binary target is not actually binary or multi-class has < 3 classes
    """
    from sklearn.preprocessing import LabelEncoder
    
    if task_type == "reg":
        # For regression, ensure numeric type
        y_vec = y_raw.astype(float)
        
    elif task_type == "binary":
        # For binary classification, handle both string and numeric labels
        if y_raw.dtype.kind in "OUS":  # If string type
            y_vec = LabelEncoder().fit_transform(y_raw)
        else:
            uniq = np.unique(y_raw)
            # Convert to 0/1 if already binary, otherwise encode
            y_vec = y_raw.astype(int) if set(uniq) <= {0,1} else LabelEncoder().fit_transform(y_raw)
        # Verify binary encoding
        if not set(np.unique(y_vec)) <= {0,1}:
            raise ValueError(f"{target_name} is not binary.")
        
    else:  # multi-class classification
        # Encode string labels to integers
        y_vec = LabelEncoder().fit_transform(y_raw)
        # Skip if not enough classes for multi-class
        if len(np.unique(y_vec)) < 3:
            raise ValueError(f"{target_name}: only {len(np.unique(y_vec))} classes; insufficient for multi-class.")
    
    return y_vec


def setup_training_environment(args, model_type="graph"):
    """Common setup for both graph and tabular training scripts.
    
    Args:
        args: Parsed command line arguments
        model_type: Either "graph" or "tabular" for model-specific setup
        
    Returns:
        Dictionary containing all setup information
    """
    import logging
    
    # Load configuration
    config = load_config()
    
    # Set up paths and parameters
    chemprop_dir = Path.cwd()
    
    # Get paths from config with defaults
    paths = config.get('PATHS', {})
    data_dir = chemprop_dir / paths.get('data_dir', 'data')
    results_dir = chemprop_dir / paths.get('results_dir', 'results')
    
    if model_type == "graph":
        checkpoint_dir = chemprop_dir / paths.get('checkpoint_dir', 'checkpoints') / args.model_name
        feat_select_dir = None  # Not used for graph models
    else:  # tabular
        checkpoint_dir = None
        feat_select_dir = chemprop_dir / paths.get('feat_select_dir', 'out') / "tabular" / args.dataset_name
    
    # Create necessary directories
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if feat_select_dir:  # Only create for tabular models
        feat_select_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file paths
    input_path = chemprop_dir / data_dir / f"{args.dataset_name}.csv"
    
    # Get model-specific configuration
    if model_type == "graph":
        model_config = config['MODELS'].get(args.model_name, {})
        smiles_column = model_config.get('smiles_column', 'smiles')
        ignore_columns = model_config.get('ignore_columns', [])
    else:  # tabular
        smiles_column = 'smiles'
        ignore_columns = ['WDMPNN_Input']
    
    # Set parameters from config with defaults
    GLOBAL_CONFIG = config.get('GLOBAL', {})
    SEED = GLOBAL_CONFIG.get('SEED', 42)
    REPLICATES = GLOBAL_CONFIG.get('REPLICATES', 5)
    
    if model_type == "graph":
        EPOCHS = GLOBAL_CONFIG.get('EPOCHS', 300)
        PATIENCE = GLOBAL_CONFIG.get('PATIENCE', 30)
        num_workers = min(GLOBAL_CONFIG.get('NUM_WORKERS', 8), os.cpu_count() or 1)
    else:
        EPOCHS = None
        PATIENCE = None
        num_workers = None
    
    # Get dataset descriptors from config
    DATASET_DESCRIPTORS = config.get('DATASET_DESCRIPTORS', {}).get(args.dataset_name, [])
    descriptor_columns =  []
    
    return {
        'config': config,
        'chemprop_dir': chemprop_dir,
        'data_dir': data_dir,
        'results_dir': results_dir,
        'checkpoint_dir': checkpoint_dir,
        'feat_select_dir': feat_select_dir,
        'input_path': input_path,
        'smiles_column': smiles_column,
        'ignore_columns': ignore_columns,
        'descriptor_columns': descriptor_columns,
        'SEED': SEED,
        'REPLICATES': REPLICATES,
        'EPOCHS': EPOCHS,
        'PATIENCE': PATIENCE,
        'num_workers': num_workers,
        'DATASET_DESCRIPTORS': DATASET_DESCRIPTORS
    }


def load_and_preprocess_data(args, setup_info):
    """Load and preprocess data with common filtering logic.

    - Applies dataset-specific ignores from config
    - Handles homopolymer (single smiles) and copolymer (A/B + fractions)
    - For copolymers: infers/normalizes fracB and creates a 'group_key'
    - Auto-detects target columns after exclusions
    """
    import logging
    import numpy as np
    import pandas as pd

    logger = logging.getLogger(__name__)

    # ---------------------- Load raw CSV ----------------------
    df_input = pd.read_csv(setup_info['input_path'])

    # ---------------------- Apply dataset_ignore ----------------------
    cfg = setup_info.get('config', {})
    ds_ignore = cfg.get('DATASET_IGNORE', {}).get(args.dataset_name, []) or []
    logger.info(f"Dataset ignore config for '{args.dataset_name}': {ds_ignore}")
    if ds_ignore:
        drop_cols = [c for c in ds_ignore if c in df_input.columns]
        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} dataset_ignore columns: {drop_cols}")
            df_input = df_input.drop(columns=drop_cols, errors="ignore")
        else:
            logger.info(f"No columns to drop from dataset_ignore list: {ds_ignore}")
    else:
        logger.info(f"No dataset_ignore configuration found for '{args.dataset_name}'")

    # ---------------------- Model-specific wDMPNN filters (unchanged) ----------------------
    if args.dataset_name == "insulator" and hasattr(args, 'model_name') and args.model_name == "wDMPNN":
        df_input = filter_insulator_data(args, df_input, setup_info['smiles_column'])

    if not (hasattr(args, 'model_name') and args.model_name == "wDMPNN"):
        drop_idx, excluded_smis = load_drop_indices(setup_info['chemprop_dir'], args.dataset_name)
        if drop_idx:
            logger.info(f"Dropping {len(drop_idx)} rows from {args.dataset_name} due to exclusions.")
            df_input = df_input.drop(index=drop_idx, errors="ignore").reset_index(drop=True)

    # ---------------------- Copolymer handling ----------------------
    if args.polymer_type == "copolymer":
        # Accept both naming conventions
        sA_col = "smilesA" if "smilesA" in df_input.columns else ("smiles_A" if "smiles_A" in df_input.columns else None)
        sB_col = "smilesB" if "smilesB" in df_input.columns else ("smiles_B" if "smiles_B" in df_input.columns else None)
        if sA_col is None or sB_col is None:
            raise KeyError("Copolymer mode expects 'smilesA'/'smilesB' or 'smiles_A'/'smiles_B' columns.")

        # Fractions: accept A/B or A_B; infer fracB if missing
        if "fracA" in df_input.columns or "fracB" in df_input.columns:
            df_input["fracA"] = pd.to_numeric(df_input.get("fracA"), errors="coerce")
            if "fracB" in df_input.columns:
                df_input["fracB"] = pd.to_numeric(df_input.get("fracB"), errors="coerce")
            else:
                df_input["fracB"] = 1.0 - df_input["fracA"]
        elif "frac_A" in df_input.columns or "frac_B" in df_input.columns:
            df_input["fracA"] = pd.to_numeric(df_input.get("frac_A"), errors="coerce")
            if "frac_B" in df_input.columns:
                df_input["fracB"] = pd.to_numeric(df_input.get("frac_B"), errors="coerce")
            else:
                df_input["fracB"] = 1.0 - df_input["fracA"]
        else:
            raise KeyError("Copolymer mode expects 'fracA'/'fracB' or 'frac_A'/'frac_B' columns.")

        # Validate / normalize fractions (robust to tiny numeric noise)
        if df_input[["fracA", "fracB"]].isna().any().any():
            raise ValueError("Found NaNs in fracA/fracB after coercion.")

        ssum = (df_input["fracA"].astype(float) + df_input["fracB"].astype(float)).values
        if not np.isfinite(ssum).all() or np.any(ssum <= 0):
            raise ValueError("Invalid fractions: non-finite or non-positive totals in fracA+fracB.")

        # Normalize so fracA+fracB == 1.0 exactly
        df_input["fracA"] = (df_input["fracA"].astype(float) / ssum)
        df_input["fracB"] = 1.0 - df_input["fracA"]

        # Canonicalize pair & build group_key (A+B == B+A) to avoid leakage across splits
        def _canon_pair(a, b, wa, wb):
            a = "" if pd.isna(a) else str(a)
            b = "" if pd.isna(b) else str(b)
            if b < a:
                return b, a, wb, wa
            return a, b, wa, wb

        def _round6(x):  # stable key vs tiny fp jitter
            try:
                return round(float(x), 6)
            except Exception:
                return x

        canA, canB, fA_list, fB_list = [], [], [], []
        for a, b, wa, wb in zip(df_input[sA_col].astype(str), df_input[sB_col].astype(str),
                                df_input["fracA"].values, df_input["fracB"].values):
            a2, b2, wa2, wb2 = _canon_pair(a, b, wa, wb)
            canA.append(a2); canB.append(b2); fA_list.append(wa2); fB_list.append(wb2)

        df_input["smilesA"] = canA  # normalized names going forward
        df_input["smilesB"] = canB
        df_input["fracA"] = np.asarray(fA_list, float)
        df_input["fracB"] = np.asarray(fB_list, float)

        df_input["group_key"] = [
            f"{a}|||{b}|||{_round6(wa)}|||{_round6(wb)}"
            for a, b, wa, wb in zip(df_input["smilesA"], df_input["smilesB"], df_input["fracA"], df_input["fracB"])
        ]

        # ---------------------- Target column detection (copolymer) ----------------------
        exclude_cols = {
            # smiles/fractions, both raw variants and normalized
            "smilesA", "smilesB", "fracA", "fracB",
            sA_col, sB_col, "frac_A", "frac_B",
            # descriptors & ignores
            *setup_info.get('DATASET_DESCRIPTORS', []),
            *setup_info.get('ignore_columns', []),
            # internal helper col
            "group_key"
        }
        target_columns = [c for c in df_input.columns if c not in exclude_cols]

    else:
        # ---------------------- Homopolymer target detection ----------------------
        smiles_col = setup_info.get('smiles_column', 'smiles')
        exclude_cols = set([smiles_col]) if smiles_col else set()
        exclude_cols.update(setup_info.get('DATASET_DESCRIPTORS', []))
        exclude_cols.update(setup_info.get('ignore_columns', []))

        if smiles_col and smiles_col not in df_input.columns:
            raise KeyError(f"Homo mode expects SMILES column '{smiles_col}' present in CSV.")

        target_columns = [c for c in df_input.columns if c not in exclude_cols]

    if not target_columns:
        msg = "No target columns found after exclusions."
        if args.polymer_type == "homo":
            msg += f" Expected at least one column other than '{setup_info.get('smiles_column','smiles')}'."
        else:
            msg += " Ensure your CSV includes targets beyond smilesA/B and fracA/B."
        raise ValueError(msg)

    return df_input, target_columns



def determine_split_strategy(dataset_size, replicates):
    """Determine whether to use CV or holdout based on dataset size.
    
    Args:
        dataset_size: Number of samples in dataset
        replicates: Number of replicates from config
        
    Returns:
        Tuple of (n_splits, local_reps)
    """
    # For small datasets (<2000 samples), use 5-fold CV with 1 replicate
    # For larger datasets, use a single train/val/test split with multiple replicates
    n_splits = 5 if dataset_size < 2000 else 1
    local_reps = 1 if n_splits > 1 else replicates
    return n_splits, local_reps


def generate_data_splits(args, ys, n_splits, local_reps, seed):
    """Generate train/val/test splits with common logic.
    
    Args:
        args: Command line arguments
        ys: Target values
        n_splits: Number of splits (1 for holdout, >1 for CV)
        local_reps: Number of replicates
        seed: Random seed
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if args.task_type in ['binary', 'multi']:
        train_indices, val_indices, test_indices = make_repeated_splits(
            task_type=args.task_type,
            replicates=local_reps,
            seed=seed,
            y_class=ys,
            n_splits=n_splits
        )
    else:
        train_indices, val_indices, test_indices = make_repeated_splits(
            task_type=args.task_type,
            replicates=local_reps,
            seed=seed,
            y_reg=ys,
            n_splits=n_splits
        )
    
    return train_indices, val_indices, test_indices

def unordered_pair(a: str, b: str):
    """Stable, order-invariant key for a monomer pair."""
    return tuple(sorted((str(a), str(b))))

def make_groups_for_copolymer(df: pd.DataFrame):
    """Return array of group IDs (one per row). Prefers a provided 'group_id' column."""
    gids = compute_group_id(df)
    return gids.astype(str).values

def group_splits(df, y, task_type, n_splits, seed, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    """
    Produce train/val/test index lists with group integrity preserved.
    - For CV (n_splits > 1): GroupKFold (or StratifiedGroupKFold for classification if available).
    - For holdout (n_splits == 1): two-stage GroupShuffleSplit to get 80/10/10 by groups.
    Returns: train_indices, val_indices, test_indices as lists (len = n_splits or 1).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8
    
    groups = make_groups_for_copolymer(df)
    n = len(df)

    # ---------------------- Cross-validation path ----------------------
    if n_splits > 1:
        if task_type != "reg" and HAVE_SGKF:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            cv_iter = splitter.split(np.zeros(n), y, groups)
        else:
            splitter = GroupKFold(n_splits=n_splits)  # deterministic, no shuffle
            cv_iter = splitter.split(np.zeros(n), groups=groups)

        tr_list, va_list, te_list = [], [], []
        # proportion actually available for training folds in CV (rest is test)
        train_total = 1.0 - 1.0 / n_splits
        # inner val fraction relative to TRAIN so that global val ≈ val_frac
        inner_val = min(max(val_frac / train_total, 0.0), 1.0)

        for k, (tr_idx, te_idx) in enumerate(cv_iter):
            # carve validation from training by groups
            inner = GroupShuffleSplit(n_splits=1, train_size=1.0 - inner_val,
                                      random_state=seed + k)
            tr2_rel, va_rel = next(inner.split(tr_idx, groups=groups[tr_idx]))
            tr2_idx = tr_idx[tr2_rel]
            va_idx = tr_idx[va_rel]

            tr_list.append(tr2_idx)
            va_list.append(va_idx)
            te_list.append(te_idx)

        return tr_list, va_list, te_list

    # ---------------------- Holdout path (80/10/10 by default) ----------------------
    outer = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_idx, temp_idx = next(outer.split(np.zeros(n), groups=groups))

    temp_groups = groups[temp_idx]
    # fraction of TEMP that should become VAL (so VAL:TEST ≈ val_frac:test_frac)
    inner_train_size = val_frac / (val_frac + test_frac)
    inner = GroupShuffleSplit(n_splits=1, train_size=inner_train_size, random_state=seed + 1)
    va_rel, te_rel = next(inner.split(temp_idx, groups=temp_groups))

    va_idx = temp_idx[va_rel]
    te_idx = temp_idx[te_rel]

    return [tr_idx], [va_idx], [te_idx]

def save_aggregate_results(results_list, results_dir, model_name, dataset_name, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, logger):
    """Save results using target-specific filenames to prevent overwriting.
    
    Args:
        results_list: List of result DataFrames
        results_dir: Results directory path
        model_name: Model name for directory structure
        dataset_name: Dataset name for filename
        desc_suffix: Descriptor suffix for filename
        rdkit_suffix: RDKit suffix for filename
        batch_norm_suffix: Batch normalization suffix for filename
        size_suffix: Train size suffix for filename
        logger: Logger instance
    """
    if model_name.lower() == "tabular":
        model_results_dir = results_dir / "tabular"
        base_filename = f"{dataset_name}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}"
    else:
        model_results_dir = results_dir / model_name
        base_filename = f"{dataset_name}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}_results"
    
    model_results_dir.mkdir(exist_ok=True)
    
    # Save results to file(s)
    if results_list:
        current_aggregate_df = pd.concat(results_list, ignore_index=True)
        
        # Organize columns: target (if exists), split, then metrics
        base_cols = []
        if 'target' in current_aggregate_df.columns:
            base_cols.append("target")
        base_cols.append("split")
        
        if "model" in current_aggregate_df.columns:
            base_cols.append("model")
        
        metric_cols = sorted([c for c in current_aggregate_df.columns if c not in base_cols])
        current_aggregate_df = current_aggregate_df[base_cols + metric_cols]
        
        if 'target' in current_aggregate_df.columns:
            # Save each target to a separate file
            for target in current_aggregate_df['target'].unique():
                target_df = current_aggregate_df[current_aggregate_df['target'] == target]
                target_csv = model_results_dir / f"{base_filename}_{target}.csv"
                target_df.to_csv(target_csv, index=False)
                logger.info(f"Saved target results -> {target_csv}")
        else:
            # Save all results to a single file when no target is specified
            output_csv = model_results_dir / f"{base_filename}.csv"
            current_aggregate_df.to_csv(output_csv, index=False)
            logger.info(f"Saved results -> {output_csv}")
        
        # Skip combined file since targets run in parallel


def load_best_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
    
    # Prioritize validation-best checkpoints over last.ckpt
    best_ckpts = [f for f in ckpts if f.startswith("best-")]
    if best_ckpts:
        # Parse validation loss from checkpoint names and select the one with lowest loss
        import re
        best_checkpoint = None
        lowest_val_loss = float('inf')
        
        for ckpt_name in best_ckpts:
            # Extract validation loss from filename: best-[epoch]-[val_loss].ckpt
            # Handle various possible formats like best-epoch=10-val_loss=0.123.ckpt or best-10-0.123.ckpt
            val_loss_match = re.search(r'(?:val_loss=|-)([0-9]+\.?[0-9]*)(?:\.ckpt|$)', ckpt_name)
            if val_loss_match:
                try:
                    val_loss = float(val_loss_match.group(1))
                    if val_loss < lowest_val_loss:
                        lowest_val_loss = val_loss
                        best_checkpoint = ckpt_name
                except ValueError:
                    continue
        
        if best_checkpoint:
            return ckpt_dir / best_checkpoint
        else:
            # Fallback to alphabetical sorting if parsing fails
            best_ckpts.sort()
            return ckpt_dir / best_ckpts[-1]
    
    # Fallback to lexicographically last checkpoint
    ckpts.sort()
    return ckpt_dir / ckpts[-1]


# def get_encodings_from_loader(model, loader):
#     import torch
#     encs = []
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         for batch in loader:
#             # 1) Get the graph object
#             bmg = getattr(batch, "bmg", None)
#             if bmg is None:
#                 raise ValueError(f"Batch has no 'bmg': {batch}")

#             # 2) Move it to the right device
#             if hasattr(bmg, "to") and callable(getattr(bmg, "to")):
#                 # Many Chemprop BatchMolGraph versions implement .to(device)
#                 bmg = bmg.to(device)
#             else:
#                 # Fallback: move known tensor fields manually
#                 for name in ("V", "E", "edge_index", "batch"):
#                     t = getattr(bmg, name, None)
#                     if isinstance(t, torch.Tensor):
#                         setattr(bmg, name, t.to(device, non_blocking=True))

#             # 3) Move optional descriptor tensors
#             V_d = getattr(batch, "V_d", None)
#             if isinstance(V_d, torch.Tensor):
#                 V_d = V_d.to(device, non_blocking=True)

#             X_d = getattr(batch, "X_d", None)
#             if isinstance(X_d, torch.Tensor):
#                 X_d = X_d.to(device, non_blocking=True)
            
#             enc = model.fingerprint(bmg,V_d,X_d)
#             encs.append(enc)
#     return torch.cat(encs, dim=0).cpu().numpy()
def get_encodings_from_loader(model, loader):
    import torch  # Local import for heavy dependency
    encs = []
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(loader):
            # 1) Graph
            bmg = getattr(batch, "bmg", None)
            if bmg is None:
                raise ValueError(f"[batch {j}] has no 'bmg'")

            # 2) Move to device without losing the object if .to() is in-place
            if hasattr(bmg, "to") and callable(bmg.to):
                ret = bmg.to(device)          # may return None (in-place)
                if ret is not None:
                    bmg = ret
            # Ensure key tensors are on device
            bmg.V = bmg.V.to(device, non_blocking=True)
            bmg.edge_index = bmg.edge_index.to(device, non_blocking=True)
            if getattr(bmg, "batch", None) is not None:
                bmg.batch = bmg.batch.to(device, non_blocking=True)
            # Provide empty edge features if missing (concat expects E)
            if getattr(bmg, "E", None) is None:
                n_edges = bmg.edge_index.size(1)
                bmg.E = bmg.V.new_empty((n_edges, 0))
            else:
                bmg.E = bmg.E.to(device, non_blocking=True)

            # 3) Optional extras
            V_d = getattr(batch, "V_d", None)
            if isinstance(V_d, torch.Tensor):
                V_d = V_d.to(device, non_blocking=True)
            else:
                V_d = None

            X_d = getattr(batch, "X_d", None)
            if isinstance(X_d, torch.Tensor):
                X_d = X_d.to(device, non_blocking=True)
            else:
                X_d = None

            # 4) Encode
            enc = model.fingerprint(bmg, V_d, X_d)
            encs.append(enc)

    return torch.cat(encs, dim=0).cpu().numpy()

    

def load_drop_indices(root_dir, dataset_name: str): 
    """Reads <dataset>_skipped_indices.txt and <dataset>_excluded_problematic_smiles.txt.
    Returns (sorted_indices, excluded_smiles_list)."""
    skip_path = Path(root_dir / f"{dataset_name}_skipped_indices.txt")
    prob_path = Path(root_dir / f"{dataset_name}_excluded_problematic_smiles.txt")

    idxs = set()
    excluded_smis = []

    # one index per line
    if skip_path.exists():
        with skip_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        idxs.add(int(line))
                    except ValueError:
                        pass  # ignore malformed lines

    # "idx,smiles" per line
    if prob_path.exists():
        with prob_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split only on first comma (SMILES can sometimes contain commas if quoted, but your file format uses a single comma)
                parts = line.split(",", 1)
                if len(parts) == 2:
                    idx_str, smi = parts
                    excluded_smis.append(smi)
                    try:
                        idxs.add(int(idx_str))
                    except ValueError:
                        pass

    return sorted(idxs), excluded_smis

def upsert_csv(out_csv: Path, new_df: pd.DataFrame, key_cols: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        old = pd.read_csv(out_csv)

        # Union of columns, keeping key cols first
        all_cols = key_cols + [c for c in (set(old.columns) | set(new_df.columns)) if c not in key_cols]
        old = old.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)

        # Index on keys
        old_i = old.set_index(key_cols)
        new_i = new_df.set_index(key_cols)

        # Start with old, add any new rows/cols
        combined = old_i.combine_first(new_i)

        # Overwrite with new where provided
        combined.update(new_i)

        combined.reset_index().to_csv(out_csv, index=False)
    else:
        new_df.to_csv(out_csv, index=False)



def filter_insulator_data(args, df_input, smiles_column):
    """Filter insulator dataset for wDMPNN model by removing invalid SMILES."""
    # Copy of original data for index reference
    df_orig = df_input.copy()

    # Identify rows without '*' in the SMILES string
    missing_star_mask = ~df_orig[smiles_column].astype(str).str.contains(r"\*", regex=True)
    missing_star_indices = df_orig[missing_star_mask].index.tolist()

    # Identify rows with specific problematic SMILES
    excluded_smis = {
        "[*:1]CC(C)/C=C\\[*:2]C|1.0|<1-2:0.5:0.5",
        "[*:1]CC/C=C\\[*:2]Cl|1.0|<1-2:0.5:0.5",
        "[*:1]C1CCC(C=[*:2])C1|1.0|<1-2:0.5:0.5"
    }
    problem_smiles_mask = df_orig[smiles_column].isin(excluded_smis)
    problem_smiles_indices = df_orig[problem_smiles_mask].index.tolist()

    # Save missing '*' indices
    with open(f"{args.dataset_name}_skipped_indices.txt", "w") as f:
        for idx in missing_star_indices:
            f.write(f"{idx}\n")

    # Save excluded problematic SMILES (index + SMILES string)
    with open(f"{args.dataset_name}_excluded_problematic_smiles.txt", "w") as f:
        for idx in problem_smiles_indices:
            smi = df_orig.loc[idx, smiles_column]
            f.write(f"{idx},{smi}\n")

    # Apply filtering on original DataFrame
    final_mask = ~(missing_star_mask | problem_smiles_mask)
    filtered_df = df_orig[final_mask].reset_index(drop=True)
    
    print(f"Filtered out {len(df_orig) - len(filtered_df)} invalid SMILES from insulator dataset")
    return filtered_df


# Import organization note:
# Main imports are now at the top of the file
# This block should be removed as these imports are already handled above

def select_features_remove_constant_and_correlated(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    *,
    corr_threshold: float = 0.95,
    method: str = "pearson",   # "pearson" or "spearman"
    min_unique: int = 2,       # drop columns with < min_unique non-NA distinct values
    positive_class: Optional[Union[str, int, float]] = None,  # only used if y is binary and you care
    verbose: bool = True
) -> Dict[str, Union[List[str], Callable[[pd.DataFrame], pd.DataFrame]]]:
    # Get logger from calling module context
    import inspect
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        caller_module = inspect.getmodule(caller_frame)
        if caller_module and hasattr(caller_module, 'logger'):
            logger = caller_module.logger
        else:
            # Fallback to a generic logger
            logger = logging.getLogger(__name__)
    finally:
        del frame
    X = X_train.copy()

    # 1) Drop constant / low-unique
    non_na_uniques = X.nunique(dropna=True)
    drop_constant = non_na_uniques[non_na_uniques < min_unique].index.tolist()

    # Columns for correlation step (numeric, after removing constants)
    num_cols = X.drop(columns=drop_constant, errors="ignore") \
                .select_dtypes(include=[np.number]) \
                .columns.tolist()

    # Helper: association with y for tie-breaking
    assoc: pd.Series
    if y_train is not None and len(num_cols) > 0:
        y = y_train
        if np.issubdtype(y.dtype, np.number):
            # numeric target (regression) or already-encoded classification
            if method == "spearman":
                assoc = X[num_cols].rank().corrwith(y.rank(), method="spearman").abs()
            else:
                assoc = X[num_cols].corrwith(y, method="pearson").abs()
        else:
            # categorical y (you said you pre-encode, so this branch won't run in your setup)
            if y.nunique(dropna=True) == 2:
                if positive_class is not None:
                    y01 = (y == positive_class).astype(int)
                else:
                    # stable encoding via factorize (0/1), no dependence on "first unique"
                    codes, _ = pd.factorize(y, sort=True)
                    y01 = pd.Series(codes, index=y.index).astype(int)
                assoc = X[num_cols].corrwith(y01, method="pearson").abs()
            else:
                # multi-class: spearman on codes
                codes, _ = pd.factorize(y, sort=True)
                yc = pd.Series(codes, index=y.index).astype(float)
                assoc = X[num_cols].rank().corrwith(yc.rank(), method="spearman").abs()

        assoc = assoc.fillna(0.0)
    else:
        assoc = pd.Series(0.0, index=num_cols, dtype=float)

    # 2) Drop highly correlated features (absolute correlation)
    drop_corr: List[str] = []
    if len(num_cols) > 1:
        corr = X[num_cols].corr(method=method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        pairs = (
            upper.stack()
                 .reset_index()
                 .rename(columns={"level_0":"f1","level_1":"f2",0:"abs_corr"})
                 .sort_values("abs_corr", ascending=False)
        )

        kept = set(num_cols)
        for f1, f2, r in pairs[["f1","f2","abs_corr"]].itertuples(index=False):
            if r < corr_threshold:
                break
            if f1 not in kept or f2 not in kept:
                continue

            a1, a2 = float(assoc.get(f1, 0.0)), float(assoc.get(f2, 0.0))
            if a1 > a2:
                drop = f2
            elif a2 > a1:
                drop = f1
            else:
                # tie-breaker: more missingness (won't matter if you truly have none), then name
                na1 = int(X[f1].isna().sum())
                na2 = int(X[f2].isna().sum())
                if na1 != na2:
                    drop = f1 if na1 > na2 else f2
                else:
                    drop = f1 if f1 < f2 else f2

            drop_corr.append(drop)
            kept.discard(drop)

    dropped = sorted(set(drop_constant) | set(drop_corr))
    kept_cols = [c for c in X.columns if c not in dropped]

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=dropped, errors="ignore")

    if verbose:
        if drop_constant:
            logger.info(f"[FS] dropped {len(drop_constant)} constant/low-unique features")
        if drop_corr:
            logger.info(f"[FS] dropped {len(drop_corr)} highly correlated features (≥{corr_threshold})")
        logger.info(f"[FS] kept {len(kept)} features")

    return {
        "dropped_constant": drop_constant,
        "dropped_correlated": drop_corr,
        "dropped_all": dropped,
        "kept": kept_cols,
        "transform": transform,
    }


def save_predictions(y_true, y_pred, predictions_dir, dataset_name, target, model_name, 
                    desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, split_idx, logger,
                    test_ids=None):
    """Save predictions for learning curve analysis.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values  
        predictions_dir: Base predictions directory
        dataset_name: Dataset name
        target: Target name
        model_name: Model name (e.g., 'DMPNN')
        desc_suffix: Descriptor suffix
        rdkit_suffix: RDKit suffix
        batch_norm_suffix: Batch norm suffix
        size_suffix: Train size suffix
        split_idx: Split index
        logger: Logger instance
        test_ids: Optional list of IDs/identifiers for order verification
    """
    import numpy as np
    from pathlib import Path
    
    # Create predictions directory structure
    model_pred_dir = predictions_dir / model_name
    model_pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename with all relevant identifiers
    filename = f"{dataset_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}__split{split_idx}.npz"
    pred_file = model_pred_dir / filename
    
    # Prepare data to save
    save_data = {
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred),
        'metadata': {
            'dataset': dataset_name,
            'target': target,
            'model': model_name,
            'split': split_idx,
            'desc_suffix': desc_suffix,
            'rdkit_suffix': rdkit_suffix,
            'batch_norm_suffix': batch_norm_suffix,
            'size_suffix': size_suffix
        }
    }
    
    # Add IDs if provided
    if test_ids is not None:
        save_data['test_ids'] = np.array(test_ids, dtype=object)
        logger.info(f"Saving predictions with {len(test_ids)} IDs for order verification")
    
    # Save predictions as compressed numpy arrays
    np.savez_compressed(pred_file, **save_data)
    
    logger.info(f"Saved predictions -> {pred_file}")

