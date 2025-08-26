"""Utility functions for model training and data processing.

This module provides various utility functions for:
- Configuration management
- Data preprocessing and loading
- Model building and training
- Feature selection and processing
- File I/O operations
"""

# Standard library imports
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

# Third-party imports (lightweight)
import numpy as np
import pandas as pd


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
            'DATASET_DESCRIPTORS': Dict of dataset descriptors
        }
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    import yaml  # Local import for optional dependency
    
    if config_path is None:
        config_path = Path(__file__).parent / 'train_config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Convert to the expected format with consistent key casing
    result = {
        'GLOBAL': {k.upper(): v for k, v in config.get('global', {}).items()},
        'MODELS': {k.upper(): v for k, v in config.get('models', {}).items()},
        'DATASET_DESCRIPTORS': config.get('dataset_descriptors', {})
    }
    
    # Handle dynamic values
    if 'num_workers' in result['GLOBAL']:
        num_workers = result['GLOBAL'].pop('num_workers')
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
    # Validate input
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(f"df_input must be a pandas DataFrame, got {type(df_input).__name__}")
    
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
        try:
            from chemprop import featurizers
            from chemprop.utils import make_mol
            
            # Use original SMILES for wDMPNN model if available
            use_smiles = df_input["smiles"].values if hasattr(args, 'model_name') and args.model_name == "wDMPNN" else smis
            
            # Convert SMILES to molecules
            mols = []
            for smi in use_smiles:
                try:
                    mol = make_mol(smi, keep_h=False, add_h=False, ignore_stereo=False)
                    mols.append(mol)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to process SMILES {smi}: {str(e)}")
                    mols.append(None)
            
            # Compute RDKit descriptors
            feat = featurizers.RDKit2DFeaturizer()
            rdkit_data = []
            for mol in mols:
                try:
                    rdkit_data.append(feat(mol) if mol is not None else np.full(feat.dim(), np.nan))
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to compute RDKit features: {str(e)}")
                    rdkit_data.append(np.full(feat.dim(), np.nan))
            
            rdkit_data = np.array(rdkit_data, dtype=np.float32)
            
        except ImportError as e:
            if verbose:
                print(f"Warning: RDKit features requested but could not import required modules: {e}")
    
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
    datapoint_class = (
        data.MoleculeDatapoint 
        if model_name.upper() == "DMPNN" 
        else data.PolymerDatapoint
    )
    
    if combined_descriptor_data is not None:
        return [
            datapoint_class.from_smi(smi, y, x_d=desc) 
            for smi, y, desc in zip(smis, ys, combined_descriptor_data)
            if smi and pd.notna(y).all()  # Skip invalid SMILES or NaN targets
        ]
    else:
        return [
            datapoint_class.from_smi(smi, y)
            for smi, y in zip(smis, ys)
            if smi and pd.notna(y).all()  # Skip invalid SMILES or NaN targets
        ]


        


def get_metric_list(
    task_type: str, 
    target: Optional[str] = None, 
    n_classes: Optional[int] = None, 
    df_input: Optional[pd.DataFrame] = None
) -> List[Any]:
    """Get a list of metrics appropriate for the given task type.
    
    Args:
        task_type: Type of task - 'reg', 'binary', or 'multi'
        target: Name of the target column (required for binary tasks with df_input)
        n_classes: Number of classes (required for multi-class tasks)
        df_input: DataFrame containing target values (required for binary tasks)
        
    Returns:
        List of metric objects for the specified task type
        
    Raises:
        ValueError: If task_type is invalid or required arguments are missing
        ImportError: If chemprop is not available
    """
    try:
        from chemprop import nn  # Local import for heavy dependency
    except ImportError as e:
        raise ImportError(
            "chemprop package is required for metrics. "
            "Install with: pip install chemprop"
        ) from e
    
    if task_type == 'reg':
        return [
            nn.metrics.MAE(), 
            nn.metrics.RMSE(), 
            nn.metrics.R2Score()
        ]
    
    elif task_type == 'binary':
        if df_input is not None and target is not None:
            unique_classes = df_input[target].dropna().unique()
            has_both_classes = len(unique_classes) > 1
        else:
            has_both_classes = False
            
        metrics = [
            nn.metrics.BinaryAccuracy(), 
            nn.metrics.BinaryF1Score()
        ]
        
        # Add AUROC only if both classes are present
        if has_both_classes:
            metrics.append(nn.metrics.BinaryAUROC())
            
        return metrics
    
    elif task_type == 'multi':
        if n_classes is None:
            raise ValueError("n_classes must be provided for multi-class tasks")
            
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
            
        return [
            nn.metrics.MulticlassAccuracy(
                num_classes=n_classes, 
                average='macro'
            ),
            nn.metrics.MulticlassF1Score(
                num_classes=n_classes, 
                average='macro'
            ),
            nn.metrics.MulticlassAUROC(
                num_classes=n_classes, 
                average='macro'
            )
        ]
    
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            "Must be one of: 'reg', 'binary', 'multi'"
        )

def build_model_and_trainer(
    args: Any,
    combined_descriptor_data: Optional[np.ndarray],
    n_classes: Optional[int],
    scaler: Optional[Any],
    X_d_transform: Optional[Any],
    checkpoint_path: Union[str, Path],
    batch_norm: bool = True,
    metric_list: Optional[List[Any]] = None,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    max_epochs: int = 100,
    gradient_clip_val: float = 10.0,
    **trainer_kwargs
) -> Tuple[Any, Any]:  # Returns (model, trainer)
    """Build and configure a chemprop model and PyTorch Lightning trainer.
    
    Args:
        args: Command line arguments or config object with model parameters
        combined_descriptor_data: Combined RDKit and custom descriptors
        n_classes: Number of classes (for classification tasks)
        scaler: Scaler object for regression tasks
        X_d_transform: Transform for descriptor data
        checkpoint_path: Directory to save model checkpoints
        batch_norm: Whether to use batch normalization
        metric_list: List of metrics to track during training
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        max_epochs: Maximum number of training epochs
        gradient_clip_val: Maximum gradient norm for gradient clipping
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
        from chemprop import nn, models
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
    
    # Select Message Passing Scheme
    if args.model_name == "wDMPNN":
        mp = nn.WeightedBondMessagePassing(d_v=72, d_e=86)
    elif args.model_name == "DMPNN":
        mp = nn.BondMessagePassing()
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    
    # Calculate input dimension for FFN
    descriptor_dim = combined_descriptor_data.shape[1] if combined_descriptor_data is not None else 0
    input_dim = mp.output_dim + descriptor_dim
    
    # Configure output transform for regression
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
    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")
    
    # Create aggregation and model
    agg = nn.MeanAggregation()
    mpnn = models.MPNN(
        message_passing=mp, 
        aggregation=agg, 
        ffn=ffn, 
        batch_norm=batch_norm, 
        metrics=metric_list or [],
        X_d_transform=X_d_transform
    )
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Configure model checkpointing
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
    
    # Configure learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
        log_momentum=True
    )
    
    # Configure callbacks
    callbacks = [checkpointing, early_stop, lr_monitor]
    
    # Configure logging
    logger = pl.loggers.CSVLogger(
        save_dir=str(checkpoint_path),
        name="logs",
        version=""  # Use empty string to avoid creating versioned subdirectories
    )
    
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
        **trainer_kwargs  # Allow overriding any trainer parameter
    )
    
    return mpnn, trainer

from typing import Any, List, Optional, Tuple, Union, Dict
import numpy as np

from typing import Any, List, Optional, Tuple, Union
import numpy as np

from typing import Any, List, Optional, Tuple, Union
import numpy as np

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


def load_best_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
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



def filter_insulator_data(df_input, smiles_column):
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
    logger = logging.getLogger("tabular-baselines")
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

