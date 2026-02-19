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
    pick_best_checkpoint,
    get_encodings_from_loader,
)


from chemprop import data, featurizers, models, nn

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_config_from_checkpoint_path(checkpoint_path: str) -> dict:
    """Extract training configuration from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Configuration with keys 'descriptors', 'rdkit', 'batch_norm', 'train_size'
    """
    # Extract the experiment name from the path
    # Format: /path/to/checkpoints/MODEL/dataset__target__[desc]__[rdkit]__[batch_norm]__[sizeN]__repN/...
    path_parts = Path(checkpoint_path).parts
    
    # Find the experiment directory (contains dataset name and suffixes)
    experiment_dir = None
    for part in path_parts:
        if '__rep' in part:
            experiment_dir = part
            break
    
    if not experiment_dir:
        logger.warning(f"Could not extract experiment configuration from checkpoint path: {checkpoint_path}")
        return {'descriptors': False, 'rdkit': False, 'batch_norm': False, 'train_size': 'full'}
    
    # Parse the experiment directory name
    config = {
        'descriptors': '__desc' in experiment_dir,
        'rdkit': '__rdkit' in experiment_dir,
        'batch_norm': '__batch_norm' in experiment_dir,
        'train_size': 'full'
    }
    
    # Extract train_size if present
    if '__size' in experiment_dir:
        import re
        size_match = re.search(r'__size(\d+)', experiment_dir)
        if size_match:
            config['train_size'] = size_match.group(1)
    
    return config


# === Modular Functions to Eliminate Code Duplication ===

def get_metric_columns(task_type: str, results_df: pd.DataFrame) -> list:
    """Get metric column names based on task type."""
    if task_type == "reg":
        metric_cols = ["test/mae", "test/r2", "test/rmse"]
    else:
        metric_cols = ["test/accuracy", "test/f1"]
        if "test/roc_auc" in results_df.columns:
            metric_cols.append("test/roc_auc")
    return metric_cols


def build_results_filename(args, results_dir: Path, descriptor_columns: list = None) -> Path:
    """Build results filename with appropriate suffixes."""
    # Create model results directory
    model_results_dir = results_dir / args.model_name
    model_results_dir.mkdir(exist_ok=True)
    
    # Build filename with suffixes
    filename_parts = [args.dataset_name]
    
    # Add descriptor suffixes (for DMPNN pipeline)
    if descriptor_columns:
        desc_suffix = "__desc" if descriptor_columns else ""
        if desc_suffix:
            filename_parts.append("desc")
    
    if hasattr(args, 'incl_rdkit') and args.incl_rdkit:
        filename_parts.append("rdkit")
    
    if hasattr(args, 'batch_norm') and args.batch_norm:
        filename_parts.append("batch_norm")
    
    # Add target suffix if specific target
    if args.target:
        filename_parts.append(args.target)
    
    # Join with double underscores and add suffix
    if len(filename_parts) == 1:
        filename = f"{filename_parts[0]}_baseline.csv"
    else:
        filename = "__".join(filename_parts) + "_baseline.csv"
    
    return model_results_dir / filename


def save_evaluation_results(results_df: pd.DataFrame, args, results_dir: Path, 
                          descriptor_columns: list = None, model_name: str = None) -> Path:
    """Save evaluation results with proper formatting and return the output path."""
    if results_df.empty:
        logger.warning("No results to save - empty DataFrame")
        return None
    
    # Build output filename
    out_csv = build_results_filename(args, results_dir, descriptor_columns)
    
    # Organize columns: target, split, then metrics, then model
    base_cols = ["target", "split"]
    metric_cols = get_metric_columns(args.task_type, results_df)
    extra_cols = [c for c in results_df.columns if c not in base_cols + metric_cols]
    results_df = results_df[base_cols + metric_cols + extra_cols]
    
    # Save to CSV
    results_df.to_csv(out_csv, index=False)
    
    return out_csv


def print_evaluation_summary(results_df: pd.DataFrame, model_name: str = None):
    """Print summary statistics for evaluation results."""
    model_label = f"{model_name} " if model_name else ""
    logger.info(f"\n=== {model_label}Evaluation Summary ===")
    
    for col in results_df.columns:
        if col.startswith('test/'):
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            logger.info(f"{col}: {mean_val:.4f} ± {std_val:.4f}")


def save_and_summarize_results(results_df: pd.DataFrame, args, results_dir: Path,
                             descriptor_columns: list = None, model_name: str = None) -> Path:
    """Complete results saving and summary printing."""
    if results_df.empty:
        logger.warning("No results to save")
        return None
    
    # Save results
    out_csv = save_evaluation_results(results_df, args, results_dir, descriptor_columns, model_name)
    
    if out_csv:
        logger.info(f"✅ {model_name or args.model_name} results saved to: {out_csv}")
        
        # Print summary statistics
        print_evaluation_summary(results_df, model_name)
    
    return out_csv

def has_embedding(embeddings_dir: Path, split, args, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix):
    embedding_prefix = f"{args.dataset_name}__{args.model_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}"
            
    have_perm = all((embeddings_dir / f"{embedding_prefix}__{k}_split_{split}.npy").exists()
                    for k in ["X_train", "X_val", "X_test", "feature_mask"])
    return have_perm

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
parser.add_argument('--model_name', type=str, default="DMPNN",
                    help='Name of the model to use')
parser.add_argument("--polymer_type", type=str, choices=["homo", "copolymer"], default="homo",
                    help='Type of polymer: "homo" for homopolymer or "copolymer" for copolymer')
parser.add_argument('--target', type=str, default=None,
                    help='Specific target to evaluate (if not provided, evaluates all targets)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to specific checkpoint file to evaluate (overrides automatic checkpoint discovery)')
parser.add_argument('--preprocessing_path', type=str, default=None,
                    help='Path to preprocessing directory (overrides automatic preprocessing path discovery)')
parser.add_argument('--train_size', type=str, default=None,
                    help='Training size used during model training (for path matching)')
parser.add_argument('--export_embeddings', action='store_true',
                    help='Load pre-exported embeddings if available (speeds up evaluation)')


args = parser.parse_args()

# Auto-detect task type for specific datasets
if args.dataset_name == 'polyinfo' and args.task_type == 'reg':
    args.task_type = 'multi'
    logger.info(f"Auto-detected task type for {args.dataset_name}: {args.task_type}")

logger.info("\n=== Evaluation Configuration ===")
logger.info(f"Dataset       : {args.dataset_name}")
logger.info(f"Task type     : {args.task_type}")
logger.info(f"Model         : {args.model_name}")
logger.info(f"Target        : {args.target if args.target else 'All targets'}")
logger.info(f"Descriptors   : {'Enabled' if args.incl_desc else 'Disabled'}")
logger.info(f"RDKit desc.   : {'Enabled' if args.incl_rdkit else 'Disabled'}")
logger.info(f"Batch norm    : {'Enabled' if args.batch_norm else 'Disabled'}")
if args.train_size is not None:
    logger.info(f"Train size    : {args.train_size}")
if args.export_embeddings:
    logger.info(f"Load embeddings: Enabled")
if args.save_predictions:
    logger.info(f"Save predictions: Enabled")
if args.pretrain_monomer:
    logger.info(f"Pretrain monomer: Enabled")
logger.info("===============================\n")

# Setup evaluation environment with model-specific configuration
from utils import setup_model_environment

# Determine model type for environment setup
if args.model_name in ["DMPNN", "wDMPNN", "DMPNN_DiffPool", "PPG"]:
    model_type = "dmpnn"
elif args.model_name == "AttentiveFP":
    model_type = "attentivefp"
elif args.model_name == "Graphormer":
    model_type = "graphormer"
else:
    # Default to dmpnn for unknown models
    model_type = "dmpnn"
    logger.warning(f"Unknown model type {args.model_name}, using DMPNN environment setup")

setup_info = setup_model_environment(args, model_type)

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
embeddings_dir = results_dir / "embeddings"

# Check model configuration
model_config = config['MODELS'].get(args.model_name, {})
if not model_config:
    logger.info(f"Warning: No configuration found for model '{args.model_name}'. Using defaults.")

# === Set Random Seed ===
set_seed(SEED)

# === Model-Specific Pipeline Routing ===
if args.model_name == "AttentiveFP":
    logger.info("🔄 AttentiveFP evaluation pipeline")
 
    logger.info("🎉 AttentiveFP evaluation complete!")
    sys.exit(0)  # Exit after AttentiveFP evaluation
    
elif args.model_name == "Graphormer":
    logger.error("❌ Graphormer evaluation not supported")
    logger.info("Graphormer requires:")
    logger.info("  1. DGL graph data format from train_graphormer.py")
    logger.info("  2. Custom collation functions")
    logger.info("  3. Accelerate-based model loading")
    logger.info("  4. Graph-level embedding extraction")
    logger.error("Please use DMPNN, wDMPNN, DMPNN_DiffPool, PPG, or AttentiveFP models")
    sys.exit(1)
    
else:
    logger.info("🔄 Using DMPNN/Lightning-based evaluation pipeline")

# === Continue with Standard Pipeline (works for DMPNN variants, partially for AttentiveFP) ===

# === Load and Preprocess Data ===
df_input, target_columns = load_and_preprocess_data(args, setup_info)

# Filter target columns if specific target is provided
if args.target:
    if args.target not in target_columns:
        logger.error(f"Target '{args.target}' not found in dataset. Available targets: {target_columns}")
        sys.exit(1)
    target_columns = [args.target]
    logger.info(f"Evaluating single target: {args.target}")

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

# Choose featurizer based on model type
small_molecule_models = ["DMPNN", "DMPNN_DiffPool", "PPG"]
featurizer = (
    featurizers.SimpleMoleculeMolGraphFeaturizer() 
    if args.model_name in small_molecule_models 
    else featurizers.PolymerMolGraphFeaturizer()
)

# Prepare results list for tabular format (same as train_tabular.py)
all_results = []

# Initialize result storage for all targets
rep_model_to_row = {}

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
        
        # Extract configuration from checkpoint path if provided
        if args.checkpoint_path:
            checkpoint_config = extract_config_from_checkpoint_path(args.checkpoint_path)
            logger.info(f"Extracted config from checkpoint path: {checkpoint_config}")
            
            # Create a temporary args object with the extracted configuration
            import copy
            temp_args = copy.deepcopy(args)
            temp_args.incl_rdkit = checkpoint_config['rdkit']
            temp_args.batch_norm = checkpoint_config['batch_norm']
            temp_args.train_size = checkpoint_config['train_size']
            
            # Update descriptor_columns based on extracted config
            if checkpoint_config['descriptors'] and not descriptor_columns:
                logger.warning("Checkpoint was trained with descriptors, but no descriptor columns provided!")
            elif not checkpoint_config['descriptors'] and descriptor_columns:
                logger.info("Checkpoint was trained without descriptors, ignoring provided descriptor columns")
                descriptor_columns = None
        else:
            temp_args = args

        # Load preprocessing metadata directly (same as train_graph.py)
        split_preprocessing_metadata = {}
        for i in range(REPLICATES):
            # Build experiment paths to get proper suffixes using extracted config
            checkpoint_path, auto_preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix = build_experiment_paths(
                temp_args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            
            # Use provided preprocessing path or automatic path
            if args.preprocessing_path:
                provided_path = Path(args.preprocessing_path)
                
                # Check if the provided path is for a specific replicate (ends with __repN)
                if '__rep' in provided_path.name:
                    # Extract the base path by removing the __repN suffix and build path for current replicate
                    import re
                    base_path_name = re.sub(r'__rep\d+$', '', provided_path.name)
                    base_directory = provided_path.parent
                    current_rep_name = f"{base_path_name}__rep{i}"
                    preprocessing_path = base_directory / current_rep_name
                    logger.info(f"Extracted base from provided path: {base_path_name}")
                    logger.info(f"Using preprocessing path for rep {i}: {preprocessing_path}")
                else:
                    # Provided path is a base directory, append the full experiment name
                    base_name = f"{args.dataset_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}__rep{i}"
                    preprocessing_path = provided_path / base_name
                    logger.info(f"Using provided preprocessing base directory: {args.preprocessing_path}")
                    logger.info(f"Full preprocessing path: {preprocessing_path}")
            else:
                preprocessing_path = auto_preprocessing_path
                logger.info(f"Using automatic preprocessing path: {preprocessing_path}")
            
            # Check if preprocessing files exist
            metadata_path = preprocessing_path / f"preprocessing_metadata_split_{i}.json"
            scaler_file = preprocessing_path / "descriptor_scaler.pkl"
            imputer_file = preprocessing_path / "descriptor_imputer.pkl"
            correlation_mask_file = preprocessing_path / "correlation_mask.npy"
            
            preprocessing_files_exist = (
                metadata_path.exists() and 
                scaler_file.exists() and
                imputer_file.exists() and
                correlation_mask_file.exists()
            )
            
            if preprocessing_files_exist:
                logger.info(f"✓ Found preprocessing files for split {i} at {preprocessing_path}")
            else:
                logger.warning(f"⚠ Preprocessing files incomplete for split {i} at {preprocessing_path}")
                logger.warning(f"  metadata: {metadata_path.exists()}, scaler: {scaler_file.exists()}, "
                             f"imputer: {imputer_file.exists()}, mask: {correlation_mask_file.exists()}")
            
            # Load the metadata file directly
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    split_preprocessing_metadata[i] = json.load(f)
                logger.info(f"Loaded preprocessing metadata for split {i}")
            else:
                logger.warning(f"No preprocessing metadata found at {metadata_path}")
                split_preprocessing_metadata[i] = None
        
        # Remove constant features using saved metadata (same as train_graph.py)
        if split_preprocessing_metadata[0] is not None:
            constant_features = split_preprocessing_metadata[0]['data_info']['constant_features_removed']
            if constant_features:
                logger.info(f"Removing {len(constant_features)} constant features from full dataset")
                orig_Xd = np.delete(orig_Xd, constant_features, axis=1)
        
    # === Evaluation Loop ===
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
            if args.preprocessing_path:
                # Use provided base path but with proper experiment naming
                base_name = f"{args.dataset_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}__rep{i}"
                preprocessing_path = Path(args.preprocessing_path) / base_name
            else:
                checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix = build_experiment_paths(
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
        DS = data.MoleculeDataset if MODEL_NAME in small_molecule_models else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)
        
        # Apply descriptor scaling using saved scaler (same as train_graph.py)
        if combined_descriptor_data is not None:
            if args.preprocessing_path:
                # Use provided base path but with proper experiment naming
                base_name = f"{args.dataset_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{size_suffix}__rep{i}"
                preprocessing_path = Path(args.preprocessing_path) / base_name
            else:
                checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix = build_experiment_paths(
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
        
        # Build checkpoint path using the same logic as training scripts
        if args.checkpoint_path:
            # Legacy support: if checkpoint_path is provided, use the existing logic
            # Extract the base experiment pattern from the provided checkpoint path
            # and build the correct path for this replicate
            provided_path = Path(args.checkpoint_path)
            
            # Try to find the experiment directory by looking for __rep pattern in the path
            # Walk up the path until we find a directory with __rep in its name
            current_path = provided_path
            exp_dir = None
            
            # If it's a file, start from its parent directory
            if provided_path.is_file():
                current_path = provided_path.parent
            
            # Walk up the directory tree to find the experiment directory
            for parent in [current_path] + list(current_path.parents):
                if "__rep" in parent.name:
                    exp_dir = parent
                    break
            
            if exp_dir:
                exp_dir_name = exp_dir.name
                # Replace any existing rep number with the current replicate number
                import re
                new_exp_dir_name = re.sub(r'__rep\d+', f'__rep{i}', exp_dir_name)
                checkpoint_path = exp_dir.parent / new_exp_dir_name
                logger.info(f"Built checkpoint path for rep {i}: {checkpoint_path}")
            else:
                # Special case: check if the file itself has __rep pattern (e.g., AttentiveFP)
                if provided_path.is_file() and "__rep" in provided_path.name:
                    import re
                    new_filename = re.sub(r'__rep\d+', f'__rep{i}', provided_path.name)
                    checkpoint_path = provided_path.parent / new_filename
                    logger.info(f"Built checkpoint file path for rep {i}: {checkpoint_path}")
                else:
                    # Check if this is a base pattern that we need to append __rep{i} to
                    # This handles cases like "/checkpoints/DMPNN/htpmd__Conductivity"
                    base_pattern = provided_path.name
                    if not base_pattern.endswith(f"__rep{i}"):
                        # Append the replicate suffix
                        checkpoint_path = provided_path.parent / f"{base_pattern}__rep{i}"
                        logger.info(f"Built checkpoint path for rep {i}: {checkpoint_path}")
                    else:
                        # Already has rep suffix, use as-is
                        checkpoint_path = provided_path
                        logger.info(f"Using provided checkpoint path: {checkpoint_path}")
        else:
            # Use the same checkpoint path building logic as training scripts
            if args.model_name == "AttentiveFP":
                # AttentiveFP uses a different checkpoint structure
                try:
                    from train_attentivefp import create_checkpoint_name
                    checkpoint_name = create_checkpoint_name(args, target, i)
                    checkpoint_path = checkpoint_dir / "AttentiveFP" / checkpoint_name / "best.pt"
                except ImportError:
                    # Fallback: build AttentiveFP path manually
                    parts = [args.dataset_name, target]
                    if args.train_size and args.train_size != "full":
                        parts.append(f"size{args.train_size}")
                    parts.append(f"rep{i}")
                    checkpoint_name = "__".join(parts)
                    checkpoint_path = checkpoint_dir / "AttentiveFP" / checkpoint_name / "best.pt"
            else:
                # DMPNN, wDMPNN, DMPNN_DiffPool use build_experiment_paths
                checkpoint_path, _, _, _, _, _, _, _ = build_experiment_paths(
                    args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
                )
        
        # Check if embeddings already exist from train_graph.py --export_embeddings FIRST
        # OR if we've already extracted them in this evaluation run
        embeddings_dir = Path(checkpoint_path) / "embeddings"
        embedding_files_exist = (
            (embeddings_dir / f"X_train_split_{i}.npy").exists() and
            (embeddings_dir / f"X_val_split_{i}.npy").exists() and
            (embeddings_dir / f"X_test_split_{i}.npy").exists() and
            (embeddings_dir / f"feature_mask_split_{i}.npy").exists()
        )
        
        # Also check for temporary embeddings from this evaluation session
        # Use a target-agnostic path since the same model can be used for all targets
        checkpoint_path_str = str(checkpoint_path)
        if "__rep" in checkpoint_path_str:
            # Remove everything between dataset and rep to get base model path
            # e.g., "htpmd__Conductivity__desc__rdkit__rep0" -> "htpmd__rep0"
            import re
            match = re.match(r'^(.+?)__.*?__(rep\d+)$', checkpoint_path_str)
            if match:
                dataset_part = match.group(1)
                rep_part = match.group(2)
                base_path = f"{dataset_part}__{rep_part}"
                base_checkpoint_path = Path(checkpoint_path_str).parent / base_path
            else:
                base_checkpoint_path = checkpoint_path
        else:
            base_checkpoint_path = checkpoint_path
            
        temp_embeddings_dir = base_checkpoint_path / "temp_embeddings"
        temp_embedding_files_exist = (
            (temp_embeddings_dir / f"X_train_split_{i}.npy").exists() and
            (temp_embeddings_dir / f"X_val_split_{i}.npy").exists() and
            (temp_embeddings_dir / f"X_test_split_{i}.npy").exists() and
            (temp_embeddings_dir / f"feature_mask_split_{i}.npy").exists()
        )
        
        # Use either permanent or temporary embeddings
        if embedding_files_exist:
            use_embeddings_dir = embeddings_dir
        elif temp_embedding_files_exist:
            use_embeddings_dir = temp_embeddings_dir
            embedding_files_exist = True
        else:
            use_embeddings_dir = None
        
        if embedding_files_exist:
            logger.info(f"✅ Found existing embeddings at {use_embeddings_dir} - loading directly (fast path)")
            X_train = np.load(use_embeddings_dir / f"X_train_split_{i}.npy")
            X_val = np.load(use_embeddings_dir / f"X_val_split_{i}.npy")
            X_test = np.load(use_embeddings_dir / f"X_test_split_{i}.npy")
            keep = np.load(use_embeddings_dir / f"feature_mask_split_{i}.npy")
            
            logger.info(f"Loaded embeddings - kept dims: {int(keep.sum())} / {len(keep)}")
            logger.info(f"  - X_train: {X_train.shape}")
            logger.info(f"  - X_val: {X_val.shape}")
            logger.info(f"  - X_test: {X_test.shape}")
        else:
            logger.info("❌ No existing embeddings found, need to extract from checkpoint (slow path)")
            logger.info("🔄 NOTE: This script ONLY loads trained checkpoints for evaluation - NO TRAINING occurs")
            
            # Use provided checkpoint file or discover best checkpoint
            if args.checkpoint_path:
                # Always discover the best checkpoint in the replicate-specific directory
                # (we've already built the correct checkpoint_path above)
                last_ckpt = pick_best_checkpoint(Path(checkpoint_path))
                if last_ckpt is None:
                    logger.warning(f"No checkpoint found at {checkpoint_path}; skipping rep {i} for target {target}.")
                    continue
                logger.info(f"Using discovered checkpoint file for rep {i}: {last_ckpt}")
            else:
                # Only now check for checkpoint since we need to extract embeddings
                last_ckpt = pick_best_checkpoint(Path(checkpoint_path))
                if last_ckpt is None:
                    # no checkpoint → skip this replicate (leave row without this target's metrics)
                    logger.warning(f"No checkpoint found at {checkpoint_path}; skipping rep {i} for target {target}.")
                    continue
            
            # Load encoder and make fingerprints (map to CPU if CUDA not available)
            import torch
            map_location = torch.device('cpu') if not torch.cuda.is_available() else None
            
            logger.info("📥 Loading trained model from checkpoint for embedding extraction...")
            
            # Handle different checkpoint formats and model types
            # This section only handles DMPNN variants since AttentiveFP and Graphormer exit early
            if args.model_name == "DMPNN_DiffPool":
                # DMPNN_DiffPool has different architecture - need to create model first
                logger.info("Creating DMPNN_DiffPool model architecture...")
                
                # Import required modules
                from chemprop import nn
                
                # Create the same architecture as in training (from utils.py)
                base_mp_cls = nn.BondMessagePassing
                
                # Use same defaults as in utils.py build_model_and_trainer
                depth = getattr(args, "diffpool_depth", 1)
                ratio = getattr(args, "diffpool_ratio", 0.5)
                
                mp = nn.BondMessagePassingWithDiffPool(
                    base_mp_cls=base_mp_cls,
                    depth=depth,
                    ratio=ratio
                )
                agg = nn.IdentityAggregation()
                
                # Calculate input dimension for FFN (same as in utils.py)
                descriptor_dim = combined_descriptor_data.shape[1] if combined_descriptor_data is not None else 0
                input_dim = mp.output_dim + descriptor_dim
                
                # Create predictor based on task type (same as in utils.py)
                if args.task_type == 'reg':
                    predictor = nn.RegressionFFN(
                        output_transform=None,  # Will be loaded from checkpoint
                        n_tasks=1, 
                        input_dim=input_dim
                    )
                elif args.task_type == 'binary':
                    predictor = nn.BinaryClassificationFFN(input_dim=input_dim)
                elif args.task_type == 'multi':
                    # For multi-class, we need n_classes - use a reasonable default
                    n_classes = max(n_classes_per_target.values()) if n_classes_per_target else 2
                    predictor = nn.MulticlassClassificationFFN(
                        n_classes=n_classes, 
                        input_dim=input_dim
                    )
                else:
                    # Default to regression
                    predictor = nn.RegressionFFN(
                        output_transform=None,
                        n_tasks=1, 
                        input_dim=input_dim
                    )
                
                # Create MPNN with DiffPool architecture
                mpnn = models.MPNN(
                    message_passing=mp,
                    agg=agg,
                    predictor=predictor,
                    batch_norm=args.batch_norm,
                    metrics=[]
                )
                
                # Load checkpoint manually
                checkpoint = torch.load(last_ckpt, map_location=map_location)
                mpnn.load_state_dict(checkpoint["state_dict"], strict=False)
                mpnn.eval()
                logger.info("✅ DMPNN_DiffPool model loaded in evaluation mode")
                
            else:
                # Standard DMPNN variants (DMPNN, wDMPNN, PPG)
                logger.info(f"Loading {args.model_name} Lightning checkpoint: {last_ckpt}")
                mpnn = models.MPNN.load_from_checkpoint(str(last_ckpt), map_location=map_location)
                mpnn.eval()  # Ensure evaluation mode
                logger.info(f"✅ {args.model_name} model loaded in evaluation mode")
                
            logger.info("🧠 Extracting embeddings from trained model...")
            X_train = get_encodings_from_loader(mpnn, train_loader)
            X_val = get_encodings_from_loader(mpnn, val_loader)
            X_test = get_encodings_from_loader(mpnn, test_loader)

            eps = 1e-8  # or 1e-6 if you want to be stricter
            std_train = X_train.std(axis=0)
            keep = std_train > eps

            X_train = X_train[:, keep]
            X_val   = X_val[:, keep]
            X_test  = X_test[:, keep]

            logger.info(f"Extracted embeddings - kept dims: {int(keep.sum())} / {len(keep)}")
            
            # Save temporary embeddings for reuse in subsequent targets
            # Use the same target-agnostic path as above
            temp_embeddings_dir = base_checkpoint_path / "temp_embeddings"
            temp_embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(temp_embeddings_dir / f"X_train_split_{i}.npy", X_train)
            np.save(temp_embeddings_dir / f"X_val_split_{i}.npy", X_val)
            np.save(temp_embeddings_dir / f"X_test_split_{i}.npy", X_test)
            np.save(temp_embeddings_dir / f"feature_mask_split_{i}.npy", keep)
            
            logger.info(f"💾 Saved temporary embeddings to {temp_embeddings_dir} for reuse")

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
        
       

        # Build requested baselines with scaler info (same as train_tabular.py)
        num_classes = len(np.unique(y_train_scaled)) if args.task_type != "reg" else None
        rep_model_to_row = fit_and_score_baselines(X_train, y_train, X_val, y_val, X_test, y_test, args.task_type, num_classes)
        

# Convert to train_graph.py format with appropriate metrics for task type
eval_results = []
for (rep_idx, model_name), row_data in rep_model_to_row.items():
    for target in target_columns:
        if args.task_type == "reg" and f"{target}_R2" in row_data:
            # Regression metrics
            eval_results.append({
                'target': target,
                'split': rep_idx,
                'test/mae': row_data[f"{target}_MAE"],
                'test/r2': row_data[f"{target}_R2"],
                'test/rmse': row_data[f"{target}_RMSE"],
                'model': model_name
            })
        elif args.task_type in ["binary", "multi"] and f"{target}_ACC" in row_data:
            # Classification metrics
            result_row = {
                'target': target,
                'split': rep_idx,
                'test/accuracy': row_data[f"{target}_ACC"],
                'test/f1': row_data[f"{target}_F1"],
                'model': model_name
            }
            # Add ROC-AUC if available
            if f"{target}_ROC_AUC" in row_data:
                result_row['test/roc_auc'] = row_data[f"{target}_ROC_AUC"]
            eval_results.append(result_row)

if eval_results:
    results_df = pd.DataFrame(eval_results)
    save_and_summarize_results(results_df, args, results_dir, descriptor_columns)

# Clean up temporary embeddings - DISABLED to preserve embeddings for reuse
# logger.info("🧹 Cleaning up temporary embeddings...")
# import shutil
# 
# # Get unique base checkpoint paths (target-agnostic) to avoid duplicate cleanup
# cleaned_paths = set()
# 
# for target in target_columns:
#     for i in range(REPLICATES):
#         if args.checkpoint_path:
#             # Use the same path building logic as above
#             provided_path = Path(args.checkpoint_path)
#             current_path = provided_path
#             exp_dir = None
#             
#             if provided_path.is_file():
#                 current_path = provided_path.parent
#             
#             for parent in [current_path] + list(current_path.parents):
#                 if "__rep" in parent.name:
#                     exp_dir = parent
#                     break
#             
#             if exp_dir:
#                 exp_dir_name = exp_dir.name
#                 import re
#                 new_exp_dir_name = re.sub(r'__rep\d+', f'__rep{i}', exp_dir_name)
#                 checkpoint_path = exp_dir.parent / new_exp_dir_name
#             else:
#                 if provided_path.is_file() and "__rep" in provided_path.name:
#                     import re
#                     new_filename = re.sub(r'__rep\d+', f'__rep{i}', provided_path.name)
#                     checkpoint_path = provided_path.parent / new_filename
#                 else:
#                     checkpoint_path = provided_path
#         else:
#             checkpoint_path, _, _, _, _, _ = build_experiment_paths(
#                 args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
#             )
#         
#         # Convert to target-agnostic path for cleanup
#         checkpoint_path_str = str(checkpoint_path)
#         if "__rep" in checkpoint_path_str:
#             # Use same logic as above for consistency
#             import re
#             match = re.match(r'^(.+?)__.*?__(rep\d+)$', checkpoint_path_str)
#             if match:
#                 dataset_part = match.group(1)
#                 rep_part = match.group(2)
#                 base_path = f"{dataset_part}__{rep_part}"
#                 base_checkpoint_path = Path(checkpoint_path_str).parent / base_path
#             else:
#                 base_checkpoint_path = checkpoint_path
#         else:
#             base_checkpoint_path = checkpoint_path
#             
#         temp_embeddings_dir = base_checkpoint_path / "temp_embeddings"
#         
#         # Only clean each unique path once
#         if str(temp_embeddings_dir) not in cleaned_paths and temp_embeddings_dir.exists():
#             try:
#                 shutil.rmtree(temp_embeddings_dir)
#                 cleaned_paths.add(str(temp_embeddings_dir))
#                 logger.info(f"🗑️ Cleaned up {temp_embeddings_dir}")
#             except OSError:
#                 pass  # Ignore if cleanup fails

logger.info("✅ Evaluation complete!")