#!/usr/bin/env python3
"""
Quick test of feature space analysis for TC dataset only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from typing import Dict, List, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import umap
import warnings
import logging
warnings.filterwarnings('ignore')

# Import tabular utilities
import sys
sys.path.append('/Users/u6788552/Desktop/experiments/dmpnn/scripts/python')
from tabular_utils import build_features, preprocess_descriptor_data
from utils import set_seed

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_data_path(dataset_name: str, datasets_config: Dict[str, Any]) -> Path:
    """Get the path to the dataset CSV file from config."""
    base_path = Path("/Users/u6788552/Desktop/experiments/dmpnn")
    
    if dataset_name not in datasets_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
    
    dataset_file = datasets_config[dataset_name]['file_path']
    return base_path / dataset_file

def get_target_columns(dataset_name: str, datasets_config: Dict[str, Any]) -> List[str]:
    """Get target columns for a dataset from config."""
    if dataset_name not in datasets_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
    
    return datasets_config[dataset_name]['targets']

def load_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Load dataset-specific information from train_config.yaml."""
    config_path = "/Users/u6788552/Desktop/experiments/dmpnn/scripts/python/train_config.yaml"
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Determine polymer type based on dataset name
    copolymer_datasets = ['ea_ip']
    
    # Get descriptor columns
    descriptor_columns = []
    if 'dataset_descriptors' in train_config and dataset_name in train_config['dataset_descriptors']:
        descriptor_columns = train_config['dataset_descriptors'][dataset_name]
    
    # Determine SMILES column
    smiles_column = 'smiles'
    if dataset_name.lower() in ['ea_ip']:
        smiles_column = 'smi'
    
    return {
        'polymer_type': 'copolymer' if dataset_name in copolymer_datasets else 'homo',
        'descriptor_columns': descriptor_columns,
        'smiles_column': smiles_column
    }

def plot_feature_variance_histogram(features: np.ndarray, feature_names: List[str], 
                                  dataset_name: str, target_name: str, output_dir: str) -> None:
    """Plot histogram of feature variances after preprocessing."""
    
    # Calculate variances
    variances = np.var(features, axis=0)
    
    # Filter out zero-variance features for better visualization
    zero_var_mask = variances < 1e-10
    zero_var_count = np.sum(zero_var_mask)
    non_zero_variances = variances[~zero_var_mask]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    if len(non_zero_variances) > 0:
        plt.hist(non_zero_variances, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    else:
        plt.text(0.5, 0.5, 'All features have zero variance', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    
    # Add statistics
    if len(non_zero_variances) > 0:
        mean_var = np.mean(non_zero_variances)
        median_var = np.median(non_zero_variances)
        
        plt.axvline(mean_var, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_var:.2e}')
        plt.axvline(median_var, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_var:.2e}')
    else:
        mean_var = 0.0
        median_var = 0.0
    
    plt.title(f'{dataset_name.upper()} - {target_name}\nFeature Variance Distribution (After Preprocessing)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Feature Variance', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    
    if len(non_zero_variances) > 0:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    # Format axes
    if len(non_zero_variances) > 0:
        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Add text statistics
    stats_text = f'Total features: {len(variances)}\nZero variance: {zero_var_count}\nNon-zero variance: {len(non_zero_variances)}'
    if len(non_zero_variances) > 0:
        stats_text += f'\nMean: {mean_var:.2e}\nMedian: {median_var:.2e}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{dataset_name}_{target_name}_feature_variance_histogram.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature variance histogram: {output_file}")
    plt.close()

def main():
    """Test the fixed preprocessing for TC dataset."""
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "/Users/u6788552/Desktop/experiments/dmpnn/analysis/dataset_config.yaml"
    config = load_config(config_path)
    
    # Test TC dataset only
    dataset_name = 'tc'
    output_dir = "plots/feature_space_analysis_test"
    
    print(f"Testing fixed preprocessing for {dataset_name.upper()}")
    print("="*60)
    
    # Load dataset
    data_path = get_data_path(dataset_name, config['datasets'])
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Get dataset info
    dataset_info = load_dataset_info(dataset_name)
    polymer_type = dataset_info.get('polymer_type', 'homo')
    descriptor_columns = dataset_info.get('descriptor_columns', [])
    smiles_column = dataset_info.get('smiles_column', 'smiles')
    
    # Get target columns
    target_cols = get_target_columns(dataset_name, config['datasets'])
    valid_targets = [col for col in target_cols if col in df.columns and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"Found {len(valid_targets)} valid targets: {valid_targets}")
    
    # Create train/val/test split
    np.random.seed(42)
    n_samples = len(df)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Generate features
    print("Generating features...")
    ab_block, descriptor_block, feat_names = build_features(
        df, train_idx, descriptor_columns, polymer_type, 
        use_rdkit=True, use_ab=True, smiles_column=smiles_column
    )
    
    print(f"Total features generated: {len(feat_names)}")
    print(f"AB block shape: {ab_block.shape if ab_block is not None else 'None'}")
    print(f"Descriptor block shape: {descriptor_block.shape if descriptor_block is not None else 'None'}")
    
    # Preprocess features
    print("Preprocessing features...")
    if descriptor_block is not None:
        non_ab_names = [name for name in feat_names if not name.startswith('AB_')]
        
        (desc_tr_selected, desc_val_selected, desc_te_selected, selected_desc_names, 
         preprocessing_metadata, imputer, constant_mask, corr_mask) = preprocess_descriptor_data(
             descriptor_block, train_idx, val_idx, test_idx, 
             non_ab_names,
             logger
         )
        
        print(f"\nPreprocessing Results:")
        print(f"Features before any selection: {preprocessing_metadata['n_desc_before_any_selection']}")
        print(f"Features after constant removal: {preprocessing_metadata['n_desc_after_constant_removal']}")
        print(f"Features after correlation removal: {preprocessing_metadata['n_desc_after_corr_removal']}")
        print(f"Features after final zero-var removal: {preprocessing_metadata['n_desc_after_final_zero_var_removal']}")
        print(f"Constant features removed: {len(preprocessing_metadata['constant_features_removed'])}")
        print(f"Correlated features removed: {len(preprocessing_metadata['correlated_features_removed'])}")
        print(f"Zero-var features removed after imputation: {len(preprocessing_metadata['zero_var_after_impute_removed'])}")
        
        if preprocessing_metadata['zero_var_after_impute_removed']:
            print(f"Zero-var features: {preprocessing_metadata['zero_var_after_impute_removed']}")
        
        # Combine with AB block
        if ab_block is not None:
            features_combined = np.concatenate([ab_block[train_idx], desc_tr_selected], axis=1)
            final_feat_names = ([name for name in feat_names if name.startswith('AB_')] + 
                               selected_desc_names)
        else:
            features_combined = desc_tr_selected
            final_feat_names = selected_desc_names
        
        print(f"\nFinal feature matrix shape: {features_combined.shape}")
        
        # Check variance of final features
        final_variances = np.var(features_combined, axis=0)
        zero_var_final = np.sum(final_variances < 1e-10)
        print(f"Zero variance features in final matrix: {zero_var_final}")
        
        # Plot variance histogram for the first target
        if valid_targets:
            target = valid_targets[0]
            print(f"\nCreating variance histogram for target: {target}")
            plot_feature_variance_histogram(features_combined, final_feat_names, dataset_name, target, output_dir)
            
            print(f"\n✅ Test complete! Check the plot: plots/feature_space_analysis_test/{dataset_name}_{target}_feature_variance_histogram.png")
        else:
            print("No valid targets found for plotting")
    else:
        print("No descriptor features found!")

if __name__ == "__main__":
    main()
