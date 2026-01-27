#!/usr/bin/env python3
"""
Comprehensive feature space analysis for tabular models.

This script analyzes the feature space used in tabular training, including:
1. Feature family summary (AB block, RDKit, descriptors)
2. Feature variance histograms after preprocessing
3. PCA 2D scatter plots colored by target
4. UMAP 2D embedding plots colored by target
5. t-SNE 2D embedding plots colored by target
6. Top 5 feature-target correlations bar charts

Uses YAML configuration to specify datasets and targets.
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
from sklearn.manifold import TSNE
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

# Dataset-specific color palette (consistent with compare_tabular_vs_graph.py)
DATASET_COLORS = {
    'tc': '#1f77b4',              # Blue
    'insulator': '#ff7f0e',       # Orange
    'htpmd': '#2ca02c',           # Green
    'polyinfo': '#d62728',        # Red
    'camb3lyp': '#9467bd',
    'cam_b3lyp': '#9467bd',        # Purple
    'opv_camb3lyp': '#9467bd',    # Purple (same as camb3lyp)
    'ea_ip': '#8c564b',           # Brown
    'pae_tg_mono211': '#e377c2',  # Pink
    'pae_tg_paper211': '#7f7f7f', # Gray
}

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
    # Copolymer datasets typically have fraction columns
    copolymer_datasets = ['ea_ip']  # Add more as needed
    
    # Get descriptor columns
    descriptor_columns = []
    if 'dataset_descriptors' in train_config and dataset_name in train_config['dataset_descriptors']:
        descriptor_columns = train_config['dataset_descriptors'][dataset_name]
    
    # Determine SMILES column (default to 'smiles' for most datasets)
    smiles_column = 'smiles'
    if dataset_name.lower() in ['ea_ip']:
        smiles_column = 'smi'  # Some datasets use 'smi' instead of 'smiles'
    
    return {
        'polymer_type': 'copolymer' if dataset_name in copolymer_datasets else 'homo',
        'descriptor_columns': descriptor_columns,
        'smiles_column': smiles_column
    }

def analyze_feature_families(ab_block: np.ndarray, descriptor_block: np.ndarray, 
                           rdkit_block: np.ndarray, feat_names: List[str]) -> pd.DataFrame:
    """Create summary table of feature counts per family."""
    
    feature_data = []
    
    # AB block features
    if ab_block is not None:
        ab_count = ab_block.shape[1]
        feature_data.append({
            'Feature_Family': 'AB_Block',
            'Feature_Count': ab_count,
            'Percentage': (ab_count / len(feat_names)) * 100,
            'Description': 'Polymer composition features (fractions, counts)'
        })
    
    # RDKit features
    if rdkit_block is not None:
        rdkit_count = rdkit_block.shape[1]
        feature_data.append({
            'Feature_Family': 'RDKit',
            'Feature_Count': rdkit_count,
            'Percentage': (rdkit_count / len(feat_names)) * 100,
            'Description': 'Molecular descriptors from RDKit'
        })
    
    # Descriptor features
    if descriptor_block is not None:
        desc_count = descriptor_block.shape[1]
        feature_data.append({
            'Feature_Family': 'Descriptors',
            'Feature_Count': desc_count,
            'Percentage': (desc_count / len(feat_names)) * 100,
            'Description': 'Pre-computed molecular descriptors'
        })
    
    # Total
    feature_data.append({
        'Feature_Family': 'Total',
        'Feature_Count': len(feat_names),
        'Percentage': 100.0,
        'Description': 'All features combined'
    })
    
    return pd.DataFrame(feature_data)

def plot_feature_variance_histogram(features: np.ndarray, feature_names: List[str], 
                                  dataset_name: str, target_name: str, output_dir: str) -> None:
    """Plot histogram of feature variances after preprocessing."""
    
    # Calculate variances
    variances = np.var(features, axis=0)
    
    # Categorize features
    exact_zero_mask = variances == 0.0
    low_variance_mask = (variances > 0.0) & (variances < 1e-10)
    normal_variance_mask = variances >= 1e-10
    
    exact_zero_count = np.sum(exact_zero_mask)
    low_variance_count = np.sum(low_variance_mask)
    normal_variance_count = np.sum(normal_variance_mask)
    
    # For histogram, show only features with meaningful variance
    meaningful_variances = variances[normal_variance_mask]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    if len(meaningful_variances) > 0:
        plt.hist(meaningful_variances, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        
        # Add statistics
        mean_var = np.mean(meaningful_variances)
        median_var = np.median(meaningful_variances)
        
        plt.axvline(mean_var, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_var:.2e}')
        plt.axvline(median_var, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_var:.2e}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No features with meaningful variance', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    
    plt.title(f'{dataset_name.upper()} - {target_name}\nFeature Variance Distribution (Smart Filtering)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Feature Variance', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format axes
    if len(meaningful_variances) > 0:
        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Add comprehensive statistics
    stats_text = f'Total features: {len(variances)}\n'
    stats_text += f'Exact zero variance: {exact_zero_count}\n'
    stats_text += f'Low variance (<1e-10): {low_variance_count}\n'
    stats_text += f'Meaningful variance (≥1e-10): {normal_variance_count}'
    
    if len(meaningful_variances) > 0:
        stats_text += f'\nMean: {mean_var:.2e}\nMedian: {median_var:.2e}'
    
    # Break down by feature type
    ab_indices = [i for i, name in enumerate(feature_names) if name.startswith('AB_')]
    desc_indices = [i for i, name in enumerate(feature_names) if not name.startswith('AB_')]
    
    ab_exact_zero = sum(1 for i in ab_indices if exact_zero_mask[i])
    ab_low_variance = sum(1 for i in ab_indices if low_variance_mask[i])
    ab_normal = sum(1 for i in ab_indices if normal_variance_mask[i])
    
    desc_exact_zero = sum(1 for i in desc_indices if exact_zero_mask[i])
    desc_low_variance = sum(1 for i in desc_indices if low_variance_mask[i])
    desc_normal = sum(1 for i in desc_indices if normal_variance_mask[i])
    
    stats_text += f'\n\nAB features: {ab_exact_zero} zero, {ab_low_variance} low, {ab_normal} normal'
    stats_text += f'\nDescriptors: {desc_exact_zero} zero, {desc_low_variance} low, {desc_normal} normal'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # Save plot
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{dataset_name}_{target_name}_feature_variance_histogram.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature variance histogram: {output_file}")
    plt.close()

def plot_pca_scatter(features: np.ndarray, targets: np.ndarray, dataset_name: str, 
                    target_name: str, output_dir: str) -> None:
    """Create PCA 2D scatter plot colored by target values."""
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=targets, cmap='viridis', alpha=0.6, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{target_name} Value', fontsize=12)
    
    # Add labels and title
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title(f'{dataset_name.upper()} - {target_name}\nPCA 2D Projection', 
             fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    plt.text(0.02, 0.98, f'Explained variance: {explained_var:.1f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save plot
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{dataset_name}_{target_name}_pca_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PCA scatter plot: {output_file}")
    plt.close()

def plot_umap_embedding(features: np.ndarray, targets: np.ndarray, dataset_name: str, 
                       target_name: str, output_dir: str) -> None:
    """Create UMAP 2D embedding plot colored by target values."""
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    features_umap = reducer.fit_transform(features_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Use dataset-specific color
    dataset_color = DATASET_COLORS.get(dataset_name, '#1f77b4')
    
    # Create scatter plot with dataset-specific color
    scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], 
                         c=dataset_color, alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
    
    # # OLD: Color by target values with viridis colormap
    # scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], 
    #                      c=targets, cmap='viridis', alpha=0.6, s=50)
    # 
    # # Add colorbar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label(f'{target_name} Value', fontsize=12)
    
    # Add labels and title
    # plt.xlabel('UMAP 1', fontsize=12)
    # plt.ylabel('UMAP 2', fontsize=12)
    plt.title(f'{dataset_name.upper()}',  # OLD: - {target_name}\nUMAP 2D Embedding
             fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)

    # plt.xlim(-15, 35)
    # plt.ylim(-20, 30)
    # plt.gca().set_aspect('equal', adjustable='box')
                      
    # Save plot
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{dataset_name}_{target_name}_umap_embedding.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP embedding plot: {output_file}")
    plt.close()

def plot_tsne_embedding(features: np.ndarray, targets: np.ndarray, dataset_name: str, 
                       target_name: str, output_dir: str) -> None:
    """Create t-SNE 2D embedding plot colored by target values."""
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Limit samples for t-SNE performance if dataset is large
    max_samples = 5000
    if len(features_scaled) > max_samples:
        print(f"Subsampling {max_samples} points for t-SNE analysis (from {len(features_scaled)})")
        indices = np.random.choice(len(features_scaled), max_samples, replace=False)
        features_scaled = features_scaled[indices]
        targets = targets[indices]
    
    # Apply t-SNE
    perplexity = min(30, len(features_scaled) // 4)  # Ensure perplexity is valid
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_tsne = tsne.fit_transform(features_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                         c=targets, cmap='viridis', alpha=0.6, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{target_name} Value', fontsize=12)
    
    # Add labels and title
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title(f'{dataset_name.upper()} - {target_name}\nt-SNE 2D Embedding', 
             fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add sample info if subsampled
    if len(features_scaled) < len(features):
        plt.text(0.02, 0.98, f'Showing {len(features_scaled)} of {len(features)} samples', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save plot
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{dataset_name}_{target_name}_tsne_embedding.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE embedding plot: {output_file}")
    plt.close()

def plot_top_feature_correlations(features: np.ndarray, feature_names: List[str], 
                                targets: np.ndarray, dataset_name: str, target_name: str, 
                                output_dir: str) -> None:
    """Create bar chart of top 5 feature-target correlations."""
    
    # Calculate correlations
    correlations = []
    p_values = []
    
    for i in range(features.shape[1]):
        corr, p_val = pearsonr(features[:, i], targets)
        if not np.isnan(corr):
            correlations.append(abs(corr))  # Use absolute value for ranking
            p_values.append(p_val)
        else:
            correlations.append(0.0)
            p_values.append(1.0)
    
    correlations = np.array(correlations)
    p_values = np.array(p_values)
    
    # Get top 5 features
    top_indices = np.argsort(correlations)[-5:][::-1]  # Top 5 in descending order
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    top_features = [feature_names[i] for i in top_indices]
    top_corrs = [correlations[i] * np.sign(pearsonr(features[:, i], targets)[0]) for i in top_indices]
    top_pvals = [p_values[i] for i in top_indices]
    
    # Create color map based on correlation direction
    colors = ['red' if corr < 0 else 'blue' for corr in top_corrs]
    
    # Create bar plot
    bars = plt.bar(range(len(top_features)), top_corrs, color=colors, alpha=0.7, edgecolor='black')
    
    # Add significance stars
    for i, (bar, p_val) in enumerate(zip(bars, top_pvals)):
        if p_val < 0.001:
            star_text = '***'
        elif p_val < 0.01:
            star_text = '**'
        elif p_val < 0.05:
            star_text = '*'
        else:
            star_text = ''
        
        if star_text:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    star_text, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize plot
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Correlation with Target', fontsize=12)
    plt.title(f'{dataset_name.upper()} - {target_name}\nTop 5 Feature-Target Correlations', 
             fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
    
    # Add grid and horizontal line at y=0
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add legend for correlation direction
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Positive correlation'),
                      Patch(facecolor='red', alpha=0.7, label='Negative correlation')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add text for significance
    plt.text(0.02, 0.98, 'Significance: *** p<0.001, ** p<0.01, * p<0.05', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{dataset_name}_{target_name}_top_correlations.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved top correlations plot: {output_file}")
    plt.close()
    
    return top_features, top_corrs, top_pvals

def analyze_dataset_features(dataset_name: str, config: Dict[str, Any], output_dir: str, logger: logging.Logger) -> List[Dict]:
    """Perform comprehensive feature analysis for a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    data_path = get_data_path(dataset_name, config['datasets'])
    if not data_path.exists():
        print(f"Warning: Dataset file not found: {data_path}")
        return []
    
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
    
    if not valid_targets:
        print(f"No valid targets found for {dataset_name}")
        return []
    
    print(f"Found {len(valid_targets)} valid targets: {valid_targets}")
    
    # Create train/val/test split (use first split for analysis)
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
    
    # For simplicity, treat all descriptor block as "descriptors" 
    # (RDKit features are included in the descriptor block)
    rdkit_block = None
    rdkit_indices = []
    
    # Preprocess features
    print("Preprocessing features...")
    if descriptor_block is not None:
        # Get non-AB feature names
        non_ab_names = [name for name in feat_names if not name.startswith('AB_')]
        
        (desc_tr_selected, desc_val_selected, desc_te_selected, selected_desc_names, 
         preprocessing_metadata, imputer, constant_mask, corr_mask) = preprocess_descriptor_data(
             descriptor_block, train_idx, val_idx, test_idx, 
             non_ab_names,
             logger  # Pass the logger
         )
    else:
        desc_tr_selected = None
        selected_desc_names = []
    
    # Combine all features
    if ab_block is not None and desc_tr_selected is not None:
        # AB + descriptors
        features_combined = np.concatenate([ab_block[train_idx], desc_tr_selected], axis=1)
        final_feat_names = ([name for name in feat_names if name.startswith('AB_')] + 
                           selected_desc_names)
    elif ab_block is not None:
        # Only AB
        features_combined = ab_block[train_idx]
        final_feat_names = [name for name in feat_names if name.startswith('AB_')]
    elif desc_tr_selected is not None:
        # Only descriptors
        features_combined = desc_tr_selected
        final_feat_names = selected_desc_names
    else:
        print("No features found!")
        return []
    
    # FINAL STEP: Smart zero-variance filtering
    # Remove exact-zero variance features but keep low-variance AB features
    print("Performing smart zero-variance filtering on combined features...")
    final_variances = np.var(features_combined, axis=0)
    
    # Separate AB and descriptor features
    ab_indices = [i for i, name in enumerate(final_feat_names) if name.startswith('AB_')]
    desc_indices = [i for i, name in enumerate(final_feat_names) if not name.startswith('AB_')]
    
    # Remove exact-zero variance features (variance == 0.0)
    exact_zero_mask = final_variances == 0.0
    
    # For descriptors: remove exact-zero variance (same as before)
    # For AB features: remove only exact-zero variance, keep low-variance
    final_keep_mask = ~exact_zero_mask
    
    # Log what's being removed
    final_zero_var_removed = [name for name, keep in zip(final_feat_names, final_keep_mask) if not keep]
    
    if final_zero_var_removed:
        print(f"Removing {len(final_zero_var_removed)} exact-zero variance features:")
        ab_zero_var = [name for name in final_zero_var_removed if name.startswith('AB_')]
        desc_zero_var = [name for name in final_zero_var_removed if not name.startswith('AB_')]
        if ab_zero_var:
            print(f"  AB features: {len(ab_zero_var)} (exact zero variance)")
        if desc_zero_var:
            print(f"  Descriptor features: {len(desc_zero_var)} (exact zero variance)")
    
    # Count low-variance features that are being kept
    low_variance_mask = (final_variances > 0.0) & (final_variances < 1e-10)
    low_variance_count = np.sum(low_variance_mask)
    ab_low_variance = sum(1 for i in ab_indices if low_variance_mask[i])
    desc_low_variance = sum(1 for i in desc_indices if low_variance_mask[i])
    
    if low_variance_count > 0:
        print(f"Keeping {low_variance_count} low-variance features:")
        if ab_low_variance > 0:
            print(f"  AB features: {ab_low_variance} (low but non-zero variance)")
        if desc_low_variance > 0:
            print(f"  Descriptor features: {desc_low_variance} (low but non-zero variance)")
    
    # Apply final filtering
    features_combined = features_combined[:, final_keep_mask]
    final_feat_names = [name for name, keep in zip(final_feat_names, final_keep_mask) if keep]
    
    print(f"Final feature matrix shape after smart filtering: {features_combined.shape}")
    
    # Update feature family summary to reflect final removal
    ab_count = len([name for name in final_feat_names if name.startswith('AB_')])
    desc_count = len([name for name in final_feat_names if not name.startswith('AB_')])
    
    # Recreate feature family summary with corrected counts
    feature_summary = pd.DataFrame({
        'Feature_Family': ['AB_Block', 'Descriptors', 'Total'],
        'Feature_Count': [ab_count, desc_count, len(final_feat_names)],
        'Percentage': [ab_count/len(final_feat_names)*100, desc_count/len(final_feat_names)*100, 100.0],
        'Description': [
            'Polymer composition features (fractions, counts)',
            'Pre-computed molecular descriptors', 
            'All features combined'
        ]
    })
    
    # Save feature summary
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / f"{dataset_name}_feature_family_summary.csv"
    feature_summary.to_csv(summary_file, index=False)
    print(f"Saved feature family summary: {summary_file}")
    
    # Analyze each target
    analysis_results = []
    
    for target in valid_targets:
        print(f"\nAnalyzing target: {target}")
        
        # Get target values
        y = df[target].values
        y_train = y[train_idx]
        
        # Remove samples with NaN target
        valid_mask = ~np.isnan(y_train)
        if valid_mask.sum() < 10:
            print(f"Skipping {target}: insufficient valid samples")
            continue
        
        features_valid = features_combined[valid_mask]
        y_valid = y_train[valid_mask]
        
        # # 2. Feature variance histogram
        # print("2. Creating feature variance histogram...")
        # plot_feature_variance_histogram(features_valid, final_feat_names, dataset_name, target, output_dir)
        
        # # 3. PCA scatter plot
        # print("3. Creating PCA scatter plot...")
        # plot_pca_scatter(features_valid, y_valid, dataset_name, target, output_dir)
        
        # 4. UMAP embedding plot
        print("4. Creating UMAP embedding plot...")
        plot_umap_embedding(features_valid, y_valid, dataset_name, target, output_dir)
        
        # # 5. t-SNE embedding plot
        # print("5. Creating t-SNE embedding plot...")
        # plot_tsne_embedding(features_valid, y_valid, dataset_name, target, output_dir)
        
        # #s
        break
    
    return analysis_results

def main():
    """Main function to perform comprehensive feature analysis."""
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "/Users/u6788552/Desktop/experiments/dmpnn/analysis/dataset_config.yaml"
    config = load_config(config_path)
    
    # Define output directory for this script
    output_dir = "plots/feature_space_analysis"
    
    print("Comprehensive Feature Space Analysis")
    print("="*60)
    print(f"Config file: {config_path}")
    print(f"Datasets to process: {list(config['datasets'].keys())}")
    print(f"Output directory: {output_dir}")
    
    # Process each dataset
    datasets = list(config['datasets'].keys())
    all_results = []
    
    for dataset in datasets:
        try:
            results = analyze_dataset_features(dataset, config, output_dir, logger)
            all_results.extend(results)
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary table of top correlations
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Save summary
        output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / "top_feature_correlations_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved correlations summary: {summary_file}")
        
        # Print summary of top correlations
        print("\n" + "="*80)
        print("TOP FEATURE-TARGET CORRELATIONS SUMMARY")
        print("="*80)
        
        # Show top 10 correlations overall
        top_overall = summary_df.nlargest(10, 'Absolute_Correlation')
        print(top_overall[['Dataset', 'Target', 'Rank', 'Feature', 'Correlation', 'P_Value']].to_string(index=False, float_format='%.3f'))
    
    print(f"\n" + "="*60)
    print("Feature space analysis complete!")
    print(f"Plots saved in: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
