#!/usr/bin/env python3
"""
Comprehensive feature space analysis for graph models.

This script analyzes the embedding space used in graph model training, including:
1. PCA 2D scatter plots colored by target (full dataset + train/val/test subplots)
2. UMAP 2D embedding plots colored by target (full dataset + train/val/test subplots)  
3. t-SNE 2D embedding plots colored by target (full dataset + train/val/test subplots)

Uses saved embeddings from train_graph.py --export_embeddings for analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from typing import Dict, List, Any, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import warnings
import logging
from glob import glob
import re
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def discover_embeddings(embeddings_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Discover available embeddings in the embeddings directory.
    
    Returns:
        Dict mapping dataset__model__target__variant to split information
    """
    embeddings_info = {}
    
    # Pattern to match embedding files
    # More flexible pattern to handle special characters in target names
    pattern = r"(.+?)__(.+?)__(.+?)__X_train_split_(\d+)\.npy"
    
    train_files = list(embeddings_dir.glob("*__X_train_split_*.npy"))
    
    for train_file in train_files:
        match = re.match(pattern, train_file.name)
        if not match:
            continue
            
        dataset = match.group(1)
        model = match.group(2)
        target_and_suffixes = match.group(3)
        split_idx = match.group(4)
        
        # Parse target and suffixes from the combined string
        # Split by __ and find where the target ends and suffixes begin
        parts = target_and_suffixes.split("__")
        
        # Common suffixes to look for
        suffix_keywords = ['desc', 'rdkit', 'batch_norm'] + [f'size{i}' for i in range(100, 20000, 100)]
        
        # Find the first suffix keyword
        target_parts = []
        suffix_parts = []
        found_suffix = False
        
        for part in parts:
            if not found_suffix and (part in suffix_keywords or part.startswith('size')):
                found_suffix = True
                suffix_parts.append(part)
            elif found_suffix:
                suffix_parts.append(part)
            else:
                target_parts.append(part)
        
        target = "__".join(target_parts)
        variant = "__".join(suffix_parts) if suffix_parts else "base"
        key = f"{dataset}__{model}__{target}__{variant}"
        
        if key not in embeddings_info:
            embeddings_info[key] = {
                'dataset': dataset,
                'model': model, 
                'target': target,
                'variant': variant,
                'splits': []
            }
        
        # Check if all required files exist for this split
        # Construct the base filename pattern
        if variant == "base":
            base_pattern = f"{dataset}__{model}__{target}"
        else:
            base_pattern = f"{dataset}__{model}__{target}__{variant}"
            
        val_file = embeddings_dir / f"{base_pattern}__X_val_split_{split_idx}.npy"
        test_file = embeddings_dir / f"{base_pattern}__X_test_split_{split_idx}.npy"
        
        if all(f.exists() for f in [train_file, val_file, test_file]):
            embeddings_info[key]['splits'].append({
                'split_idx': int(split_idx),
                'train_file': train_file,
                'val_file': val_file,
                'test_file': test_file,
            })
    
    # Filter to only include experiments with at least 1 split (flexible)
    complete_embeddings = {}
    for key, info in embeddings_info.items():
        if len(info['splits']) >= 1:  # Accept experiments with at least 1 split
            # Sort splits by index
            info['splits'].sort(key=lambda x: x['split_idx'])
            complete_embeddings[key] = info
            
    return complete_embeddings

def load_target_data(dataset_name: str, target_name: str, config: Dict[str, Any]) -> np.ndarray:
    """Load target values from the original dataset."""
    # Get dataset path
    base_path = Path("/Users/u6788552/Desktop/experiments/dmpnn")
    
    # Handle dataset name mapping
    dataset_mapping = {
        'opv_camb3lyp': 'cam_b3lyp',
        'cam_b3lyp': 'cam_b3lyp'
    }
    
    config_dataset_name = dataset_mapping.get(dataset_name, dataset_name)
    
    if config_dataset_name not in config['datasets']:
        print(f"Warning: Dataset '{dataset_name}' (mapped to '{config_dataset_name}') not found in configuration")
        # Return dummy data for visualization
        return np.random.randn(1000)
    
    dataset_file = config['datasets'][config_dataset_name]['file_path']
    data_path = base_path / dataset_file
    
    if not data_path.exists():
        print(f"Warning: Dataset file not found: {data_path}")
        return np.random.randn(1000)
    
    df = pd.read_csv(data_path)
    
    if target_name not in df.columns:
        print(f"Warning: Target '{target_name}' not found in dataset '{dataset_name}'")
        return np.random.randn(len(df))
    
    return df[target_name].values

def load_embeddings_and_targets(embedding_info: Dict[str, Any], config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings and targets for the first available split.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (from first split only)
    """
    from pathlib import Path
    import sys

    # Locate project root relative to THIS file
    THIS_FILE = Path(__file__).resolve()
    PROJ_ROOT = THIS_FILE.parents[1]  # adjust if depth differs
    # Expecting utils at proj_root/scripts/python/utils.py
    UTILS_DIR = PROJ_ROOT / "scripts" / "python"
    sys.path.insert(0, str(UTILS_DIR))
    import argparse
    from utils import generate_data_splits, determine_split_strategy
    parser = argparse.ArgumentParser(description="Load embeddings and targets for the first available split.")
    parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi', 'mixed-reg-multi'], 
                        default='reg', help='Type of task: regression, binary, multi-class, or mixed')

    # Load target data
    y_full = load_target_data(embedding_info['dataset'], embedding_info['target'], config)
    args = parser.parse_args()
    args.dataset_name = embedding_info['dataset']
    
    SEED = 42
    REPLICATES = 5
    n_splits, local_reps = determine_split_strategy(len(y_full), REPLICATES)
    if embedding_info['dataset'] == "polyinfo":
        args.task_type = "multi"
    train_indices, val_indices, test_indices = generate_data_splits(args, y_full, n_splits, local_reps, SEED)


    # --- only FIRST split ---
    split_info = embedding_info['splits'][0]
    X_train = np.load(split_info['train_file'])
    X_val   = np.load(split_info['val_file'])
    X_test  = np.load(split_info['test_file'])

    print(f"Using split {split_info['split_idx']}: "
          f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")


    y_train = y_full[train_indices[0]]
    y_val   = y_full[val_indices[0]]
    y_test  = y_full[test_indices[0]]

    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_pca_analysis(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                     dataset_name: str, model_name: str, target_name: str, variant: str,
                     output_dir: str) -> None:
    """Create PCA 2D scatter plots."""
    variant_suffix = f"_{variant}" if variant != "base" else ""
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_file = output_path / f"{dataset_name}_{model_name}_{target_name}{variant_suffix}_pca_analysis.png"
    if output_file.exists():
        return
    # Combine all data for full dataset plot
    X_full = np.vstack([X_train, X_val, X_test])
    y_full = np.concatenate([y_train, y_val, y_test])
    split_labels = (['train'] * len(X_train) + 
                   ['val'] * len(X_val) + 
                   ['test'] * len(X_test))
    
    # Standardize features
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_full_scaled)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name.upper()} - {model_name} - {target_name}\nPCA Analysis ({variant})', 
                fontsize=16, fontweight='bold')
    
    # Full dataset plot (colored by target)
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_full, cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('Full Dataset (colored by target)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=target_name)
    
    # Split-specific plots
    split_colors = {'train': 'blue', 'val': 'orange', 'test': 'red'}
    split_data = {
        'train': (X_pca[:len(X_train)], y_train),
        'val': (X_pca[len(X_train):len(X_train)+len(X_val)], y_val),
        'test': (X_pca[len(X_train)+len(X_val):], y_test)
    }
    
    for idx, (split_name, (X_split, y_split)) in enumerate(split_data.items()):
        row, col = (0, 1) if idx == 0 else (1, idx-1)
        ax = axes[row, col]
        
        scatter = ax.scatter(X_split[:, 0], X_split[:, 1], 
                           c=y_split, cmap='viridis', alpha=0.7, s=25)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.set_title(f'{split_name.capitalize()} Set')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label=target_name)
    
    # Add explained variance info
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    fig.text(0.02, 0.02, f'Total explained variance: {explained_var:.1f}%', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PCA analysis: {output_file}")
    plt.close()

def plot_umap_analysis(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                      dataset_name: str, model_name: str, target_name: str, variant: str,
                      output_dir: str) -> None:
    """Create UMAP 2D embedding plots."""
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    variant_suffix = f"_{variant}" if variant != "base" else ""
    output_file = output_path / f"{dataset_name}_{model_name}_{target_name}{variant_suffix}_umap_analysis.png"
    if output_file.exists():
        return
    # Combine all data
    X_full = np.vstack([X_train, X_val, X_test])
    y_full = np.concatenate([y_train, y_val, y_test])
    
    # Standardize features
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_full_scaled)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name.upper()} - {model_name} - {target_name}\nUMAP Analysis ({variant})', 
                fontsize=16, fontweight='bold')
    
    # Full dataset plot
    ax = axes[0, 0]
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y_full, cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Full Dataset (colored by target)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=target_name)
    
    # Split-specific plots
    split_data = {
        'train': X_umap[:len(X_train)],
        'val': X_umap[len(X_train):len(X_train)+len(X_val)],
        'test': X_umap[len(X_train)+len(X_val):]
    }
    split_targets = {'train': y_train, 'val': y_val, 'test': y_test}
    
    for idx, split_name in enumerate(['train', 'val', 'test']):
        row, col = (0, 1) if idx == 0 else (1, idx-1)
        ax = axes[row, col]
        
        X_split = split_data[split_name]
        y_split = split_targets[split_name]
        
        scatter = ax.scatter(X_split[:, 0], X_split[:, 1], 
                           c=y_split, cmap='viridis', alpha=0.7, s=25)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{split_name.capitalize()} Set')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label=target_name)
    
    plt.tight_layout()
    
    # Save plot
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP analysis: {output_file}")
    plt.close()

def plot_tsne_analysis(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                      dataset_name: str, model_name: str, target_name: str, variant: str,
                      output_dir: str) -> None:
    """Create t-SNE 2D embedding plots."""
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    variant_suffix = f"_{variant}" if variant != "base" else ""
    output_file = output_path / f"{dataset_name}_{model_name}_{target_name}{variant_suffix}_tsne_analysis.png"
    if output_file.exists():
        return

    # Combine all data
    X_full = np.vstack([X_train, X_val, X_test])
    y_full = np.concatenate([y_train, y_val, y_test])
    
    # Standardize features
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    
    # Apply t-SNE (limit samples if too large for performance)
    max_samples = 5000
    if len(X_full_scaled) > max_samples:
        print(f"Subsampling {max_samples} points for t-SNE analysis (from {len(X_full_scaled)})")
        indices = np.random.choice(len(X_full_scaled), max_samples, replace=False)
        X_tsne_input = X_full_scaled[indices]
        y_tsne_input = y_full[indices]
        
        # Track which split each sample belongs to
        split_labels = (['train'] * len(X_train) + 
                       ['val'] * len(X_val) + 
                       ['test'] * len(X_test))
        split_labels_sampled = [split_labels[i] for i in indices]
    else:
        X_tsne_input = X_full_scaled
        y_tsne_input = y_full
        split_labels_sampled = (['train'] * len(X_train) + 
                               ['val'] * len(X_val) + 
                               ['test'] * len(X_test))
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne_input)//4))
    X_tsne = tsne.fit_transform(X_tsne_input)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name.upper()} - {model_name} - {target_name}\nt-SNE Analysis ({variant})', 
                fontsize=16, fontweight='bold')
    
    # Full dataset plot
    ax = axes[0, 0]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne_input, cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Full Dataset (colored by target)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=target_name)
    
    # Split-specific plots
    split_colors = {'train': 'blue', 'val': 'orange', 'test': 'red'}
    
    for idx, split_name in enumerate(['train', 'val', 'test']):
        row, col = (0, 1) if idx == 0 else (1, idx-1)
        ax = axes[row, col]
        
        # Get indices for this split
        split_mask = np.array(split_labels_sampled) == split_name
        if np.sum(split_mask) == 0:
            ax.text(0.5, 0.5, f'No {split_name} samples', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{split_name.capitalize()} Set')
            continue
            
        X_split = X_tsne[split_mask]
        y_split = y_tsne_input[split_mask]
        
        scatter = ax.scatter(X_split[:, 0], X_split[:, 1], 
                           c=y_split, cmap='viridis', alpha=0.7, s=25)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f'{split_name.capitalize()} Set')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label=target_name)
    
    plt.tight_layout()
    
    # Save plot
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE analysis: {output_file}")
    plt.close()

def analyze_embedding_experiment(embedding_key: str, embedding_info: Dict[str, Any], 
                               config: Dict[str, Any], output_dir: str) -> None:
    """Analyze a single embedding experiment."""
    
    dataset_name = embedding_info['dataset']
    model_name = embedding_info['model']
    target_name = embedding_info['target']
    variant = embedding_info['variant']
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset_name} | {model_name} | {target_name} | {variant}")
    print(f"{'='*80}")
    
    try:
        # Load embeddings and targets
        print("Loading embeddings and target data...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_embeddings_and_targets(
            embedding_info, config
        )
        
        print(f"Loaded embeddings - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Check for valid target values
        if np.all(np.isnan(y_train)) or len(np.unique(y_train[~np.isnan(y_train)])) < 2:
            print("Warning: Insufficient valid target values, using dummy values for visualization")
            # Use embedding coordinates as proxy targets for visualization
            y_train = X_train[:, 0] if X_train.shape[1] > 0 else np.random.randn(len(X_train))
            y_val = X_val[:, 0] if X_val.shape[1] > 0 else np.random.randn(len(X_val))
            y_test = X_test[:, 0] if X_test.shape[1] > 0 else np.random.randn(len(X_test))
        
        # 1. PCA Analysis
        print("Creating PCA analysis...")
        plot_pca_analysis(X_train, X_val, X_test, y_train, y_val, y_test,
                         dataset_name, model_name, target_name, variant, output_dir)
        
        # 2. UMAP Analysis
        print("Creating UMAP analysis...")
        plot_umap_analysis(X_train, X_val, X_test, y_train, y_val, y_test,
                          dataset_name, model_name, target_name, variant, output_dir)
        
        # 3. t-SNE Analysis
        print("Creating t-SNE analysis...")
        plot_tsne_analysis(X_train, X_val, X_test, y_train, y_val, y_test,
                          dataset_name, model_name, target_name, variant, output_dir)
        
        print(f"✓ Completed analysis for {embedding_key}")
        
    except Exception as e:
        print(f"✗ Error analyzing {embedding_key}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to perform comprehensive graph feature space analysis."""
    
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = "/Users/u6788552/Desktop/experiments/dmpnn/analysis/dataset_config.yaml"
    config = load_config(config_path)
    
    # Define paths
    embeddings_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn/results/embeddings")
    output_dir = "plots/graph_feature_space_analysis"
    
    print("Graph Model Feature Space Analysis")
    print("="*80)
    print(f"Config file: {config_path}")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Output directory: {output_dir}")
    
    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        return
    
    # Discover available embeddings
    print("\nDiscovering available embeddings...")
    embeddings_info = discover_embeddings(embeddings_dir)
    
    # Exclude any variants that include a size token like "size500"
    embeddings_info = {
        k: v for k, v in embeddings_info.items()
        if not any(part.startswith('size') for part in v['variant'].split('__'))
    }

    if not embeddings_info:
        print("No embedding experiments found")
        return
    
    print(f"Found {len(embeddings_info)} complete embedding experiments:")
    for key, info in embeddings_info.items():
        print(f"  - {key} ({len(info['splits'])} splits)")
    
    # Analyze each embedding experiment
    for embedding_key, embedding_info in embeddings_info.items():
        analyze_embedding_experiment(embedding_key, embedding_info, config, output_dir)
    
    print(f"\n" + "="*80)
    print("Graph feature space analysis complete!")
    print(f"Plots saved in: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
