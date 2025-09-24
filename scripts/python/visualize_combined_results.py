#!/usr/bin/env python3
"""
Combined Results Visualization Script
Creates comparison plots between tabular and graph model results.

Usage: python3 scripts/python/visualize_combined_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse

def parse_filename(filename: str) -> tuple:
    """Parse CSV filename to extract dataset and feature information."""
    # Remove .csv extension
    base = filename.replace('.csv', '')
    
    # Handle tabular files (both with and without _tabular prefix)
    if '_descriptors_rdkit_ab' in base:
        dataset = base.replace('_descriptors_rdkit_ab', '')
        features = 'AB+Desc+RDKit'
    elif '_descriptors_rdkit' in base:
        dataset = base.replace('_descriptors_rdkit', '')
        features = 'AB+Desc+RDKit'
    elif '_descriptors_ab' in base:
        dataset = base.replace('_descriptors_ab', '')
        features = 'AB+Desc'
    elif '_rdkit_ab' in base:
        dataset = base.replace('_rdkit_ab', '')
        features = 'AB+RDKit'
    elif '_descriptors' in base:
        dataset = base.replace('_descriptors', '')
        features = 'AB+Desc'
    elif '_rdkit' in base:
        dataset = base.replace('_rdkit', '')
        features = 'AB+RDKit'
    elif '_ab' in base:
        dataset = base.replace('_ab', '')
        features = 'AB'
    elif '_tabular' in base:
        dataset = base.replace('_tabular', '')
        features = 'AB'
    else:
        # Handle basic dataset files
        dataset = base
        features = 'AB'
    
    return dataset, features

def parse_model_filename(filename: str, method: str) -> tuple:
    """Parse model CSV filename to extract dataset and feature information."""
    # Remove .csv extension and method suffix first
    base = filename.replace('.csv', '')
    
    if method == 'Graph':
        base = base.replace('_results', '')
    elif method == 'Baseline':
        base = base.replace('_baseline', '')
    
    # Handle batch normalization
    batch_norm = False
    if '_batch_norm' in base:
        base = base.replace('_batch_norm', '')
        batch_norm = True
    
    # Handle different feature combinations
    if '__desc__rdkit' in base:
        dataset = base.replace('__desc__rdkit', '')
        features = f'{method}+Desc+RDKit'
    elif '__desc' in base:
        dataset = base.replace('__desc', '')
        features = f'{method}+Desc'
    elif '__rdkit' in base:
        dataset = base.replace('__rdkit', '')
        features = f'{method}+RDKit'
    else:
        dataset = base
        features = method
    
    # Clean up any trailing underscores in dataset name
    dataset = dataset.rstrip('_')
    
    # Add batch norm to features if present
    if batch_norm and method in ['Graph', 'Baseline']:  # Only add (BN) for Graph and Baseline methods
        features = f"{features} (BN)"
    
    return dataset, features

def load_results_by_method(results_dir: Path, method: str) -> Dict[str, pd.DataFrame]:
    """Load results CSV files for a specific method (Graph, Baseline, or Tabular)."""
    results = {}
    
    if method in ['Graph', 'Baseline']:
        # First pass: collect all CSV files
        csv_files = []
        for model_name in ['DMPNN', 'wDMPNN', 'PPG']:
            model_dir = results_dir / model_name
            if not model_dir.exists():
                continue
                
            suffix = '_results.csv' if method == 'Graph' else '_baseline.csv'
            csv_files.extend(list(model_dir.glob(f"*{suffix}")))
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                dataset, features = parse_model_filename(csv_file.name, method)
                
                # Extract model name from path
                model_name = csv_file.parent.name
                
                df = pd.read_csv(csv_file)
                
                # Skip empty DataFrames
                if df.empty:
                    print(f"Warning: Empty CSV file: {csv_file}")
                    continue
                
                # Rename columns to match expected format
                if 'test/mae' in df.columns:
                    df = df.rename(columns={
                        'test/mae': 'mae', 
                        'test/r2': 'r2', 
                        'test/rmse': 'rmse'
                    })
                elif 'test/multiclass-accuracy' in df.columns:
                    df = df.rename(columns={
                        'test/multiclass-accuracy': 'acc', 
                        'test/multiclass-f1': 'f1_macro', 
                        'test/multiclass-roc': 'logloss'
                    })
                elif 'test/accuracy' in df.columns:
                    df = df.rename(columns={
                        'test/accuracy': 'acc', 
                        'test/f1': 'f1_macro', 
                        'test/roc_auc': 'logloss'
                    })
                
                # Add metadata
                df['dataset'] = dataset
                df['features'] = features
                df['method'] = f"{method}_{model_name}"  # Distinguish between DMPNN and wDMPNN
                
                # Add target column if missing (some wDMPNN files don't have it)
                if 'target' not in df.columns:
                    # For classification datasets, use a default target name
                    if any(col in df.columns for col in ['acc', 'f1_macro', 'logloss']):
                        df['target'] = 'Class'
                    else:
                        # For regression, try to infer from dataset name or use default
                        df['target'] = 'Target'
                
                if method == 'Graph':
                    df['model'] = model_name
                elif method == 'Baseline':
                    # For Baseline, combine encoder name with baseline model
                    df['encoder'] = model_name  # Store the encoder (DMPNN/wDMPNN)
                    df['baseline_model'] = df['model']  # Store original model (Linear/RF/XGB)
                    df['model'] = df['model'] + f'-{model_name}'  # Combine for unique identification
                
                # Convert MSE to RMSE if MSE column exists
                if 'mse' in df.columns:
                    df['rmse'] = np.sqrt(df['mse'])
                
                if dataset not in results:
                    results[dataset] = []
                results[dataset].append(df)
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                continue
    
    elif method == 'Tabular':
        # Load from tabular directory
        tabular_dir = results_dir / 'tabular'
        if tabular_dir.exists():
            for csv_file in tabular_dir.glob("*.csv"):
                dataset, features = parse_filename(csv_file.name)
                
                df = pd.read_csv(csv_file)
                df['dataset'] = dataset
                df['features'] = features
                df['method'] = method
                
                # Convert MSE to RMSE if MSE column exists
                if 'mse' in df.columns:
                    df['rmse'] = np.sqrt(df['mse'])
                
                if dataset not in results:
                    results[dataset] = []
                results[dataset].append(df)
        
        # Also check root directory for backward compatibility
        for csv_file in results_dir.glob("*_tabular*.csv"):
            dataset, features = parse_filename(csv_file.name)
            
            df = pd.read_csv(csv_file)
            df['dataset'] = dataset
            df['features'] = features
            df['method'] = method
            
            # Convert MSE to RMSE if MSE column exists
            if 'mse' in df.columns:
                df['rmse'] = np.sqrt(df['mse'])
            
            if dataset not in results:
                results[dataset] = []
            results[dataset].append(df)
    
    # Concatenate all feature combinations for each dataset
    for dataset in results:
        results[dataset] = pd.concat(results[dataset], ignore_index=True)
    
    return results

def load_combined_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load and combine tabular, graph, and baseline results."""
    tabular_results = load_results_by_method(results_dir, 'Tabular')
    graph_results = load_results_by_method(results_dir, 'Graph')
    baseline_results = load_results_by_method(results_dir, 'Baseline')
    
    # Combine results
    combined_results = {}
    all_datasets = set(tabular_results.keys()) | set(graph_results.keys()) | set(baseline_results.keys())
    
    for dataset in all_datasets:
        dataset_dfs = []
        
        if dataset in tabular_results:
            dataset_dfs.append(tabular_results[dataset])
        
        if dataset in graph_results:
            dataset_dfs.append(graph_results[dataset])
            
        if dataset in baseline_results:
            dataset_dfs.append(baseline_results[dataset])
        
        if dataset_dfs:
            combined_results[dataset] = pd.concat(dataset_dfs, ignore_index=True)
    
    return combined_results

def detect_task_type(data: pd.DataFrame) -> str:
    """Detect if dataset is regression or classification based on available columns."""
    columns = set(data.columns.str.lower())
    
    # Classification metrics
    if any(col in columns for col in ['acc', 'f1_macro', 'logloss', 'roc_auc', 'prec', 'rec']):
        return 'classification'
    # Regression metrics  
    elif any(col in columns for col in ['mae', 'r2', 'mse', 'rmse']):
        return 'regression'
    else:
        return 'unknown'

def get_metrics_for_task(task_type: str) -> List[str]:
    """Get appropriate metrics for task type."""
    if task_type == 'classification':
        return ['acc', 'f1_macro', 'logloss']
    elif task_type == 'regression':
        return ['mae', 'r2', 'rmse']
    else:
        return []

def create_combined_comparison_plots(data: pd.DataFrame, dataset: str, metric: str, output_dir: Path):
    """Create bar plots comparing both tabular and graph feature combinations for a specific metric."""
    
    # Get unique targets and feature combinations in desired order
    # Handle mixed data types (strings and NaN) by filtering out NaN values
    unique_targets = data['target'].dropna().unique()
    targets = sorted([str(t) for t in unique_targets])
    
    # Define desired feature order for combined plots, including batch norm variants
    base_features = [
        'AB', 'AB+RDKit', 'AB+Desc+RDKit',  # Tabular features
        'Graph', 'Graph+RDKit', 'Graph+Desc+RDKit',  # Graph features
        'Graph (BN)', 'Graph+RDKit (BN)', 'Graph+Desc+RDKit (BN)',  # Graph with batch norm
        'Baseline', 'Baseline+RDKit', 'Baseline+Desc+RDKit',  # Baseline features
        'Baseline (BN)', 'Baseline+RDKit (BN)', 'Baseline+Desc+RDKit (BN)'  # Baseline with batch norm
    ]
    
    # Get available features and sort them according to our desired order
    available_features = data['features'].unique()
    features = [f for f in base_features if f in available_features]
    
    # Check if metric exists in data
    if metric not in data.columns:
        print(f"Warning: Metric '{metric}' not found in {dataset} data. Skipping.")
        return
    
    # Calculate means across splits
    summary = data.groupby(['target', 'features', 'model', 'method'])[metric].agg(['mean', 'std']).reset_index()
    
    # Create subplots
    n_targets = len(targets)
    n_cols = min(3, n_targets)  # Max 3 columns
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 8*n_rows))
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_targets > 1 else axes
    
    # Define colors for different methods and models with distinct colors for batch norm variants
    colors = {
        'Tabular': {'Linear': '#1f77b4', 'RF': '#ff7f0e', 'XGB': '#2ca02c', 'LogReg': '#1f77b4'},
        'Graph_DMPNN': '#d62728',
        'Graph_wDMPNN': '#9467bd', 
        'Graph_PPG': '#8c564b',
        'Baseline_DMPNN': '#17becf',
        'Baseline_wDMPNN': '#bcbd22',
        'Baseline_PPG': '#e377c2'
    }
    
    for i, target in enumerate(targets):
        ax = axes_flat[i]
        
        # Filter data for this target
        target_data = summary[summary['target'] == target]
        
        # Create grouped bar plot
        x_pos = np.arange(len(features))
        
        # Get unique models and methods
        unique_combinations = target_data[['model', 'method']].drop_duplicates()
        
        # Calculate bar width and positions
        n_bars = len(unique_combinations)
        bar_width = 0.8 / n_bars if n_bars > 0 else 0.8
        
        for j, (_, row) in enumerate(unique_combinations.iterrows()):
            model = row['model']
            method = row['method']
            
            model_data = target_data[(target_data['model'] == model) & (target_data['method'] == method)]
            
            # Get color - handle both new flat structure and old nested structure
            if method == 'Tabular':
                color = colors.get('Tabular', {}).get(model, '#1f77b4')
            else:
                # For Graph_DMPNN, Baseline_wDMPNN, etc.
                color = colors.get(method, '#1f77b4')
            
            means = []
            stds = []
            for feature in features:
                feature_data = model_data[model_data['features'] == feature]
                if len(feature_data) > 0:
                    means.append(feature_data['mean'].iloc[0])
                    stds.append(feature_data['std'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Calculate bar positions
            bar_positions = x_pos + (j - n_bars/2 + 0.5) * bar_width
            
            ax.bar(bar_positions, means, bar_width, 
                  yerr=stds, capsize=3, label=f'{method}-{model}', alpha=0.8, color=color)
        
        ax.set_xlabel('Feature Combination')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{target}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_targets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'{dataset} - {metric.upper()} Combined Comparison (Tabular vs Graph)', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{dataset}_{metric}_combined_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize combined tabular and graph results')
    parser.add_argument('--results_dir', type=str, default=None, 
                       help='Directory containing result CSV files (optional, defaults to ../../results)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (optional, defaults to ../../plots/combined)')
    args = parser.parse_args()
    
    # Set up paths relative to script location
    script_dir = Path(__file__).parent
    
    # Default results_dir to ../../results relative to script
    if args.results_dir is None:
        results_dir = script_dir.parent.parent / "results"
    else:
        results_dir = Path(args.results_dir)
    
    # Default output_dir to ../../plots/combined relative to script  
    if args.output_dir is None:
        output_dir = script_dir.parent.parent / "plots" / "combined"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading combined results...")
    results = load_combined_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for datasets: {list(results.keys())}")
    
    # Process each dataset
    for dataset, data in results.items():
        print(f"\nProcessing {dataset}...")
        
        # Detect task type and get metrics
        task_type = detect_task_type(data)
        metrics = get_metrics_for_task(task_type)
        
        print(f"Detected task type: {task_type}")
        print(f"Using metrics: {metrics}")
        
        # Create comparison plots for each metric
        for metric in metrics:
            if metric in data.columns:
                create_combined_comparison_plots(data, dataset, metric, output_dir)
            else:
                print(f"Warning: Metric '{metric}' not found in {dataset} data. Skipping.")
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
