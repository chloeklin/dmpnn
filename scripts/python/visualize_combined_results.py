#!/usr/bin/env python3
"""
Combined Results Visualization Script
Creates comparison plots between tabular and graph model results.
Also exports consolidated CSV files with mean and std for all metrics.

Usage: python3 scripts/python/visualize_combined_results.py
       python3 scripts/python/visualize_combined_results.py --results_dir path/to/results --output_dir path/to/output
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
import sys

# Import combine_target_results function
try:
    from combine_target_results import combine_results
except ImportError:
    print("Error: Could not import combine_results from combine_target_results.py")
    print("Make sure combine_target_results.py is in the same directory.")
    sys.exit(1)

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

def parse_model_filename(filename: str, method: str, model_name: str = None) -> tuple:
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
    
    # For baseline methods, include the model name
    if method == 'Baseline' and model_name:
        method_name = f'Baseline_{model_name}'
    else:
        method_name = method
    
    # Handle different feature combinations
    if '__desc__rdkit' in base:
        dataset = base.replace('__desc__rdkit', '')
        features = f'{method_name}+Desc+RDKit'
    elif '__desc' in base:
        dataset = base.replace('__desc', '')
        features = f'{method_name}+Desc'
    elif '__rdkit' in base:
        dataset = base.replace('__rdkit', '')
        features = f'{method_name}+RDKit'
    else:
        dataset = base
        features = method_name
    
    # Clean up any trailing underscores in dataset name
    dataset = dataset.rstrip('_')
    
    # Add batch norm to features if present
    if batch_norm and method in ['Graph', 'Baseline']:  # Only add (BN) for Graph and Baseline methods
        features = f"{features} (BN)"
    
    return dataset, features

def apply_opv_target_filtering(results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Filter OPV datasets to only include specific targets."""
    # Define allowed targets for OPV dataset
    opv_allowed_targets = {
        'optical_lumo',
        'gap', 
        'homo',
        'lumo',
        'spectral_overlap',
        'delta_optical_lumo',
        'homo_extrapolated',
        'gap_extrapolated'
    }
    
    filtered_results = {}
    for dataset, df in results.items():
        # Check if this is an OPV dataset
        is_opv_dataset = 'opv' in dataset.lower()
        
        if is_opv_dataset and 'target' in df.columns:
            # Filter to only allowed targets
            original_targets = set(df['target'].unique())
            df_filtered = df[df['target'].isin(opv_allowed_targets)]
            filtered_targets = set(df_filtered['target'].unique())
            
            removed_targets = original_targets - filtered_targets
            if removed_targets:
                print(f"Filtered OPV dataset '{dataset}': removed targets {sorted(removed_targets)}")
                print(f"Kept targets: {sorted(filtered_targets)}")
            
            filtered_results[dataset] = df_filtered
        else:
            # Keep all targets for non-OPV datasets
            filtered_results[dataset] = df
    
    return filtered_results

def load_results_by_method(results_dir: Path, method: str) -> Dict[str, pd.DataFrame]:
    """Load results CSV files for a specific method (Graph, Baseline, or Tabular)."""
    results = {}
    
    if method in ['Graph', 'Baseline']:
        # First pass: collect all CSV files
        csv_files = []
        for model_name in ['DMPNN', 'wDMPNN', 'DMPNN_DiffPool', 'PPG', 'AttentiveFP']:
            model_dir = results_dir / model_name
            if not model_dir.exists():
                continue
                
            suffix = '_results.csv' if method == 'Graph' else '_baseline.csv'
            csv_files.extend(list(model_dir.glob(f"*{suffix}")))
        
        # Process each CSV file
        for csv_file in csv_files:
            # Skip files with __size in the name (learning curve experiments)
            if '__size' in csv_file.name:
                continue
                
            try:
                # Extract model name from path first
                model_name = csv_file.parent.name
                
                dataset, features = parse_model_filename(csv_file.name, method, model_name)
                
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
                elif 'MAE' in df.columns:
                    # Handle uppercase columns (e.g., from AttentiveFP)
                    df = df.rename(columns={
                        'MAE': 'mae',
                        'R2': 'r2',
                        'RMSE': 'rmse',
                        'MSE': 'mse'
                    })
                elif 'test/multiclass-accuracy' in df.columns:
                    df = df.rename(columns={
                        'test/multiclass-accuracy': 'acc', 
                        'test/multiclass-f1': 'f1_macro', 
                        'test/multiclass-roc': 'roc_auc'
                    })
                elif 'test/accuracy' in df.columns:
                    df = df.rename(columns={
                        'test/accuracy': 'acc', 
                        'test/f1': 'f1_macro', 
                        'test/roc_auc': 'roc_auc'
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
                    # For Baseline, include encoder name in method to distinguish between different baselines
                    df['encoder'] = model_name  # Store the encoder (DMPNN/wDMPNN)
                    df['baseline_model'] = df['model']  # Store original model (Linear/RF/XGB)
                    # Group by baseline model type, not encoder
                    df['model'] = df['model']  # Keep original baseline model name
                    df['method'] = f"Baseline_{model_name}"  # Include encoder name in method
                
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
                # Skip files with __size in the name (learning curve experiments)
                if '__size' in csv_file.name:
                    continue
                    
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
            # Skip files with __size in the name (learning curve experiments)
            if '__size' in csv_file.name:
                continue
                
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
    
    # Apply OPV target filtering
    results = apply_opv_target_filtering(results)
    
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
        return ['acc', 'f1_macro', 'logloss', 'roc_auc']  # Include both metrics
    elif task_type == 'regression':
        return ['mae', 'r2', 'rmse']
    else:
        return []

def export_consolidated_csv(data: pd.DataFrame, dataset: str, metrics: List[str], output_dir: Path):
    """Export consolidated CSV with mean and std for each metric across all feature combinations."""
    
    
    # Calculate summary statistics for all metrics
    agg_dict = {}
    for metric in metrics:
        if metric in data.columns:
            agg_dict[metric] = ['mean', 'std']
    
    if not agg_dict:
        print(f"Warning: No valid metrics found for {dataset}. Skipping CSV export.")
        return
    
    # Group by target, features, model, and method
    summary = data.groupby(['target', 'features', 'model', 'method']).agg(agg_dict).reset_index()
    
    # Flatten multi-level column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in summary.columns.values]
    
    # Sort by target and features for better readability
    summary = summary.sort_values(['target', 'features', 'method', 'model'])
    
    # Save to CSV
    output_file = output_dir / f'{dataset}_consolidated_results.csv'
    summary.to_csv(output_file, index=False)
    
    print(f"Saved consolidated CSV: {output_file}")
    
    return summary

def create_combined_comparison_plots_with_suffix(data: pd.DataFrame, dataset: str, metric: str, output_dir: Path, suffix: str = ''):
    """Create bar plots comparing both tabular and graph feature combinations for a specific metric."""
    _create_comparison_plots_internal(data, dataset, metric, output_dir, suffix)

def create_combined_comparison_plots(data: pd.DataFrame, dataset: str, metric: str, output_dir: Path):
    """Create bar plots comparing both tabular and graph feature combinations for a specific metric."""
    _create_comparison_plots_internal(data, dataset, metric, output_dir, '')

def _create_comparison_plots_internal(data: pd.DataFrame, dataset: str, metric: str, output_dir: Path, suffix: str = ''):
    """Internal function to create bar plots with optional filename suffix."""
    
    # Get unique targets and feature combinations in desired order
    # Handle mixed data types (strings and NaN) by filtering out NaN values
    unique_targets = data['target'].dropna().unique()
    targets = sorted([str(t) for t in unique_targets])
    
    # Define desired feature order for combined plots, including batch norm variants
    base_features = [
        'AB', 'AB+RDKit', 'AB+Desc+RDKit',  # Tabular features
        'Baseline_DMPNN', 'Baseline_DMPNN+RDKit', 'Baseline_DMPNN+Desc+RDKit',  # DMPNN Baseline features
        'Baseline_DMPNN (BN)', 'Baseline_DMPNN+RDKit (BN)', 'Baseline_DMPNN+Desc+RDKit (BN)',  # DMPNN Baseline with batch norm
        'Baseline_wDMPNN', 'Baseline_wDMPNN+RDKit', 'Baseline_wDMPNN+Desc+RDKit',  # wDMPNN Baseline features
        'Baseline_wDMPNN (BN)', 'Baseline_wDMPNN+RDKit (BN)', 'Baseline_wDMPNN+Desc+RDKit (BN)',  # wDMPNN Baseline with batch norm
        'Baseline_DMPNN_DiffPool', 'Baseline_DMPNN_DiffPool+RDKit', 'Baseline_DMPNN_DiffPool+Desc+RDKit',  # DMPNN_DiffPool Baseline
        'Baseline_PPG', 'Baseline_PPG+RDKit', 'Baseline_PPG+Desc+RDKit',  # PPG Baseline features
        'Baseline_AttentiveFP', 'Baseline_AttentiveFP+RDKit', 'Baseline_AttentiveFP+Desc+RDKit',  # AttentiveFP Baseline features
        'Graph', 'Graph+RDKit', 'Graph+Desc+RDKit',  # Graph features
        'Graph (BN)', 'Graph+RDKit (BN)', 'Graph+Desc+RDKit (BN)'  # Graph with batch norm
    ]
    
    # Get available features and sort them according to our desired order
    available_features = data['features'].unique()
    features = [f for f in base_features if f in available_features]
    
    # Check if metric exists in data
    if metric not in data.columns:
        print(f"Warning: Metric '{metric}' not found in {dataset} data. Skipping.")
        return
    
    # Check if we have any targets to plot
    if not targets:
        print(f"Warning: No allowed targets found in {dataset} data for metric '{metric}'. Skipping plot creation.")
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
        # Tabular models (keep existing colors)
        'Tabular': {'Linear': '#1f77b4', 'RF': '#ff7f0e', 'XGB': '#2ca02c', 'LogReg': '#1f77b4'},
        
        # Baseline models (different colors for different encoders)
        'Baseline_DMPNN': {'Linear': '#87CEEB', 'RF': '#FFB347', 'XGB': '#90EE90', 'LogReg': '#87CEEB'},
        'Baseline_wDMPNN': {'Linear': '#DDA0DD', 'RF': '#F0E68C', 'XGB': '#98FB98', 'LogReg': '#DDA0DD'},
        'Baseline_DMPNN_DiffPool': {'Linear': '#B0E0E6', 'RF': '#FFDAB9', 'XGB': '#AFEEEE', 'LogReg': '#B0E0E6'},
        'Baseline_PPG': {'Linear': '#D2B48C', 'RF': '#F5DEB3', 'XGB': '#E0FFFF', 'LogReg': '#D2B48C'},
        'Baseline_AttentiveFP': {'Linear': '#F0B27A', 'RF': '#F7DC6F', 'XGB': '#ABEBC6', 'LogReg': '#F0B27A'},
        
        # Graph models (keep existing colors)
        'Graph_DMPNN': '#d62728',
        'Graph_wDMPNN': '#9467bd', 
        'Graph_PPG': '#8c564b',
        'Graph_AttentiveFP': '#e377c2',
        'Graph_DMPNN_DiffPool': '#17becf'
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
            
            # Get color - handle tabular, baseline, and graph methods
            if method == 'Tabular':
                color = colors.get('Tabular', {}).get(model, '#1f77b4')
            elif method.startswith('Baseline_'):
                # For Baseline_DMPNN, Baseline_wDMPNN, etc.
                color = colors.get(method, {}).get(model, '#87CEEB')
            else:
                # For Graph_DMPNN, Graph_wDMPNN, etc.
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
    
    # Determine title based on what's included
    has_tabular = 'Tabular' in data['method'].unique()
    if suffix == '_graph_only' or not has_tabular:
        title = f'{dataset} - {metric.upper()} Comparison (Graph Models Only)'
    else:
        title = f'{dataset} - {metric.upper()} Combined Comparison (Tabular vs Graph)'
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{dataset}_{metric}_combined_comparison{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")


def _select_best_tabular_and_graph_models(data: pd.DataFrame, metric: str, task_type: str) -> pd.DataFrame:
    """Select best Tabular and best Graph models per target for a given metric.

    This looks across all feature combinations and models and picks, for each target:
      - the best Tabular entry (method == 'Tabular')
      - the best Graph entry (method starts with 'Graph_')

    The definition of "best" depends on task type and metric name:
      - Regression: lower is better for mae/rmse/mse, higher is better for r2
      - Classification: higher is better for acc/f1_macro/roc_auc, lower is better for logloss
    """

    if metric not in data.columns:
        return pd.DataFrame()

    # Aggregate over splits first (keep mean and std so we can show variance)
    grouped = (
        data.groupby(['target', 'features', 'model', 'method'])[metric]
        .agg(['mean', 'std'])
        .reset_index()
    )
    # Helper to decide if higher is better for this metric
    metric_lower_is_better = {
        'mae': True,
        'rmse': True,
        'mse': True,
        'logloss': True,
    }

    metric_higher_is_better = {
        'r2': True,
        'acc': True,
        'f1_macro': True,
        'roc_auc': True,
    }

    if metric in metric_lower_is_better:
        best_fn = lambda x: x.nsmallest(1, 'mean')
    elif metric in metric_higher_is_better:
        best_fn = lambda x: x.nlargest(1, 'mean')
    else:
        # Fallback: use lower-is-better for unknown regression metrics,
        # higher-is-better for unknown classification metrics.
        if task_type == 'classification':
            best_fn = lambda x: x.nlargest(1, 'mean')
        else:
            best_fn = lambda x: x.nsmallest(1, 'mean')

    records = []

    for target, target_df in grouped.groupby('target'):
        # Best Tabular
        tab_df = target_df[target_df['method'] == 'Tabular']
        if not tab_df.empty:
            best_tab = best_fn(tab_df).iloc[0]
            records.append({
                'target': target,
                'method_group': 'Tabular',
                'method': best_tab['method'],
                'model': best_tab['model'],
                'features': best_tab['features'],
                'metric_mean': best_tab['mean'],
                'metric_std': best_tab['std'],
            })

        # Best Graph (methods like Graph_DMPNN, Graph_wDMPNN, etc.)
        graph_df = target_df[target_df['method'].astype(str).str.startswith('Graph_')]
        if not graph_df.empty:
            best_graph = best_fn(graph_df).iloc[0]
            records.append({
                'target': target,
                'method_group': 'Graph',
                'method': best_graph['method'],
                'model': best_graph['model'],
                'features': best_graph['features'],
                'metric_mean': best_graph['mean'],
                'metric_std': best_graph['std'],
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def create_best_model_comparison_plots(data: pd.DataFrame, dataset: str, metric: str, task_type: str, output_dir: Path):
    """Create simple plots comparing best Tabular vs best Graph models per target.

    Each plot shows, for a given dataset and metric, bars for the best Tabular
    model and the best Graph model for each target (where available).
    """

    best_df = _select_best_tabular_and_graph_models(data, metric, task_type)
    if best_df.empty:
        print(f"Warning: No best-model data available for {dataset} and metric '{metric}'. Skipping best-model plot.")
        return

    targets = sorted(best_df['target'].astype(str).unique())
    n_targets = len(targets)
    x = np.arange(n_targets)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_targets * 1.2), 6))

    # Prepare values and stds for Tabular and Graph, and legend labels
    tab_values = []
    graph_values = []
    tab_stds = []
    graph_stds = []
    tab_labels = []
    graph_labels = []
    for t in targets:
        t_df = best_df[best_df['target'].astype(str) == t]

        tab_row = t_df[t_df['method_group'] == 'Tabular']
        graph_row = t_df[t_df['method_group'] == 'Graph']

        if not tab_row.empty:
            tab_values.append(tab_row['metric_mean'].iloc[0])
            tab_stds.append(tab_row['metric_std'].iloc[0])
            tab_labels.append(f"Tabular") #- {tab_row['model'].iloc[0]} ({tab_row['features'].iloc[0]})
        else:
            tab_values.append(np.nan)
            tab_stds.append(np.nan)
            tab_labels.append("")

        if not graph_row.empty:
            graph_values.append(graph_row['metric_mean'].iloc[0])
            graph_stds.append(graph_row['metric_std'].iloc[0])
            graph_labels.append(f"Graph") #{graph_row['method'].iloc[0]}
        else:
            graph_values.append(np.nan)
            graph_stds.append(np.nan)
            graph_labels.append("")

    tab_vals = np.array(tab_values, dtype=float)
    graph_vals = np.array(graph_values, dtype=float)
    tab_err = np.array(tab_stds, dtype=float)
    graph_err = np.array(graph_stds, dtype=float)

    # Colors consistent with other plots
    tab_color = '#1f77b4'
    graph_color = '#d62728'

    # Use error bars to show variance across splits
    tab_bars = ax.bar(
        x - width/2,
        tab_vals,
        width,
        yerr=tab_err,
        capsize=3,
        label=None,
        color=tab_color,
    )
    graph_bars = ax.bar(
        x + width/2,
        graph_vals,
        width,
        yerr=graph_err,
        capsize=3,
        label=None,
        color=graph_color,
    )

    # Build legend entries from the actual model/method names
    legend_entries = []
    legend_labels = []

    # Tabular legend: use the (unique) set of non-empty labels
    for lbl in sorted(set(l for l in tab_labels if l)):
        legend_entries.append(
            plt.Line2D([0], [0], marker='s', color='w', label=lbl, markerfacecolor=tab_color, markersize=10)
        )
        legend_labels.append(lbl)

    # Graph legend: use the (unique) set of non-empty labels
    for lbl in sorted(set(l for l in graph_labels if l)):
        legend_entries.append(
            plt.Line2D([0], [0], marker='s', color='w', label=lbl, markerfacecolor=graph_color, markersize=10)
        )
        legend_labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{dataset} - Best Tabular vs Best Graph ({metric.upper()})')
    if legend_entries:
        ax.legend(legend_entries, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()

    output_file = output_dir / f'{dataset}_{metric}_best_tabular_vs_graph.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved best-model comparison plot: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize combined tabular and graph results')
    parser.add_argument('--results_dir', type=str, default=None, 
                       help='Directory containing result CSV files (optional, defaults to ../../results)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (optional, defaults to ../../plots/combined)')
    parser.add_argument('--exclude-tabular', action='store_true',
                       help='Exclude tabular baseline models from plots (useful when they have very different scales)')
    parser.add_argument('--exclude-models', type=str, nargs='+', default=[],
                       help='Exclude specific models from plots (e.g., --exclude-models Linear LogReg)')
    parser.add_argument('--dataset', type=str, nargs='+', default=[],
                       help='Only process specific datasets (e.g., --dataset tc insulator)')
    parser.add_argument('--best-tabular-graph-only', action='store_true',
                       help='Only plot the best Tabular model vs the best Graph model per dataset/target/metric')
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
        base_output_dir = script_dir.parent.parent / "plots" / "combined"
    else:
        base_output_dir = Path(args.output_dir)
    
    # Create subdirectory based on filtering flags for organization
    subdirs = []
    if args.exclude_tabular:
        subdirs.append("no_tabular")
    if args.exclude_models:
        # Create safe directory name from excluded models
        excluded_str = "_".join(args.exclude_models).replace("/", "_").replace(" ", "_")
        subdirs.append(f"exclude_{excluded_str}")
    if args.dataset:
        # Create directory name for specific datasets
        datasets_str = "_".join(args.dataset)
        subdirs.append(f"datasets_{datasets_str}")
    if args.best_tabular_graph_only:
        subdirs.append("best_tabular_vs_graph")
    
    # Build final output directory with subdirectories
    if subdirs:
        output_dir = base_output_dir / "_".join(subdirs)
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, combine target-specific results into single files
    print("Combining target-specific result files...")
    
    # List of model directories to process
    model_dirs = ['DMPNN', 'wDMPNN', 'DMPNN_DiffPool', 'PPG', 'AttentiveFP']
    
    # Combine results in each model subdirectory first
    for model_name in model_dirs:
        model_dir = results_dir / model_name
        if model_dir.exists():
            print(f"  Processing {model_name} results...")
            try:
                combine_results(str(model_dir))
                print(f"  ✅ {model_name} results combined successfully!")
            except Exception as e:
                print(f"  Warning: Could not combine {model_name} results: {e}")
    
    # Also combine any results in the main results directory
    try:
        combine_results(str(results_dir))
        print("✅ Main directory results combined successfully!")
    except Exception as e:
        print(f"Warning: Could not combine main directory results: {e}")
        print("Continuing with existing combined files...")
    
    print("Loading combined results...")
    results = load_combined_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for datasets: {list(results.keys())}")
    
    # Filter datasets if specified
    if args.dataset:
        results = {k: v for k, v in results.items() if k in args.dataset}
        if not results:
            print(f"No results found for specified datasets: {args.dataset}")
            return
        print(f"Processing only: {list(results.keys())}")
    
    # Process each dataset
    for dataset, data in results.items():
        print(f"\nProcessing {dataset}...")
        
        # Detect task type and get metrics
        task_type = detect_task_type(data)
        metrics = get_metrics_for_task(task_type)
        
        print(f"Detected task type: {task_type}")
        print(f"Using metrics: {metrics}")
        
        # Export consolidated CSV (always include all data)
        export_consolidated_csv(data, dataset, metrics, output_dir)
        
        # Check if tabular models exist
        has_tabular = 'Tabular' in data['method'].unique()
        
        # Create comparison plots for each metric
        for metric in metrics:
            if metric in data.columns:
                if args.best_tabular_graph_only:
                    # Only create simplified best Tabular vs best Graph plots
                    create_best_model_comparison_plots(data, dataset, metric, task_type, output_dir)
                else:
                    # Apply exclusions
                    plot_data = data.copy()

                    # Exclude entire tabular method if flag is set
                    if args.exclude_tabular:
                        plot_data = plot_data[plot_data['method'] != 'Tabular']

                    # Exclude specific models
                    if args.exclude_models:
                        plot_data = plot_data[~plot_data['model'].isin(args.exclude_models)]

                    # Create main plot with exclusions applied
                    create_combined_comparison_plots(plot_data, dataset, metric, output_dir)

                    # If tabular exists and not excluded, also create a graph-only plot for better visibility
                    if has_tabular and not args.exclude_tabular and not args.exclude_models:
                        graph_only_data = data[data['method'] != 'Tabular']
                        if not graph_only_data.empty:
                            # Save with different filename
                            create_combined_comparison_plots_with_suffix(graph_only_data, dataset, metric, output_dir, '_graph_only')
            else:
                print(f"Warning: Metric '{metric}' not found in {dataset} data. Skipping.")
    
    print(f"\nAll plots and consolidated CSVs saved to: {output_dir}")

if __name__ == "__main__":
    main()
