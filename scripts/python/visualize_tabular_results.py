#!/usr/bin/env python3
"""
Visualization script for tabular method results comparison.
Creates bar plots comparing different feature combinations across datasets.

Usage:
    python visualize_tabular_results.py --results_dir results/ --output_dir plots/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple

# Set style
plt.style.use('seaborn-v0_8')
# Use standard colors: blue, orange, green for the three models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # matplotlib default colors
sns.set_palette(colors)

def parse_filename(filename: str) -> Tuple[str, str]:
    """Parse dataset name and feature combination from filename."""
    # Expected format: {dataset}_tabular{_descriptors}{_rdkit}.csv
    base = filename.replace('.csv', '').replace('_tabular', '')
    
    if '_descriptors_rdkit' in filename:
        dataset = base.replace('_descriptors_rdkit', '')
        features = 'AB+Desc+RDKit'
    elif '_descriptors' in filename:
        dataset = base.replace('_descriptors', '')
        features = 'AB+Desc'
    elif '_rdkit' in filename:
        dataset = base.replace('_rdkit', '')
        features = 'AB+RDKit'
    else:
        dataset = base
        features = 'AB'
    
    return dataset, features

def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all tabular results CSV files."""
    results = {}
    
    for csv_file in results_dir.glob("*_tabular*.csv"):
        dataset, features = parse_filename(csv_file.name)
        
        df = pd.read_csv(csv_file)
        df['dataset'] = dataset
        df['features'] = features
        
        if dataset not in results:
            results[dataset] = []
        results[dataset].append(df)
    
    # Concatenate all feature combinations for each dataset
    for dataset in results:
        results[dataset] = pd.concat(results[dataset], ignore_index=True)
    
    return results

def detect_task_type(data: pd.DataFrame) -> str:
    """Detect if dataset is regression or classification based on available columns."""
    columns = set(data.columns.str.lower())
    
    # Classification metrics
    if any(col in columns for col in ['acc', 'f1_macro', 'logloss', 'roc_auc', 'prec', 'rec']):
        return 'classification'
    # Regression metrics  
    elif any(col in columns for col in ['mae', 'r2', 'rmse']):
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

def get_models_for_task(task_type: str) -> List[str]:
    """Get appropriate model names for task type."""
    if task_type == 'classification':
        return ['LogReg', 'RF', 'XGB']
    elif task_type == 'regression':
        return ['Linear', 'RF', 'XGB']
    else:
        return []

def create_comparison_plots(data: pd.DataFrame, dataset: str, metric: str, output_dir: Path):
    """Create bar plots comparing feature combinations for a specific metric."""
    
    # Detect task type and get appropriate models
    task_type = detect_task_type(data)
    models = get_models_for_task(task_type)
    
    # Get unique targets and feature combinations in desired order
    targets = sorted(data['target'].unique())
    
    # Define desired feature order
    feature_order = ['AB', 'AB+RDKit', 'AB+Desc+RDKit']
    available_features = data['features'].unique()
    features = [f for f in feature_order if f in available_features]
    
    # Check if metric exists in data
    if metric not in data.columns:
        print(f"Warning: Metric '{metric}' not found in {dataset} data. Skipping.")
        return
    
    # Calculate means across splits
    summary = data.groupby(['target', 'features', 'model'])[metric].agg(['mean', 'std']).reset_index()
    
    # Create subplots
    n_targets = len(targets)
    n_cols = min(3, n_targets)  # Max 3 columns
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_targets > 1 else axes
    
    for i, target in enumerate(targets):
        ax = axes_flat[i]
        
        # Filter data for this target
        target_data = summary[summary['target'] == target]
        
        # Create bar plot
        x_pos = np.arange(len(features))
        width = 0.25
        
        for j, model in enumerate(models):
            model_data = target_data[target_data['model'] == model]
            
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
            
            ax.bar(x_pos + j*width, means, width, 
                  yerr=stds, capsize=3, label=model, alpha=0.8)
        
        ax.set_xlabel('Feature Combination')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{target}')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_targets, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'{dataset} - {metric.upper()} Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{dataset}_{metric}_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def create_summary_table(data: pd.DataFrame, dataset: str, output_dir: Path):
    """Create summary table with best results for each target."""
    
    # Detect task type and get appropriate metrics
    task_type = detect_task_type(data)
    available_metrics = get_metrics_for_task(task_type)
    
    # Only use metrics that exist in the data
    metrics_to_use = {metric: ['mean', 'std'] for metric in available_metrics if metric in data.columns}
    
    if not metrics_to_use:
        print(f"Warning: No valid metrics found for {dataset}")
        return None
    
    # Calculate means across splits
    summary = data.groupby(['target', 'features', 'model']).agg(metrics_to_use).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Find best model for each target and feature combination
    best_results = []
    for target in summary['target'].unique():
        for features in summary['features'].unique():
            subset = summary[(summary['target'] == target) & 
                           (summary['features'] == features)]
            
            if len(subset) == 0:
                continue
            
            result_entry = {
                'target': target,
                'features': features
            }
            
            # Add best results for each available metric
            if task_type == 'regression':
                if 'r2_mean' in subset.columns:
                    best_r2_idx = subset['r2_mean'].idxmax()
                    best_r2 = subset.loc[best_r2_idx]
                    result_entry['best_r2_model'] = best_r2['model']
                    result_entry['best_r2'] = f"{best_r2['r2_mean']:.3f} ± {best_r2['r2_std']:.3f}"
                
                if 'mae_mean' in subset.columns:
                    best_mae_idx = subset['mae_mean'].idxmin()
                    best_mae = subset.loc[best_mae_idx]
                    result_entry['best_mae_model'] = best_mae['model']
                    result_entry['best_mae'] = f"{best_mae['mae_mean']:.4f} ± {best_mae['mae_std']:.4f}"
                    
            elif task_type == 'classification':
                if 'acc_mean' in subset.columns:
                    best_acc_idx = subset['acc_mean'].idxmax()
                    best_acc = subset.loc[best_acc_idx]
                    result_entry['best_acc_model'] = best_acc['model']
                    result_entry['best_acc'] = f"{best_acc['acc_mean']:.3f} ± {best_acc['acc_std']:.3f}"
                
                if 'f1_macro_mean' in subset.columns:
                    best_f1_idx = subset['f1_macro_mean'].idxmax()
                    best_f1 = subset.loc[best_f1_idx]
                    result_entry['best_f1_model'] = best_f1['model']
                    result_entry['best_f1'] = f"{best_f1['f1_macro_mean']:.3f} ± {best_f1['f1_macro_std']:.3f}"
            
            best_results.append(result_entry)
    
    best_df = pd.DataFrame(best_results)
    
    # Save summary table
    output_file = output_dir / f'{dataset}_summary.csv'
    best_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    return best_df

def main():
    parser = argparse.ArgumentParser(description='Visualize tabular method results')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Directory containing result CSV files')
    parser.add_argument('--output_dir', type=str, default='plots/tabular/',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No tabular results found!")
        return
    
    print(f"Found results for datasets: {list(results.keys())}")
    
    # Create plots for each dataset and metric
    for dataset, data in results.items():
        print(f"\nProcessing {dataset}...")
        
        # Detect task type and get appropriate metrics
        task_type = detect_task_type(data)
        metrics = get_metrics_for_task(task_type)
        
        print(f"Detected task type: {task_type}")
        print(f"Using metrics: {metrics}")
        
        # Create comparison plots for each metric
        for metric in metrics:
            create_comparison_plots(data, dataset, metric, output_dir)
        
        # Create summary table
        create_summary_table(data, dataset, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
