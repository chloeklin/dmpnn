#!/usr/bin/env python3
"""
Plot histograms of target values to visualize label diversity for specified datasets.

Uses YAML configuration to specify datasets and target columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import yaml
from typing import Dict, List, Any

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

def plot_target_histograms(dataset_name: str, config: Dict[str, Any], output_dir: str) -> None:
    """Plot histograms for all target columns in a dataset using config."""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load data
    data_path = get_data_path(dataset_name, config['datasets'])
    if not data_path.exists():
        print(f"Warning: Dataset file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Get target columns from config
    target_cols = get_target_columns(dataset_name, config['datasets'])
    print(f"Target columns from config: {target_cols}")
    
    # Validate that target columns exist in the dataset
    valid_targets = []
    for target in target_cols:
        if target not in df.columns:
            print(f"Warning: Target column '{target}' not found in dataset")
        elif df[target].dtype not in ['float64', 'int64', 'float32', 'int32']:
            print(f"Warning: Target column '{target}' is not numeric")
        elif df[target].nunique() < config['settings']['filter']['min_unique_values']:
            print(f"Warning: Target column '{target}' has insufficient unique values")
        else:
            valid_targets.append(target)
    
    if not valid_targets:
        print(f"No valid target columns found for {dataset_name}")
        return
    
    print(f"Found {len(valid_targets)} valid target columns: {valid_targets}")
    
    # Create output directory
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot settings
    plot_settings = config['settings']['plot']
    max_cols = plot_settings['max_cols_per_row']
    subplot_size = plot_settings['subplot_size']
    dpi = plot_settings['dpi']
    
    # Create combined plot
    n_targets = len(valid_targets)
    n_cols = min(max_cols, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(subplot_size[0]*n_cols, subplot_size[1]*n_rows))
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, target in enumerate(valid_targets):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Get non-null values
        values = df[target].dropna()
        
        if len(values) == 0:
            ax.text(0.5, 0.5, f"No data\nfor {target}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{target} (no data)")
            continue
        
        # Plot histogram
        n_bins = min(plot_settings['default_bins'], max(10, len(values) // 20))
        ax.hist(values, bins=n_bins, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        n_valid = len(values)
        cv = std_val / mean_val if mean_val != 0 else np.inf
        
        # Title with statistics
        title = f"{target}\n"
        title += f"n={n_valid}, μ={mean_val:.2e}, σ={std_val:.2e}\n"
        title += f"range=[{min_val:.2e}, {max_val:.2e}]"
        ax.set_title(title, fontsize=10)
        
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format scientific notation if needed
        if max_val - min_val > 1000 or abs(mean_val) < 0.01:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Hide empty subplots
    for i in range(n_targets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            fig.delaxes(axes[col])
        else:
            fig.delaxes(axes[row, col])
    
    plt.suptitle(f"Target Value Distribution - {dataset_name.upper()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save combined plot
    output_file = output_path / f"{dataset_name}_target_histograms.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved combined histogram plot: {output_file}")
    
    # Create individual plots if specified
    if plot_settings['create_individual_plots']:
        for target in valid_targets:
            values = df[target].dropna()
            if len(values) == 0:
                continue
                
            plt.figure(figsize=(8, 6))
            n_bins = min(plot_settings['default_bins'], max(10, len(values) // 20))
            plt.hist(values, bins=n_bins, alpha=0.7, edgecolor='black', color='steelblue')
            
            mean_val = values.mean()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            n_valid = len(values)
            
            plt.title(f"{dataset_name.upper()} - {target}\n"
                     f"n={n_valid}, μ={mean_val:.2e}, σ={std_val:.2e}, range=[{min_val:.2e}, {max_val:.2e}]")
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add statistics lines if specified
            if plot_settings['show_statistics']:
                plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2e}')
                plt.axvline(values.median(), color='orange', linestyle='--', alpha=0.8, label=f'Median: {values.median():.2e}')
                plt.legend()
            
            # Format scientific notation if needed
            if max_val - min_val > 1000 or abs(mean_val) < 0.01:
                plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            individual_output = output_path / f"{dataset_name}_{target}_histogram.png"
            plt.savefig(individual_output, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            print(f"  Individual plot saved: {individual_output}")
    
    plt.close()

def create_summary_table(datasets: List[str], config: Dict[str, Any], output_dir: str) -> pd.DataFrame:
    """Create a summary table of target statistics across all datasets."""
    summary_data = []
    max_cv = config['settings']['filter']['max_cv_threshold']
    
    for dataset in datasets:
        data_path = get_data_path(dataset, config['datasets'])
        if not data_path.exists():
            continue
            
        df = pd.read_csv(data_path)
        target_cols = get_target_columns(dataset, config['datasets'])
        
        for target in target_cols:
            if target not in df.columns:
                continue
                
            values = df[target].dropna()
            if len(values) == 0 or values.nunique() < config['settings']['filter']['min_unique_values']:
                continue
            
            mean_val = values.mean()
            std_val = values.std()
            cv = std_val / mean_val if mean_val != 0 else np.inf
            
            summary_data.append({
                'Dataset': dataset.upper(),
                'Target': target,
                'N_Samples': len(values),
                'Mean': mean_val,
                'Std': std_val,
                'Min': values.min(),
                'Max': values.max(),
                'Range': values.max() - values.min(),
                'CV': cv,
                'High_Variance': cv > max_cv
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary table if specified
        if config['settings']['summary']['create_summary']:
            output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            summary_file = output_path / f"target_summary_table.{config['settings']['summary']['format']}"
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSummary table saved: {summary_file}")
        
        return summary_df
    
    return None

def main():
    """Main function to generate histograms using configuration."""
    # Load configuration
    config_path = "/Users/u6788552/Desktop/experiments/dmpnn/analysis/dataset_config.yaml"
    config = load_config(config_path)
    
    # Define output directory for this script
    output_dir = "plots/target_histograms"
    
    print("Generating target value histograms using configuration:")
    print(f"Config file: {config_path}")
    print(f"Datasets to process: {list(config['datasets'].keys())}")
    print(f"Output directory: {output_dir}")
    
    # Process each dataset
    datasets = list(config['datasets'].keys())
    
    for dataset in datasets:
        try:
            plot_target_histograms(dataset, config, output_dir)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    # Create summary table
    try:
        summary_df = create_summary_table(datasets, config, output_dir)
        if summary_df is not None:
            print("\n" + "="*80)
            print("TARGET VALUE SUMMARY TABLE")
            print("="*80)
            print(summary_df.to_string(index=False, float_format='%.2e'))
    except Exception as e:
        print(f"Error creating summary table: {e}")
    
    print(f"\n" + "="*60)
    print("Histogram generation complete!")
    print(f"Plots saved in: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
