#!/usr/bin/env python3
"""
Plot heatmaps of Pearson correlation coefficients between targets for datasets with multiple targets.

Uses YAML configuration to specify datasets and target columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from typing import Dict, List, Any
from scipy.stats import pearsonr

# Set style for better plots
plt.style.use('default')
sns.set_palette("coolwarm")

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

def calculate_correlation_matrix(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """Calculate Pearson correlation matrix for target columns."""
    # Extract target columns and drop rows with any NaN values
    target_data = df[target_cols].dropna()
    
    if len(target_data) == 0:
        return pd.DataFrame()
    
    # Calculate correlation matrix
    corr_matrix = target_data.corr(method='pearson')
    
    return corr_matrix

def calculate_p_values(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """Calculate p-values for Pearson correlations."""
    # Extract target columns and drop rows with any NaN values
    target_data = df[target_cols].dropna()
    
    if len(target_data) == 0:
        return pd.DataFrame()
    
    # Calculate p-values matrix
    n_targets = len(target_cols)
    p_values = pd.DataFrame(index=target_cols, columns=target_cols, dtype=float)
    
    for i, col1 in enumerate(target_cols):
        for j, col2 in enumerate(target_cols):
            if i == j:
                p_values.iloc[i, j] = 0.0
            else:
                _, p_val = pearsonr(target_data[col1], target_data[col2])
                p_values.iloc[i, j] = p_val
    
    return p_values

def plot_correlation_heatmap(dataset_name: str, config: Dict[str, Any], output_dir: str) -> None:
    """Plot correlation heatmap for a dataset with multiple targets."""
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
    
    if len(valid_targets) < 2:
        print(f"Dataset {dataset_name} has fewer than 2 valid targets. Skipping correlation analysis.")
        return
    
    print(f"Found {len(valid_targets)} valid target columns: {valid_targets}")
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df, valid_targets)
    if corr_matrix.empty:
        print(f"No valid data for correlation analysis in {dataset_name}")
        return
    
    # Calculate p-values
    p_values = calculate_p_values(df, valid_targets)
    
    # Create output directory
    output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot settings
    plot_settings = config['settings']['plot']
    dpi = plot_settings['dpi']
    
    # Create heatmap
    plt.figure(figsize=(max(8, len(valid_targets) * 1.2), max(6, len(valid_targets) * 1.0)))
    
    # Create mask for upper triangle (optional, for cleaner look)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap with annotations
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 10})
    
    plt.title(f'Pearson Correlation Matrix - {dataset_name.upper()}\n'
              f'Targets: {", ".join(valid_targets)}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save correlation heatmap
    corr_output = output_path / f"{dataset_name}_correlation_heatmap.png"
    plt.savefig(corr_output, dpi=dpi, bbox_inches='tight')
    print(f"Saved correlation heatmap: {corr_output}")
    
    # Create p-value heatmap
    plt.figure(figsize=(max(8, len(valid_targets) * 1.2), max(6, len(valid_targets) * 1.0)))
    
    # Create mask for diagonal (p-values are always 0 on diagonal)
    p_mask = np.eye(len(p_values), dtype=bool)
    
    # Create p-value heatmap with significance threshold
    sns.heatmap(p_values, 
                mask=p_mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0.05,
                square=True, 
                fmt='.3f',
                cbar_kws={'shrink': 0.8, 'label': 'p-value'},
                annot_kws={'size': 10})
    
    plt.title(f'Correlation P-Values - {dataset_name.upper()}\n'
              f'Targets: {", ".join(valid_targets)}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save p-value heatmap
    pval_output = output_path / f"{dataset_name}_pvalues_heatmap.png"
    plt.savefig(pval_output, dpi=dpi, bbox_inches='tight')
    print(f"Saved p-value heatmap: {pval_output}")
    
    # Create detailed correlation table
    correlation_data = []
    for i, col1 in enumerate(valid_targets):
        for j, col2 in enumerate(valid_targets):
            if i < j:  # Only include upper triangle (avoid duplicates)
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_values.iloc[i, j]
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                correlation_data.append({
                    'Dataset': dataset_name.upper(),
                    'Target_1': col1,
                    'Target_2': col2,
                    'Correlation': corr_val,
                    'P_Value': p_val,
                    'Significance': significance,
                    'Significant': p_val < 0.05
                })
    
    plt.close('all')  # Close all figures
    
    return correlation_data

def create_correlation_summary(datasets: List[str], config: Dict[str, Any], output_dir: str) -> pd.DataFrame:
    """Create a summary table of all significant correlations across datasets."""
    all_correlations = []
    
    for dataset in datasets:
        try:
            correlation_data = plot_correlation_heatmap(dataset, config, output_dir)
            if correlation_data:
                all_correlations.extend(correlation_data)
        except Exception as e:
            print(f"Error processing correlations for {dataset}: {e}")
    
    if all_correlations:
        summary_df = pd.DataFrame(all_correlations)
        
        # Save summary table
        output_path = Path("/Users/u6788552/Desktop/experiments/dmpnn") / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_path / f"correlation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nCorrelation summary saved: {summary_file}")
        
        return summary_df
    
    return None

def main():
    """Main function to generate correlation heatmaps using configuration."""
    # Load configuration
    config_path = "/Users/u6788552/Desktop/experiments/dmpnn/analysis/dataset_config.yaml"
    config = load_config(config_path)
    
    # Define output directory for this script
    output_dir = "plots/correlation_heatmaps"
    
    print("Generating correlation heatmaps using configuration:")
    print(f"Config file: {config_path}")
    print(f"Datasets to process: {list(config['datasets'].keys())}")
    print(f"Output directory: {output_dir}")
    
    # Process each dataset
    datasets = list(config['datasets'].keys())
    
    # Filter datasets that have multiple targets
    multi_target_datasets = []
    for dataset in datasets:
        target_cols = get_target_columns(dataset, config['datasets'])
        if len(target_cols) >= 2:
            multi_target_datasets.append(dataset)
        else:
            print(f"\nSkipping {dataset}: only {len(target_cols)} target(s) found")
    
    if not multi_target_datasets:
        print("\nNo datasets with multiple targets found for correlation analysis.")
        return
    
    print(f"\nDatasets with multiple targets: {multi_target_datasets}")
    
    # Create correlation summary
    try:
        summary_df = create_correlation_summary(multi_target_datasets, config, output_dir)
        if summary_df is not None:
            print("\n" + "="*80)
            print("CORRELATION SUMMARY TABLE")
            print("="*80)
            
            # Show significant correlations only
            significant_corr = summary_df[summary_df['Significant']].copy()
            if len(significant_corr) > 0:
                print(significant_corr.to_string(index=False, float_format='%.3f'))
                
                print(f"\nSummary Statistics:")
                print(f"Total correlations analyzed: {len(summary_df)}")
                print(f"Significant correlations (p < 0.05): {len(significant_corr)}")
                print(f"Highly significant (p < 0.001): {len(significant_corr[significant_corr['P_Value'] < 0.001])}")
                
                # Show strongest correlations
                print(f"\nStrongest correlations (|r| > 0.5):")
                strong_corr = significant_corr[abs(significant_corr['Correlation']) > 0.5]
                if len(strong_corr) > 0:
                    print(strong_corr.sort_values('Correlation', key=abs, ascending=False).to_string(index=False, float_format='%.3f'))
                else:
                    print("No strong correlations found (|r| > 0.5)")
            else:
                print("No significant correlations found (p < 0.05)")
                
    except Exception as e:
        print(f"Error creating correlation summary: {e}")
    
    print(f"\n" + "="*60)
    print("Correlation heatmap generation complete!")
    print(f"Plots saved in: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
