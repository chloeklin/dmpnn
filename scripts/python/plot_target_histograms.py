#!/usr/bin/env python3
"""
Plot histograms of target values to visualize label diversity for specified datasets.

Datasets: tc, htpmd, insulator, cam_b3lyp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def get_data_path(dataset_name):
    """Get the path to the dataset CSV file."""
    base_path = Path("/Users/u6788552/Desktop/experiments/dmpnn/data")
    
    dataset_paths = {
        "tc": base_path / "tc.csv",
        "htpmd": base_path / "htpmd.csv", 
        "insulator": base_path / "insulator.csv",
        "cam_b3lyp": base_path / "opv_camb3lyp.csv"
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_paths[dataset_name]

def identify_target_columns(df, dataset_name):
    """Identify target columns in the dataset."""
    # Common columns to exclude
    exclude_cols = {'smiles', 'smi', 'SMILES', 'molecule_id', 'id', 'index', 
                   'fracA', 'fracB', 'frac_A', 'frac_B', 'A', 'B', 'sA', 'sB',
                   'dataset_source', 'Molecule', 'InChI', 'RDKit', 'WDMPNN_Input'}
    
    # Dataset-specific target identification
    if dataset_name == "tc":
        # TC dataset has TC as the main target column
        target_cols = ['TC'] if 'TC' in df.columns else []
    elif dataset_name == "htpmd":
        # htpmd dataset has multiple property targets
        potential_targets = ['Conductivity', 'TFSI Diffusivity', 'Li Diffusivity', 'Poly Diffusivity', 'Transference Number']
        target_cols = [col for col in potential_targets if col in df.columns]
    elif dataset_name == "insulator":
        # insulator dataset has bandgap_chain as target
        target_cols = ['bandgap_chain'] if 'bandgap_chain' in df.columns else []
    elif dataset_name == "cam_b3lyp":
        # cam_b3lyp dataset (opv_camb3lyp) has multiple electronic property targets
        potential_targets = ['optical_lumo', 'gap', 'homo', 'lumo', 'spectral_overlap', 
                           'delta_homo', 'delta_lumo', 'delta_optical_lumo',
                           'homo_extrapolated', 'lumo_extrapolated', 'gap_extrapolated', 
                           'optical_lumo_extrapolated']
        target_cols = [col for col in potential_targets if col in df.columns]
    else:
        # Generic approach: find numeric columns that aren't excluded
        target_cols = []
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                # Check if column has reasonable variation (not all same values)
                if df[col].nunique() > 1:
                    target_cols.append(col)
    
    # Filter out columns that don't have variation
    valid_targets = []
    for col in target_cols:
        if df[col].nunique() > 1:
            valid_targets.append(col)
    
    return valid_targets

def plot_target_histograms(dataset_name):
    """Plot histograms for all target columns in a dataset."""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load data
    data_path = get_data_path(dataset_name)
    if not data_path.exists():
        print(f"Warning: Dataset file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Identify target columns
    target_cols = identify_target_columns(df, dataset_name)
    print(f"Found {len(target_cols)} target columns: {target_cols}")
    
    if not target_cols:
        print(f"No target columns found for {dataset_name}")
        return
    
    # Create output directory
    output_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn/plots/target_histograms")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot histograms for each target
    n_targets = len(target_cols)
    n_cols = min(3, n_targets)  # Max 3 columns per row
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, target in enumerate(target_cols):
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
        n_bins = min(50, max(10, len(values) // 20))
        ax.hist(values, bins=n_bins, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        n_valid = len(values)
        
        # Title with statistics
        title = f"{target}\n"
        title += f"n={n_valid}, μ={mean_val:.2e}, σ={std_val:.2e}\n"
        title += f"range=[{min_val:.2e}, {max_val:.2e}]"
        ax.set_title(title, fontsize=10)
        
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        if max_val - min_val > 1000:
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
    
    # Save plot
    output_file = output_dir / f"{dataset_name}_target_histograms.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot: {output_file}")
    
    # Also create individual plots for better visibility
    for target in target_cols:
        values = df[target].dropna()
        if len(values) == 0:
            continue
            
        plt.figure(figsize=(8, 6))
        n_bins = min(50, max(10, len(values) // 20))
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
        
        # Add vertical lines for mean and median
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2e}')
        plt.axvline(values.median(), color='orange', linestyle='--', alpha=0.8, label=f'Median: {values.median():.2e}')
        plt.legend()
        
        # Format scientific notation if needed
        if max_val - min_val > 1000:
            plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        individual_output = output_dir / f"{dataset_name}_{target}_histogram.png"
        plt.savefig(individual_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Individual plot saved: {individual_output}")
    
    plt.close()

def create_summary_table():
    """Create a summary table of target statistics across all datasets."""
    datasets = ["tc", "htpmd", "insulator", "cam_b3lyp"]
    summary_data = []
    
    for dataset in datasets:
        data_path = get_data_path(dataset)
        if not data_path.exists():
            continue
            
        df = pd.read_csv(data_path)
        target_cols = identify_target_columns(df, dataset)
        
        for target in target_cols:
            values = df[target].dropna()
            if len(values) == 0:
                continue
                
            summary_data.append({
                'Dataset': dataset.upper(),
                'Target': target,
                'N_Samples': len(values),
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Range': values.max() - values.min(),
                'CV': values.std() / values.mean() if values.mean() != 0 else np.inf
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary table
        output_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn/plots/target_histograms")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_dir / "target_summary_table.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary table saved: {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("TARGET VALUE SUMMARY TABLE")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.2e'))
        
        return summary_df
    
    return None

def main():
    """Main function to generate histograms for all datasets."""
    print("Generating target value histograms for datasets: tc, htpmd, insulator, cam_b3lyp")
    
    # Create plots for each dataset
    datasets = ["tc", "htpmd", "insulator", "cam_b3lyp"]
    
    for dataset in datasets:
        try:
            plot_target_histograms(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    # Create summary table
    try:
        create_summary_table()
    except Exception as e:
        print(f"Error creating summary table: {e}")
    
    print("\n" + "="*60)
    print("Histogram generation complete!")
    print("Plots saved in: plots/target_histograms/")
    print("="*60)

if __name__ == "__main__":
    main()
