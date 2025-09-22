#!/usr/bin/env python3
"""
Plot Learning Curves for OPV Dataset DMPNN Results

This script analyzes DMPNN results on the OPV dataset across different training sizes
and creates learning curve plots showing how performance metrics (MAE, R², RMSE) 
vary with training set size.

Usage:
    python plot_opv_learning_curves.py
    
The script will:
1. Load all OPV result files from results/DMPNN/
2. Extract training sizes and performance metrics
3. Create separate plots for each variant (original, RDKit)
4. Show error bars representing standard deviation across 5 replicates
5. Save plots as high-quality PNG files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_train_size_from_filename(filename):
    """Extract training size from filename. Returns 'full' for files without size suffix."""
    if '__size' in filename:
        match = re.search(r'__size(\d+)', filename)
        if match:
            return int(match.group(1))
    return 'full'

def extract_variant_from_filename(filename):
    """Extract variant info (rdkit, batch_norm, etc.) from filename."""
    variants = []
    if '__rdkit' in filename:
        variants.append('rdkit')
    if '__batch_norm' in filename:
        variants.append('batch_norm')
    return '_'.join(variants) if variants else 'original'

def load_opv_results(results_dir):
    """Load all OPV DMPNN result files and organize by variant and training size."""
    results_dir = Path(results_dir)
    
    # Find all OPV result files (excluding baseline files)
    opv_files = list(results_dir.glob('opv_camb3lyp*_results.csv'))
    
    print(f"Found {len(opv_files)} OPV result files:")
    for f in sorted(opv_files):
        print(f"  - {f.name}")
    
    # Organize results by variant and training size
    results_data = defaultdict(dict)  # {variant: {train_size: DataFrame}}
    
    # Track which targets are available for size-specific vs full datasets
    size_specific_targets = set()
    full_targets = set()
    
    for file_path in opv_files:
        filename = file_path.name
        train_size = extract_train_size_from_filename(filename)
        variant = extract_variant_from_filename(filename)
        
        try:
            df = pd.read_csv(file_path)
            results_data[variant][train_size] = df
            
            # Track targets
            targets = set(df['target'].unique())
            if train_size == 'full':
                full_targets.update(targets)
            else:
                size_specific_targets.update(targets)
            
            print(f"Loaded {filename}: variant='{variant}', train_size={train_size}, shape={df.shape}, targets={len(targets)}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"\nTarget analysis:")
    print(f"  - Full dataset targets ({len(full_targets)}): {sorted(full_targets)}")
    print(f"  - Size-specific targets ({len(size_specific_targets)}): {sorted(size_specific_targets)}")
    print(f"  - Common targets ({len(size_specific_targets & full_targets)}): {sorted(size_specific_targets & full_targets)}")
    
    return results_data, size_specific_targets

def calculate_metrics_summary(results_data, common_targets):
    """Calculate mean and std for each metric across replicates, only for common targets."""
    summary_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # {variant: {train_size: {target: {metric: (mean, std)}}}}
    
    for variant, size_data in results_data.items():
        for train_size, df in size_data.items():
            # Group by target and calculate statistics - only for common targets
            for target in df['target'].unique():
                if target not in common_targets:
                    continue  # Skip targets not available in size-specific datasets
                    
                target_data = df[df['target'] == target]
                
                # Calculate mean and std for each metric
                for metric in ['test/mae', 'test/r2', 'test/rmse']:
                    if metric in target_data.columns:
                        values = target_data[metric].values
                        # Filter out extreme outliers (likely failed runs)
                        if metric == 'test/r2':
                            values = values[values > -10]  # Remove extreme negative R² values
                        elif metric in ['test/mae', 'test/rmse']:
                            values = values[values < np.percentile(values, 95)]  # Remove top 5% outliers
                        
                        if len(values) > 0:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            summary_data[variant][train_size][target][metric] = (mean_val, std_val)
    
    return summary_data

def create_learning_curve_plots(summary_data, common_targets, output_dir='plots'):
    """Create individual learning curve plots for each target-metric combination with all variants."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define metrics and their properties
    metrics_info = {
        'test/mae': {'name': 'Mean Absolute Error (MAE)', 'better': 'lower', 'color': '#1f77b4'},
        'test/r2': {'name': 'R² Score', 'better': 'higher', 'color': '#ff7f0e'},
        'test/rmse': {'name': 'Root Mean Square Error (RMSE)', 'better': 'lower', 'color': '#2ca02c'}
    }
    
    # Define variant colors and styles
    variant_styles = {
        'original': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'Original'},
        'rdkit': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'label': 'RDKit'},
        'batch_norm': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'label': 'Batch Norm'},
        'rdkit_batch_norm': {'color': '#d62728', 'marker': 'D', 'linestyle': ':', 'label': 'RDKit + Batch Norm'}
    }
    
    print(f"Creating plots for {len(common_targets)} targets and {len(metrics_info)} metrics...")
    
    # Create a plot for each target-metric combination
    for target in sorted(common_targets):
        for metric, metric_info in metrics_info.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot each variant
            for variant in sorted(summary_data.keys()):
                if variant not in variant_styles:
                    continue
                    
                variant_data = summary_data[variant]
                style = variant_styles[variant]
                
                # Get training sizes and sort them
                train_sizes = list(variant_data.keys())
                train_sizes_numeric = []
                for size in train_sizes:
                    if size == 'full':
                        # Estimate full size based on largest explicit size
                        numeric_sizes = [s for s in train_sizes if s != 'full']
                        if numeric_sizes:
                            train_sizes_numeric.append(max(numeric_sizes) * 1.2)  # Estimate
                        else:
                            train_sizes_numeric.append(15000)  # Default estimate
                    else:
                        train_sizes_numeric.append(size)
                
                # Sort by numeric value
                sorted_indices = np.argsort(train_sizes_numeric)
                train_sizes_sorted = [train_sizes[i] for i in sorted_indices]
                train_sizes_numeric_sorted = [train_sizes_numeric[i] for i in sorted_indices]
                
                # Collect data points for this variant
                x_vals = []
                y_means = []
                y_stds = []
                
                for size_idx, train_size in enumerate(train_sizes_sorted):
                    if (train_size in variant_data and 
                        target in variant_data[train_size] and 
                        metric in variant_data[train_size][target]):
                        
                        mean_val, std_val = variant_data[train_size][target][metric]
                        x_vals.append(train_sizes_numeric_sorted[size_idx])
                        y_means.append(mean_val)
                        y_stds.append(std_val)
                
                if len(x_vals) > 0:
                    # Plot line with error bars
                    ax.errorbar(x_vals, y_means, yerr=y_stds, 
                               marker=style['marker'], 
                               color=style['color'],
                               linestyle=style['linestyle'],
                               linewidth=2.5, 
                               markersize=8,
                               label=style['label'],
                               capsize=5, 
                               capthick=2,
                               alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Training Set Size', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_info['name'], fontsize=14, fontweight='bold')
            ax.set_title(f'{target.replace("_", " ").title()} - {metric_info["name"]} vs Training Size', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            
            # Set x-axis ticks
            if len(x_vals) > 0:  # Only if we have data
                all_x_vals = []
                for variant_data in summary_data.values():
                    train_sizes = list(variant_data.keys())
                    for size in train_sizes:
                        if size == 'full':
                            numeric_sizes = [s for s in train_sizes if s != 'full']
                            if numeric_sizes:
                                all_x_vals.append(max(numeric_sizes) * 1.2)
                        else:
                            all_x_vals.append(size)
                
                unique_x_vals = sorted(list(set(all_x_vals)))
                ax.set_xticks(unique_x_vals)
                ax.set_xticklabels([str(int(x)) if x != max(unique_x_vals) else 'Full' 
                                   for x in unique_x_vals], rotation=45)
            
            # Add legend
            ax.legend(loc='best', fontsize=12, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save plot
            target_clean = target.replace('_', '-')
            metric_clean = metric.split('/')[-1]
            output_file = output_dir / f'opv_{target_clean}_{metric_clean}_learning_curve.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()  # Close to save memory
    
    print(f"\n✅ Created {len(common_targets) * len(metrics_info)} individual learning curve plots!")

def create_summary_table(summary_data, output_dir='plots'):
    """Create a summary table showing best performance for each target."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Collect data for summary table
    summary_rows = []
    
    for variant in sorted(summary_data.keys()):
        variant_data = summary_data[variant]
        
        # Get all targets
        all_targets = set()
        for size_data in variant_data.values():
            all_targets.update(size_data.keys())
        
        for target in sorted(all_targets):
            row = {'Variant': variant.title(), 'Target': target.replace('_', ' ').title()}
            
            # Find best performance for each metric
            for metric in ['test/mae', 'test/r2', 'test/rmse']:
                best_size = None
                best_value = None
                
                for train_size, size_data in variant_data.items():
                    if target in size_data and metric in size_data[target]:
                        mean_val, std_val = size_data[target][metric]
                        
                        if best_value is None:
                            best_value = mean_val
                            best_size = train_size
                        else:
                            # For MAE and RMSE, lower is better; for R², higher is better
                            if ((metric in ['test/mae', 'test/rmse'] and mean_val < best_value) or
                                (metric == 'test/r2' and mean_val > best_value)):
                                best_value = mean_val
                                best_size = train_size
                
                if best_value is not None:
                    row[f'{metric.split("/")[1].upper()}'] = f'{best_value:.4f}'
                    row[f'{metric.split("/")[1].upper()}_size'] = str(best_size)
            
            summary_rows.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    output_file = output_dir / 'opv_performance_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Saved summary table: {output_file}")
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))

def main():
    """Main function to run the learning curve analysis."""
    print("🚀 Starting OPV Learning Curve Analysis")
    print("=" * 50)
    
    # Load results
    results_dir = Path('results/DMPNN')
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found!")
        return
    
    print(f"Loading results from: {results_dir}")
    results_data, common_targets = load_opv_results(results_dir)
    
    if not results_data:
        print("No results found!")
        return
    
    print(f"\nFound {len(results_data)} variants:")
    for variant, size_data in results_data.items():
        print(f"  - {variant}: {len(size_data)} training sizes")
    
    print(f"\nUsing {len(common_targets)} common targets for learning curves:")
    for target in sorted(common_targets):
        print(f"  - {target}")
    
    # Calculate summary statistics
    print("\n📊 Calculating summary statistics...")
    summary_data = calculate_metrics_summary(results_data, common_targets)
    
    # Create plots
    print("\n📈 Creating learning curve plots...")
    create_learning_curve_plots(summary_data, common_targets)
    
    # Create summary table
    print("\n📋 Creating performance summary...")
    create_summary_table(summary_data)
    
    print("\n✅ Analysis complete!")
    print("Check the 'plots/' directory for generated figures and summary table.")

if __name__ == "__main__":
    main()
