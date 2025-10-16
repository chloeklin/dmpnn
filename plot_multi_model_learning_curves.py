#!/usr/bin/env python3
"""
Plot Learning Curves for Multiple Models on OPV Dataset

This script compares learning curves across different models (DMPNN, AttentiveFP, etc.)
showing how performance metrics vary with training set size.

Usage:
    python plot_multi_model_learning_curves.py
    
The script will:
1. Load results from multiple model directories (DMPNN, AttentiveFP, etc.)
2. Extract training sizes and performance metrics
3. Create plots comparing all models for each target-metric combination
4. Show error bars representing standard deviation across replicates
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

# Configuration: Add or remove models here
MODEL_CONFIGS = {
    'DMPNN': {
        'dir': 'results/DMPNN',
        'pattern': 'opv_camb3lyp*_results.csv',
        'color': '#1f77b4',
        'marker': 'o',
        'linestyle': '-',
        'is_tabular': False
    },
    'Tabular-Linear': {
        'dir': 'results/tabular',
        'pattern': 'opv_camb3lyp_descriptors_rdkit_ab*.csv',
        'color': '#ff7f0e',
        'marker': 's',
        'linestyle': '--',
        'is_tabular': True,
        'tabular_model': 'Linear'
    },
    'Tabular-RF': {
        'dir': 'results/tabular',
        'pattern': 'opv_camb3lyp_descriptors_rdkit_ab*.csv',
        'color': '#2ca02c',
        'marker': '^',
        'linestyle': '-.',
        'is_tabular': True,
        'tabular_model': 'RF'
    },
    'Tabular-XGB': {
        'dir': 'results/tabular',
        'pattern': 'opv_camb3lyp_descriptors_rdkit_ab*.csv',
        'color': '#d62728',
        'marker': 'D',
        'linestyle': ':',
        'is_tabular': True,
        'tabular_model': 'XGB'
    },
    'AttentiveFP': {
        'dir': 'results/AttentiveFP',
        'pattern': 'opv_camb3lyp*.csv',
        'color': '#9467bd',
        'marker': 'v',
        'linestyle': '--',
        'is_tabular': False
    },
    # Add more models here as needed:
    # 'PAE_TG': {
    #     'dir': 'results/PAE_TG',
    #     'pattern': 'opv_camb3lyp*_results.csv',
    #     'color': '#9467bd',
    #     'marker': 'v',
    #     'linestyle': '--',
    #     'is_tabular': False
    # },
}

# Target filter: Only plot these targets
TARGET_FILTER = [
    'spectral_overlap',
    'gap',
    'homo',
    'homo_extrapolated',
    'optical_lumo',
    'lumo',
    'delta_optical_lumo',
    'gap_extrapolated'
]

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
    if '__desc' in filename:
        variants.append('desc')
    return '_'.join(variants) if variants else 'original'

def load_model_results(model_name, model_config):
    """Load all result files for a specific model."""
    results_dir = Path(model_config['dir'])
    
    if not results_dir.exists():
        print(f"⚠️  Warning: Directory {results_dir} not found for {model_name}")
        return {}, set()
    
    # Find all result files matching the pattern
    result_files = list(results_dir.glob(model_config['pattern']))
    
    print(f"\n{model_name}: Found {len(result_files)} result files")
    
    # Organize results by variant and training size
    results_data = defaultdict(dict)  # {variant: {train_size: DataFrame}}
    all_targets = set()
    
    for file_path in result_files:
        filename = file_path.name
        train_size = extract_train_size_from_filename(filename)
        variant = extract_variant_from_filename(filename)
        
        try:
            df = pd.read_csv(file_path)
            
            # Handle tabular results with multiple models per file
            if model_config.get('is_tabular', False):
                tabular_model = model_config.get('tabular_model')
                if 'model' in df.columns and tabular_model:
                    # Filter to specific tabular model
                    df = df[df['model'] == tabular_model].copy()
                    if len(df) == 0:
                        continue
            
            # Handle different column naming conventions
            # Some files have 'test/mae', others have 'MAE' or 'mae'
            column_mapping = {}
            for col in df.columns:
                if col.lower() == 'mae' or col == 'test/mae':
                    column_mapping[col] = 'test/mae'
                elif col.lower() == 'r2' or col == 'test/r2':
                    column_mapping[col] = 'test/r2'
                elif col.lower() == 'rmse' or col == 'test/rmse':
                    column_mapping[col] = 'test/rmse'
                elif col.lower() == 'mse':
                    # Calculate RMSE from MSE if RMSE not present
                    if 'test/rmse' not in df.columns and 'rmse' not in df.columns and 'RMSE' not in df.columns:
                        df['test/rmse'] = np.sqrt(df[col])
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Filter to only specified targets
            if 'target' in df.columns:
                df = df[df['target'].isin(TARGET_FILTER)].copy()
                if len(df) == 0:
                    continue
            
            # Concatenate with existing data for this variant/train_size if it exists
            if train_size in results_data[variant]:
                results_data[variant][train_size] = pd.concat([results_data[variant][train_size], df], ignore_index=True)
            else:
                results_data[variant][train_size] = df
            
            # Track targets
            if 'target' in df.columns:
                targets = set(df['target'].unique())
                all_targets.update(targets)
            
            print(f"  ✓ {filename}: variant='{variant}', train_size={train_size}, shape={df.shape}")
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
    
    return results_data, all_targets

def load_all_models():
    """Load results from all configured models."""
    all_results = {}  # {model_name: {variant: {train_size: DataFrame}}}
    common_targets = None
    
    print("=" * 70)
    print("📂 Loading Results from All Models")
    print("=" * 70)
    
    for model_name, model_config in MODEL_CONFIGS.items():
        results_data, targets = load_model_results(model_name, model_config)
        
        if results_data:
            all_results[model_name] = results_data
            
            # Find common targets across all models
            if common_targets is None:
                common_targets = targets
            else:
                common_targets = common_targets & targets
    
    if common_targets:
        print(f"\n✅ Common targets across all models ({len(common_targets)}): {sorted(common_targets)}")
    else:
        print("\n⚠️  Warning: No common targets found across models")
        common_targets = set()
    
    return all_results, common_targets

def calculate_metrics_summary(all_results, common_targets, variant_filter='original'):
    """Calculate mean and std for each metric across replicates."""
    # {model: {train_size: {target: {metric: (mean, std)}}}}
    summary_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for model_name, model_results in all_results.items():
        # Use only the specified variant (e.g., 'original' for base model without extras)
        if variant_filter not in model_results:
            print(f"⚠️  Warning: Variant '{variant_filter}' not found for {model_name}")
            continue
        
        variant_data = model_results[variant_filter]
        
        for train_size, df in variant_data.items():
            # Group by target and calculate statistics
            for target in df['target'].unique() if 'target' in df.columns else [None]:
                if target and target not in common_targets:
                    continue
                
                if target:
                    target_data = df[df['target'] == target]
                else:
                    target_data = df
                
                # Calculate mean and std for each metric
                for metric in ['test/mae', 'test/r2', 'test/rmse']:
                    if metric in target_data.columns:
                        values = target_data[metric].values
                        original_count = len(values)
                        
                        # Filter out extreme outliers (likely failed runs)
                        if metric == 'test/r2':
                            values = values[values > -10]
                        elif metric in ['test/mae', 'test/rmse']:
                            # Only filter if we have enough data points and there are true outliers
                            if len(values) >= 5:
                                percentile_95 = np.percentile(values, 95)
                                # Only filter if the 95th percentile is significantly different from median
                                median_val = np.median(values)
                                if percentile_95 > median_val * 3:  # Only filter extreme outliers
                                    values = values[values < percentile_95]
                        
                        filtered_count = len(values)
                        
                        if len(values) > 0:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            summary_data[model_name][train_size][target][metric] = (mean_val, std_val)
    
    return summary_data

def create_multi_model_learning_curves(summary_data, common_targets, output_dir='plots/multi_model'):
    """Create learning curve plots comparing multiple models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics and their properties
    metrics_info = {
        'test/mae': {'name': 'Mean Absolute Error (MAE)', 'better': 'lower'},
        'test/r2': {'name': 'R² Score', 'better': 'higher'},
        'test/rmse': {'name': 'Root Mean Square Error (RMSE)', 'better': 'lower'}
    }
    
    print(f"\n📈 Creating multi-model learning curve plots...")
    print(f"   Targets: {len(common_targets)}, Metrics: {len(metrics_info)}, Models: {len(summary_data)}")
    
    # Create a plot for each target-metric combination
    for target in sorted(common_targets):
        for metric, metric_info in metrics_info.items():
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            
            # Plot each model
            for model_name in sorted(summary_data.keys()):
                model_data = summary_data[model_name]
                model_config = MODEL_CONFIGS[model_name]
                
                # Get training sizes and sort them
                train_sizes = list(model_data.keys())
                train_sizes_numeric = []
                for size in train_sizes:
                    if size == 'full':
                        # Estimate full size
                        numeric_sizes = [s for s in train_sizes if s != 'full']
                        if numeric_sizes:
                            train_sizes_numeric.append(max(numeric_sizes) * 1.2)
                        else:
                            train_sizes_numeric.append(15000)
                    else:
                        train_sizes_numeric.append(size)
                
                # Sort by numeric value
                sorted_indices = np.argsort(train_sizes_numeric)
                train_sizes_sorted = [train_sizes[i] for i in sorted_indices]
                train_sizes_numeric_sorted = [train_sizes_numeric[i] for i in sorted_indices]
                
                # Collect data points
                x_vals = []
                y_means = []
                y_stds = []
                
                for size_idx, train_size in enumerate(train_sizes_sorted):
                    if (train_size in model_data and 
                        target in model_data[train_size] and 
                        metric in model_data[train_size][target]):
                        
                        mean_val, std_val = model_data[train_size][target][metric]
                        x_vals.append(train_sizes_numeric_sorted[size_idx])
                        y_means.append(mean_val)
                        y_stds.append(std_val)
                
                if len(x_vals) > 0:
                    # Plot line with error bars
                    ax.errorbar(x_vals, y_means, yerr=y_stds,
                               marker=model_config['marker'],
                               color=model_config['color'],
                               linestyle=model_config['linestyle'],
                               linewidth=2.5,
                               markersize=6,
                               label=model_name,
                               capsize=5,
                               capthick=2,
                               alpha=0.85)
            
            # Formatting
            ax.set_xlabel('Training Set Size', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_info['name'], fontsize=14, fontweight='bold')
            ax.set_title(f'{target.replace("_", " ").title()} - {metric_info["name"]} vs Training Size\n(Model Comparison)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xscale('log')
            
            # Set x-axis ticks
            all_x_vals = []
            for model_data in summary_data.values():
                train_sizes = list(model_data.keys())
                for size in train_sizes:
                    if size == 'full':
                        numeric_sizes = [s for s in train_sizes if s != 'full']
                        if numeric_sizes:
                            all_x_vals.append(max(numeric_sizes) * 1.2)
                    else:
                        all_x_vals.append(size)
            
            if all_x_vals:
                unique_x_vals = sorted(list(set(all_x_vals)))
                ax.set_xticks(unique_x_vals)
                ax.set_xticklabels([str(int(x)) if x != max(unique_x_vals) else 'Full' 
                                   for x in unique_x_vals], rotation=45)
            
            # Add legend
            ax.legend(loc='best', fontsize=13, framealpha=0.95, shadow=True)
            
            plt.tight_layout()
            
            # Save plot
            target_clean = target.replace('_', '-')
            metric_clean = metric.split('/')[-1]
            output_file = output_dir / f'opv_{target_clean}_{metric_clean}_multi_model.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_file.name}")
            plt.close()
    
    print(f"\n✅ Created {len(common_targets) * len(metrics_info)} multi-model learning curve plots!")

def create_comparison_summary(summary_data, common_targets, output_dir='plots/multi_model'):
    """Create a summary table comparing best performance across models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    for target in sorted(common_targets):
        for metric in ['test/mae', 'test/r2', 'test/rmse']:
            row = {'Target': target.replace('_', ' ').title(), 'Metric': metric.split('/')[-1].upper()}
            
            # Find best model for this target-metric combination
            best_model = None
            best_value = None
            
            for model_name, model_data in summary_data.items():
                for train_size, size_data in model_data.items():
                    if target in size_data and metric in size_data[target]:
                        mean_val, std_val = size_data[target][metric]
                        
                        if best_value is None:
                            best_value = mean_val
                            best_model = model_name
                        else:
                            # For MAE and RMSE, lower is better; for R², higher is better
                            if ((metric in ['test/mae', 'test/rmse'] and mean_val < best_value) or
                                (metric == 'test/r2' and mean_val > best_value)):
                                best_value = mean_val
                                best_model = model_name
                
                # Add this model's best performance
                model_best = None
                for train_size, size_data in model_data.items():
                    if target in size_data and metric in size_data[target]:
                        mean_val, std_val = size_data[target][metric]
                        if model_best is None:
                            model_best = mean_val
                        else:
                            if ((metric in ['test/mae', 'test/rmse'] and mean_val < model_best) or
                                (metric == 'test/r2' and mean_val > model_best)):
                                model_best = mean_val
                
                if model_best is not None:
                    row[model_name] = f'{model_best:.4f}'
            
            row['Best_Model'] = best_model if best_model else 'N/A'
            summary_rows.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    output_file = output_dir / 'model_comparison_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\n📋 Saved comparison summary: {output_file}")
    print("\nModel Comparison Summary (Best Performance):")
    print(summary_df.to_string(index=False))

def main():
    """Main function to run the multi-model learning curve analysis."""
    print("🚀 Starting Multi-Model Learning Curve Analysis")
    print("=" * 70)
    
    # Load results from all models
    all_results, common_targets = load_all_models()
    
    if not all_results:
        print("❌ No results found!")
        return
    
    if not common_targets:
        print("⚠️  Warning: No common targets found. Proceeding with all available targets.")
        # Use all targets from all models
        for model_results in all_results.values():
            for variant_data in model_results.values():
                for df in variant_data.values():
                    if 'target' in df.columns:
                        common_targets.update(df['target'].unique())
    
    print(f"\n📊 Found {len(all_results)} models:")
    for model_name, model_results in all_results.items():
        print(f"  - {model_name}: {len(model_results)} variants")
    
    # Calculate summary statistics (using 'original' variant for fair comparison)
    print("\n📊 Calculating summary statistics (using 'original' variant)...")
    summary_data = calculate_metrics_summary(all_results, common_targets, variant_filter='original')
    
    # Create plots
    create_multi_model_learning_curves(summary_data, common_targets)
    
    # Create comparison summary
    create_comparison_summary(summary_data, common_targets)
    
    print("\n✅ Multi-model analysis complete!")
    print("Check the 'plots/multi_model/' directory for generated figures and summary table.")

if __name__ == "__main__":
    main()
