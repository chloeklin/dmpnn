#!/usr/bin/env python3
"""
Plot HTPMD fusion variants comparison:
1. DMPNN/GIN/GAT with all fusion methods (graph only, desc, aux, film, film_fllast)
2. Same graph variants + Tabular (desc only)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
results_dir = Path(__file__).parent.parent.parent / "results"
output_dir = Path(__file__).parent.parent.parent / "plots" / "htpmd_fusion_variants"
output_dir.mkdir(parents=True, exist_ok=True)

# Define variants to load for each model
variants = {
    'graph_only': 'htpmd_results.csv',
    'desc': 'htpmd__desc_results.csv',
    'aux': 'htpmd__aux_results.csv',
    'film': 'htpmd__desc__film_results.csv',
    'film_fllast': 'htpmd__desc__film__fllast_results.csv'
}

variant_labels = {
    'graph_only': 'Graph Only',
    'desc': 'Graph+Desc',
    'aux': 'Graph+Aux',
    'film': 'Graph+FiLM',
    'film_fllast': 'Graph+FiLM(last)'
}

# Load data
all_data = []

# Load graph models with variants
for model_name in ['DMPNN', 'GIN', 'GAT']:
    model_dir = results_dir / model_name
    
    for variant_key, filename in variants.items():
        file_path = model_dir / filename
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            
            # Normalize column names
            rename_map = {}
            for col in df.columns:
                if col.startswith('test/'):
                    rename_map[col] = col.replace('test/', '')
            if rename_map:
                df = df.rename(columns=rename_map)
            
            df['model'] = model_name
            df['variant'] = variant_key
            df['variant_label'] = variant_labels[variant_key]
            df['model_variant'] = f"{model_name}_{variant_labels[variant_key]}"
            all_data.append(df)
            print(f"Loaded {model_name} - {variant_labels[variant_key]}: {len(df)} rows")

# Load tabular (desc only)
tabular_file = results_dir / "tabular" / "htpmd_descriptors.csv"
if tabular_file.exists():
    df_tab = pd.read_csv(tabular_file)
    # Compute RMSE from MSE (tabular results only have MSE)
    if 'mse' in df_tab.columns and 'rmse' not in df_tab.columns:
        df_tab['rmse'] = np.sqrt(df_tab['mse'])
    df_tab['model'] = 'Tabular'
    df_tab['variant'] = 'desc'
    df_tab['variant_label'] = 'Desc'
    df_tab['model_variant'] = 'Tabular_Desc'
    all_data.append(df_tab)
    print(f"Loaded Tabular - Desc: {len(df_tab)} rows")

# Combine all data
combined = pd.concat(all_data, ignore_index=True)

print(f"\nTotal rows: {len(combined)}")
print(f"Models: {combined['model'].unique()}")
print(f"Variants: {combined['variant_label'].unique()}")

# Get targets
targets = sorted(combined['target'].unique())
metrics = ['rmse', 'mae', 'r2']

# Color schemes - conventional journal colors
# Each model gets a base color, variants get different shades/intensities
model_base_colors = {
    'DMPNN': '#1F77B4',  # Blue
    'GIN': '#2CA02C',     # Green
    'GAT': '#D62728',     # Red
    'Tabular': '#FF7F0E'  # Orange
}

# Variant styles (intensity/shade variations)
variant_styles = {
    'graph_only': {'alpha': 0.4, 'hatch': None},
    'desc': {'alpha': 0.6, 'hatch': None},
    'aux': {'alpha': 0.8, 'hatch': None},
    'film': {'alpha': 1.0, 'hatch': None},
    'film_fllast': {'alpha': 1.0, 'hatch': '//'}
}

# Create color map for each model_variant
color_map = {}
for model in ['DMPNN', 'GIN', 'GAT']:
    base_color = model_base_colors[model]
    for variant_key in variants.keys():
        key = f"{model}_{variant_labels[variant_key]}"
        color_map[key] = base_color
color_map['Tabular_Desc'] = model_base_colors['Tabular']

# ============================================================
# PLOT 1: Graph models with all fusion variants (no tabular)
# All targets in one figure with subplots
# ============================================================
graph_only_data = combined[combined['model'] != 'Tabular'].copy()

models = ['DMPNN', 'GIN', 'GAT']
variant_order = ['graph_only', 'desc', 'aux', 'film', 'film_fllast']

for metric in metrics:
    if metric not in graph_only_data.columns:
        print(f"Skipping {metric} - not in data")
        continue
    
    # Create figure with subplots for all targets
    fig, axes = plt.subplots(1, len(targets), figsize=(20, 5))
    
    for target_idx, target in enumerate(targets):
        ax = axes[target_idx]
        
        bar_width = 0.25
        x_base = np.arange(len(variant_order))
        
        target_data = graph_only_data[graph_only_data['target'] == target]
        
        # Plot each model
        for model_idx, model in enumerate(models):
            means = []
            stds = []
            
            for variant_key in variant_order:
                variant_data = target_data[
                    (target_data['model'] == model) & 
                    (target_data['variant'] == variant_key)
                ]
                
                if not variant_data.empty:
                    means.append(variant_data[metric].mean())
                    stds.append(variant_data[metric].std())
                else:
                    means.append(0)
                    stds.append(0)
            
            x_pos = x_base + model_idx * bar_width
            ax.bar(x_pos, means, bar_width,
                   yerr=stds,
                   color=model_base_colors[model],
                   alpha=0.8,
                   capsize=3,
                   edgecolor='black',
                   linewidth=0.8,
                   label=model if target_idx == 0 else '')
        
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(target, fontsize=12, fontweight='bold')
        ax.set_xticks(x_base + bar_width)
        ax.set_xticklabels([variant_labels[v] for v in variant_order], fontsize=9, rotation=25, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend outside the plot area
    fig.legend(models, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True, fontsize=10)
    fig.suptitle(f'HTPMD: Graph Model Fusion Variants - {metric.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'htpmd_{metric}_graph_fusion_variants_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

# ============================================================
# PLOT 2: Graph variants + Tabular (desc only)
# All targets in one figure with subplots
# ============================================================
models_with_tabular = ['DMPNN', 'GIN', 'GAT', 'Tabular']

for metric in metrics:
    if metric not in combined.columns:
        print(f"Skipping {metric} - not in data")
        continue
    
    # Create figure with subplots for all targets
    fig, axes = plt.subplots(1, len(targets), figsize=(20, 5))
    
    for target_idx, target in enumerate(targets):
        ax = axes[target_idx]
        
        bar_width = 0.18
        x_base = np.arange(len(variant_order))
        
        target_data = combined[combined['target'] == target]
        
        # Plot each model
        for model_idx, model in enumerate(models_with_tabular):
            means = []
            stds = []
            
            for variant_key in variant_order:
                variant_data = target_data[
                    (target_data['model'] == model) & 
                    (target_data['variant'] == variant_key)
                ]
                
                if not variant_data.empty:
                    means.append(variant_data[metric].mean())
                    stds.append(variant_data[metric].std())
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            x_pos = x_base + model_idx * bar_width
            
            # Filter out NaN values for plotting
            valid_mask = ~np.isnan(means)
            if valid_mask.any():
                ax.bar(x_pos[valid_mask], np.array(means)[valid_mask], bar_width,
                       yerr=np.array(stds)[valid_mask],
                       color=model_base_colors[model],
                       alpha=0.8,
                       capsize=3,
                       edgecolor='black',
                       linewidth=0.8,
                       label=model if target_idx == 0 else '')
        
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(target, fontsize=12, fontweight='bold')
        ax.set_xticks(x_base + bar_width * 1.5)
        ax.set_xticklabels([variant_labels[v] for v in variant_order], fontsize=9, rotation=25, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend outside the plot area
    fig.legend(models_with_tabular, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=True, fontsize=10)
    fig.suptitle(f'HTPMD: Graph Fusion Variants + Tabular - {metric.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'htpmd_{metric}_graph_fusion_with_tabular_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

# ============================================================
# Save consolidated CSV results
# ============================================================

# Prepare consolidated data with mean and std for each model/variant/target/metric
consolidated_rows = []

for target in targets:
    target_data = combined[combined['target'] == target]
    
    for metric in metrics:
        if metric not in combined.columns:
            continue
        
        # Graph models
        for model in ['DMPNN', 'GIN', 'GAT']:
            for variant_key in variant_order:
                variant_data = target_data[
                    (target_data['model'] == model) & 
                    (target_data['variant'] == variant_key)
                ]
                
                if not variant_data.empty:
                    mean_val = variant_data[metric].mean()
                    std_val = variant_data[metric].std()
                    
                    consolidated_rows.append({
                        'target': target,
                        'model': model,
                        'variant': variant_labels[variant_key],
                        'metric': metric,
                        'mean': mean_val,
                        'std': std_val,
                        'n_splits': len(variant_data)
                    })
        
        # Tabular
        tabular_data = target_data[target_data['model'] == 'Tabular']
        if not tabular_data.empty:
            mean_val = tabular_data[metric].mean()
            std_val = tabular_data[metric].std()
            
            consolidated_rows.append({
                'target': target,
                'model': 'Tabular',
                'variant': 'Desc',
                'metric': metric,
                'mean': mean_val,
                'std': std_val,
                'n_splits': len(tabular_data)
            })

# Save consolidated results
consolidated_df = pd.DataFrame(consolidated_rows)
consolidated_file = output_dir / 'htpmd_fusion_variants_consolidated.csv'
consolidated_df.to_csv(consolidated_file, index=False)
print(f"\nSaved consolidated results: {consolidated_file}")

print(f"\nAll plots saved to: {output_dir}")
