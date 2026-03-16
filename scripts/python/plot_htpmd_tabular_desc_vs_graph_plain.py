#!/usr/bin/env python3
"""
Plot HTPMD: Tabular (descriptors only) vs Graph (no descriptors)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
results_dir = Path(__file__).parent.parent.parent / "results"
output_dir = Path(__file__).parent.parent.parent / "plots" / "htpmd_tabular_desc_vs_graph_plain"
output_dir.mkdir(parents=True, exist_ok=True)

# Load tabular results - ONLY descriptors
tabular_desc = pd.read_csv(results_dir / "tabular" / "htpmd_descriptors.csv")
# Compute RMSE from MSE (tabular results only have MSE)
if 'mse' in tabular_desc.columns and 'rmse' not in tabular_desc.columns:
    tabular_desc['rmse'] = np.sqrt(tabular_desc['mse'])
tabular_desc['method'] = 'Tabular'
tabular_desc['model'] = 'Linear'
tabular_desc['features'] = 'Desc'

# Load graph results - ONLY plain (no descriptors, no rdkit)
graph_results = []

for model_name in ['DMPNN', 'GIN', 'GAT', 'wDMPNN', 'DMPNN_DiffPool', 'AttentiveFP', 'PPG']:
    model_dir = results_dir / model_name
    result_file = model_dir / "htpmd_results.csv"
    
    if result_file.exists():
        df = pd.read_csv(result_file)
        # Skip if empty
        if df.empty:
            continue
        
        # Normalize column names - handle both 'test/metric' and 'metric' formats
        rename_map = {}
        for col in df.columns:
            if col.startswith('test/'):
                rename_map[col] = col.replace('test/', '')
        if rename_map:
            df = df.rename(columns=rename_map)
        
        df['method'] = 'Graph'
        df['model'] = model_name
        df['features'] = 'Graph'
        graph_results.append(df)
        print(f"Loaded {model_name}: {len(df)} rows")

# Combine all data
all_data = [tabular_desc] + graph_results
combined = pd.concat(all_data, ignore_index=True)

print(f"\nTotal rows: {len(combined)}")
print(f"Methods: {combined['method'].unique()}")
print(f"Models: {combined['model'].unique()}")
print(f"Targets: {combined['target'].unique() if 'target' in combined.columns else 'N/A'}")

# Get unique targets
if 'target' in combined.columns:
    targets = sorted(combined['target'].unique())
else:
    targets = ['htpmd']

# Metrics to plot
metrics = ['rmse', 'mae', 'r2']

# Color scheme - Nature/Science journal style
# Tabular: warm red/orange to stand out
# Graph models: cool blues/purples for clear distinction
colors = {
    'Tabular': '#D62728',  # Vibrant red (Nature style)
    'Graph': '#1F77B4'      # Deep blue
}

model_colors = {
    'Linear': '#D62728',        # Vibrant red (tabular - stands out)
    'DMPNN': '#1F77B4',         # Deep blue
    'GIN': '#9467BD',           # Purple
    'GAT': '#2CA02C',           # Green
    'wDMPNN': '#FF7F0E',        # Orange
    'DMPNN_DiffPool': '#8C564B', # Brown
    'AttentiveFP': '#E377C2',   # Pink
    'PPG': '#17BECF'            # Cyan
}

# Create plots for each metric
for metric in metrics:
    if metric not in combined.columns:
        print(f"Skipping {metric} - not in data")
        continue
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Group by target and model
    x_positions = []
    x_labels = []
    
    for i, target in enumerate(targets):
        target_data = combined[combined['target'] == target] if 'target' in combined.columns else combined
        
        # Get tabular value - compute mean and std across splits
        tab_data = target_data[target_data['method'] == 'Tabular']
        if not tab_data.empty:
            tab_mean = tab_data[metric].mean()
            tab_std = tab_data[metric].std()
            
            x_pos = i * 10
            ax.bar(x_pos, tab_mean, yerr=tab_std, width=0.8, 
                   color=colors['Tabular'], alpha=0.7, capsize=5,
                   label='Tabular (Desc)' if i == 0 else '')
            x_positions.append(x_pos)
            x_labels.append(f"{target}\nTabular")
        
        # Get graph models - compute mean and std across splits for each model
        graph_data = target_data[target_data['method'] == 'Graph']
        graph_models = sorted(graph_data['model'].unique())
        
        for j, model in enumerate(graph_models):
            model_data = graph_data[graph_data['model'] == model]
            if not model_data.empty:
                g_mean = model_data[metric].mean()
                g_std = model_data[metric].std()
                
                x_pos = i * 10 + j + 1.5
                ax.bar(x_pos, g_mean, yerr=g_std, width=0.8,
                       color=model_colors.get(model, '#95A5A6'), alpha=0.7, capsize=5,
                       label=model if i == 0 else '')
                x_positions.append(x_pos)
                x_labels.append(f"{target}\n{model}")
    
    # Formatting
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(f'HTPMD: Tabular (Desc only) vs Graph (Plain) - {metric.upper()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks([i * 10 + 3.5 for i in range(len(targets))])
    ax.set_xticklabels(targets, fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_file = output_dir / f'htpmd_{metric}_tabular_desc_vs_graph_plain.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

print(f"\nAll plots saved to: {output_dir}")
