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

# Load tabular (desc only) — preserve individual model names (Linear, RF, etc.)
tabular_file = results_dir / "tabular" / "htpmd_descriptors.csv"
if tabular_file.exists():
    df_tab = pd.read_csv(tabular_file)
    # Compute RMSE from MSE (tabular results only have MSE)
    if 'mse' in df_tab.columns and 'rmse' not in df_tab.columns:
        df_tab['rmse'] = np.sqrt(df_tab['mse'])
    # Preserve original sub-model name (Linear, RF, XGB…) before grouping
    df_tab = df_tab.rename(columns={'model': 'tab_model'})
    df_tab['model'] = 'Tabular'          # group identifier used for filtering
    df_tab['variant'] = 'desc'
    df_tab['variant_label'] = 'Desc'
    df_tab['model_variant'] = df_tab['tab_model'].apply(lambda m: f'Tabular_{m}')
    all_data.append(df_tab)
    tab_models_found = sorted(df_tab['tab_model'].unique())
    print(f"Loaded Tabular sub-models: {tab_models_found}  ({len(df_tab)} rows total)")

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
    'Tabular': '#FF7F0E'  # Orange (group fallback)
}

# Distinct shades for individual tabular sub-models (light → dark orange/brown)
_TAB_SHADES = ['#FFBB78', '#FF7F0E', '#D45500', '#8B3A00', '#5C2500']
tabular_model_colors = {}  # filled below once we know the sub-models

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

# ============================================================
# PLOT 1: Graph models with all fusion variants (no tabular)
# All targets in one figure with subplots
# ============================================================
graph_only_data = combined[combined['model'] != 'Tabular'].copy()

models = ['DMPNN', 'GIN', 'GAT']
variant_order = ['graph_only', 'desc', 'film', 'film_fllast', 'aux']

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
    
    # Add legend with correct colors
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=model_base_colors[m], edgecolor='black', linewidth=0.8, label=m) for m in models]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True, fontsize=10)
    fig.suptitle(f'HTPMD: Graph Model Fusion Variants - {metric.upper()}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'htpmd_{metric}_graph_fusion_variants_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

# ============================================================
# PLOT 2: Graph variants + Tabular (individual sub-models as separate bars)
# All targets in one figure with subplots
# ============================================================
graph_models = ['DMPNN', 'GIN', 'GAT']
n_graph_groups = len(variant_order)

# Resolve tabular sub-models and assign colours
tab_data_check = combined[combined['model'] == 'Tabular']
if not tab_data_check.empty and 'tab_model' in tab_data_check.columns:
    tabular_sub_models = sorted(tab_data_check['tab_model'].unique().tolist())
else:
    tabular_sub_models = []
for i, tm in enumerate(tabular_sub_models):
    tabular_model_colors[tm] = _TAB_SHADES[i % len(_TAB_SHADES)]

n_tab_models = len(tabular_sub_models)
# x-axis: graph variant groups first, then one label per tabular sub-model
x_group_labels = ([variant_labels[v] for v in variant_order]
                  + (tabular_sub_models if tabular_sub_models else ['Tabular']))

for metric in metrics:
    if metric not in combined.columns:
        print(f"Skipping {metric} - not in data")
        continue
    
    fig, axes = plt.subplots(1, len(targets), figsize=(26, 6))
    
    for target_idx, target in enumerate(targets):
        ax = axes[target_idx]
        
        n_models = len(graph_models)
        bar_width = 0.25          # width for each graph model bar
        tab_bar_width = 0.4       # wider bars for tabular sub-models
        tab_spacing = 0.55        # centre-to-centre spacing between tabular bars
        target_data = combined[combined['target'] == target]
        
        # --- Graph variant groups (positions 0..n_graph_groups-1) ---
        x_base = np.arange(n_graph_groups)
        
        for model_idx, model in enumerate(graph_models):
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
        
        # --- Tabular group (one wide bar per sub-model, well-spaced) ---
        # Start far enough right so the group doesn't crowd the last graph group
        tabular_x_start = n_graph_groups + 0.8
        tab_target = target_data[target_data['model'] == 'Tabular']
        tabular_tick_positions = []
        if not tab_target.empty and metric in tab_target.columns and 'tab_model' in tab_target.columns:
            for tab_idx, tab_model_name in enumerate(tabular_sub_models):
                tab_sub = tab_target[tab_target['tab_model'] == tab_model_name]
                x_pos = tabular_x_start + tab_idx * tab_spacing
                tabular_tick_positions.append(x_pos)
                if tab_sub.empty:
                    continue
                tab_mean = tab_sub[metric].mean()
                tab_std = tab_sub[metric].std() if len(tab_sub) > 1 else 0
                ax.bar(x_pos, tab_mean, tab_bar_width,
                       yerr=tab_std,
                       color=tabular_model_colors.get(tab_model_name, model_base_colors['Tabular']),
                       alpha=0.85,
                       capsize=3,
                       edgecolor='black',
                       linewidth=0.8,
                       label=tab_model_name if target_idx == 0 else '')
        elif not tab_target.empty and metric in tab_target.columns:
            # Fallback: no tab_model column → single averaged bar
            x_pos = tabular_x_start
            tabular_tick_positions.append(x_pos)
            ax.bar(x_pos, tab_target[metric].mean(), tab_bar_width,
                   color=model_base_colors['Tabular'], alpha=0.85, capsize=3,
                   edgecolor='black', linewidth=0.8,
                   label='Tabular' if target_idx == 0 else '')

        # x-tick positions and labels
        graph_tick_positions = x_base + bar_width * (n_models - 1) / 2
        all_ticks = list(graph_tick_positions) + tabular_tick_positions
        
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(target, fontsize=12, fontweight='bold')
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(x_group_labels, fontsize=8, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Vertical dashed separator between graph and tabular sections
        sep_x = n_graph_groups + 0.35
        ax.axvline(x=sep_x, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=model_base_colors[m], edgecolor='black', linewidth=0.8, label=m) for m in graph_models]
    for tm in tabular_sub_models:
        legend_patches.append(Patch(facecolor=tabular_model_colors.get(tm, model_base_colors['Tabular']),
                                     edgecolor='black', linewidth=0.8, label=tm))
    ncols = len(graph_models) + max(len(tabular_sub_models), 1)
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=ncols, frameon=True, fontsize=10)
    fig.suptitle(f'HTPMD: Graph Fusion Variants + Tabular Models - {metric.upper()}', fontsize=14, fontweight='bold', y=1.02)
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
        
        # Tabular — one row per sub-model
        tabular_data = target_data[target_data['model'] == 'Tabular']
        if not tabular_data.empty:
            if 'tab_model' in tabular_data.columns:
                for tm in sorted(tabular_data['tab_model'].unique()):
                    tm_data = tabular_data[tabular_data['tab_model'] == tm]
                    consolidated_rows.append({
                        'target': target,
                        'model': f'Tabular_{tm}',
                        'variant': 'Desc',
                        'metric': metric,
                        'mean': tm_data[metric].mean(),
                        'std': tm_data[metric].std(),
                        'n_splits': len(tm_data)
                    })
            else:
                consolidated_rows.append({
                    'target': target,
                    'model': 'Tabular',
                    'variant': 'Desc',
                    'metric': metric,
                    'mean': tabular_data[metric].mean(),
                    'std': tabular_data[metric].std(),
                    'n_splits': len(tabular_data)
                })

# Save consolidated results
consolidated_df = pd.DataFrame(consolidated_rows)
consolidated_file = output_dir / 'htpmd_fusion_variants_consolidated.csv'
consolidated_df.to_csv(consolidated_file, index=False)
print(f"\nSaved consolidated results: {consolidated_file}")

print(f"\nAll plots saved to: {output_dir}")
