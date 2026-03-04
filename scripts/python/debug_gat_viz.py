#!/usr/bin/env python3
"""Debug script to check why GAT results aren't showing in plots."""

import pandas as pd
from pathlib import Path
import re

def parse_model_filename(filename: str, method: str, model_name: str = None):
    """Parse model CSV filename to extract dataset and feature information."""
    base = filename.replace('.csv', '')
    
    if method == 'Graph':
        base = base.replace('_results', '')
    
    # Extract copolymer mode if present
    copoly_mode = None
    if '__copoly_' in base:
        match = re.search(r'__copoly_([a-z_]+)', base)
        if match:
            copoly_mode = match.group(1)
            base = base.replace(f'__copoly_{copoly_mode}', '')
    
    # Handle batch normalization
    batch_norm = False
    if '_batch_norm' in base or '__batch_norm' in base:
        base = base.replace('_batch_norm', '').replace('__batch_norm', '')
        batch_norm = True
    
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
    
    if copoly_mode:
        features = f"{features} ({copoly_mode})"
    
    dataset = dataset.rstrip('_')
    
    if batch_norm:
        features = f"{features} (BN)"
    
    return dataset, features

# Check GAT files
results_dir = Path("../../results")
gat_dir = results_dir / "GAT"

print("=" * 80)
print("GAT Results Diagnostic")
print("=" * 80)

if not gat_dir.exists():
    print("ERROR: GAT directory does not exist!")
    exit(1)

# Find block copolymer files
block_files = list(gat_dir.glob("block__copoly_*_results.csv"))
print(f"\nFound {len(block_files)} block copolymer GAT result files:")

all_data = []

for csv_file in block_files:
    print(f"\n{'='*80}")
    print(f"Processing: {csv_file.name}")
    print(f"{'='*80}")
    
    # Parse filename
    dataset, features = parse_model_filename(csv_file.name, 'Graph', 'GAT')
    print(f"  Dataset: {dataset}")
    print(f"  Features: {features}")
    
    # Load file
    df = pd.read_csv(csv_file)
    print(f"  Rows loaded: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Check for data
    if df.empty:
        print("  WARNING: Empty DataFrame!")
        continue
    
    # Rename columns
    if 'test/mae' in df.columns:
        df = df.rename(columns={
            'test/mae': 'mae', 
            'test/r2': 'r2', 
            'test/rmse': 'rmse'
        })
    
    # Add metadata
    df['dataset'] = dataset
    df['features'] = features
    df['method'] = f"Graph_GAT"
    df['model'] = 'GAT'
    
    print(f"  Sample data:")
    print(df[['mae', 'rmse', 'r2', 'target', 'split']].head())
    
    # Check for NaN values
    print(f"  NaN values in mae: {df['mae'].isna().sum()}")
    print(f"  NaN values in rmse: {df['rmse'].isna().sum()}")
    print(f"  NaN values in r2: {df['r2'].isna().sum()}")
    
    # Aggregate
    summary = df.groupby(['target', 'features', 'model', 'method'])['mae'].agg(['mean', 'std']).reset_index()
    print(f"\n  Aggregated data:")
    print(summary)
    
    all_data.append(df)

# Combine all data
if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*80}")
    print("COMBINED GAT DATA")
    print(f"{'='*80}")
    print(f"Total rows: {len(combined)}")
    print(f"Unique features: {combined['features'].unique()}")
    print(f"Unique models: {combined['model'].unique()}")
    print(f"Unique methods: {combined['method'].unique()}")
    
    # Final aggregation
    final_summary = combined.groupby(['target', 'features', 'model', 'method'])['mae'].agg(['mean', 'std']).reset_index()
    print(f"\nFinal aggregated summary:")
    print(final_summary)
    
    # Check if any means are zero or NaN
    if (final_summary['mean'] == 0).any():
        print("\nWARNING: Some mean values are zero!")
    if final_summary['mean'].isna().any():
        print("\nWARNING: Some mean values are NaN!")
else:
    print("\nERROR: No data loaded!")
