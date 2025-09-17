#!/usr/bin/env python3
"""
Combine all CSV files in data/ directory (except ea_ip.csv) into one unified dataset.
Pools all homopolymer monomer SMILES into one table with dataset-specific target columns.
For SMILES that only appear in one dataset, other target columns are set to NaN.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Combine all datasets into unified SMILES table')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing CSV files (default: data)')
    parser.add_argument('--output', type=str, default='data/combined_homopolymer.csv',
                       help='Output file path (default: data/combined_homopolymer.csv)')
    parser.add_argument('--exclude', type=str, nargs='+', default=['ea_ip.csv'],
                       help='Files to exclude (default: ea_ip.csv)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Dataset-specific target columns configuration
    dataset_targets = {
        'htpmd': ['Conductivity'],  # Main target from HTPMD dataset
        'insulator': ['bandgap_chain'],  # Bandgap target
        'polyinfo': ['Class'],  # Classification target
        'opv_camb3lyp': ['optical_lumo', 'gap', 'homo', 'lumo', 'spectral_overlap', 
                        'delta_optical_lumo', 'homo_extrapolated', 'gap_extrapolated'],
        'opv_b3lyp': ['optical_lumo', 'gap', 'homo', 'lumo', 'spectral_overlap',
                     'delta_homo', 'delta_lumo', 'delta_optical_lumo', 
                     'homo_extrapolated', 'lumo_extrapolated', 'gap_extrapolated', 
                     'optical_lumo_extrapolated'],
    }
    
    print("🔍 Scanning data directory for CSV files...")
    
    # Find all CSV files, excluding specified ones
    csv_files = []
    exclude_set = set(args.exclude)
    
    for csv_file in data_dir.glob("*.csv"):
        if csv_file.name not in exclude_set:
            csv_files.append(csv_file)
    
    print(f"📁 Found {len(csv_files)} CSV files to process:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Load and process each dataset
    all_data = []
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem
        print(f"\n📊 Processing {dataset_name}...")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check if smiles column exists
            if 'smiles' not in df.columns:
                print(f"  ⚠️  Warning: No 'smiles' column found in {dataset_name}, skipping...")
                continue
            
            # Get target columns for this dataset
            targets = dataset_targets.get(dataset_name, [])
            if not targets:
                print(f"  ⚠️  Warning: No target columns defined for {dataset_name}, skipping...")
                continue
            
            # Check which target columns actually exist
            existing_targets = [col for col in targets if col in df.columns]
            missing_targets = [col for col in targets if col not in df.columns]
            
            if missing_targets:
                print(f"  ⚠️  Warning: Missing target columns in {dataset_name}: {missing_targets}")
            
            if not existing_targets:
                print(f"  ⚠️  Warning: No valid target columns found in {dataset_name}, skipping...")
                continue
            
            print(f"  ✅ Using target columns: {existing_targets}")
            
            # Create dataset-specific target column names
            dataset_data = df[['smiles'] + existing_targets].copy()
            
            # Rename target columns to include dataset prefix
            rename_dict = {}
            for target in existing_targets:
                new_name = f"{dataset_name}_{target}"
                rename_dict[target] = new_name
            
            dataset_data = dataset_data.rename(columns=rename_dict)
            dataset_data['dataset_source'] = dataset_name
            
            print(f"  📝 Renamed columns: {list(rename_dict.values())}")
            print(f"  🧬 Unique SMILES: {dataset_data['smiles'].nunique()}")
            
            all_data.append(dataset_data)
            
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {str(e)}")
            continue
    
    if not all_data:
        print("\n❌ No valid datasets found to combine!")
        return
    
    print(f"\n🔗 Combining {len(all_data)} datasets...")
    
    # Start with the first dataset
    combined_df = all_data[0].copy()
    
    # Merge with remaining datasets on SMILES
    for i, dataset_data in enumerate(all_data[1:], 1):
        print(f"  Merging dataset {i+1}/{len(all_data)}...")
        combined_df = pd.merge(combined_df, dataset_data, on='smiles', how='outer', suffixes=('', f'_dup{i}'))
        
        # Handle duplicate dataset_source columns
        if f'dataset_source_dup{i}' in combined_df.columns:
            # Combine dataset sources
            combined_df['dataset_source'] = combined_df['dataset_source'].fillna('') + ',' + combined_df[f'dataset_source_dup{i}'].fillna('')
            combined_df['dataset_source'] = combined_df['dataset_source'].str.strip(',')
            combined_df = combined_df.drop(columns=[f'dataset_source_dup{i}'])
    
    print(f"\n📈 Combined dataset statistics:")
    print(f"  Total SMILES: {len(combined_df)}")
    print(f"  Unique SMILES: {combined_df['smiles'].nunique()}")
    
    # Get all target columns (excluding smiles and dataset_source)
    target_columns = [col for col in combined_df.columns if col not in ['smiles', 'dataset_source']]
    print(f"  Target columns: {len(target_columns)}")
    
    # Show coverage statistics
    print(f"\n📊 Target column coverage:")
    for col in target_columns:
        non_null_count = combined_df[col].notna().sum()
        coverage = (non_null_count / len(combined_df)) * 100
        print(f"  {col}: {non_null_count}/{len(combined_df)} ({coverage:.1f}%)")
    
    # Reorder columns: smiles first, then dataset_source, then all targets
    column_order = ['smiles', 'dataset_source'] + sorted(target_columns)
    combined_df = combined_df[column_order]
    
    # Save the combined dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined_df.to_csv(output_path, index=False)
    print(f"\n✅ Combined dataset saved to: {output_path}")
    print(f"   Final shape: {combined_df.shape}")
    
    # Show sample of the data
    print(f"\n📋 Sample of combined data:")
    print(combined_df.head(3).to_string())
    
    print(f"\n🎯 Summary:")
    print(f"  - Combined {len(all_data)} datasets")
    print(f"  - Total SMILES entries: {len(combined_df)}")
    print(f"  - Unique SMILES: {combined_df['smiles'].nunique()}")
    print(f"  - Target columns: {len(target_columns)}")
    print(f"  - Output file: {output_path}")

if __name__ == "__main__":
    main()
