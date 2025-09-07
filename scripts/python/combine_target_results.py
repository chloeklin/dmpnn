#!/usr/bin/env python3
"""
Combine target-specific result files into single CSV files.

This script looks for files matching the pattern:
  {dataset}__*_results_{target}.csv
and combines them into a single file:
  {dataset}__*_results.csv
"""
import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

def combine_results(results_dir: str):
    """Combine target-specific result files into single CSV files."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return
    
    # Group files by their base name (everything before the last underscore)
    file_groups = defaultdict(list)
    
    # Pattern to match target-specific result files
    pattern = re.compile(r'(.+_results)_.+\.csv$')
    
    for file_path in results_dir.glob('*_results_*.csv'):
        match = pattern.search(file_path.name)
        if match:
            base_name = match.group(1)
            file_groups[base_name].append(file_path)
    
    if not file_groups:
        print("No target-specific result files found.")
        return
    
    # Process each group of files
    for base_name, file_paths in file_groups.items():
        combined_df = pd.DataFrame()
        
        for file_path in file_paths:
            # Extract target name from filename
            target = file_path.stem.split('_results_')[-1]
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add target column if not present
                if 'target' not in df.columns:
                    df['target'] = target
                
                # Combine with other results
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
                print(f"Processed: {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not combined_df.empty:
            # Save combined results
            output_path = results_dir / f"{base_name}.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Saved combined results to: {output_path}")
        else:
            print(f"No valid data to combine for {base_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_directory>")
        sys.exit(1)
    
    combine_results(sys.argv[1])
