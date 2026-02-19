#!/usr/bin/env python3
"""
Combine target-specific result files into single CSV files.

This script looks for files matching the pattern:
  {dataset}__*_results_{target}.csv
and combines them into a single file:
  {dataset}__*_results.csv

For OPV datasets, only specific targets are included:
  - optical_lumo, gap, homo, lumo, spectral_overlap
  - delta_optical_lumo, homo_extrapolated, gap_extrapolated
Other datasets include all targets.
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
    
    # Define allowed targets for OPV dataset
    opv_allowed_targets = {
        'optical_lumo',
        'gap', 
        'homo',
        'lumo',
        'spectral_overlap',
        'delta_optical_lumo',
        'homo_extrapolated',
        'gap_extrapolated'
    }
    
    # Group files by their base name (everything before "__target_")
    file_groups = defaultdict(list)

    # Pattern to match filenames like:
    #   insulator__size1024__target_bandgap_chain_results.csv
    #   opv_camb3lyp__target_spectral_overlap_results.csv
    #   htpmd__target_Conductivity__film_results.csv
    #   htpmd__desc__rdkit__target_Tg__aux_results.csv
    # Group 1: base name (before __target_)
    # Group 2: target name (between __target_ and optional mode suffixes / _results)
    # Group 3: optional mode suffixes (__film, __aux, __nofusion, __aux_la0.05, etc.)
    pattern = re.compile(r'(.+?)__target_(.+?)(__(?:film|aux|nofusion)(?:_[a-zA-Z0-9.]+)*)?_results\.csv$')

    # Find all __target_..._results.csv files
    for file_path in results_dir.glob('*__target_*_results.csv'):
        match = pattern.match(file_path.name)
        if match:
            base_name = match.group(1)  # e.g., "insulator__size1024", "htpmd"
            mode_suffix = match.group(3) or ""  # e.g., "__film", "__aux", ""
            group_key = base_name + mode_suffix  # group by base + mode
            file_groups[group_key].append(file_path)

    
    # Also check for _baseline pattern files  
    pattern_baseline = re.compile(r'(.+)__.+_baseline\.csv$')
    for file_path in results_dir.glob('*_baseline.csv'):
        match = pattern_baseline.search(file_path.name)
        if match:
            base_name = match.group(1) + "_baseline"
            file_groups[base_name].append(file_path)
    
    if not file_groups:
        print("No target-specific result files found.")
        return
    
    # Process each group of files
    for base_name, file_paths in file_groups.items():
        output_path = results_dir / f"{base_name}_results.csv"

        # Load existing combined file if present; we'll append any missing targets to it
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            existing_targets = set(existing_df['target'].unique()) if 'target' in existing_df.columns else set()
        else:
            existing_df = pd.DataFrame()
            existing_targets = set()

        combined_df = existing_df.copy()
        successfully_processed = []
        
        for file_path in file_paths:
            # Extract target name from filename
            if '_baseline.csv' in file_path.name:
                target = file_path.stem.split('__')[-1].replace('_baseline', '')
            else:
                m = pattern.match(file_path.name)
                if not m:
                    continue
                target = m.group(2)  # clean target name (mode suffixes are in group 3)

            # Skip if this target is already in the existing combined file
            if target in existing_targets:
                print(f"Skipping {file_path.name}: target '{target}' already in combined file")
                continue

            # Check if this is an OPV dataset and filter targets
            is_opv_dataset = 'opv' in file_path.name.lower()
            if is_opv_dataset and target not in opv_allowed_targets:
                print(f"Skipping OPV target '{target}' (not in allowed list): {file_path.name}")
                continue
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add target column if not present
                if 'target' not in df.columns:
                    df['target'] = target
                
                # Combine with other results
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
                print(f"Processed: {file_path.name}")
                successfully_processed.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not combined_df.empty:
            # Save combined results
            output_path = results_dir / f"{base_name}_results.csv"
            combined_df.to_csv(output_path, index=False)
            print(f"Saved combined results to: {output_path}")
            
            # Delete individual files after successful combination
            for file_path in successfully_processed:
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Could not delete {file_path.name}: {e}")
        else:
            print(f"No valid data to combine for {base_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_directory>")
        sys.exit(1)
    
    combine_results(sys.argv[1])
