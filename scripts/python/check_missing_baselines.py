#!/usr/bin/env python3
"""
Script to check for missing baseline files in the results directory.
For each model subdirectory (except 'tabular'), identifies which *_results.csv files
are missing their corresponding *_baseline.csv files.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def find_missing_baselines(results_dir):
    """
    Analyze results directory to find missing baseline files.
    
    Args:
        results_dir (Path): Path to the results directory
    
    Returns:
        dict: Dictionary with model names as keys and lists of missing baselines as values
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return {}
    
    missing_baselines = defaultdict(list)
    
    # Iterate through model directories
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Skip tabular directory as requested
        if model_name == 'tabular':
            continue
            
        # Skip hidden directories
        if model_name.startswith('.'):
            continue
            
        print(f"\n📁 Analyzing model: {model_name}")
        
        # Find all *_results.csv files
        results_files = []
        baseline_files = []
        
        for csv_file in model_dir.glob("*.csv"):
            filename = csv_file.name
            if filename.endswith('_results.csv'):
                results_files.append(filename)
            elif filename.endswith('_baseline.csv'):
                baseline_files.append(filename)
        
        # Convert baseline filenames to their corresponding results pattern
        baseline_prefixes = set()
        for baseline_file in baseline_files:
            # Remove '_baseline.csv' to get the prefix
            prefix = baseline_file.replace('_baseline.csv', '')
            baseline_prefixes.add(prefix)
        
        # Check each results file for missing baseline
        for results_file in results_files:
            # Remove '_results.csv' to get the prefix
            prefix = results_file.replace('_results.csv', '')
            
            if prefix not in baseline_prefixes:
                missing_baselines[model_name].append(prefix)
                print(f"  ❌ Missing baseline for: {prefix}")
            else:
                print(f"  ✅ Has baseline: {prefix}")
    
    return dict(missing_baselines)

def print_summary(missing_baselines):
    """Print a summary of missing baselines."""
    print("\n" + "="*60)
    print("📊 MISSING BASELINES SUMMARY")
    print("="*60)
    
    if not missing_baselines:
        print("🎉 All results files have corresponding baseline files!")
        return
    
    total_missing = sum(len(baselines) for baselines in missing_baselines.values())
    print(f"Total missing baselines: {total_missing}")
    
    for model_name, missing_list in missing_baselines.items():
        if missing_list:
            print(f"\n🤖 {model_name} ({len(missing_list)} missing):")
            for missing in sorted(missing_list):
                print(f"   • {missing}_baseline.csv")

def main():
    """Main function to run the baseline checker."""
    # Default to results directory relative to script location
    script_dir = Path(__file__).parent
    default_results_dir = script_dir.parent.parent / "results"
    
    # Allow command line argument for results directory
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = default_results_dir
    
    print(f"🔍 Checking for missing baselines in: {results_dir}")
    
    missing_baselines = find_missing_baselines(results_dir)
    print_summary(missing_baselines)
    
    # Exit with error code if there are missing baselines
    if missing_baselines:
        print(f"\n💡 To generate missing baselines, run evaluate_model.py for each missing configuration.")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
