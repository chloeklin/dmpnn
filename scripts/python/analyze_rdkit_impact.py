"""
Analyze the impact of including RDKit descriptors on DMPNN model performance.
Compares results from {dataset}_results.csv vs {dataset}__rdkit_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def find_dataset_pairs(results_dir):
    """Find pairs of results files with and without rdkit."""
    results_path = Path(results_dir)
    
    # Find all base results files (without rdkit)
    base_files = []
    for f in results_path.glob("*_results.csv"):
        fname = f.stem
        # Exclude files that already have rdkit, desc, batch_norm, or size suffixes
        if "__rdkit" not in fname and "__desc" not in fname and "__batch_norm" not in fname and "__size" not in fname:
            base_files.append(f)
    
    pairs = []
    for base_file in base_files:
        dataset_name = base_file.stem.replace("_results", "")
        rdkit_file = results_path / f"{dataset_name}__rdkit_results.csv"
        
        if rdkit_file.exists():
            pairs.append({
                'dataset': dataset_name,
                'base_file': base_file,
                'rdkit_file': rdkit_file
            })
    
    return pairs

def calculate_degradation(base_df, rdkit_df):
    """Calculate performance degradation metrics."""
    results = []
    
    # Data is in long format with columns: target, split, test/mae, test/r2, test/rmse
    # Get unique targets
    base_targets = base_df['target'].unique()
    rdkit_targets = rdkit_df['target'].unique()
    
    common_targets = set(base_targets) & set(rdkit_targets)
    
    # Metrics to compare
    metrics = ['test/rmse', 'test/r2', 'test/mae']
    
    for target in sorted(common_targets):
        base_target_data = base_df[base_df['target'] == target]
        rdkit_target_data = rdkit_df[rdkit_df['target'] == target]
        
        for metric_col in metrics:
            if metric_col not in base_df.columns or metric_col not in rdkit_df.columns:
                continue
            
            base_values = base_target_data[metric_col].values
            rdkit_values = rdkit_target_data[metric_col].values
            
            # Calculate mean values
            base_mean = np.mean(base_values)
            rdkit_mean = np.mean(rdkit_values)
            
            # Extract metric name (e.g., "test/rmse" -> "rmse")
            metric_name = metric_col.split('/')[-1]
            
            # Calculate percentage change
            if metric_name.lower() == 'rmse':
                # For RMSE, increase is bad
                pct_change = ((rdkit_mean - base_mean) / base_mean) * 100
                change_type = 'increase'
            elif metric_name.lower() == 'r2':
                # For R2, decrease is bad
                pct_change = ((base_mean - rdkit_mean) / base_mean) * 100
                change_type = 'decrease'
            elif metric_name.lower() == 'mae':
                # For MAE, increase is bad
                pct_change = ((rdkit_mean - base_mean) / base_mean) * 100
                change_type = 'increase'
            else:
                # For other metrics, just calculate raw change
                pct_change = ((rdkit_mean - base_mean) / base_mean) * 100
                change_type = 'change'
            
            results.append({
                'target': target,
                'metric': metric_name,
                'base_mean': base_mean,
                'rdkit_mean': rdkit_mean,
                'pct_change': pct_change,
                'change_type': change_type
            })
    
    return results

def main():
    results_dir = "/Users/u6788552/Desktop/experiments/dmpnn/results/DMPNN"
    
    # Find all dataset pairs
    pairs = find_dataset_pairs(results_dir)
    
    print(f"Found {len(pairs)} dataset pairs for comparison\n")
    print("=" * 100)
    
    all_results = []
    
    for pair in pairs:
        dataset = pair['dataset']
        print(f"\n### Dataset: {dataset}")
        print("-" * 100)
        
        # Load the results
        base_df = pd.read_csv(pair['base_file'])
        rdkit_df = pd.read_csv(pair['rdkit_file'])
        
        # Calculate degradation
        degradation = calculate_degradation(base_df, rdkit_df)
        
        if not degradation:
            print("  No comparable metrics found")
            continue
        
        # Print results for this dataset
        for result in degradation:
            metric_upper = result['metric'].upper()
            
            if result['metric'].lower() == 'rmse':
                print(f"  {result['target']} - {metric_upper}:")
                print(f"    Without RDKit: {result['base_mean']:.4f}")
                print(f"    With RDKit:    {result['rdkit_mean']:.4f}")
                print(f"    Increase:      {result['pct_change']:+.2f}%")
            elif result['metric'].lower() == 'r2':
                print(f"  {result['target']} - {metric_upper}:")
                print(f"    Without RDKit: {result['base_mean']:.4f}")
                print(f"    With RDKit:    {result['rdkit_mean']:.4f}")
                print(f"    Decrease:      {result['pct_change']:+.2f}%")
            
            # Store for summary
            all_results.append({
                'dataset': dataset,
                **result
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(all_results)
    
    # Print overall summary
    print("\n" + "=" * 100)
    print("\n### OVERALL SUMMARY")
    print("-" * 100)
    
    if len(summary_df) > 0:
        # Group by metric type
        for metric in summary_df['metric'].unique():
            metric_data = summary_df[summary_df['metric'] == metric]
            avg_change = metric_data['pct_change'].mean()
            
            if metric.lower() == 'rmse':
                print(f"\n{metric.upper()} - Average increase across all targets: {avg_change:+.2f}%")
                print(f"  Best case (smallest increase):  {metric_data['pct_change'].min():+.2f}%")
                print(f"  Worst case (largest increase):  {metric_data['pct_change'].max():+.2f}%")
            elif metric.lower() == 'r2':
                print(f"\n{metric.upper()} - Average decrease across all targets: {avg_change:+.2f}%")
                print(f"  Best case (smallest decrease):  {metric_data['pct_change'].min():+.2f}%")
                print(f"  Worst case (largest decrease):  {metric_data['pct_change'].max():+.2f}%")
    
    # Save detailed results to CSV
    output_file = Path(results_dir) / "rdkit_impact_analysis.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")
    print("=" * 100)

if __name__ == "__main__":
    main()
