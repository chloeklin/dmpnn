#!/usr/bin/env python3
"""
Batch evaluation script that runs evaluate_model.py on all available datasets
while skipping datasets that already have existing result CSV files.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def find_datasets(data_dir):
    """Find all available CSV datasets."""
    data_path = Path(data_dir)
    datasets = []
    
    for csv_file in data_path.glob("*.csv"):
        # Skip preprocessed versions
        if "preprocessed" not in csv_file.stem:
            datasets.append(csv_file.stem)
    
    return sorted(datasets)

def check_existing_results(dataset_name, model_name, results_dir, variant_configs):
    """Check if result files already exist for a dataset/model/variant combination."""
    results_path = Path(results_dir) / model_name
    existing_files = []
    
    for variant_name, variant_args in variant_configs.items():
        # Build expected filename based on variant
        desc_suffix = "__desc" if "--incl_desc" in variant_args else ""
        rdkit_suffix = "__rdkit" if "--incl_rdkit" in variant_args else ""
        
        # Only check for baseline files (since evaluate_model.py creates _baseline.csv)
        baseline_file = results_path / f"{dataset_name}{desc_suffix}{rdkit_suffix}_baseline.csv"
        
        if baseline_file.exists():
            existing_files.append(str(baseline_file))
    
    return existing_files

def check_trained_checkpoints(dataset_name, model_name, checkpoint_dir):
    """Check if trained model checkpoints exist for a dataset/model combination."""
    checkpoint_path = Path(checkpoint_dir) / model_name
    
    # Look for any checkpoint directories that match the dataset pattern
    pattern = f"{dataset_name}__*"
    matching_dirs = list(checkpoint_path.glob(pattern))
    
    if not matching_dirs:
        return False, []
    
    # Check if any of the matching directories contain actual checkpoint files
    valid_checkpoints = []
    for checkpoint_dir in matching_dirs:
        # Look for .ckpt files in the directory
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        if ckpt_files:
            valid_checkpoints.append(str(checkpoint_dir))
    
    return len(valid_checkpoints) > 0, valid_checkpoints

def run_evaluation(dataset_name, model_name, task_type, variant_args, script_dir):
    """Run evaluate_model.py with specified parameters."""
    script_path = Path(script_dir) / "scripts" / "python" / "evaluate_model.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--dataset_name", dataset_name,
        "--task_type", task_type,
        "--model_name", model_name
    ]
    
    # Add variant-specific arguments
    cmd.extend(variant_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Successfully evaluated {dataset_name} with {' '.join(variant_args)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to evaluate {dataset_name} with {' '.join(variant_args)}")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate all datasets')
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='Directory containing CSV datasets')
    parser.add_argument('--results_dir', type=str, default='../../results',
                        help='Directory to store results')
    parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoints',
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--script_dir', type=str, default='.',
                        help='Directory containing evaluate_model.py')
    parser.add_argument('--models', type=str, nargs='+', choices=['DMPNN', 'wDMPNN'], 
                        default=['DMPNN', 'wDMPNN'], help='Models to use for evaluation')
    parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'],
                        default='reg', help='Task type')
    parser.add_argument('--force', action='store_true',
                        help='Force re-evaluation even if results exist')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be evaluated without running')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    script_dir = Path(args.script_dir).resolve()
    
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Script directory: {script_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Task type: {args.task_type}")
    print(f"Force re-evaluation: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 60)
    
    # Find all available datasets
    datasets = find_datasets(data_dir)
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    print("-" * 60)
    
    # Define evaluation variants
    variant_configs = {
        "original": [],
        "rdkit": ["--incl_rdkit"],
        # Add more variants as needed
        # "desc": ["--incl_desc"],
        # "desc_rdkit": ["--incl_desc", "--incl_rdkit"],
    }
    
    total_evaluations = 0
    skipped_evaluations = 0
    successful_evaluations = 0
    failed_evaluations = 0
    
    # Process each dataset with each model
    for dataset in datasets:
        print(f"\n📊 Processing dataset: {dataset}")
        
        for model_name in args.models:
            print(f"\n🤖 Model: {model_name}")
            
            # Check if trained checkpoints exist
            has_checkpoints, checkpoint_paths = check_trained_checkpoints(dataset, model_name, checkpoint_dir)
            
            if not has_checkpoints:
                print(f"⚠️  Skipping {dataset} ({model_name}) - no trained checkpoints found")
                skipped_evaluations += len(variant_configs)
                continue
            
            print(f"✅ Found {len(checkpoint_paths)} checkpoint directories")
            
            # Check existing results for all variants
            existing_files = check_existing_results(dataset, model_name, results_dir, variant_configs)
            
            if existing_files and not args.force:
                print(f"⏭️  Skipping {dataset} ({model_name}) - existing results found:")
                for file in existing_files:
                    print(f"   - {file}")
                skipped_evaluations += len(variant_configs)
                continue
            
            # Run evaluation for each variant
            for variant_name, variant_args in variant_configs.items():
                total_evaluations += 1
                
                print(f"\n🔄 Evaluating {dataset} ({model_name}) with variant: {variant_name}")
                
                if args.dry_run:
                    print(f"   [DRY RUN] Would run: {dataset} {model_name} {variant_args}")
                    continue
                
                success = run_evaluation(dataset, model_name, args.task_type, 
                                       variant_args, script_dir)
                
                if success:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total datasets found: {len(datasets)}")
    print(f"Total evaluations planned: {total_evaluations}")
    print(f"Skipped (existing results): {skipped_evaluations}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    
    if args.dry_run:
        print("\n🔍 This was a dry run - no evaluations were actually performed.")
    
    if failed_evaluations > 0:
        print(f"\n⚠️  {failed_evaluations} evaluations failed. Check the error messages above.")
        sys.exit(1)
    else:
        print(f"\n✅ All evaluations completed successfully!")

if __name__ == "__main__":
    main()
