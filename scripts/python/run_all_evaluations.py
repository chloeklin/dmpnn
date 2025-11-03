#!/usr/bin/env python3
"""
Automated evaluation script for all available checkpoints and preprocessing files.
Runs evaluate_model.py for all valid model/dataset/target combinations.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set
import argparse


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """Parse experiment name to extract components.
    
    Format: {dataset}__{desc}__{rdkit}__{batch_norm}__{size{N}}__rep{i}
    Example: insulator__rdkit__batch_norm__rep0
    """
    # Remove rep suffix first
    rep_match = re.search(r'__rep(\d+)$', exp_name)
    if not rep_match:
        return {}
    
    rep_num = rep_match.group(1)
    base_name = exp_name[:-len(f"__rep{rep_num}")]
    
    # Split by double underscore
    parts = base_name.split('__')
    if len(parts) < 1:
        return {}
    
    result = {
        'dataset': parts[0],
        'replicate': rep_num,
        'has_desc': False,
        'has_rdkit': False,
        'has_batch_norm': False,
        'train_size': None
    }
    
    # Check for descriptor and other suffixes
    if len(parts) > 1:
        for part in parts[1:]:
            if part == 'desc':
                result['has_desc'] = True
            elif part == 'rdkit':
                result['has_rdkit'] = True
            elif part == 'batch_norm':
                result['has_batch_norm'] = True
            elif part.startswith('size'):
                result['train_size'] = part  # e.g., 'size500'
    
    return result


def find_available_experiments(checkpoint_dir: Path, preprocessing_dir: Path) -> List[Dict[str, str]]:
    """Find all experiments with checkpoints. Preprocessing files are optional for graph-only models.
    
    Note: Preprocessing files are now stored at dataset-level (not per-model) since they are
    identical across DMPNN, wDMPNN, and DMPNN_DiffPool.
    """
    
    experiments = []
    
    # Scan checkpoint directories
    for model_dir in checkpoint_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            exp_name = exp_dir.name
            exp_info = parse_experiment_name(exp_name)
            
            if not exp_info:
                continue
                
            # Check if checkpoint files exist
            best_ckpt = list(exp_dir.glob('best-*.ckpt'))
            if not best_ckpt:
                continue
                
            # Check if corresponding preprocessing files exist (optional for graph-only models)
            # Preprocessing is now at dataset-level (shared across models)
            preprocess_path = preprocessing_dir / exp_name
            has_preprocessing = False
            
            if preprocess_path.exists():
                # Check required preprocessing files
                required_files = [
                    'preprocessing_metadata_split_*.json',
                    'descriptor_scaler.pkl',
                    'correlation_mask.npy',
                    'constant_features_removed.npy'
                ]
                
                has_all_files = True
                for pattern in required_files:
                    if not list(preprocess_path.glob(pattern)):
                        has_all_files = False
                        break
                        
                has_preprocessing = has_all_files
                
            exp_info.update({
                'model': model_name,
                'exp_name': exp_name,
                'checkpoint_path': exp_dir,
                'preprocessing_path': preprocess_path if has_preprocessing else None,
                'has_preprocessing': has_preprocessing
            })
            
            experiments.append(exp_info)
    
    return experiments


def group_experiments_by_dataset_model(experiments: List[Dict[str, str]]) -> Dict[Tuple, List[Dict[str, str]]]:
    """Group experiments by (dataset, model, has_desc, has_rdkit, has_batch_norm, train_size) for batch evaluation."""
    
    groups = {}
    for exp in experiments:
        key = (exp['dataset'], exp['model'], exp['has_desc'], exp['has_rdkit'], 
               exp['has_batch_norm'], exp['train_size'])
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)
    
    return groups


def build_result_filename(dataset: str, has_desc: bool, has_rdkit: bool,
                         has_batch_norm: bool = False, train_size: str = None) -> str:
    """Build the expected result filename based on configuration.
    
    Note: evaluate_model.py generates files ending with '_baseline.csv'
    """
    parts = [dataset]
    if has_desc:
        parts.append('desc')
    if has_rdkit:
        parts.append('rdkit')
    if has_batch_norm:
        parts.append('batch_norm')
    if train_size:
        parts.append(train_size)
    
    return '__'.join(parts) + '_baseline.csv'


def check_result_exists(results_dir: Path, model: str, dataset: str, 
                       has_desc: bool, has_rdkit: bool,
                       has_batch_norm: bool = False, train_size: str = None) -> bool:
    """Check if result file already exists."""
    result_file = build_result_filename(dataset, has_desc, has_rdkit, has_batch_norm, train_size)
    result_path = results_dir / model / result_file
    return result_path.exists()


def run_evaluation(dataset: str, model: str, has_desc: bool, has_rdkit: bool,
                  has_batch_norm: bool = False, train_size: str = None,
                  results_dir: Path = None, force: bool = False,
                  dry_run: bool = False, verbose: bool = False) -> bool:
    """Run evaluate_model.py for a specific configuration."""
    
    # Check if result already exists
    if not force and results_dir and check_result_exists(results_dir, model, dataset, 
                                                          has_desc, has_rdkit, 
                                                          has_batch_norm, train_size):
        if verbose:
            suffix = f"desc={has_desc}, rdkit={has_rdkit}, batch_norm={has_batch_norm}"
            if train_size:
                suffix += f", {train_size}"
            print(f"⏭️  Skipping: {dataset} {model} ({suffix}) - result file already exists")
        return True
    
    cmd = [
        sys.executable, 'scripts/python/evaluate_model.py',
        '--dataset_name', dataset,
        '--model_name', model,
        '--task_type', 'multi' if dataset == 'polyinfo' else 'reg'
    ]
    
    if has_desc:
        cmd.append('--incl_desc')
    if has_rdkit:
        cmd.append('--incl_rdkit')
    if has_batch_norm:
        cmd.append('--batch_norm')
    if train_size:
        # Extract numeric part from 'size500' -> '500'
        size_num = train_size.replace('size', '')
        cmd.extend(['--train_size', size_num])
    
    cmd_str = ' '.join(cmd)
    
    if dry_run:
        print(f"[DRY RUN] Would run: {cmd_str}")
        return True
    
    if verbose:
        print(f"Running: {cmd_str}")
    
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, check=True)
        if verbose:
            suffix = f"desc={has_desc}, rdkit={has_rdkit}, batch_norm={has_batch_norm}"
            if train_size:
                suffix += f", {train_size}"
            print(f"✅ Success: {dataset} {model} ({suffix})")
        return True
    except subprocess.CalledProcessError as e:
        suffix = f"desc={has_desc}, rdkit={has_rdkit}, batch_norm={has_batch_norm}"
        if train_size:
            suffix += f", {train_size}"
        print(f"❌ Failed: {dataset} {model} ({suffix})")
        if verbose:
            print(f"Error: {e}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
        return False


def get_available_models(results_dir: Path) -> List[str]:
    """Dynamically detect available models from results directory."""
    if not results_dir.exists():
        return []
    
    models = []
    for item in results_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            models.append(item.name)
    
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(description='Run evaluations for all available checkpoints')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be run without executing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--dataset', type=str,
                       help='Only run evaluations for specific dataset')
    parser.add_argument('--model', type=str,
                       help='Only run evaluations for specific model (dynamically detected from results/)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-evaluation even if result files already exist')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue running even if some evaluations fail')
    
    args = parser.parse_args()
    
    # Set up paths
    root_dir = Path.cwd()
    checkpoint_dir = root_dir / 'checkpoints'
    preprocessing_dir = root_dir / 'preprocessing'
    results_dir = root_dir / 'results'
    
    # Get available models dynamically
    available_models = get_available_models(results_dir)
    if available_models:
        print(f"📦 Available models: {', '.join(available_models)}")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return 1
        
    if not preprocessing_dir.exists():
        print(f"❌ Preprocessing directory not found: {preprocessing_dir}")
        return 1
    
    # Find available experiments
    print("🔍 Scanning for available experiments...")
    experiments = find_available_experiments(checkpoint_dir, preprocessing_dir)
    
    if not experiments:
        print("❌ No valid experiments found with both checkpoints and preprocessing files")
        return 1
    
    # Filter by user criteria
    if args.dataset:
        experiments = [exp for exp in experiments if exp['dataset'] == args.dataset]
    if args.model:
        experiments = [exp for exp in experiments if exp['model'] == args.model]
    
    if not experiments:
        print("❌ No experiments match the specified criteria")
        return 1
    
    # Group by (dataset, model) for unique evaluations
    groups = group_experiments_by_dataset_model(experiments)
    
    print(f"📊 Found {len(experiments)} experiment directories")
    print(f"🎯 Will run {len(groups)} unique evaluations")
    
    if args.verbose:
        print("\nExperiment summary:")
        for (dataset, model, has_desc, has_rdkit, has_batch_norm, train_size), exps in groups.items():
            suffix = f"desc={has_desc}, rdkit={has_rdkit}, batch_norm={has_batch_norm}"
            if train_size:
                suffix += f", {train_size}"
            print(f"  {dataset} + {model}: {len(exps)} replicates ({suffix})")
    
    # Run evaluations
    print(f"\n{'🔍 DRY RUN MODE' if args.dry_run else '🚀 RUNNING EVALUATIONS'}")
    print("=" * 50)
    
    success_count = 0
    total_count = len(groups)
    
    for (dataset, model, has_desc, has_rdkit, has_batch_norm, train_size), exps in groups.items():
        
        success = run_evaluation(
            dataset=dataset,
            model=model, 
            has_desc=has_desc,
            has_rdkit=has_rdkit,
            has_batch_norm=has_batch_norm,
            train_size=train_size,
            results_dir=results_dir,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
        elif not args.continue_on_error:
            print(f"\n❌ Stopping due to error. Use --continue-on-error to continue.")
            break
    
    # Summary
    print("\n" + "=" * 50)
    if args.dry_run:
        print(f"🔍 DRY RUN COMPLETE: Would run {total_count} evaluations")
    else:
        print(f"✅ EVALUATION COMPLETE: {success_count}/{total_count} successful")
        if success_count < total_count:
            print(f"❌ {total_count - success_count} evaluations failed")
    
    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
