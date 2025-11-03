#!/usr/bin/env python3
"""
Embeddings-Only Training Runner

Runs train_graph.py with --export_embeddings flag for specified configurations.
Stops after training and embedding extraction (no evaluation).

Usage:
    python3 run_embeddings_only.py --dataset insulator --model DMPNN --incl_rdkit
    python3 run_embeddings_only.py --config batch_experiments.yaml --model wDMPNN
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_embeddings_training(dataset, model, **kwargs):
    """Run train_graph.py with embeddings export for a single configuration."""
    
    # Build command arguments
    cmd = [
        sys.executable, 'scripts/python/train_graph.py',
        '--dataset_name', dataset,
        '--model_name', model,
        '--export_embeddings'  # Always include for embeddings-only mode
    ]
    
    # Add optional arguments
    if kwargs.get('task_type') and kwargs['task_type'] != 'reg':
        cmd.extend(['--task_type', kwargs['task_type']])
    
    if kwargs.get('incl_rdkit', False):
        cmd.append('--incl_rdkit')
    
    if kwargs.get('incl_desc', False):
        cmd.append('--incl_desc')
    
    if kwargs.get('incl_ab', False):
        cmd.append('--incl_ab')
    
    if kwargs.get('batch_norm', False):
        cmd.append('--batch_norm')
    
    if kwargs.get('pretrain_monomer', False):
        cmd.append('--pretrain_monomer')
    
    if kwargs.get('target'):
        cmd.extend(['--target', kwargs['target']])
    
    if kwargs.get('train_size'):
        cmd.extend(['--train_size', str(kwargs['train_size'])])
    
    # Log configuration
    logger.info(f"Running embeddings-only training:")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Task Type: {kwargs.get('task_type', 'reg')}")
    logger.info(f"  Target: {kwargs.get('target', 'All targets')}")
    logger.info(f"  RDKit: {kwargs.get('incl_rdkit', False)}")
    logger.info(f"  Descriptors: {kwargs.get('incl_desc', False)}")
    logger.info(f"  Batch Norm: {kwargs.get('batch_norm', False)}")
    logger.info(f"  Train Size: {kwargs.get('train_size', 'full')}")
    
    # Run the command
    cmd_str = ' '.join(cmd)
    logger.info(f"Command: {cmd_str}")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ Embeddings training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Embeddings training failed with exit code {e.returncode}")
        return False


def load_config(config_file):
    """Load experiment configurations from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('experiments', [])
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Run embeddings-only training for DMPNN models')
    
    # Single experiment arguments
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name (DMPNN, wDMPNN, etc.)')
    
    # Batch configuration
    parser.add_argument('--config', type=str, help='YAML config file with batch experiments')
    parser.add_argument('--model_filter', type=str, help='Filter experiments by model name')
    
    # Optional flags
    parser.add_argument('--incl_rdkit', action='store_true', help='Include RDKit descriptors')
    parser.add_argument('--incl_desc', action='store_true', help='Include dataset-specific descriptors')
    parser.add_argument('--incl_ab', action='store_true', help='Include atom/bond pooled features')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--pretrain_monomer', action='store_true', help='Train multitask monomer model')
    parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default='reg',
                       help='Task type')
    parser.add_argument('--target', type=str, help='Specific target to train')
    parser.add_argument('--train_size', type=str, help='Training size (e.g., "500", "full")')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config and not (args.dataset and args.model):
        logger.error("Either --config (for batch) or --dataset and --model (for single) must be specified")
        return 1
    
    success_count = 0
    total_count = 0
    
    if args.config:
        # Batch mode
        experiments = load_config(args.config)
        if not experiments:
            logger.error("No experiments found in config file")
            return 1
        
        logger.info(f"Running {len(experiments)} experiments from config: {args.config}")
        
        # Track unique combinations to avoid duplicates
        seen_combinations = set()
        
        for exp in experiments:
            dataset = exp.get('dataset')
            model = exp.get('model')
            
            if not dataset or not model:
                logger.warning(f"Skipping incomplete experiment: {exp}")
                continue
            
            # Skip tabular models (no embeddings)
            if model == 'tabular':
                logger.info(f"Skipping {dataset}-{model} (tabular model doesn't support embeddings)")
                continue
            
            # Apply model filter
            if args.model_filter and model != args.model_filter:
                logger.info(f"Skipping {dataset}-{model} (model filter: {args.model_filter})")
                continue
            
            # Handle targets
            targets = exp.get('targets', [])
            single_target = exp.get('target')
            
            if targets:
                target_list = targets if isinstance(targets, list) else [targets]
            elif single_target:
                target_list = [single_target]
            else:
                target_list = [None]
            
            # Build experiment configuration
            exp_kwargs = {
                'task_type': exp.get('task_type', 'reg'),
                'incl_rdkit': exp.get('incl_rdkit', False),
                'incl_desc': exp.get('incl_desc', False),
                'incl_ab': exp.get('incl_ab', False),
                'batch_norm': exp.get('batch_norm', False),
                'pretrain_monomer': exp.get('pretrain_monomer', False),
                'train_size': exp.get('train_size'),
                'polymer_type': exp.get('polymer_type')
            }
            
            # Run for each target (unique combinations only)
            for target in target_list:
                # Create unique combination key
                combo_key = (dataset, model, target, 
                            exp_kwargs['task_type'],
                            exp_kwargs['incl_rdkit'],
                            exp_kwargs['incl_desc'], 
                            exp_kwargs['incl_ab'],
                            exp_kwargs['batch_norm'],
                            exp_kwargs['train_size'],
                            exp_kwargs['polymer_type'])
                
                if combo_key in seen_combinations:
                    target_info = f" [target: {target}]" if target else " [all targets]"
                    logger.info(f"Skipping duplicate: {dataset} {model}{target_info}")
                    continue
                
                seen_combinations.add(combo_key)
                
                total_count += 1
                exp_kwargs['target'] = target
                
                target_info = f" [target: {target}]" if target else " [all targets]"
                logger.info(f"Processing: {dataset} {model}{target_info}")
                
                if run_embeddings_training(dataset, model, **exp_kwargs):
                    success_count += 1
    
    else:
        # Single experiment mode
        total_count = 1
        exp_kwargs = {
            'task_type': args.task_type,
            'incl_rdkit': args.incl_rdkit,
            'incl_desc': args.incl_desc,
            'incl_ab': args.incl_ab,
            'batch_norm': args.batch_norm,
            'pretrain_monomer': args.pretrain_monomer,
            'target': args.target,
            'train_size': args.train_size
        }
        
        if run_embeddings_training(args.dataset, args.model, **exp_kwargs):
            success_count = 1
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"Embeddings training complete: {success_count}/{total_count} successful")
    
    if success_count < total_count:
        logger.warning(f"❌ {total_count - success_count} experiments failed")
        return 1
    else:
        logger.info("✅ All experiments completed successfully!")
        logger.info("Embeddings saved to: results/embeddings/")
        return 0


if __name__ == '__main__':
    sys.exit(main())
