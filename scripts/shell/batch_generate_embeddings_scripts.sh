#!/bin/bash

# Batch Training Script Generator for Embeddings Only
# Reads experiment configurations from a YAML file and generates training scripts that stop after embedding extraction
#
# Usage: ./batch_generate_embeddings_scripts.sh [config_file.yaml] [--no-submit] [--model MODEL_NAME]
# Default config: batch_experiments.yaml
#
# YAML Format:
# experiments:
#   - dataset: polyinfo
#     model: DMPNN
#     walltime: "3:00:00"
#     incl_rdkit: true
#     incl_desc: true
#     task_type: reg
#   - dataset: insulator
#     model: wDMPNN
#     walltime: "1:30:00"
#     incl_rdkit: true
#     batch_norm: true
#     task_type: reg

# Check if python3 and yq are available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

# Show help if requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [config_file.yaml] [--no-submit] [--model MODEL_NAME]"
    echo ""
    echo "Default config: batch_experiments.yaml"
    echo ""
    echo "Examples:"
    echo "  $0                              # Use default config"
    echo "  $0 --no-submit                  # Use default, don't submit jobs"
    echo "  $0 --model DMPNN                # Only generate DMPNN scripts"
    echo "  $0 custom.yaml                  # Use custom config file"
    echo "  $0 custom.yaml --no-submit      # Custom config, no submit"
    echo ""
    echo "YAML file format:"
    echo "experiments:"
    echo "  - dataset: polyinfo"
    echo "    model: DMPNN"
    echo "    walltime: \"3:00:00\""
    echo "    incl_rdkit: true"
    echo "    incl_desc: true"
    echo "    task_type: reg"
    echo "  - dataset: insulator"
    echo "    model: wDMPNN"
    echo "    walltime: \"1:30:00\""
    echo "    incl_rdkit: true"
    echo "    batch_norm: true"
    echo "    task_type: reg"
    echo ""
    echo "Optional flags: incl_rdkit, incl_desc, incl_ab, batch_norm (true/false)"
    echo "task_type defaults to 'reg' if not specified"
    echo ""
    echo "Additional options:"
    echo "  --no-submit       Generate scripts without submitting to queue"
    echo "  --model MODEL     Only generate scripts for specified model (DMPNN, wDMPNN, etc.)"
    echo ""
    echo "Note: This script automatically adds export_embeddings flag"
    exit 1
fi

# Default config file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/batch_experiments.yaml"
CONFIG_FILE="$DEFAULT_CONFIG"
NO_SUBMIT=""
MODEL_FILTER=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-submit)
            NO_SUBMIT="--no-submit"
            shift
            ;;
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            # Help already shown above
            exit 0
            ;;
        *.yaml|*.yml)
            # Config file specified
            CONFIG_FILE="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "Batch generating embeddings-only training scripts from: $CONFIG_FILE"
if [[ -n "$MODEL_FILTER" ]]; then
    echo "Model filter: $MODEL_FILTER"
fi
echo "=================================================="

# Parse YAML and generate scripts using Python
python3 << EOF
import yaml
import subprocess
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    if 'experiments' not in config:
        print("Error: YAML file must contain 'experiments' key")
        sys.exit(1)
    
    # Track unique dataset-target-model combinations to avoid duplicates
    seen_combinations = set()
    
    for exp in config['experiments']:
        # Required fields
        dataset = exp.get('dataset')
        model = exp.get('model')
        walltime = exp.get('walltime')
        
        if not all([dataset, model, walltime]):
            print(f"Skipping incomplete experiment: {exp}")
            continue
        
        # Apply model filter if specified
        model_filter = '$MODEL_FILTER'
        if model_filter and model != model_filter:
            print(f"Skipping {dataset}-{model} (model filter: {model_filter})")
            continue
        
        # Only process graph models (not tabular)
        if model == 'tabular':
            print(f"Skipping {dataset}-{model} (tabular model doesn't support embeddings)")
            continue
        
        # Handle targets: can be a list, single value, or omitted
        targets = exp.get('targets')  # List of targets
        single_target = exp.get('target')  # Single target
        
        # Determine target list
        target_list = []
        if targets:  # If 'targets' list is provided
            target_list = targets if isinstance(targets, list) else [targets]
        elif single_target:  # If single 'target' is provided
            target_list = [single_target]
        else:  # No targets specified - use all targets (one script)
            target_list = [None]
        
        # Build arguments (without target-specific args)
        args = ['./generate_embeddings_script.sh', dataset, model, walltime]
        
        # Add optional flags
        if exp.get('incl_rdkit', False):
            args.append('incl_rdkit')
        
        if exp.get('incl_desc', False):
            args.append('incl_desc')
        
        if exp.get('incl_ab', False):
            args.append('incl_ab')
        
        if exp.get('batch_norm', False):
            args.append('batch_norm')
        
        # Add task type if not default
        task_type = exp.get('task_type', 'reg')
        if task_type != 'reg':
            args.append(task_type)
        
        # Add polymer type if specified
        polymer_type = exp.get('polymer_type')
        if polymer_type:
            args.append(polymer_type)
        
        # Add train_size if specified
        train_size = exp.get('train_size')
        if train_size:
            args.append(f'train_size={train_size}')
        
        # Generate scripts for unique combinations only
        for target in target_list:
            # Create unique combination key
            combo_key = (dataset, model, target, 
                        exp.get('incl_rdkit', False),
                        exp.get('incl_desc', False), 
                        exp.get('incl_ab', False),
                        exp.get('batch_norm', False),
                        task_type,
                        polymer_type,
                        train_size)
            
            if combo_key in seen_combinations:
                target_info = f" [target: {target}]" if target else " [all targets]"
                print(f"Skipping duplicate: {dataset} {model}{target_info}")
                continue
            
            seen_combinations.add(combo_key)
            
            target_args = args.copy()
            
            if target:
                target_args.append(f'target={target}')
            
            # Add no-submit flag if specified
            if '$NO_SUBMIT':
                target_args.append('$NO_SUBMIT')
            
            target_info = f" [target: {target}]" if target else " [all targets]"
            print(f"Generating embeddings script: {dataset} {model}{target_info}")
            
            # Call the generate_embeddings_script.sh
            result = subprocess.run(target_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            if result.returncode != 0:
                print(f"  ❌ Error: {result.stderr}")
            else:
                print(f"  ✅ {result.stdout.strip()}")
            
            print()

except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

EOF

echo "=================================================="
echo "Batch embeddings script generation complete!"
