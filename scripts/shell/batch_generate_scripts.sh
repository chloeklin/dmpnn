#!/bin/bash

# Batch Training Script Generator
# Reads experiment configurations from a YAML file and generates training scripts
#
# Usage: ./batch_generate_scripts.sh <config_file.yaml> [--no-submit]
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
#     model: tabular
#     walltime: "1:30:00"
#     incl_desc: true
#     incl_ab: true
#     task_type: binary

# Check if python3 and yq are available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file.yaml> [--no-submit]"
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
    echo "    model: tabular"
    echo "    walltime: \"1:30:00\""
    echo "    incl_desc: true"
    echo "    incl_ab: true"
    echo "    task_type: binary"
    echo ""
    echo "Optional flags: incl_rdkit, incl_desc, incl_ab (true/false)"
    echo "task_type defaults to 'reg' if not specified"
    exit 1
fi

CONFIG_FILE="$1"
NO_SUBMIT=""

# Check for --no-submit flag
if [[ "$2" == "--no-submit" ]]; then
    NO_SUBMIT="--no-submit"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "Batch generating training scripts from: $CONFIG_FILE"
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
    
    for exp in config['experiments']:
        # Required fields
        dataset = exp.get('dataset')
        model = exp.get('model')
        walltime = exp.get('walltime')
        
        if not all([dataset, model, walltime]):
            print(f"Skipping incomplete experiment: {exp}")
            continue
        
        # Build arguments
        args = ['./scripts/shell/generate_training_script.sh', dataset, model, walltime]
        
        # Add optional flags
        if exp.get('incl_rdkit', False):
            args.append('incl_rdkit')
        
        if exp.get('incl_desc', False):
            args.append('incl_desc')
        
        if exp.get('incl_ab', False):
            args.append('incl_ab')
        
        # Add task type if not default
        task_type = exp.get('task_type', 'reg')
        if task_type != 'reg':
            args.append(task_type)
        
        # Add no-submit flag if specified
        if '$NO_SUBMIT':
            args.append('$NO_SUBMIT')
        
        print(f"Generating: {' '.join(args[1:])}")
        
        # Call the generate_training_script.sh
        result = subprocess.run(args, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error generating script for {dataset}-{model}: {result.stderr}")
        else:
            print(result.stdout)
        
        print()

except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

EOF

echo "=================================================="
echo "Batch script generation complete!"
