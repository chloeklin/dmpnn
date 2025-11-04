#!/bin/bash

# Debug script to see what's happening with generate_evaluation_scripts.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🔍 Debug: Scanning checkpoints..."
echo "Project root: $PROJECT_ROOT"

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
echo "Checkpoint dir: $CHECKPOINT_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo ""
echo "📁 Model directories found:"
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model=$(basename "$model_dir")
    echo "  Model: $model"
    
    # Count experiments for this model
    exp_count=0
    checkpoint_count=0
    
    for exp_dir in "$model_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        exp_count=$((exp_count + 1))
        
        # Check if checkpoint exists
        if ls "$exp_dir"/best-*.ckpt >/dev/null 2>&1; then
            checkpoint_count=$((checkpoint_count + 1))
            echo "    ✅ $exp_name (has checkpoint)"
        else
            echo "    ❌ $exp_name (no checkpoint)"
        fi
    done
    
    echo "    Total experiments: $exp_count, With checkpoints: $checkpoint_count"
    echo ""
done

echo ""
echo "🧪 Testing parse function on sample experiments..."

# Function to parse experiment name (copied from original script)
parse_experiment_name() {
    local exp_name=$1
    local dataset=""
    local has_desc=false
    local has_rdkit=false
    local has_batch_norm=false
    local train_size=""
    
    # Remove rep suffix
    exp_name=$(echo "$exp_name" | sed 's/__rep[0-9]*$//')
    
    # Split by __
    IFS='__' read -ra PARTS <<< "$exp_name"
    dataset="${PARTS[0]}"
    
    # Check for suffixes
    for part in "${PARTS[@]:1}"; do
        case "$part" in
            desc)
                has_desc=true
                ;;
            rdkit)
                has_rdkit=true
                ;;
            batch_norm)
                has_batch_norm=true
                ;;
            size*)
                train_size="$part"
                ;;
        esac
    done
    
    echo "$dataset|$has_desc|$has_rdkit|$has_batch_norm|$train_size"
}

# Test parsing on a few real experiment names
sample_count=0
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    for exp_dir in "$model_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        
        # Only test first few
        if [ $sample_count -ge 5 ]; then
            break 2
        fi
        
        echo "Testing: $exp_name"
        result=$(parse_experiment_name "$exp_name")
        IFS='|' read -r dataset has_desc has_rdkit has_batch_norm train_size <<< "$result"
        echo "  -> Dataset: '$dataset', Desc: $has_desc, RDKit: $has_rdkit, BatchNorm: $has_batch_norm, Size: '$train_size'"
        
        if [ -z "$dataset" ]; then
            echo "  ❌ PARSING FAILED - empty dataset!"
        else
            echo "  ✅ Parsing OK"
        fi
        
        sample_count=$((sample_count + 1))
        echo ""
    done
done
