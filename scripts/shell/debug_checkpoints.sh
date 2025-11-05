#!/bin/bash

# Debug script to check checkpoint directory structure
# Usage: ./debug_checkpoints.sh [model_name]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

MODEL_FILTER="$1"

echo "🔍 Debugging checkpoint directory: $CHECKPOINTS_DIR"
echo "=================================================="

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "❌ Checkpoints directory not found!"
    exit 1
fi

echo "📂 Top-level contents:"
ls -la "$CHECKPOINTS_DIR"
echo ""

for model_dir in "$CHECKPOINTS_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model=$(basename "$model_dir")
    
    if [ -n "$MODEL_FILTER" ] && [ "$model" != "$MODEL_FILTER" ]; then
        continue
    fi
    
    echo "📂 Model: $model"
    echo "  Directory: $model_dir"
    
    # Count total directories
    total_dirs=$(find "$model_dir" -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Total subdirectories: $((total_dirs-1))"
    
    # Count rep directories
    rep_dirs=$(find "$model_dir" -maxdepth 1 -type d -name "*__rep[0-4]" 2>/dev/null | wc -l)
    echo "  Rep directories: $rep_dirs"
    
    # Show first few examples
    echo "  First 5 examples:"
    find "$model_dir" -maxdepth 1 -type d -name "*__rep[0-4]" 2>/dev/null | head -5 | while read dir; do
        echo "    $(basename "$dir")"
    done
    
    # Check for any issues
    if [ $rep_dirs -eq 0 ]; then
        echo "  ⚠️  No rep directories found - checking naming pattern:"
        find "$model_dir" -maxdepth 1 -type d 2>/dev/null | head -10 | while read dir; do
            name=$(basename "$dir")
            if [ "$name" != "$model" ]; then
                echo "    $name"
            fi
        done
    fi
    
    echo ""
done

echo "=================================================="
echo "✅ Debug complete!"
