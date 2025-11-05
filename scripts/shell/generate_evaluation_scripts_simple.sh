#!/bin/bash

# Ultra-simple evaluation script generator
# Avoids all complex operations that might hang on network filesystems

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
EVAL_SCRIPTS_DIR="$PROJECT_ROOT/scripts/shell/eval_scripts"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/python/evaluate_model.py"

# Command line options
FORCE=false
DRY_RUN=false
AUTO_SUBMIT=false
SPECIFIC_MODEL=""
SPECIFIC_DATASET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --auto-submit)
            AUTO_SUBMIT=true
            shift
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --dataset)
            SPECIFIC_DATASET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force            Overwrite existing evaluation scripts"
            echo "  --dry-run          Show what would be generated without creating files"
            echo "  --auto-submit      Automatically submit generated scripts with qsub"
            echo "  --model MODEL      Generate scripts only for specific model"
            echo "  --dataset DATASET  Generate scripts only for specific dataset"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🔍 Scanning checkpoints directory: $CHECKPOINTS_DIR"
echo "📝 Output directory: $EVAL_SCRIPTS_DIR"

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "❌ Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
fi

# Create output directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$EVAL_SCRIPTS_DIR"
fi

# Function to generate evaluation script
generate_eval_script() {
    local model="$1"
    local dataset="$2"
    local exp_name="$3"
    
    # Build script name
    local script_name="eval_${dataset}_${model}"
    
    # Add suffixes based on experiment name
    if [[ "$exp_name" == *"__desc"* ]]; then
        script_name="${script_name}_desc"
    fi
    if [[ "$exp_name" == *"__rdkit"* ]]; then
        script_name="${script_name}_rdkit"
    fi
    if [[ "$exp_name" == *"__batch_norm"* ]]; then
        script_name="${script_name}_batch_norm"
    fi
    
    script_name="${script_name}.sh"
    local script_path="$EVAL_SCRIPTS_DIR/$script_name"
    
    # Check if script already exists
    if [ -f "$script_path" ] && [ "$FORCE" = false ]; then
        echo "  ⏭️  Script exists: $script_name"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "  📝 Would create: $script_name"
        echo "     Model: $model, Dataset: $dataset, Experiment: $exp_name"
        return
    fi
    
    # Build evaluation arguments - let evaluate_model.py figure out the flags
    local eval_args="--model_name $model --dataset_name $dataset"
    
    # Generate the script
    cat > "$script_path" << EOF
#!/bin/bash
#PBS -N eval_${dataset}_${model}
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M \${USER}@student.unsw.edu.au
#PBS -m abe

# Auto-generated evaluation script
# Model: $model
# Dataset: $dataset
# Experiment: $exp_name
# Generated: $(date)

cd \$PBS_O_WORKDIR

# Load environment
module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

# Run evaluation (evaluate_model.py will auto-detect configuration)
python3 $PYTHON_SCRIPT \\
    $eval_args

echo "Evaluation completed for $model on $dataset ($exp_name)"
EOF
    
    chmod +x "$script_path"
    echo "  ✅ Created: $script_name"
    
    # Auto-submit if requested
    if [ "$AUTO_SUBMIT" = true ]; then
        echo "  🚀 Submitting: $script_name"
        qsub "$script_path"
    fi
}

# Track generated experiments
GENERATED_CONFIGS_FILE=$(mktemp)
trap "rm -f $GENERATED_CONFIGS_FILE" EXIT

total_scripts=0
total_experiments=0

echo ""
echo "🔍 Scanning for completed experiments..."

# Iterate through model directories
for model_dir in "$CHECKPOINTS_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model=$(basename "$model_dir")
    
    # Skip if specific model requested and this isn't it
    if [ -n "$SPECIFIC_MODEL" ] && [ "$model" != "$SPECIFIC_MODEL" ]; then
        continue
    fi
    
    echo ""
    echo "📂 Model: $model"
    
    # Simple approach: just look for rep0 directories and assume they exist
    exp_count=0
    complete_count=0
    
    for exp_dir in "$model_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        exp_count=$((exp_count + 1))
        
        # Only process rep0 directories to avoid duplicates
        if [[ ! "$exp_name" =~ __rep0$ ]]; then
            continue
        fi
        
        # Remove the __rep0 suffix to get the base experiment name
        base_exp_name="${exp_name%__rep0}"
        
        # Parse dataset from experiment name
        dataset=$(echo "$base_exp_name" | cut -d'__' -f1)
        
        # Skip if specific dataset requested and this isn't it
        if [ -n "$SPECIFIC_DATASET" ] && [ "$dataset" != "$SPECIFIC_DATASET" ]; then
            continue
        fi
        
        # Create config key to avoid duplicates
        config_key="${model}|${dataset}|${base_exp_name}"
        
        # Skip if we've already processed this configuration
        if grep -Fxq "$config_key" "$GENERATED_CONFIGS_FILE" 2>/dev/null; then
            continue
        fi
        
        echo "$config_key" >> "$GENERATED_CONFIGS_FILE"
        ((complete_count++))
        
        echo "  ✅ $dataset (experiment: $base_exp_name)"
        
        # Generate script
        generate_eval_script "$model" "$dataset" "$base_exp_name"
        ((total_scripts++))
        
        # Progress indicator
        if [ $((complete_count % 10)) -eq 0 ]; then
            echo "    ... processed $complete_count experiments"
        fi
    done
    
    echo "  📊 Model $model: $exp_count total, $complete_count complete"
    ((total_experiments += complete_count))
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 GENERATION SUMMARY"
echo "════════════════════════════════════════════════════════════"
if [ -n "$SPECIFIC_MODEL" ]; then
    echo "Model filter: $SPECIFIC_MODEL"
fi
if [ -n "$SPECIFIC_DATASET" ]; then
    echo "Dataset filter: $SPECIFIC_DATASET"
fi
echo "Complete experiments found: $total_experiments"
echo "Evaluation scripts generated: $total_scripts"
echo "Output directory: $EVAL_SCRIPTS_DIR"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - No files were created"
    echo "Remove --dry-run to generate actual scripts"
fi

echo ""
echo "✅ Script generation complete!"
