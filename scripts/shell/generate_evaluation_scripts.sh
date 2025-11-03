#!/bin/bash

# Generate individual evaluation shell scripts for each model/dataset/configuration combination
# This script scans checkpoints and creates PBS job scripts for evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/evaluation_config.yaml"
OUTPUT_DIR="$SCRIPT_DIR"

# PBS configuration
PBS_QUEUE="gpuvolta"
PBS_PROJECT="um09"
PBS_NCPUS="12"
PBS_NGPUS="1"
PBS_MEM="100GB"
PBS_STORAGE="scratch/um09+gdata/dk92"
PBS_JOBFS="100GB"

# Default walltime
DEFAULT_WALLTIME="2:00:00"

# Parse command line arguments
FORCE=false
DRY_RUN=false

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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force      Overwrite existing evaluation scripts"
            echo "  --dry-run    Show what would be generated without creating files"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to get walltime from config
get_walltime() {
    local dataset=$1
    if [ -f "$CONFIG_FILE" ]; then
        local walltime=$(grep "^  ${dataset}:" "$CONFIG_FILE" | cut -d'"' -f2)
        if [ -n "$walltime" ]; then
            echo "$walltime"
            return
        fi
    fi
    echo "$DEFAULT_WALLTIME"
}

# Function to parse experiment name
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

# Function to build result filename
build_result_filename() {
    local dataset=$1
    local has_desc=$2
    local has_rdkit=$3
    local has_batch_norm=$4
    local train_size=$5
    
    local parts=("$dataset")
    
    if [ "$has_desc" = true ]; then
        parts+=("desc")
    fi
    
    if [ "$has_rdkit" = true ]; then
        parts+=("rdkit")
    fi
    
    if [ "$has_batch_norm" = true ]; then
        parts+=("batch_norm")
    fi
    
    if [ -n "$train_size" ]; then
        parts+=("$train_size")
    fi
    
    # Join with __
    local result=$(IFS=__; echo "${parts[*]}")
    echo "${result}_baseline.csv"
}

# Function to generate evaluation script
generate_eval_script() {
    local model=$1
    local dataset=$2
    local has_desc=$3
    local has_rdkit=$4
    local has_batch_norm=$5
    local train_size=$6
    local checkpoint_path=$7
    local preprocessing_path=$8
    
    # Build script name
    local script_name="eval_${dataset}_${model}"
    
    if [ "$has_desc" = true ]; then
        script_name="${script_name}_desc"
    fi
    
    if [ "$has_rdkit" = true ]; then
        script_name="${script_name}_rdkit"
    fi
    
    if [ "$has_batch_norm" = true ]; then
        script_name="${script_name}_batch_norm"
    fi
    
    if [ -n "$train_size" ]; then
        script_name="${script_name}_${train_size}"
    fi
    
    script_name="${script_name}.sh"
    local script_path="$OUTPUT_DIR/$script_name"
    
    # Check if script already exists
    if [ -f "$script_path" ] && [ "$FORCE" = false ]; then
        echo "⏭️  Skipping (exists): $script_name"
        return
    fi
    
    # Get walltime for dataset
    local walltime=$(get_walltime "$dataset")
    
    # Build evaluate_model.py arguments
    local eval_args="--model_name $model --dataset_name $dataset"
    
    if [ "$has_desc" = true ]; then
        eval_args="$eval_args --incl_descriptors"
    fi
    
    if [ "$has_rdkit" = true ]; then
        eval_args="$eval_args --incl_rdkit"
    fi
    
    if [ -n "$checkpoint_path" ]; then
        eval_args="$eval_args --checkpoint_path $checkpoint_path"
    fi
    
    if [ -n "$preprocessing_path" ]; then
        eval_args="$eval_args --preprocessing_path $preprocessing_path"
    fi
    
    # Build result filename
    local result_file=$(build_result_filename "$dataset" "$has_desc" "$has_rdkit" "$has_batch_norm" "$train_size")
    
    if [ "$DRY_RUN" = true ]; then
        echo "Would generate: $script_name"
        echo "  Model: $model"
        echo "  Dataset: $dataset"
        echo "  Desc: $has_desc, RDKit: $has_rdkit, BatchNorm: $has_batch_norm"
        echo "  Train size: ${train_size:-full}"
        echo "  Walltime: $walltime"
        echo "  Result: $result_file"
        echo ""
        return
    fi
    
    # Generate the script
    cat > "$script_path" << EOF
#!/bin/bash

#PBS -q $PBS_QUEUE
#PBS -P $PBS_PROJECT
#PBS -l ncpus=$PBS_NCPUS
#PBS -l ngpus=$PBS_NGPUS
#PBS -l mem=$PBS_MEM
#PBS -l walltime=$walltime
#PBS -l storage=$PBS_STORAGE
#PBS -l jobfs=$PBS_JOBFS
#PBS -N eval-${dataset}-${model}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: $dataset
# Model: $model
# Descriptors: $has_desc
# RDKit: $has_rdkit
# Batch Norm: $has_batch_norm
# Train Size: ${train_size:-full}
# Expected Result: results/$model/$result_file

echo "Starting evaluation..."
echo "Model: $model"
echo "Dataset: $dataset"
echo "Configuration: desc=$has_desc, rdkit=$has_rdkit, batch_norm=$has_batch_norm"

python3 scripts/python/evaluate_model.py \\
    $eval_args

echo "Evaluation complete!"
echo "Results saved to: results/$model/$result_file"
EOF

    chmod +x "$script_path"
    echo "✅ Generated: $script_name"
}

# Main execution
echo "🔍 Scanning checkpoints directory..."
cd "$PROJECT_ROOT" || exit 1

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -d "$PREPROCESSING_DIR" ]; then
    echo "❌ Preprocessing directory not found: $PREPROCESSING_DIR"
    exit 1
fi

# Track unique configurations (using a simple list instead of associative array for compatibility)
CONFIGS_FILE=$(mktemp)
trap "rm -f $CONFIGS_FILE" EXIT

# Scan checkpoint directories
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model=$(basename "$model_dir")
    
    for exp_dir in "$model_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        
        # Check if checkpoint exists
        if ! ls "$exp_dir"/best-*.ckpt >/dev/null 2>&1; then
            continue
        fi
        
        # Parse experiment name
        IFS='|' read -r dataset has_desc has_rdkit has_batch_norm train_size <<< "$(parse_experiment_name "$exp_name")"
        
        if [ -z "$dataset" ]; then
            continue
        fi
        
        # Check if preprocessing exists
        preprocess_path="$PREPROCESSING_DIR/$exp_name"
        if [ ! -d "$preprocess_path" ]; then
            preprocess_path=""
        fi
        
        # Create unique config key
        config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}|${train_size}"
        
        # Skip if we've already seen this configuration
        if grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null; then
            continue
        fi
        
        echo "$config_key" >> "$CONFIGS_FILE"
        
        # Generate script for this configuration
        generate_eval_script "$model" "$dataset" "$has_desc" "$has_rdkit" "$has_batch_norm" "$train_size" "$exp_dir" "$preprocess_path"
    done
done

# Summary
if [ -f "$CONFIGS_FILE" ]; then
    TOTAL_CONFIGS=$(wc -l < "$CONFIGS_FILE" | tr -d ' ')
else
    TOTAL_CONFIGS=0
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 GENERATION SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo "Total unique configurations: $TOTAL_CONFIGS"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a DRY RUN. No files were created."
    echo "Run without --dry-run to generate the scripts."
else
    echo "Scripts generated in: $OUTPUT_DIR"
    echo ""
    echo "To submit all evaluations:"
    echo "  for script in $OUTPUT_DIR/eval_*.sh; do qsub \$script; done"
fi

exit 0
