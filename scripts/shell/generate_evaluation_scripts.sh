#!/bin/bash

# Generate individual evaluation shell scripts for each model/dataset/configuration combination
# This script scans checkpoints and creates PBS job scripts for evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/evaluation_config.yaml"
OUTPUT_DIR="$SCRIPT_DIR"

echo "🔍 Debug: Script directory: $SCRIPT_DIR"
echo "🔍 Debug: Project root: $PROJECT_ROOT"

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
AUTO_SUBMIT=false
SKIP_BATCH_NORM=false

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
        --skip-batch-norm)
            SKIP_BATCH_NORM=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force            Overwrite existing evaluation scripts"
            echo "  --dry-run          Show what would be generated without creating files"
            echo "  --auto-submit      Automatically submit generated scripts with qsub"
            echo "  --skip-batch-norm  Skip configurations with batch normalization"
            echo "  --help, -h         Show this help message"
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
        # Look for the dataset under the walltime section
        local walltime=$(grep -A 20 "^walltime:" "$CONFIG_FILE" | grep "^  ${dataset}:" | cut -d'"' -f2)
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
    
    echo "DEBUG: generate_eval_script called with:"
    echo "  model='$model'"
    echo "  dataset='$dataset'" 
    echo "  has_desc='$has_desc'"
    echo "  has_rdkit='$has_rdkit'"
    echo "  has_batch_norm='$has_batch_norm'"
    echo "  train_size='$train_size'"
    echo "  checkpoint_path='$checkpoint_path'"
    echo "  preprocessing_path='$preprocessing_path'"
    
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
        echo "$script_path"  # Still output the path for tracking
        return
    fi
    
    # Get walltime for dataset
    local walltime=$(get_walltime "$dataset")
    
    # Build evaluate_model.py arguments
    # NOTE: We only pass model, dataset, and paths. Configuration (desc, rdkit, batch_norm)
    # is automatically extracted from the checkpoint path by evaluate_model.py
    local eval_args="--model_name $model --dataset_name $dataset"
    
    if [ -n "$checkpoint_path" ]; then
        eval_args="$eval_args --checkpoint_path \"$checkpoint_path\""
    fi
    
    if [ -n "$preprocessing_path" ]; then
        eval_args="$eval_args --preprocessing_path \"$preprocessing_path\""
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
    
    echo "DEBUG: About to create script: $script_path"
    echo "DEBUG: Script name: $script_name"
    echo "DEBUG: Walltime: $walltime"
    echo "DEBUG: Result file: $result_file"
    echo "DEBUG: eval_args: $eval_args"
    
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
# Configuration: Auto-detected from checkpoint path
# Expected Result: results/$model/$result_file

echo "Starting evaluation..."
echo "Model: $model"
echo "Dataset: $dataset"
echo "Configuration: Auto-detected from checkpoint path"

python3 scripts/python/evaluate_model.py \\
    $eval_args

echo "Evaluation complete!"
echo "Results saved to: results/$model/$result_file"
EOF

    echo "DEBUG: Script creation completed, checking file..."
    if [ -f "$script_path" ]; then
        echo "DEBUG: Script file exists"
        chmod +x "$script_path"
        echo "✅ Generated: $script_name"
        echo "$script_path"  # Output script path for tracking
    else
        echo "DEBUG: Script file does not exist - creation failed"
        return 1
    fi
}

# Main execution
echo "🔍 Scanning checkpoints directory..."
cd "$PROJECT_ROOT" || exit 1

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"

echo "🔍 Debug: Checkpoint directory: $CHECKPOINT_DIR"
echo "🔍 Debug: Preprocessing directory: $PREPROCESSING_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "📁 Available directories in PROJECT_ROOT:"
    ls -la "$PROJECT_ROOT" || echo "Cannot list PROJECT_ROOT contents"
    exit 1
fi

if [ ! -d "$PREPROCESSING_DIR" ]; then
    echo "⚠️  Preprocessing directory not found: $PREPROCESSING_DIR"
    echo "   (This is optional, continuing without it)"
fi

# Track unique configurations (using a simple list instead of associative array for compatibility)
CONFIGS_FILE=$(mktemp)
SCRIPTS_FILE=$(mktemp)  # Track generated scripts for auto-submission
trap "rm -f $CONFIGS_FILE $SCRIPTS_FILE" EXIT

# Scan checkpoint directories
echo "📂 Scanning model directories in: $CHECKPOINT_DIR"
model_count=0
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model=$(basename "$model_dir")
    model_count=$((model_count + 1))
    echo "🔍 Found model directory: $model"
    
    exp_count=0
    checkpoint_count=0
    for exp_dir in "$model_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        exp_count=$((exp_count + 1))
        
        # Check if checkpoint exists (try multiple patterns)
        checkpoint_found=false
        checkpoint_path=""
        
        # Pattern 1: best-*.ckpt (DMPNN/wDMPNN expected)
        if ls "$exp_dir"/best-*.ckpt >/dev/null 2>&1; then
            checkpoint_found=true
            checkpoint_path=$(ls "$exp_dir"/best-*.ckpt | head -1)
        # Pattern 2: best.pt (AttentiveFP)
        elif ls "$exp_dir"/best.pt >/dev/null 2>&1; then
            checkpoint_found=true
            checkpoint_path=$(ls "$exp_dir"/best.pt | head -1)
        # Pattern 3: logs/checkpoints/epoch=*-step=*.ckpt (Lightning checkpoints)
        elif ls "$exp_dir"/logs/checkpoints/epoch=*-step=*.ckpt >/dev/null 2>&1; then
            checkpoint_found=true
            checkpoint_path=$(ls "$exp_dir"/logs/checkpoints/epoch=*-step=*.ckpt | head -1)
        # Pattern 4: last.ckpt (fallback)
        elif ls "$exp_dir"/last.ckpt >/dev/null 2>&1; then
            checkpoint_found=true
            checkpoint_path=$(ls "$exp_dir"/last.ckpt | head -1)
        fi
        
        if [ "$checkpoint_found" = false ]; then
            echo "  ❌ $exp_name (no checkpoint found)"
            continue
        fi
        
        checkpoint_count=$((checkpoint_count + 1))
        echo "  ✅ $exp_name (checkpoint: $(basename "$checkpoint_path"))"
        
        # Parse experiment name
        IFS='|' read -r dataset has_desc has_rdkit has_batch_norm train_size <<< "$(parse_experiment_name "$exp_name")"
        
        echo "    Parsed: dataset='$dataset', desc=$has_desc, rdkit=$has_rdkit, batch_norm=$has_batch_norm, size='$train_size'"
        
        if [ -z "$dataset" ]; then
            echo "    ❌ Skipping: empty dataset after parsing"
            continue
        fi
        
        # Skip batch norm configurations if requested
        if [ "$SKIP_BATCH_NORM" = true ] && [ "$has_batch_norm" = true ]; then
            echo "    ⏭️  Skipping: batch norm configuration (--skip-batch-norm enabled)"
            continue
        fi
        
        # Check if preprocessing exists
        preprocess_path="$PREPROCESSING_DIR/$exp_name"
        if [ ! -d "$preprocess_path" ]; then
            preprocess_path=""
        fi
        
        # Create unique config key
        config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}|${train_size}"
        echo "    Config key: $config_key"
        
        # Skip if we've already seen this configuration
        if grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null; then
            echo "    ⏭️  Skipping: duplicate configuration"
            continue
        fi
        
        echo "$config_key" >> "$CONFIGS_FILE"
        echo "    ✅ Added new configuration"
        
        # Generate script for this configuration and capture the script path
        echo "    🔧 Generating script..."
        echo "    DEBUG: Calling generate_eval_script with parameters:"
        echo "      model='$model'"
        echo "      dataset='$dataset'"
        echo "      has_desc='$has_desc'"
        echo "      has_rdkit='$has_rdkit'"
        echo "      has_batch_norm='$has_batch_norm'"
        echo "      train_size='$train_size'"
        echo "      checkpoint_path='$checkpoint_path'"
        echo "      preprocess_path='$preprocess_path'"
        
        # Call the function and capture both output and errors
        script_output=$(generate_eval_script "$model" "$dataset" "$has_desc" "$has_rdkit" "$has_batch_norm" "$train_size" "$checkpoint_path" "$preprocess_path" 2>&1)
        script_exit_code=$?
        
        echo "    DEBUG: Function exit code: $script_exit_code"
        echo "    DEBUG: Function output:"
        echo "$script_output" | sed 's/^/      /'
        
        # Extract the script path from the last line that looks like a path
        script_path=$(echo "$script_output" | grep -E '^/.*\.sh$' | tail -n1)
        if [ -z "$script_path" ]; then
            # If no path found, try the last line
            script_path=$(echo "$script_output" | tail -n1)
        fi
        echo "    DEBUG: Extracted script_path: '$script_path'"
        
        # Check if we got a valid script path and file exists
        if [[ "$script_path" =~ ^/.+\.sh$ ]] && [ -f "$script_path" ]; then
            echo "$script_path" >> "$SCRIPTS_FILE"
            echo "    📝 Script ready: $(basename "$script_path")"
        elif echo "$script_output" | grep -q "Skipping (exists)"; then
            echo "    ⏭️  Script already exists, skipping"
        else
            echo "    ❌ Script generation failed"
        fi
    done
    
    echo "  📊 Model $model: $exp_count experiments, $checkpoint_count with checkpoints"
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
    
    # Auto-submit if requested
    if [ "$AUTO_SUBMIT" = true ]; then
        if [ -f "$SCRIPTS_FILE" ] && [ -s "$SCRIPTS_FILE" ]; then
            echo "🚀 Auto-submitting generated scripts..."
            submitted_count=0
            
            while IFS= read -r script_path; do
                if [ -f "$script_path" ]; then
                    echo "  Submitting: $(basename "$script_path")"
                    job_id=$(qsub "$script_path" 2>/dev/null)
                    if [ -n "$job_id" ]; then
                        echo "    ✅ Job ID: $job_id"
                        ((submitted_count++))
                    else
                        echo "    ❌ Failed to submit"
                    fi
                fi
            done < "$SCRIPTS_FILE"
            
            echo ""
            echo "📊 SUBMISSION SUMMARY"
            echo "Total scripts submitted: $submitted_count"
            echo ""
            echo "To check job status:"
            echo "  qstat -u $USER"
            echo ""
            echo "To view results:"
            echo "  ls -la results/"
        else
            echo "⚠️  No scripts were generated to submit"
        fi
    else
        echo "To submit all evaluations:"
        echo "  for script in $OUTPUT_DIR/eval_*.sh; do qsub \$script; done"
        echo ""
        echo "Or use auto-submit:"
        echo "  $0 --auto-submit"
    fi
fi

exit 0
