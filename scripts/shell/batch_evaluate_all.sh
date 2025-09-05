#!/bin/bash

# Batch evaluation shell script that submits individual jobs for each dataset/model/variant combination
# Usage: ./batch_evaluate_all.sh [--dry-run] [--force]

set -e  # Exit on error

# Default values
DRY_RUN=false
FORCE=false
DATA_DIR="data"
RESULTS_DIR="results"
CHECKPOINT_DIR="checkpoints"
SCRIPT_DIR="."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --script-dir)
            SCRIPT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dry-run           Show what would be evaluated without running"
            echo "  --force             Force re-evaluation even if results exist"
            echo "  --data-dir DIR      Directory containing CSV datasets (default: ../../data)"
            echo "  --results-dir DIR   Directory to store results (default: ../../results)"
            echo "  --checkpoint-dir DIR Directory containing checkpoints (default: ../../checkpoints)"
            echo "  --script-dir DIR    Directory containing evaluate_model.py (default: .)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert to absolute paths
DATA_DIR=$(cd "$DATA_DIR" && pwd)
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)
CHECKPOINT_DIR=$(cd "$CHECKPOINT_DIR" && pwd)
SCRIPT_DIR=$(cd "$SCRIPT_DIR" && pwd)

echo "============================================================"
echo "📋 BATCH EVALUATION CONFIGURATION"
echo "============================================================"
echo "Data directory: $DATA_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Dry run: $DRY_RUN"
echo "Force re-evaluation: $FORCE"
echo "============================================================"

# Find all datasets
DATASETS=()
for csv_file in "$DATA_DIR"/*.csv; do
    if [[ -f "$csv_file" ]]; then
        basename=$(basename "$csv_file" .csv)
        # Skip preprocessed versions
        if [[ ! "$basename" =~ preprocessed ]]; then
            DATASETS+=("$basename")
        fi
    fi
done

echo "Found ${#DATASETS[@]} datasets: ${DATASETS[*]}"
echo "============================================================"

# Define models and variants
MODELS=("DMPNN" "wDMPNN")
VARIANTS=(
    "original:"
    "rdkit:--incl_rdkit"
)

# Counters
TOTAL_JOBS=0
SKIPPED_JOBS=0
SUBMITTED_JOBS=0

# Function to check if checkpoints exist
check_checkpoints() {
    local dataset="$1"
    local model="$2"
    local checkpoint_path="$CHECKPOINT_DIR/$model"
    
    if [[ ! -d "$checkpoint_path" ]]; then
        return 1
    fi
    
    # Look for checkpoint directories matching the dataset pattern
    # Pattern: dataset__target__[desc|rdkit]__rep0 or dataset__target__rep0
    local count=$(find "$checkpoint_path" -maxdepth 1 -name "${dataset}__*" -type d | wc -l)
    if [[ $count -eq 0 ]]; then
        return 1
    fi
    
    # Check if any checkpoint directories contain .ckpt files
    local ckpt_count=0
    while IFS= read -r -d '' dir; do
        if [[ -n $(find "$dir" -name "*.ckpt" -type f 2>/dev/null) ]]; then
            ckpt_count=$((ckpt_count + 1))
        fi
    done < <(find "$checkpoint_path" -maxdepth 1 -name "${dataset}__*" -type d -print0)
    
    if [[ $ckpt_count -eq 0 ]]; then
        return 1
    fi
    
    return 0
}

# Function to check if results already exist
check_existing_results() {
    local dataset="$1"
    local model="$2"
    local variant_args="$3"
    local results_path="$RESULTS_DIR/$model"
    
    # Build expected filename based on variant
    local desc_suffix=""
    local rdkit_suffix=""
    
    if [[ "$variant_args" == *"--incl_desc"* ]]; then
        desc_suffix="__desc"
    fi
    if [[ "$variant_args" == *"--incl_rdkit"* ]]; then
        rdkit_suffix="__rdkit"
    fi
    
    local baseline_file="$results_path/${dataset}${desc_suffix}${rdkit_suffix}_baseline.csv"
    
    if [[ -f "$baseline_file" ]]; then
        return 0  # File exists
    else
        return 1  # File doesn't exist
    fi
}

# Function to get walltime for a dataset from evaluation_config.yaml
get_walltime() {
    local dataset="$1"
    local config_file="$SCRIPT_DIR/scripts/shell/evaluation_config.yaml"
    
    if [[ -f "$config_file" ]]; then
        # Try to extract walltime for the specific dataset
        local walltime=$(python3 -c "
import yaml
try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    walltime = config.get('walltime', {}).get('$dataset')
    if walltime:
        print(walltime)
    else:
        print(config.get('default_walltime', '2:00:00'))
except:
    print('2:00:00')
")
        echo "$walltime"
    else
        echo "2:00:00"  # Default fallback
    fi
}

# Function to submit evaluation job
submit_job() {
    local dataset="$1"
    local model="$2"
    local variant_name="$3"
    local variant_args="$4"
    
    local cmd="python3 scripts/python/evaluate_model.py --dataset_name $dataset --task_type reg --model_name $model"
    
    if [[ -n "$variant_args" ]]; then
        cmd="$cmd $variant_args"
    fi
    
    # Get walltime for this dataset
    local walltime=$(get_walltime "$dataset")
    
    echo "🔄 Generating PBS job script: $dataset ($model) - $variant_name (walltime: $walltime)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "   [DRY RUN] Would generate PBS script for: $cmd"
        return 0
    fi
    
    # Create PBS job script filename
    local job_script="eval_${dataset}_${model}_${variant_name}.sh"
    
    # Generate the PBS script
    cat > "$job_script" << EOF
#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=$walltime
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval_${model}_${dataset}_${variant_name}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# Evaluation
$cmd


##TODO

# Add additional experiments here as needed

EOF
    
    chmod +x "$job_script"
    
    echo "   📄 Generated PBS script: $job_script"
    
    # Submit job to PBS queue
    JOB_ID=$(qsub "$job_script")
    if [ $? -eq 0 ]; then
        echo "   ✅ Job submitted successfully: $JOB_ID"
        echo "   📊 Monitor with: qstat -u $USER"
    else
        echo "   ❌ Error: Failed to submit job"
        return 1
    fi
    
    return 0
}

# Main evaluation loop
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "📊 Processing dataset: $dataset"
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo "🤖 Model: $model"
        
        # Check if checkpoints exist
        if ! check_checkpoints "$dataset" "$model"; then
            echo "   ⚠️  Skipping $dataset ($model) - no trained checkpoints found"
            echo "   🔍 Debug: Looking in $CHECKPOINT_DIR/$model"
            if [[ -d "$CHECKPOINT_DIR/$model" ]]; then
                echo "   📁 Available directories:"
                ls -la "$CHECKPOINT_DIR/$model" 2>/dev/null | head -10
            else
                echo "   ❌ Checkpoint directory does not exist: $CHECKPOINT_DIR/$model"
                echo "   📁 Available model directories in $CHECKPOINT_DIR:"
                ls -la "$CHECKPOINT_DIR" 2>/dev/null | head -10
            fi
            SKIPPED_JOBS=$((SKIPPED_JOBS + 2))  # 2 variants per model
            continue
        fi
        
        local checkpoint_count=$(find "$CHECKPOINT_DIR/$model" -maxdepth 1 -name "${dataset}__*" -type d | wc -l)
        echo "   ✅ Found $checkpoint_count checkpoint directories"
        
        # Show which checkpoint directories were found for debugging
        echo "   📁 Checkpoint directories found:"
        find "$CHECKPOINT_DIR/$model" -maxdepth 1 -name "${dataset}__*" -type d | while read dir; do
            echo "      $(basename "$dir")"
        done
        
        # Process each variant
        for variant_spec in "${VARIANTS[@]}"; do
            IFS=':' read -r variant_name variant_args <<< "$variant_spec"
            
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            # Check if results already exist
            if check_existing_results "$dataset" "$model" "$variant_args"; then
                if [[ "$FORCE" != "true" ]]; then
                    echo "   ⏭️  Skipping $dataset ($model) - $variant_name: results already exist"
                    SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
                    continue
                fi
            fi
            
            # Submit the job
            if submit_job "$dataset" "$model" "$variant_name" "$variant_args"; then
                SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                # Add small delay between job submissions
                sleep 1
            fi
        done
    done
done

# Summary
echo ""
echo "============================================================"
echo "📋 EVALUATION SUMMARY"
echo "============================================================"
echo "Total datasets found: ${#DATASETS[@]}"
echo "Total jobs planned: $TOTAL_JOBS"
echo "Skipped jobs: $SKIPPED_JOBS"
echo "Submitted jobs: $SUBMITTED_JOBS"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "🔍 This was a dry run - no jobs were actually submitted."
elif [[ $SUBMITTED_JOBS -gt 0 ]]; then
    echo ""
    echo "✅ $SUBMITTED_JOBS evaluation jobs have been submitted!"
    echo "📊 Monitor progress with: ps aux | grep evaluate_model"
    echo "📝 Check logs in: /tmp/eval_*.log"
    echo "🔄 Wait for all jobs to complete before checking results"
else
    echo ""
    echo "ℹ️  No new evaluations needed - all results already exist!"
fi

echo "============================================================"
