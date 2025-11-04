#!/bin/bash

# Simple Evaluation Script Generator
# Creates PBS job scripts for model evaluation based on existing checkpoints

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🚀 Generating evaluation scripts..."
echo "📁 Project root: $PROJECT_ROOT"

# Configuration
PBS_QUEUE="gpuvolta"
PBS_PROJECT="um09"
PBS_NCPUS="12"
PBS_NGPUS="1"
PBS_MEM="100GB"
PBS_STORAGE="scratch/um09+gdata/dk92"
PBS_JOBFS="100GB"

# Walltime mapping function
get_walltime() {
    local dataset=$1
    case "$dataset" in
        polyinfo) echo "4:00:00" ;;
        insulator) echo "2:00:00" ;;
        htpmd) echo "4:00:00" ;;
        opv_camb3lyp) echo "25:00:00" ;;
        opv_b3lyp) echo "12:00:00" ;;
        tc) echo "2:00:00" ;;
        *) echo "2:00:00" ;;
    esac
}

# Function to parse experiment name
parse_experiment() {
    local exp_name=$1
    
    # Remove rep suffix
    exp_name=$(echo "$exp_name" | sed 's/__rep[0-9]*$//')
    
    # Split by __
    IFS='__' read -ra PARTS <<< "$exp_name"
    local dataset="${PARTS[0]}"
    
    # Check for flags
    local has_desc=false
    local has_rdkit=false
    local has_batch_norm=false
    local train_size=""
    
    for part in "${PARTS[@]:1}"; do
        case "$part" in
            desc) has_desc=true ;;
            rdkit) has_rdkit=true ;;
            batch_norm) has_batch_norm=true ;;
            size*) train_size="$part" ;;
        esac
    done
    
    echo "$dataset|$has_desc|$has_rdkit|$has_batch_norm|$train_size"
}

# Function to generate evaluation script
generate_script() {
    local model=$1
    local dataset=$2
    local has_desc=$3
    local has_rdkit=$4
    local has_batch_norm=$5
    local train_size=$6
    
    # Build script name
    local script_name="eval_${dataset}_${model}"
    [[ "$has_desc" == "true" ]] && script_name="${script_name}_desc"
    [[ "$has_rdkit" == "true" ]] && script_name="${script_name}_rdkit"
    [[ "$has_batch_norm" == "true" ]] && script_name="${script_name}_batch_norm"
    [[ -n "$train_size" ]] && script_name="${script_name}_${train_size}"
    script_name="${script_name}.sh"
    
    local walltime=$(get_walltime "$dataset")
    
    # Build evaluate_model.py arguments
    local eval_args="--model_name $model --dataset_name $dataset"
    [[ "$has_desc" == "true" ]] && eval_args="$eval_args --incl_descriptors"
    [[ "$has_rdkit" == "true" ]] && eval_args="$eval_args --incl_rdkit"
    
    # Build result filename
    local result_file="${dataset}"
    [[ "$has_desc" == "true" ]] && result_file="${result_file}__desc"
    [[ "$has_rdkit" == "true" ]] && result_file="${result_file}__rdkit"
    [[ "$has_batch_norm" == "true" ]] && result_file="${result_file}__batch_norm"
    [[ -n "$train_size" ]] && result_file="${result_file}__${train_size}"
    result_file="${result_file}_baseline.csv"
    
    # Create the PBS script
    cat > "$script_name" << EOF
#!/bin/bash
#PBS -P $PBS_PROJECT
#PBS -q $PBS_QUEUE
#PBS -l walltime=$walltime
#PBS -l ncpus=$PBS_NCPUS
#PBS -l ngpus=$PBS_NGPUS
#PBS -l mem=$PBS_MEM
#PBS -l jobfs=$PBS_JOBFS
#PBS -l storage=$PBS_STORAGE
#PBS -N eval_${dataset}_${model}

# Change to project directory
cd $PROJECT_ROOT

# Activate conda environment
source ~/.bashrc
conda activate chemprop

# Run evaluation
python3 scripts/python/evaluate_model.py \\
    $eval_args

echo "Evaluation complete!"
echo "Results saved to: results/$model/$result_file"
EOF

    chmod +x "$script_name"
    echo "✅ Generated: $script_name"
}

# Main execution
cd "$SCRIPT_DIR"

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo ""
echo "📊 Scanning for unique configurations..."

# Track unique configurations
CONFIGS_FILE=$(mktemp)
trap "rm -f $CONFIGS_FILE" EXIT
script_count=0

# Scan checkpoint directories
for model_dir in "$CHECKPOINT_DIR"/*; do
    [ ! -d "$model_dir" ] && continue
    
    model=$(basename "$model_dir")
    echo "🔍 Processing model: $model"
    
    for exp_dir in "$model_dir"/*; do
        [ ! -d "$exp_dir" ] && continue
        
        exp_name=$(basename "$exp_dir")
        
        # Check if checkpoint exists
        if ! ls "$exp_dir"/best-*.ckpt >/dev/null 2>&1; then
            continue
        fi
        
        # Parse experiment name
        IFS='|' read -r dataset has_desc has_rdkit has_batch_norm train_size <<< "$(parse_experiment "$exp_name")"
        
        [ -z "$dataset" ] && continue
        
        # Create unique config key
        config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}|${train_size}"
        
        # Skip if already seen
        if grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null; then
            continue
        fi
        
        echo "$config_key" >> "$CONFIGS_FILE"
        
        # Generate script
        generate_script "$model" "$dataset" "$has_desc" "$has_rdkit" "$has_batch_norm" "$train_size"
        script_count=$((script_count + 1))
    done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 GENERATION SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo "Total evaluation scripts generated: $script_count"
echo ""
echo "📋 Next steps:"
echo "  Submit all scripts:    for f in eval_*.sh; do qsub \"\$f\"; done"
echo "  Submit specific ones:  qsub eval_insulator_DMPNN.sh"
echo "  List generated scripts: ls -la eval_*.sh"
echo ""
echo "🎯 All evaluation scripts generated successfully!"
