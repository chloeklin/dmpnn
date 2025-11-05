#!/bin/bash

# Generate evaluation scripts - Fixed version based on original approach
# Simplified to avoid hanging on large directories

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
TARGET_SPECIFIC=false

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
        --target-specific)
            TARGET_SPECIFIC=true
            shift
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
            echo "  --target-specific  Generate individual scripts per target"
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

# Function to parse experiment name (simplified from original)
parse_experiment_name() {
    local exp_name=$1
    local dataset=""
    local target=""
    local has_desc=false
    local has_rdkit=false
    local has_batch_norm=false
    
    # Remove rep suffix
    exp_name=$(echo "$exp_name" | sed 's/__rep[0-9]*$//')
    
    # Split by __
    IFS='__' read -ra PARTS <<< "$exp_name"
    dataset="${PARTS[0]}"
    
    # For multi-word targets, collect all parts until we hit a flag
    target_parts=()
    for i in "${!PARTS[@]}"; do
        if [ $i -eq 0 ]; then
            continue  # Skip dataset
        fi
        
        part="${PARTS[$i]}"
        case "$part" in
            desc)
                has_desc=true
                break
                ;;
            rdkit)
                has_rdkit=true
                break
                ;;
            batch_norm)
                has_batch_norm=true
                break
                ;;
            size*)
                break
                ;;
            *)
                target_parts+=("$part")
                ;;
        esac
    done
    
    # Join target parts with spaces
    target=""
    for part in "${target_parts[@]}"; do
        if [ -z "$target" ]; then
            target="$part"
        else
            target="$target $part"
        fi
    done
    
    # Continue parsing for remaining flags
    for part in "${PARTS[@]}"; do
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
        esac
    done
    
    echo "$dataset|$target|$has_desc|$has_rdkit|$has_batch_norm"
}

# Function to generate evaluation script
generate_eval_script() {
    local model="$1"
    local dataset="$2"
    local target="$3"
    local has_desc="$4"
    local has_rdkit="$5"
    local has_batch_norm="$6"
    
    # Build script name
    if [ "$TARGET_SPECIFIC" = true ] && [ -n "$target" ]; then
        local script_name="eval_${dataset}_${target}_${model}"
    else
        local script_name="eval_${dataset}_${model}"
    fi
    
    if [ "$has_desc" = true ]; then
        script_name="${script_name}_desc"
    fi
    
    if [ "$has_rdkit" = true ]; then
        script_name="${script_name}_rdkit"
    fi
    
    if [ "$has_batch_norm" = true ]; then
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
        echo "     Model: $model, Dataset: $dataset"
        if [ -n "$target" ]; then
            echo "     Target: $target"
        else
            echo "     Targets: ALL"
        fi
        echo "     Flags: desc=$has_desc, rdkit=$has_rdkit, batch_norm=$has_batch_norm"
        return
    fi
    
    # Build evaluation arguments
    local eval_args="--model_name $model --dataset_name $dataset"
    
    if [ "$TARGET_SPECIFIC" = true ] && [ -n "$target" ]; then
        eval_args="$eval_args --target \"$target\""
    fi
    
    if [ "$has_desc" = true ]; then
        eval_args="$eval_args --incl_desc"
    fi
    
    if [ "$has_rdkit" = true ]; then
        eval_args="$eval_args --incl_rdkit"
    fi
    
    if [ "$has_batch_norm" = true ]; then
        eval_args="$eval_args --batch_norm"
    fi
    
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
$(if [ -n "$target" ]; then echo "# Target: $target"; else echo "# Targets: ALL targets in dataset"; fi)
# Generated: $(date)

cd \$PBS_O_WORKDIR

# Load environment
module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

# Run evaluation
python3 $PYTHON_SCRIPT \\
    $eval_args

$(if [ -n "$target" ]; then echo "echo \"Evaluation completed for $model on $dataset::$target\""; else echo "echo \"Evaluation completed for $model on $dataset (all targets)\""; fi)
EOF
    
    chmod +x "$script_path"
    echo "  ✅ Created: $script_name"
    
    # Auto-submit if requested
    if [ "$AUTO_SUBMIT" = true ]; then
        echo "  🚀 Submitting: $script_name"
        qsub "$script_path"
    fi
}

# Track unique configurations (simple approach like original)
CONFIGS_FILE=$(mktemp)
trap "rm -f $CONFIGS_FILE" EXIT

total_scripts=0
total_complete_experiments=0

echo ""
echo "🔍 Scanning for completed experiments..."

# Iterate through model directories (like original)
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
    
    exp_count=0
    checkpoint_count=0
    
    # Simple directory iteration (like original - this is what worked!)
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
        
        # Check if all 5 replicates exist (simple check)
        all_reps_exist=true
        for i in {0..4}; do
            rep_dir="${exp_dir%__rep0}__rep${i}"
            if [ ! -d "$rep_dir" ]; then
                all_reps_exist=false
                break
            fi
            
            # Check if checkpoint exists in this rep
            checkpoint_found=false
            if ls "$rep_dir"/best-*.ckpt >/dev/null 2>&1 || \
               ls "$rep_dir"/best.pt >/dev/null 2>&1 || \
               ls "$rep_dir"/logs/checkpoints/epoch=*-step=*.ckpt >/dev/null 2>&1 || \
               ls "$rep_dir"/last.ckpt >/dev/null 2>&1; then
                checkpoint_found=true
            fi
            
            if [ "$checkpoint_found" = false ]; then
                all_reps_exist=false
                break
            fi
        done
        
        if [ "$all_reps_exist" = true ]; then
            checkpoint_count=$((checkpoint_count + 1))

            # Parse experiment name
            IFS='|' read -r dataset target has_desc has_rdkit has_batch_norm <<< "$(parse_experiment_name "$exp_name")"

            # Skip if specific dataset requested and this isn't it
            if [ -n "$SPECIFIC_DATASET" ] && [ "$dataset" != "$SPECIFIC_DATASET" ]; then
                continue
            fi

            # Create unique config key
            if [ "$TARGET_SPECIFIC" = true ]; then
                config_key="${model}|${dataset}|${target}|${has_desc}|${has_rdkit}|${has_batch_norm}"
            else
                config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}"
            fi

            # Skip if we've already seen this configuration (before printing ✅)
            if grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null; then
                continue
            fi

            echo "$config_key" >> "$CONFIGS_FILE"
            ((total_complete_experiments++))

            echo "  ✅ $dataset::$target (5/5 replicates)"
            generate_eval_script "$model" "$dataset" "$target" "$has_desc" "$has_rdkit" "$has_batch_norm"
            ((total_scripts++))

        fi

    done
    
    echo "  📊 Model $model: $exp_count experiments, $checkpoint_count complete"
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
echo "Complete experiments found: $total_complete_experiments"
echo "Evaluation scripts generated: $total_scripts"
echo "Output directory: $EVAL_SCRIPTS_DIR"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - No files were created"
    echo "Remove --dry-run to generate actual scripts"
fi

echo ""
echo "✅ Script generation complete!"
