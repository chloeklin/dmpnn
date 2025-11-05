#!/bin/bash

# New Generate Evaluation Scripts
# Automatically scans checkpoints/ directory and generates evaluation scripts
# for completed training runs (all 5 replicates present)

set -e

# Enable debugging if DEBUG=1
if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

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
            echo "Automatically generates evaluation scripts by scanning checkpoints/ directory"
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

# Show what's in the checkpoints directory
echo "📂 Available models:"
for model_dir in "$CHECKPOINTS_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model=$(basename "$model_dir")
        count=$(find "$model_dir" -maxdepth 1 -type d | wc -l)
        echo "  - $model ($((count-1)) checkpoint directories)"
    fi
done

# Create output directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$EVAL_SCRIPTS_DIR"
fi

# Function to check if all 5 replicates exist for a given pattern
check_all_replicates_exist() {
    local model="$1"
    local pattern="$2"  # e.g., "htpmd__Conductivity" or "htpmd__Conductivity__desc__rdkit"
    
    local model_dir="$CHECKPOINTS_DIR/$model"
    local count=0
    
    for i in {0..4}; do
        local checkpoint_dir="$model_dir/${pattern}__rep${i}"
        if [ -d "$checkpoint_dir" ]; then
            # Check if checkpoint files exist
            if ls "$checkpoint_dir"/*.ckpt >/dev/null 2>&1 || ls "$checkpoint_dir"/*.pt >/dev/null 2>&1 || [ -d "$checkpoint_dir/logs/checkpoints" ]; then
                ((count++))
            fi
        fi
    done
    
    if [ $count -eq 5 ]; then
        return 0  # All 5 replicates exist
    else
        return 1  # Missing replicates
    fi
}

# Function to extract dataset and target from checkpoint pattern
parse_checkpoint_pattern() {
    local pattern="$1"
    
    # Split by __ and extract components
    IFS='__' read -ra PARTS <<< "$pattern"
    
    local dataset="${PARTS[0]}"
    
    # For target, we need to handle multi-word targets
    # Find where the flags start (desc, rdkit, batch_norm, size)
    local target_parts=()
    local desc_flag=""
    local rdkit_flag=""
    local flag_started=false
    
    for i in "${!PARTS[@]}"; do
        if [ $i -eq 0 ]; then
            continue  # Skip dataset
        fi
        
        part="${PARTS[$i]}"
        case "$part" in
            "desc")
                desc_flag="--incl_desc"
                flag_started=true
                ;;
            "rdkit")
                rdkit_flag="--incl_rdkit"
                flag_started=true
                ;;
            "batch_norm")
                # batch_norm is a training flag, not part of target name
                flag_started=true
                ;;
            "size"*)
                # Skip size specifications for now
                flag_started=true
                ;;
            *)
                if [ "$flag_started" = false ]; then
                    target_parts+=("$part")
                fi
                ;;
        esac
    done
    
    # Join target parts with spaces
    local target=""
    for part in "${target_parts[@]}"; do
        if [ -z "$target" ]; then
            target="$part"
        else
            target="$target $part"
        fi
    done
    
    echo "$dataset|$target|$desc_flag|$rdkit_flag"
}

# Function to generate evaluation script
generate_eval_script() {
    local model="$1"
    local dataset="$2"
    local target="$3"
    local desc_flag="$4"
    local rdkit_flag="$5"
    local checkpoint_pattern="$6"
    
    # Create script filename (no target since we're doing all targets)
    local script_name="eval_${dataset}_${model}"
    if [ -n "$desc_flag" ]; then
        script_name="${script_name}_desc"
    fi
    if [ -n "$rdkit_flag" ]; then
        script_name="${script_name}_rdkit"
    fi
    script_name="${script_name}.sh"
    
    local script_path="$EVAL_SCRIPTS_DIR/$script_name"
    
    # Check if script already exists
    if [ -f "$script_path" ] && [ "$FORCE" = false ]; then
        echo "  ⏭️  Script exists: $script_name (use --force to overwrite)"
        return
    fi
    
    # Verify that rep0 exists as a sanity check
    local rep0_dir="$CHECKPOINTS_DIR/$model/${checkpoint_pattern}__rep0"
    if [ ! -d "$rep0_dir" ]; then
        echo "  ❌ Rep0 directory not found: $rep0_dir"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "  📝 Would create: $script_name"
        echo "     Model: $model, Dataset: $dataset, Targets: ALL"
        echo "     Flags: $desc_flag $rdkit_flag"
        echo "     Note: evaluate_model.py will auto-build checkpoint paths"
        return
    fi
    
    # Generate the script content
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
# Targets: ALL targets in dataset
# Generated: $(date)

cd \$PBS_O_WORKDIR

# Load environment
module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

# Run evaluation (evaluate_model.py will build checkpoint paths automatically)
python3 $PYTHON_SCRIPT \\
    --model_name $model \\
    --dataset_name $dataset \\
    $desc_flag \\
    $rdkit_flag

echo "Evaluation completed for $model on $dataset (all targets)"
EOF
    
    chmod +x "$script_path"
    echo "  ✅ Created: $script_name"
    
    # Auto-submit if requested
    if [ "$AUTO_SUBMIT" = true ]; then
        echo "  🚀 Submitting: $script_name"
        qsub "$script_path"
    fi
}

# Main scanning logic
total_scripts=0
total_complete_experiments=0

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
    
    # Collect all checkpoint patterns (without rep suffix)
    patterns_file=$(mktemp)
    checkpoint_count=0
    
    echo "  🔍 Scanning checkpoint directories..."
    
    for checkpoint_dir in "$model_dir"/*; do
        if [ ! -d "$checkpoint_dir" ]; then
            continue
        fi
        
        checkpoint_name=$(basename "$checkpoint_dir")
        ((checkpoint_count++))
        
        # Extract pattern by removing __repX suffix (more precise regex)
        if echo "$checkpoint_name" | grep -E "__rep[0-4]$" >/dev/null; then
            pattern=$(echo "$checkpoint_name" | sed -E 's/__rep[0-4]$//')
            echo "$pattern" >> "$patterns_file"
            if [ "${DEBUG:-0}" = "1" ]; then
                echo "    Found: $checkpoint_name -> $pattern"
            fi
        else
            if [ "${DEBUG:-0}" = "1" ]; then
                echo "    Skipped: $checkpoint_name (no rep suffix)"
            fi
        fi
        
        # Safety check - don't process too many at once
        if [ $checkpoint_count -gt 100 ]; then
            echo "    ⚠️  Too many checkpoints, stopping scan for this model"
            break
        fi
    done
    
    if [ $checkpoint_count -eq 0 ]; then
        echo "  📭 No checkpoint directories found"
        continue
    fi
    
    # Get unique patterns
    if [ -s "$patterns_file" ]; then
        unique_patterns=$(sort "$patterns_file" | uniq)
    else
        unique_patterns=""
    fi
    rm -f "$patterns_file"
    
    # Group patterns by dataset and flags (ignoring target)
    combinations_file=$(mktemp)
    
    for pattern in $unique_patterns; do
        # Parse the pattern
        IFS='|' read -ra PARSED <<< "$(parse_checkpoint_pattern "$pattern")"
        dataset="${PARSED[0]}"
        target="${PARSED[1]}"
        desc_flag="${PARSED[2]}"
        rdkit_flag="${PARSED[3]}"
        
        # Skip if specific dataset requested and this isn't it
        if [ -n "$SPECIFIC_DATASET" ] && [ "$dataset" != "$SPECIFIC_DATASET" ]; then
            continue
        fi
        
        # Create a key for dataset + flags combination (ignoring target)
        flag_key=""
        if [ -n "$desc_flag" ]; then
            flag_key="${flag_key}__desc"
        fi
        if [ -n "$rdkit_flag" ]; then
            flag_key="${flag_key}__rdkit"
        fi
        
        combination_key="${dataset}${flag_key}"
        
        # Store combination info: key|pattern|dataset|desc_flag|rdkit_flag
        echo "$combination_key|$pattern|$dataset|$desc_flag|$rdkit_flag" >> "$combinations_file"
    done
    
    # Get unique combinations
    unique_combinations=$(cut -d'|' -f1 "$combinations_file" | sort | uniq)
    
    # Process each unique dataset+flags combination
    for combination_key in $unique_combinations; do
        # Get all patterns for this combination
        patterns_in_combo=$(grep "^$combination_key|" "$combinations_file" | cut -d'|' -f2)
        
        # Get dataset and flags from first pattern
        first_line=$(grep "^$combination_key|" "$combinations_file" | head -1)
        dataset=$(echo "$first_line" | cut -d'|' -f3)
        desc_flag=$(echo "$first_line" | cut -d'|' -f4)
        rdkit_flag=$(echo "$first_line" | cut -d'|' -f5)
        
        echo "  🎯 Checking: $dataset (all targets, flags: $desc_flag $rdkit_flag)"
        
        # Check if ALL patterns in this combination have complete replicates
        all_complete=true
        total_patterns=0
        complete_patterns=0
        
        for pattern in $patterns_in_combo; do
            ((total_patterns++))
            if check_all_replicates_exist "$model" "$pattern"; then
                ((complete_patterns++))
            else
                all_complete=false
            fi
        done
        
        if [ "$all_complete" = true ] && [ $complete_patterns -gt 0 ]; then
            echo "    ✅ Complete ($complete_patterns/$total_patterns target patterns, 5/5 replicates each)"
            ((total_complete_experiments++))
            
            # Generate evaluation script for this dataset+flags combination
            first_pattern=$(echo $patterns_in_combo | cut -d' ' -f1)
            generate_eval_script "$model" "$dataset" "" "$desc_flag" "$rdkit_flag" "$first_pattern"
            ((total_scripts++))
        else
            echo "    ⚠️  Incomplete ($complete_patterns/$total_patterns target patterns complete)"
        fi
    done
    
    rm -f "$combinations_file"
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

if [ "$AUTO_SUBMIT" = true ] && [ "$DRY_RUN" = false ]; then
    echo ""
    echo "🚀 Auto-submitted $total_scripts evaluation jobs"
fi

echo ""
echo "✅ Script generation complete!"
