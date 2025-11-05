#!/bin/bash
# Generate evaluation scripts — robust version (handles spaces, multiword targets, safe filenames)

set -e

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
EVAL_SCRIPTS_DIR="$PROJECT_ROOT/scripts/shell/eval_scripts"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/python/evaluate_model.py"

# ──────────────────────────────────────────────────────────────
# Command-line options
# ──────────────────────────────────────────────────────────────
FORCE=false
DRY_RUN=false
AUTO_SUBMIT=false
SPECIFIC_MODEL=""
SPECIFIC_DATASET=""
TARGET_SPECIFIC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --auto-submit) AUTO_SUBMIT=true; shift ;;
        --model) SPECIFIC_MODEL="$2"; shift 2 ;;
        --dataset) SPECIFIC_DATASET="$2"; shift 2 ;;
        --target-specific) TARGET_SPECIFIC=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --force            Overwrite existing evaluation scripts
  --dry-run          Show what would be generated without creating files
  --auto-submit      Automatically submit generated scripts with qsub
  --model MODEL      Generate scripts only for specific model
  --dataset DATASET  Generate scripts only for specific dataset
  --target-specific  Generate individual scripts per target
  --help, -h         Show this help message
EOF
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "🔍 Scanning checkpoints directory: $CHECKPOINTS_DIR"
echo "📝 Output directory: $EVAL_SCRIPTS_DIR"

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "❌ Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
fi
[ "$DRY_RUN" = false ] && mkdir -p "$EVAL_SCRIPTS_DIR"

# ──────────────────────────────────────────────────────────────
# Parse experiment name (handles spaces + flags)
# ──────────────────────────────────────────────────────────────
parse_experiment_name() {
    local exp_name=$1
    local dataset target has_desc=false has_rdkit=false has_batch_norm=false
    exp_name=$(echo "$exp_name" | sed 's/__rep[0-9]*$//')

    IFS='__' read -ra PARTS <<< "$exp_name"
    dataset="${PARTS[0]}"

    # Collect target pieces until hitting a flag
    local target_parts=()
    for ((i=1;i<${#PARTS[@]};i++)); do
        part="${PARTS[$i]}"
        case "$part" in
            desc|rdkit|batch_norm|size*) break ;;
            *) target_parts+=("$part") ;;
        esac
    done
    target="$(printf "%s " "${target_parts[@]}" | sed 's/ $//')"

    # Detect flags anywhere
    for part in "${PARTS[@]}"; do
        case "$part" in
            desc) has_desc=true ;;
            rdkit) has_rdkit=true ;;
            batch_norm) has_batch_norm=true ;;
        esac
    done

    echo "$dataset|$target|$has_desc|$has_rdkit|$has_batch_norm"
}

# ──────────────────────────────────────────────────────────────
# Generate PBS evaluation script
# ──────────────────────────────────────────────────────────────
generate_eval_script() {
    local model="$1" dataset="$2" target="$3" has_desc="$4" has_rdkit="$5" has_batch_norm="$6"

    # Safe filename (spaces→underscores)
    local safe_target="${target// /_}"

    if [ "$TARGET_SPECIFIC" = true ] && [ -n "$target" ]; then
        local script_name="eval_${dataset}_${safe_target}_${model}"
    else
        local script_name="eval_${dataset}_${model}"
    fi
    [ "$has_desc" = true ] && script_name="${script_name}_desc"
    [ "$has_rdkit" = true ] && script_name="${script_name}_rdkit"
    [ "$has_batch_norm" = true ] && script_name="${script_name}_batch_norm"
    script_name="${script_name}.sh"
    local script_path="$EVAL_SCRIPTS_DIR/$script_name"

    if [ -f "$script_path" ] && [ "$FORCE" = false ]; then
        echo "  ⏭️  Script exists: $script_name"
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  📝 Would create: $script_name"
        echo "     Model: $model | Dataset: $dataset | Target: ${target:-ALL}"
        echo "     Flags: desc=$has_desc rdkit=$has_rdkit batch_norm=$has_batch_norm"
        return
    fi

    # Build evaluation args
    local eval_args="--model_name \"$model\" --dataset_name \"$dataset\""
    [ "$TARGET_SPECIFIC" = true ] && [ -n "$target" ] && eval_args="$eval_args --target \"$target\""
    [ "$has_desc" = true ] && eval_args="$eval_args --incl_desc"
    [ "$has_rdkit" = true ] && eval_args="$eval_args --incl_rdkit"
    [ "$has_batch_norm" = true ] && eval_args="$eval_args --batch_norm"

    cat >"$script_path" <<EOF
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
# Target: ${target:-ALL targets}
# Generated: $(date)

cd \$PBS_O_WORKDIR
module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

python3 $PYTHON_SCRIPT $eval_args

echo "✅ Evaluation completed for $model on $dataset${target:+::$target}"
EOF

    chmod +x "$script_path"
    echo "  ✅ Created: $script_name"
    [ "$AUTO_SUBMIT" = true ] && { echo "  🚀 Submitting: $script_name"; qsub "$script_path"; }
}

# ──────────────────────────────────────────────────────────────
# Main scan loop
# ──────────────────────────────────────────────────────────────
CONFIGS_FILE=$(mktemp); trap "rm -f $CONFIGS_FILE" EXIT
total_scripts=0 total_complete_experiments=0
echo ""; echo "🔍 Scanning for completed experiments..."

for model_dir in "$CHECKPOINTS_DIR"/*; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")
    [ -n "$SPECIFIC_MODEL" ] && [ "$model" != "$SPECIFIC_MODEL" ] && continue

    echo ""; echo "📂 Model: $model"
    exp_count=0 checkpoint_count=0

    # Handle spaces safely
    while IFS= read -r -d '' exp_dir; do
        exp_name=$(basename "$exp_dir"); ((exp_count++))
        all_reps_exist=true
        for i in {0..4}; do
            rep_dir="${exp_dir%__rep0}__rep${i}"
            [ -d "$rep_dir" ] || { all_reps_exist=false; break; }
            # Flexible checkpoint detection
            if find "$rep_dir" -maxdepth 2 -type f \
                \( -name "best.pt" -o -name "best*.ckpt" -o -name "last.ckpt" -o -path "*/logs/checkpoints/epoch=*-step=*.ckpt" \) \
                | grep -q .; then
                checkpoint_found=true
            fi

        done
        [ "$all_reps_exist" = true ] || continue
        ((checkpoint_count++))

        IFS='|' read -r dataset target has_desc has_rdkit has_batch_norm <<<"$(parse_experiment_name "$exp_name")"
        [ -n "$SPECIFIC_DATASET" ] && [ "$dataset" != "$SPECIFIC_DATASET" ] && continue

        if [ "$TARGET_SPECIFIC" = true ]; then
            config_key="${model}|${dataset}|${target}|${has_desc}|${has_rdkit}|${has_batch_norm}"
        else
            config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}"
        fi
        grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null && continue
        echo "$config_key" >>"$CONFIGS_FILE"

        echo "  ✅ $dataset::${target:-ALL} (5/5 replicates)"
        generate_eval_script "$model" "$dataset" "$target" "$has_desc" "$has_rdkit" "$has_batch_norm"
        ((total_complete_experiments++)); ((total_scripts++))
    done < <(find "$model_dir" -mindepth 1 -maxdepth 1 -type d -name "*__rep0" -print0)

    echo "  📊 Model $model: $exp_count experiments, $checkpoint_count complete"
done

echo
