#!/bin/bash

# Generate evaluation scripts - Debug version (handles spaces, flexible checkpoints)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
EVAL_SCRIPTS_DIR="$PROJECT_ROOT/scripts/shell/eval_scripts"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/python/evaluate_model.py"

FORCE=false
DRY_RUN=false
AUTO_SUBMIT=false
SPECIFIC_MODEL=""
SPECIFIC_DATASET=""
TARGET_SPECIFIC=false

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --auto-submit) AUTO_SUBMIT=true; shift ;;
        --model) SPECIFIC_MODEL="$2"; shift 2 ;;
        --dataset) SPECIFIC_DATASET="$2"; shift 2 ;;
        --target-specific) TARGET_SPECIFIC=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --force, --dry-run, --auto-submit, --model, --dataset, --target-specific"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "🔍 Scanning checkpoints directory: $CHECKPOINTS_DIR"
echo "📝 Output directory: $EVAL_SCRIPTS_DIR"

mkdir -p "$EVAL_SCRIPTS_DIR"

# ---------- Helper: Parse experiment name ----------
parse_experiment_name() {
    local exp_name=$1
    local dataset=""
    local target=""
    local has_desc=false
    local has_rdkit=false
    local has_batch_norm=false

    echo "DEBUG: Parsing experiment name: '$exp_name'"

    exp_name=$(echo "$exp_name" | sed 's/__rep[0-9]*$//')
    IFS='__' read -ra PARTS <<< "$exp_name"
    dataset="${PARTS[0]}"
    target_parts=()

    for i in "${!PARTS[@]}"; do
        if [ $i -eq 0 ]; then continue; fi
        part="${PARTS[$i]}"
        case "$part" in
            desc|rdkit|batch_norm|size*) break ;;
            *) target_parts+=("$part") ;;
        esac
    done

    target="${target_parts[*]}"
    for part in "${PARTS[@]}"; do
        [[ "$part" == "desc" ]] && has_desc=true
        [[ "$part" == "rdkit" ]] && has_rdkit=true
        [[ "$part" == "batch_norm" ]] && has_batch_norm=true
    done

    echo "DEBUG: Parsed -> dataset='$dataset', target='$target', desc=$has_desc, rdkit=$has_rdkit, batch_norm=$has_batch_norm"
    echo "$dataset|$target|$has_desc|$has_rdkit|$has_batch_norm"
}

# ---------- Helper: Generate eval script ----------
generate_eval_script() {
    local model="$1" dataset="$2" target="$3" has_desc="$4" has_rdkit="$5" has_batch_norm="$6"

    echo "DEBUG: Generating eval script for model='$model', dataset='$dataset', target='$target'"

    local script_name="eval_${dataset}"
    [ "$TARGET_SPECIFIC" = true ] && [ -n "$target" ] && script_name+="_${target}"
    script_name+="_${model}"
    [ "$has_desc" = true ] && script_name+="_desc"
    [ "$has_rdkit" = true ] && script_name+="_rdkit"
    [ "$has_batch_norm" = true ] && script_name+="_batch_norm"
    script_name="${script_name// /_}.sh"
    local script_path="$EVAL_SCRIPTS_DIR/$script_name"

    echo "DEBUG: Output path: $script_path"

    if [ -f "$script_path" ] && [ "$FORCE" = false ]; then
        echo "  ⏭️  Script exists: $script_name"
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "  📝 Would create: $script_name"
        return
    fi

    cat > "$script_path" << EOF
#!/bin/bash
#PBS -N eval_${dataset}_${model}
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

module load python3/3.12.1
source ~/dmpnn-venv/bin/activate
cd \$PBS_O_WORKDIR

python3 $PYTHON_SCRIPT --model_name "$model" --dataset_name "$dataset" ${target:+--target "$target"} \
    $( [ "$has_desc" = true ] && echo "--incl_desc" ) \
    $( [ "$has_rdkit" = true ] && echo "--incl_rdkit" ) \
    $( [ "$has_batch_norm" = true ] && echo "--batch_norm" )

echo "✅ Evaluation complete: $model / $dataset / ${target:-ALL}"
EOF

    chmod +x "$script_path"
    echo "  ✅ Created: $script_name"
}

# ---------- Main ----------
echo ""
echo "🔍 Scanning for completed experiments..."
CONFIGS_FILE=$(mktemp)
trap "rm -f $CONFIGS_FILE" EXIT

for model
