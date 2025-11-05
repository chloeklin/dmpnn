#!/bin/bash

# Generate evaluation scripts - Debug & robust version (spaces-safe, nested ckpts)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
EVAL_SCRIPTS_DIR="$PROJECT_ROOT/scripts/shell/eval_scripts"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/python/evaluate_model.py"

# ── Options ───────────────────────────────────────────────────────────────────
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
    --model) SPECIFIC_MODEL="${2:-}"; shift 2 ;;
    --dataset) SPECIFIC_DATASET="${2:-}"; shift 2 ;;
    --target-specific) TARGET_SPECIFIC=true; shift ;;
    --help|-h)
      cat <<EOF
Usage: $0 [options]

Options:
  --force             Overwrite existing eval scripts
  --dry-run           Print actions without writing files
  --auto-submit       qsub each generated script
  --model NAME        Only scan this model (e.g., AttentiveFP, DMPNN, DMPNN_DiffPool)
  --dataset NAME      Only include this dataset (exact match)
  --target-specific   One script per target (if target is present in name)
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  case_end:
  esac
done

echo "🔍 Scanning checkpoints directory: $CHECKPOINTS_DIR"
echo "📝 Output directory: $EVAL_SCRIPTS_DIR"
mkdir -p "$EVAL_SCRIPTS_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Parse "dataset__target__desc__repX" (spaces supported in target)
parse_experiment_name() {
  local exp_name=$1
  local dataset="" target="" has_desc=false has_rdkit=false has_batch_norm=false

  echo "DEBUG: Parsing experiment: '$exp_name'"

  # Strip __rep#
  exp_name="$(echo "$exp_name" | sed 's/__rep[0-9][0-9]*$//')"
  IFS='__' read -r -a PARTS <<< "$exp_name"

  dataset="${PARTS[0]}"
  local target_parts=()
  local i
  for (( i=1; i<${#PARTS[@]}; i++ )); do
    case "${PARTS[$i]}" in
      desc|rdkit|batch_norm|size*) break ;;
      *) target_parts+=("${PARTS[$i]}") ;;
    esac
  done

  # Join target parts with spaces
  if ((${#target_parts[@]})); then
    target="${target_parts[*]}"
  else
    target=""
  fi

  # Flags anywhere
  for p in "${PARTS[@]}"; do
    [[ $p == desc ]] && has_desc=true
    [[ $p == rdkit ]] && has_rdkit=true
    [[ $p == batch_norm ]] && has_batch_norm=true
  done

  echo "DEBUG: Parsed -> dataset='$dataset' target='$target' desc=$has_desc rdkit=$has_rdkit batch_norm=$has_batch_norm"
  printf "%s|%s|%s|%s|%s" "$dataset" "$target" "$has_desc" "$has_rdkit" "$has_batch_norm"
}

# Return 0 if any checkpoint file exists; print list for debugging
rep_has_checkpoint() {
  local rep_dir=$1
  echo "DEBUG:       Scanning for checkpoints in: $rep_dir"

  # Collect with find (recursive), support common patterns
  mapfile -d '' ckpts < <(find "$rep_dir" -type f -print0 \
    | xargs -0 -I{} bash -c '
        f="$1";
        case "$f" in
          */best.pt|*/best*.ckpt|*/last.ckpt|*/logs/checkpoints/epoch=*step=*.ckpt) printf "%s\0" "$f" ;;
        esac
      ' _ {})

  if ((${#ckpts[@]} == 0)); then
    echo "DEBUG:         ❌ No checkpoint files matched patterns"
    return 1
  fi

  echo "DEBUG:         ✅ Found ${#ckpts[@]} checkpoint file(s):"
  for c in "${ckpts[@]}"; do
    echo "DEBUG:            - $(basename "$c")"
  done
  return 0
}

generate_eval_script() {
  local model="$1" dataset="$2" target="$3" has_desc="$4" has_rdkit="$5" has_batch_norm="$6"

  echo "DEBUG: Generating script for model='$model' dataset='$dataset' target='${target:-ALL}' flags: desc=$has_desc rdkit=$has_rdkit batch_norm=$has_batch_norm"

  local script_name="eval_${dataset}"
  if $TARGET_SPECIFIC && [[ -n "${target}" ]]; then
    script_name+="_${target}"
  fi
  script_name+="_${model}"
  $has_desc && script_name+="_desc"
  $has_rdkit && script_name+="_rdkit"
  $has_batch_norm && script_name+="_batch_norm"
  script_name="${script_name// /_}.sh"

  local script_path="$EVAL_SCRIPTS_DIR/$script_name"

  if [[ -f "$script_path" && $FORCE == false ]]; then
    echo "  ⏭️  Script exists: $(basename "$script_path")"
    return
  fi

  if $DRY_RUN; then
    echo "  📝 DRY-RUN would create: $(basename "$script_path")"
    return
  fi

  cat > "$script_path" <<EOF
#!/bin/bash
#PBS -N eval_${dataset// /_}_${model}
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

cd \$PBS_O_WORKDIR

python3 "$PYTHON_SCRIPT" \\
  --model_name "$model" \\
  --dataset_name "$dataset" \\
  ${target:+--target "$target"} \\
  $( $has_desc && echo "--incl_desc" ) \\
  $( $has_rdkit && echo "--incl_rdkit" ) \\
  $( $has_batch_norm && echo "--batch_norm" )

echo "✅ Evaluation done for $model on $dataset ${target:+(target: $target)}"
EOF

  chmod +x "$script_path"
  echo "  ✅ Created: $(basename "$script_path")"

  if $AUTO_SUBMIT; then
    echo "  🚀 Submitting: $(basename "$script_path")"
    qsub "$script_path" || echo "  ⚠️ qsub failed"
  fi
}

# ── Main scan ─────────────────────────────────────────────────────────────────
echo ""
echo "🔍 Scanning for completed experiments..."

CONFIGS_FILE=$(mktemp)
trap 'rm -f "$CONFIGS_FILE"' EXIT

# Each model (e.g., AttentiveFP, DMPNN, DMPNN_DiffPool)
mapfile -d '' model_dirs < <(find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

for model_dir in "${model_dirs[@]}"; do
  model="$(basename "$model_dir")"
  [[ -n "$SPECIFIC_MODEL" && "$model" != "$SPECIFIC_MODEL" ]] && continue

  echo ""
  echo "📂 Model: $model"
  echo "DEBUG: Model dir: $model_dir"

  # All rep0 experiment directories under this model
  mapfile -d '' rep0_dirs < <(find "$model_dir" -mindepth 1 -maxdepth 1 -type d -name "*__rep0" -print0)
  echo "DEBUG: Found ${#rep0_dirs[@]} rep0 directories."

  for rep0 in "${rep0_dirs[@]}"; do
    exp_name="$(basename "$rep0")"
    base="${rep0%__rep0}"
    echo "DEBUG: Checking base experiment: '$exp_name' (base='$base')"

    # Verify all 5 reps and checkpoints
    all_reps=true
    for i in {0..4}; do
      rep_dir="${base}__rep${i}"
      echo "DEBUG:   Replicate dir: $rep_dir"
      if [[ ! -d "$rep_dir" ]]; then
        echo "DEBUG:     ❌ Missing replicate directory"
        all_reps=false
        break
      fi
      if ! rep_has_checkpoint "$rep_dir"; then
        echo "DEBUG:     ❌ Missing checkpoint in replicate"
        all_reps=false
        break
      fi
    done

    if ! $all_reps; then
      echo "DEBUG: Skipping '$exp_name' (incomplete)"
      continue
    fi

    IFS='|' read -r dataset target has_desc has_rdkit has_batch_norm <<< "$(parse_experiment_name "$exp_name")"

    # Optional dataset filter
    [[ -n "$SPECIFIC_DATASET" && "$dataset" != "$SPECIFIC_DATASET" ]] && { echo "DEBUG: Skipping dataset '$dataset' (filter)"; continue; }

    # Uniqueness key (avoid duplicates)
    if $TARGET_SPECIFIC; then
      config_key="${model}|${dataset}|${target}|${has_desc}|${has_rdkit}|${has_batch_norm}"
    else
      config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}"
    fi

    if grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null; then
      echo "DEBUG: Duplicate config: $config_key"
      continue
    fi
    echo "$config_key" >> "$CONFIGS_FILE"

    echo "  ✅ $dataset::${target:-ALL} (5/5 replicates + checkpoints)"
    generate_eval_script "$model" "$dataset" "$target" "$has_desc" "$has_rdkit" "$has_batch_norm"
  done
done

echo ""
echo "✅ Script generation complete! Output: $EVAL_SCRIPTS_DIR"
