#!/bin/bash
# Generate evaluation scripts - Debug & robust version (spaces-safe, nested ckpts)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
EVAL_SCRIPTS_DIR="$PROJECT_ROOT/scripts/shell/eval_scripts"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/python/evaluate_model.py"

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
  --target-specific   One script per target (if target is part of experiment name)
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "🔍 Scanning checkpoints directory: $CHECKPOINTS_DIR"
echo "📝 Output directory: $EVAL_SCRIPTS_DIR"
mkdir -p "$EVAL_SCRIPTS_DIR"

# ── Debug helper ──────────────────────────────────────────────────────────────
debug() { echo "DEBUG: $*" >&2; }

# ── Helpers ───────────────────────────────────────────────────────────────────

# Parse "dataset__target__desc__repX" (spaces supported in target)
# IMPORTANT: Only the LAST echo (the pipe-delimited line) goes to stdout.
parse_experiment_name() {
  local exp_name=$1
  local dataset="" target="" has_desc=false has_rdkit=false has_batch_norm=false

  debug "Parsing experiment: '$exp_name'"

  # Strip __rep#
  exp_name="$(echo "$exp_name" | sed 's/__rep[0-9]\+$//')"
  local SEP=$'\x1f'
  local tmp="${exp_name//__/$SEP}"
  IFS=$SEP read -r -a PARTS <<< "$tmp"
  IFS=$' \t\n'  # restore default IFS



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

  debug "Parsed -> dataset='$dataset' target='$target' desc=$has_desc rdkit=$has_rdkit batch_norm=$has_batch_norm"
  # This is the ONLY stdout from this function:
  printf "%s|%s|%s|%s|%s" "$dataset" "$target" "$has_desc" "$has_rdkit" "$has_batch_norm"
}

# Return 0 if any checkpoint file exists; print list for debugging (to stderr)
rep_has_checkpoint() {
  local rep_dir=$1
  debug "  Scan checkpoints in: $rep_dir"

  # collect recursively; spaces-safe
  mapfile -d '' -t files < <(find "$rep_dir" -type f -print0)

  local found=()
  local f
  for f in "${files[@]}"; do
    case "$f" in
      */best.pt|*/best*.ckpt|*/last.ckpt|*/logs/checkpoints/epoch=*step=*.ckpt|*/logs/checkpoints/*.ckpt)
        found+=("$f")
        ;;
    esac
  done

  if ((${#found[@]} == 0)); then
    debug "    ❌ No checkpoint files matched"
    return 1
  fi

  debug "    ✅ Found ${#found[@]} checkpoint file(s):"
  for c in "${found[@]}"; do
    debug "       - $c"
  done
  return 0
}

generate_eval_script() {
  local model="$1" dataset="$2" target="$3" has_desc="$4" has_rdkit="$5" has_batch_norm="$6"

  debug "Gen script for model='$model' dataset='$dataset' target='${target:-ALL}' flags: desc=$has_desc rdkit=$has_rdkit batch_norm=$has_batch_norm"

  local script_name="eval_${dataset}"
  if $TARGET_SPECIFIC && [[ -n "${target}" ]]; then
    script_name+="_${target}"
  fi
  script_name+="_${model}"
  $has_desc && script_name+="_desc"
  $has_rdkit && script_name+="_rdkit"
  $has_batch_norm && script_name+="_batch_norm"
  # Replace spaces and slashes to be safe
  script_name="${script_name// /_}"
  script_name="${script_name//\//_}.sh"

  local script_path="$EVAL_SCRIPTS_DIR/$script_name"

  if [[ -f "$script_path" && $FORCE == false ]]; then
    echo "  ⏭️  Script exists: $(basename "$script_path")"
    return
  fi

  if $DRY_RUN; then
    echo "  📝 DRY-RUN would create: $(basename "$script_path")"
    return
  fi

  # Build optional flags for heredoc (empty expands to nothing)
  local flag_desc="" flag_rdkit="" flag_bn="" flag_target=""
  $has_desc && flag_desc="--incl_desc"
  $has_rdkit && flag_rdkit="--incl_rdkit"
  $has_batch_norm && flag_bn="--batch_norm"
  [[ -n "${target}" ]] && flag_target="--target \"$target\""

  cat > "$script_path" <<EOF
#!/bin/bash

#PBS -q $PBS_QUEUE
#PBS -P $PBS_PROJECT
#PBS -l ncpus=$PBS_NCPUS
#PBS -l ngpus=$PBS_NGPUS
#PBS -l mem=$PBS_MEM
#PBS -l walltime=$DEFAULT_WALLTIME
#PBS -l storage=$PBS_STORAGE
#PBS -l jobfs=$PBS_JOBFS
#PBS -N eval-${dataset}-${model}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


python3 $PYTHON_SCRIPT \\
  --model_name "$model" \\
  --dataset_name "$dataset" \\
  $flag_desc \\
  $flag_rdkit \\

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
mapfile -d '' -t model_dirs < <(find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

for model_dir in "${model_dirs[@]}"; do
  model="$(basename "$model_dir")"
  if [[ -n "$SPECIFIC_MODEL" && "$model" != "$SPECIFIC_MODEL" ]]; then
    debug "Skip model '$model' (filter)"
    continue
  fi

  echo ""
  echo "📂 Model: $model"
  debug "Model dir: $model_dir"

  # All rep0 experiment directories under this model
  mapfile -d '' -t rep0_dirs < <(find "$model_dir" -mindepth 1 -maxdepth 1 -type d -name "*__rep0" -print0)
  debug "Found ${#rep0_dirs[@]} rep0 directories."

  for rep0 in "${rep0_dirs[@]}"; do
    exp_name="$(basename "$rep0")"
    base="${rep0%__rep0}"
    debug "Checking base experiment: '$exp_name' (base='$base')"

    # Verify all 5 reps and checkpoints
    all_reps=true
    for i in 0 1 2 3 4; do
      rep_dir="${base}__rep${i}"
      debug "  Replicate dir: $rep_dir"
      if [[ ! -d "$rep_dir" ]]; then
        debug "    ❌ Missing replicate directory"
        all_reps=false
        break
      fi
      if ! rep_has_checkpoint "$rep_dir"; then
        debug "    ❌ Missing checkpoint in replicate"
        all_reps=false
        break
      fi
    done

    if ! $all_reps; then
      debug "Skipping '$exp_name' (incomplete)"
      continue
    fi

    IFS='|' read -r dataset target has_desc has_rdkit has_batch_norm <<< "$(parse_experiment_name "$exp_name")"

    # Optional dataset filter
    if [[ -n "$SPECIFIC_DATASET" && "$dataset" != "$SPECIFIC_DATASET" ]]; then
      debug "Skipping dataset '$dataset' (filter)"
      continue
    fi

    # Uniqueness key (avoid duplicates)
    if $TARGET_SPECIFIC; then
      config_key="${model}|${dataset}|${target}|${has_desc}|${has_rdkit}|${has_batch_norm}"
    else
      config_key="${model}|${dataset}|${has_desc}|${has_rdkit}|${has_batch_norm}"
    fi

    if grep -Fxq "$config_key" "$CONFIGS_FILE" 2>/dev/null; then
      debug "Duplicate config: $config_key"
      continue
    fi
    echo "$config_key" >> "$CONFIGS_FILE"

    echo "  ✅ $dataset::${target:-ALL} (5/5 replicates + checkpoints)"
    generate_eval_script "$model" "$dataset" "$target" "$has_desc" "$has_rdkit" "$has_batch_norm"
  done
done

echo ""
echo "✅ Script generation complete! Output: $EVAL_SCRIPTS_DIR"
