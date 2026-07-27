#!/bin/bash
set -euo pipefail

PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="gpuvolta"
NCPUS=12
NGPUS=1
MEM="100GB"
JOBFS="100GB"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
WALLTIME="12:30:00"
ESTIMATED_GPU_HOURS_PER_CELL=2.0
SEEDS="42,43,44"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/hpg_phase1"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
MANIFEST="$MANIFEST_DIR/hpg_seed_sweep_${SEEDS//,/_}.manifest"
PBS_SCRIPT="$PBS_DIR/hpg_seed_sweep_${SEEDS//,/_}.pbs"
: > "$MANIFEST"

TARGETS=(EA_vs_SHE_eV IP_vs_SHE_eV)
TARGET_ARGS=("EA vs SHE (eV)" "IP vs SHE (eV)")
MODELS=(hpg_hier wdmpnn hpg_hier_octamer hpg_hier_junction hpg_hier_attention)

for seed in ${SEEDS//,/ }; do
    [[ "$seed" =~ ^(42|43|44)$ ]] || { printf 'Unsupported seed: %s\n' "$seed" >&2; exit 2; }
    for fold in {0..8}; do
        for target_index in "${!TARGETS[@]}"; do
            target_token="${TARGETS[$target_index]}"
            target="${TARGET_ARGS[$target_index]}"
            for model in "${MODELS[@]}"; do
                output="$LOCAL_PROJECT/predictions/ea_ip_lomo/ea_ip__${target_token}__${model}__monomer_heldout__fold${fold}__s${seed}.npz"
                [[ -f "$output" ]] && continue
                if [[ "$model" == "wdmpnn" ]]; then
                    runner="scripts/python/run_wdmpnn_generalization.py"
                    args="--split_types monomer_heldout --folds ${fold} --targets '$target' --seed ${seed}"
                else
                    runner="scripts/python/run_hpg_generalization.py"
                    args="--split_types monomer_heldout --folds ${fold} --targets '$target' --models ${model} --stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --split_seed 42 --seed ${seed}"
                fi
                printf '%s\t%s\t%s\t%s\n' "$runner" "$model" "$output" "$args" >> "$MANIFEST"
            done
        done
    done
done

TASK_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
ESTIMATED_GPU_HOURS="$(awk -v n="$TASK_COUNT" -v h="$ESTIMATED_GPU_HOURS_PER_CELL" 'BEGIN { printf "%.1f", n*h }')"
printf 'Tasks: %s\nEstimated GPU hours: %s\nManifest: %s\n' "$TASK_COUNT" "$ESTIMATED_GPU_HOURS" "$MANIFEST"
if [[ "$TASK_COUNT" -eq 0 ]]; then
    printf 'All requested cells already exist; no PBS array generated.\n'
    exit 0
fi

cat > "$PBS_SCRIPT" <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N hpg_seed
#PBS -r y
#PBS -J 0-$((TASK_COUNT - 1))

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
TASK_LOG_DIR="$PROJECT_DIR/logs/hpg_phase1/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/seed_sweep_\${PBS_ARRAY_INDEX}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/hpg_phase1/manifests/$(basename "$MANIFEST")"
LINE="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "\$MANIFEST")"
IFS=\$'\t' read -r RUNNER MODEL OUTPUT ARGS <<< "\$LINE"
[[ -n "\$RUNNER" ]] || { printf 'Missing manifest entry\n' >&2; exit 2; }
if [[ -f "\$OUTPUT" ]]; then
    printf 'Skipping completed prediction: %s\n' "\$OUTPUT"
    exit 0
fi
eval "set -- \$ARGS"
printf 'runner=%s model=%s output=%s args=%s\n' "\$RUNNER" "\$MODEL" "\$OUTPUT" "\$*"
python "\$RUNNER" "\$@"
EOF
chmod +x "$PBS_SCRIPT"
printf 'PBS script generated (not submitted): %s\nUse qsub manually after reviewing the manifest.\n' "$PBS_SCRIPT"
