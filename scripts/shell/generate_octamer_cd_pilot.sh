#!/bin/bash
# Generate PBS scripts for the arms C and D pilot on R1.
#
# Arm C: 2-node transition graph + attention readout (hpg_hier + attention).
# Arm D: 8-slot octamer chain + stoichiometry-weighted / mean readout
#        (hpg_hier_octamer with --stage2_readout stoich_weighted).
#
# Pre-registered in analysis/model_diagnostics/PREREG_arms_CD_2026-08-11.md.
#
# Usage: run locally (off Gadi) to generate PBS files.  It does not submit or train.
set -euo pipefail

PROJECT="hm62"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="gpuvolta"
NCPUS=12
NGPUS=1
MEM="100GB"
JOBFS="100GB"
WALLTIME="06:00:00"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
SU_PER_RUN=41.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/octamer_cd/pilot"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
rm -f "$PBS_DIR"/*.pbs

MANIFEST="$MANIFEST_DIR/oct_cd_pilot.manifest"
: > "$MANIFEST"

SPLIT_TYPE="monomer_heldout"
SPLIT_SUBDIR="ea_ip_lomo"
TARGET_TOKEN="EA_vs_SHE_eV"
TARGET="EA vs SHE (eV)"
SEEDS=(42 43 44)
FOLDS=(0 4)

# Arm C: transition graph + attention
ARM_C_MODEL="hpg_hier_attention"
ARM_C_TOKEN="__armC"
ARM_C_ARGS="--stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16"
ARM_C_COMPARATOR="hpg_hier"

# Arm D: octamer chain + stoich-weighted / mean
ARM_D_MODEL="hpg_hier_octamer"
ARM_D_TOKEN="__armD"
ARM_D_ARGS="--stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --stage2_readout stoich_weighted"
ARM_D_COMPARATOR="hpg_hier_octamer"

PREDICTION_ROOT="$PROJECT_DIR/predictions/octamer_cd"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/octamer_cd"
LOCAL_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/octamer_cd"
LOCAL_K16_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/regen_v1"

declare -a PREEXISTING=()
declare -a MISSING_COMPARATORS=()

for seed in "${SEEDS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        # Arm C
        output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${ARM_C_MODEL}__${SPLIT_TYPE}__fold${fold}__s${seed}${ARM_C_TOKEN}.npz"
        local_output="$LOCAL_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${ARM_C_MODEL}__${SPLIT_TYPE}__fold${fold}__s${seed}${ARM_C_TOKEN}.npz"
        local_k16="$LOCAL_K16_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${ARM_C_COMPARATOR}__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"

        if [[ -e "$local_output" || -e "${local_output%.npz}.config.json" ]]; then
            PREEXISTING+=("$local_output")
        fi
        if [[ ! -f "$local_k16" ]]; then
            MISSING_COMPARATORS+=("$local_k16")
        fi

        runner="scripts/python/run_hpg_generalization.py"
        args="--split_types $SPLIT_TYPE --folds $fold --targets '$TARGET' --models $ARM_C_MODEL $ARM_C_ARGS --seed $seed --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/hpg'"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "$ARM_C_MODEL" "$TARGET_TOKEN" "$fold" "$seed" "$output|$args" >> "$MANIFEST"

        # Arm D
        output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${ARM_D_MODEL}__${SPLIT_TYPE}__fold${fold}__s${seed}${ARM_D_TOKEN}.npz"
        local_output="$LOCAL_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${ARM_D_MODEL}__${SPLIT_TYPE}__fold${fold}__s${seed}${ARM_D_TOKEN}.npz"
        local_k16="$LOCAL_K16_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${ARM_D_COMPARATOR}__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"

        if [[ -e "$local_output" || -e "${local_output%.npz}.config.json" ]]; then
            PREEXISTING+=("$local_output")
        fi
        if [[ ! -f "$local_k16" ]]; then
            MISSING_COMPARATORS+=("$local_k16")
        fi

        runner="scripts/python/run_hpg_generalization.py"
        args="--split_types $SPLIT_TYPE --folds $fold --targets '$TARGET' --models $ARM_D_MODEL $ARM_D_ARGS --seed $seed --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/hpg'"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "$ARM_D_MODEL" "$TARGET_TOKEN" "$fold" "$seed" "$output|$args" >> "$MANIFEST"
    done
done

TOTAL_RUNS="$(wc -l < "$MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((2 * 2 * 3))
[[ "$TOTAL_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s runs, generated %s\n' "$EXPECTED_RUNS" "$TOTAL_RUNS" >&2; exit 1; }

if [[ ${#PREEXISTING[@]} -gt 0 ]]; then
    printf 'ERROR: %s planned output(s) already exist:\n' "${#PREEXISTING[@]}" >&2
    printf '  %s\n' "${PREEXISTING[@]}" >&2
fi
if [[ ${#MISSING_COMPARATORS[@]} -gt 0 ]]; then
    printf 'ERROR: %s comparator K=16 output(s) are missing:\n' "${#MISSING_COMPARATORS[@]}" >&2
    printf '  %s\n' "${MISSING_COMPARATORS[@]}" >&2
fi
if [[ ${#PREEXISTING[@]} -gt 0 || ${#MISSING_COMPARATORS[@]} -gt 0 ]]; then
    exit 1
fi

write_per_task_pbs() {
    local line_num="$1"
    local name="$2"
    local pbs="$3"
    cat > "$pbs" <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N $name
#PBS -r y

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
TASK_LOG_DIR="$PROJECT_DIR/logs/octamer_cd/pilot/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/octamer_cd/pilot/manifests/$(basename "$MANIFEST")"
LINE="\$(sed -n "${line_num}p" "\$MANIFEST")"
IFS=\$'\\t' read -r RUNNER MODEL TARGET FOLD SEED PAYLOAD <<< "\$LINE"
OUTPUT="\${PAYLOAD%%|*}"
ARGS="\${PAYLOAD#*|}"
BASE_OUTPUT="\${OUTPUT%__arm*.npz}.npz"
BASE_CONFIG="\${BASE_OUTPUT%.npz}.config.json"
[[ -n "\$RUNNER" && -n "\$OUTPUT" && "\$ARGS" != "\$PAYLOAD" ]] || { printf 'Malformed manifest row: %s\\n' "\$LINE" >&2; exit 2; }
if [[ -f "\$OUTPUT" && -f "\${OUTPUT%.npz}.config.json" ]]; then
    printf 'Skipping completed cell: %s\\n' "\$OUTPUT"
    exit 0
fi
if [[ -e "\$OUTPUT" || -e "\${OUTPUT%.npz}.config.json" || -e "\$BASE_OUTPUT" || -e "\$BASE_CONFIG" ]]; then
    printf 'Partial output exists; refusing ambiguous resume: %s\\n' "\$OUTPUT" >&2
    exit 1
fi
nvidia-smi
eval "set -- \$ARGS"
printf 'runner=%s model=%s target=%s fold=%s seed=%s\\n' "\$RUNNER" "\$MODEL" "\$TARGET" "\$FOLD" "\$SEED"
python "\$RUNNER" "\$@"
test -f "\$BASE_OUTPUT" || { printf 'Runner did not create expected NPZ: %s\\n' "\$BASE_OUTPUT" >&2; exit 1; }
test -f "\$BASE_CONFIG" || { printf 'Runner did not create provenance sidecar: %s\\n' "\$BASE_CONFIG" >&2; exit 1; }
mv "\$BASE_OUTPUT" "\$OUTPUT"
mv "\$BASE_CONFIG" "\${OUTPUT%.npz}.config.json"
test -f "\$OUTPUT" || { printf 'Runner did not create expected NPZ: %s\\n' "\$OUTPUT" >&2; exit 1; }
test -f "\${OUTPUT%.npz}.config.json" || { printf 'Runner did not create provenance sidecar: %s\\n' "\${OUTPUT%.npz}.config.json" >&2; exit 1; }
EOF
    chmod +x "$pbs"
}

PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    line_num=$((TASK_INDEX + 1))
    task_pbs="$PBS_DIR/oct_cd_pilot_${TASK_INDEX}.pbs"
    write_per_task_pbs "$line_num" "oct_cd_pilot_${TASK_INDEX}" "$task_pbs"
    PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$MANIFEST"

ESTIMATED_SU=$(awk -v n="$TOTAL_RUNS" -v s="$SU_PER_RUN" 'BEGIN { printf "%.0f", n * s }')
ESTIMATED_KSU=$(awk -v su="$ESTIMATED_SU" 'BEGIN { printf "%.1f", su / 1000 }')

printf 'Arms C/D pilot cells: %s\n' "$EXPECTED_RUNS"
printf 'Arms C/D pilot runs: %s\n' "$TOTAL_RUNS"
printf 'Estimated cost: %s runs x %s SU ≈ %s kSU\n' "$TOTAL_RUNS" "$SU_PER_RUN" "$ESTIMATED_KSU"
printf 'Charging project: %s\n' "$PROJECT"
printf 'Per-task PBS files: %s\n' "${#PBS_LIST[@]}"
printf 'PBS output directory: %s\n' "$PBS_DIR"
printf 'Manifest: %s\n' "$MANIFEST"
