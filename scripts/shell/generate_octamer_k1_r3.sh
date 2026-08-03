#!/bin/bash
set -euo pipefail

PROJECT="ng76"
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
PILOT_COUNT=2


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/octamer_k1/r3"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
rm -f "$PBS_DIR"/*.pbs
FULL_MANIFEST="$MANIFEST_DIR/oct_k1_all.manifest"
PILOT_MANIFEST="$MANIFEST_DIR/oct_k1_pilot.manifest"
REMAINDER_MANIFEST="$MANIFEST_DIR/oct_k1_after_review.manifest"
: > "$FULL_MANIFEST"

MODELS=(hpg_hier_octamer)
TARGET_TOKENS=(EA_vs_SHE_eV IP_vs_SHE_eV)
TARGET_ARGS=("EA vs SHE (eV)" "IP vs SHE (eV)")
SEEDS=(42 43 44)
SPLIT_TYPE="monomer_b_heldout_clustered"
SPLIT_SUBDIR="ea_ip_lomo_b_clustered"
K1_TOKEN="__k1"
PREDICTION_ROOT="$PROJECT_DIR/predictions/octamer_k1"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/octamer_k1"
K16_PREDICTION_ROOT="$PROJECT_DIR/predictions/regen_v1"
# Pre-flight existence checks below run wherever this generator is invoked
# (typically your local machine, since results are downloaded there), while
# PREDICTION_ROOT/CHECKPOINT_ROOT above are baked into the manifest args and
# must stay as Gadi paths since the PBS jobs execute on Gadi.
LOCAL_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/octamer_k1"
LOCAL_K16_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/regen_v1"

declare -a PREEXISTING_K1=()
declare -a MISSING_K16=()

for model in "${MODELS[@]}"; do
    for target_index in "${!TARGET_TOKENS[@]}"; do
        target_token="${TARGET_TOKENS[$target_index]}"
        target="${TARGET_ARGS[$target_index]}"
        for seed in "${SEEDS[@]}"; do
            for fold in {0..8}; do
                output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${target_token}__${model}__${SPLIT_TYPE}__fold${fold}__s${seed}${K1_TOKEN}.npz"
                local_output="$LOCAL_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${target_token}__${model}__${SPLIT_TYPE}__fold${fold}__s${seed}${K1_TOKEN}.npz"
                local_k16_output="$LOCAL_K16_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${target_token}__${model}__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"

                # Pre-flight: the K=1 output must not already exist, and the
                # K=16 comparator must be present. Checked against the local
                # copy, since that's where downloaded results live.
                if [[ -e "$local_output" || -e "${local_output%.npz}.config.json" ]]; then
                    PREEXISTING_K1+=("$local_output")
                fi
                if [[ ! -f "$local_k16_output" ]]; then
                    MISSING_K16+=("$local_k16_output")
                fi

                runner="scripts/python/run_hpg_generalization.py"
                args="--split_types $SPLIT_TYPE --folds $fold --targets '$target' --models $model --stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 1 --seed $seed --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/hpg'"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "$model" "$target_token" "$fold" "$seed" "$output|$args" >> "$FULL_MANIFEST"
            done
        done
    done
done

TOTAL_RUNS="$(wc -l < "$FULL_MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((1 * 2 * 9 * 3))
[[ "$TOTAL_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s runs, generated %s\n' "$EXPECTED_RUNS" "$TOTAL_RUNS" >&2; exit 1; }

if [[ ${#PREEXISTING_K1[@]} -gt 0 ]]; then
    printf 'ERROR: %s planned K=1 output(s) already exist:\n' "${#PREEXISTING_K1[@]}" >&2
    printf '  %s\n' "${PREEXISTING_K1[@]}" >&2
fi
if [[ ${#MISSING_K16[@]} -gt 0 ]]; then
    printf 'ERROR: %s comparator K=16 output(s) are missing:\n' "${#MISSING_K16[@]}" >&2
    printf '  %s\n' "${MISSING_K16[@]}" >&2
fi
if [[ ${#PREEXISTING_K1[@]} -gt 0 || ${#MISSING_K16[@]} -gt 0 ]]; then
    exit 1
fi

# Pilot = hpg_hier_octamer EA seed 42, folds 0 (S) and 4 (D)
: > "$PILOT_MANIFEST"
awk -F'\t' '$2 == "hpg_hier_octamer" && $3 == "EA_vs_SHE_eV" && $4 == 0 && $5 == 42' "$FULL_MANIFEST" >> "$PILOT_MANIFEST"
awk -F'\t' '$2 == "hpg_hier_octamer" && $3 == "EA_vs_SHE_eV" && $4 == 4 && $5 == 42' "$FULL_MANIFEST" >> "$PILOT_MANIFEST"
awk -F'\t' '!($2 == "hpg_hier_octamer" && $3 == "EA_vs_SHE_eV" && ($4 == 0 || $4 == 4) && $5 == 42)' "$FULL_MANIFEST" > "$REMAINDER_MANIFEST"

PILOT_RUNS="$(wc -l < "$PILOT_MANIFEST" | tr -d ' ')"
[[ "$PILOT_RUNS" -eq "$PILOT_COUNT" ]] || { printf 'Expected %s pilot runs, generated %s\n' "$PILOT_COUNT" "$PILOT_RUNS" >&2; exit 1; }
REMAINDER_COUNT=$((TOTAL_RUNS - PILOT_COUNT))

write_per_task_pbs() {
    local manifest="$1"
    local task_index="$2"
    local name="$3"
    local pbs="$4"
    local line_num=$((task_index + 1))
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
TASK_LOG_DIR="$PROJECT_DIR/logs/octamer_k1/r3/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/octamer_k1/r3/manifests/$(basename "$manifest")"
LINE="\$(sed -n "${line_num}p" "\$MANIFEST")"
IFS=\$'\t' read -r RUNNER MODEL TARGET FOLD SEED PAYLOAD <<< "\$LINE"
OUTPUT="\${PAYLOAD%%|*}"
ARGS="\${PAYLOAD#*|}"
# The runner writes the base filename (without the K=1 token). We move the
# resulting .npz and sidecar into place after a successful run.
BASE_OUTPUT="\${OUTPUT%__k1.npz}.npz"
BASE_CONFIG="\${BASE_OUTPUT%.npz}.config.json"
[[ -n "\$RUNNER" && -n "\$OUTPUT" && "\$ARGS" != "\$PAYLOAD" ]] || { printf 'Malformed manifest row: %s\n' "\$LINE" >&2; exit 2; }
if [[ -f "\$OUTPUT" && -f "\${OUTPUT%.npz}.config.json" ]]; then
    printf 'Skipping completed cell: %s\n' "\$OUTPUT"
    exit 0
fi
if [[ -e "\$OUTPUT" || -e "\${OUTPUT%.npz}.config.json" || -e "\$BASE_OUTPUT" || -e "\$BASE_CONFIG" ]]; then
    printf 'Partial output exists; refusing ambiguous resume: %s\n' "\$OUTPUT" >&2
    exit 1
fi
nvidia-smi
eval "set -- \$ARGS"
printf 'runner=%s model=%s target=%s fold=%s seed=%s\n' "\$RUNNER" "\$MODEL" "\$TARGET" "\$FOLD" "\$SEED"
python "\$RUNNER" "\$@"
test -f "\$BASE_OUTPUT" || { printf 'Runner did not create expected NPZ: %s\n' "\$BASE_OUTPUT" >&2; exit 1; }
test -f "\$BASE_CONFIG" || { printf 'Runner did not create provenance sidecar: %s\n' "\$BASE_CONFIG" >&2; exit 1; }
mv "\$BASE_OUTPUT" "\$OUTPUT"
mv "\$BASE_CONFIG" "\${OUTPUT%.npz}.config.json"
test -f "\$OUTPUT" || { printf 'Runner did not create expected NPZ: %s\n' "\$OUTPUT" >&2; exit 1; }
test -f "\${OUTPUT%.npz}.config.json" || { printf 'Runner did not create provenance sidecar: %s\n' "\${OUTPUT%.npz}.config.json" >&2; exit 1; }
EOF
    chmod +x "$pbs"
}

# Generate per-task PBS files for the pilot
PILOT_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/oct_k1_p_${TASK_INDEX}.pbs"
    write_per_task_pbs "$PILOT_MANIFEST" "$TASK_INDEX" "oct_k1_p_${TASK_INDEX}" "$task_pbs"
    PILOT_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$PILOT_MANIFEST"

# Generate per-task PBS files for the remainder
REMAINDER_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/oct_k1_r_${TASK_INDEX}.pbs"
    write_per_task_pbs "$REMAINDER_MANIFEST" "$TASK_INDEX" "oct_k1_r_${TASK_INDEX}" "$task_pbs"
    REMAINDER_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$REMAINDER_MANIFEST"

ESTIMATED_SU=$(awk -v n="$TOTAL_RUNS" -v s="$SU_PER_RUN" 'BEGIN { printf "%.0f", n * s }')
ESTIMATED_KSU=$(awk -v su="$ESTIMATED_SU" 'BEGIN { printf "%.1f", su / 1000 }')

printf 'Octamer K=1 cells: %s\n' $((1 * 2 * 9))
printf 'Octamer K=1 runs: %s\n' "$TOTAL_RUNS"
printf 'Estimated cost: %s runs × %s SU ≈ %s kSU\n' "$TOTAL_RUNS" "$SU_PER_RUN" "$ESTIMATED_KSU"
printf 'Charging project: %s\n' "$PROJECT"
printf 'Pilot jobs: %s\n' "$PILOT_COUNT"
printf 'Post-pilot jobs: %s\n' "$REMAINDER_COUNT"
printf 'Pilot per-task PBS files: %s\n' "${#PILOT_PBS_LIST[@]}"
printf 'Post-pilot per-task PBS files: %s\n' "${#REMAINDER_PBS_LIST[@]}"
printf 'Fresh predictions: %s\n' "$PREDICTION_ROOT"
printf 'Fresh checkpoints: %s\n' "$CHECKPOINT_ROOT"
printf 'Sample pilot PBS: %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Sample post-pilot PBS: %s\n' "${REMAINDER_PBS_LIST[0]}"
printf 'No jobs submitted. Submit one pilot job to test: qsub %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Submit all pilots: for f in %s/oct_k1_p_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
printf 'Submit all post-pilot: for f in %s/oct_k1_r_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
printf 'Do not submit the post-pilot jobs until pilot sidecars are verified.\n'
