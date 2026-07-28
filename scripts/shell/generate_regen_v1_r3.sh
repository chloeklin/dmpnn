#!/bin/bash
set -euo pipefail

PROJECT="um09"
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
ESTIMATED_GPU_HOURS=240
PILOT_COUNT=4


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/regen_v1/r3"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
rm -f "$PBS_DIR"/*.pbs
rm -f "$MANIFEST_DIR/r3_after_review_chunk_"*
FULL_MANIFEST="$MANIFEST_DIR/r3_all.manifest"
PILOT_MANIFEST="$MANIFEST_DIR/r3_pilot.manifest"
REMAINDER_MANIFEST="$MANIFEST_DIR/r3_after_review.manifest"
: > "$FULL_MANIFEST"

MODELS=(hpg_hier wdmpnn hpg_hier_octamer hpg_hier_junction)
TARGET_TOKENS=(EA_vs_SHE_eV IP_vs_SHE_eV)
TARGET_ARGS=("EA vs SHE (eV)" "IP vs SHE (eV)")
SEEDS=(42 43 44)
SPLIT_TYPE="monomer_b_heldout_clustered"
SPLIT_SUBDIR="ea_ip_lomo_b_clustered"
PREDICTION_ROOT="$PROJECT_DIR/predictions/regen_v1"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/regen_v1"

for model in "${MODELS[@]}"; do
    for target_index in "${!TARGET_TOKENS[@]}"; do
        target_token="${TARGET_TOKENS[$target_index]}"
        target="${TARGET_ARGS[$target_index]}"
        for seed in "${SEEDS[@]}"; do
            for fold in {0..8}; do
                output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${target_token}__${model}__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"
                if [[ "$model" == "wdmpnn" ]]; then
                    runner="scripts/python/run_wdmpnn_generalization.py"
                    args="--split_types $SPLIT_TYPE --folds $fold --targets '$target' --seed $seed --split_seed 42 --patience 15 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/wdmpnn'"
                else
                    runner="scripts/python/run_hpg_generalization.py"
                    args="--split_types $SPLIT_TYPE --folds $fold --targets '$target' --models $model --stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --seed $seed --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/hpg'"
                fi
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "$model" "$target_token" "$fold" "$seed" "$output|$args" >> "$FULL_MANIFEST"
            done
        done
    done
done

TOTAL_RUNS="$(wc -l < "$FULL_MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((4 * 2 * 9 * 3))
[[ "$TOTAL_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s runs, generated %s\n' "$EXPECTED_RUNS" "$TOTAL_RUNS" >&2; exit 1; }

# Pilot = first 3 hpg_hier EA seed42 folds 0-2 + one wdmpnn EA seed42 fold0
# The shared loader is common to both runners, but wdmpnn has never trained
# on a B split under real conditions, so add it to the pilot.
PILOT_COUNT=4
head -n 3 "$FULL_MANIFEST" > "$PILOT_MANIFEST"
awk -F'\t' '$2 == "wdmpnn" && $3 == "EA_vs_SHE_eV" && $4 == 0 && $5 == 42' "$FULL_MANIFEST" >> "$PILOT_MANIFEST"
awk -F'\t' '!(NR <= 3 || ($2 == "wdmpnn" && $3 == "EA_vs_SHE_eV" && $4 == 0 && $5 == 42))' "$FULL_MANIFEST" > "$REMAINDER_MANIFEST"
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
TASK_LOG_DIR="$PROJECT_DIR/logs/regen_v1/r3/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/regen_v1/r3/manifests/$(basename "$manifest")"
LINE="\$(sed -n "${line_num}p" "\$MANIFEST")"
IFS=\$'\t' read -r RUNNER MODEL TARGET FOLD SEED PAYLOAD <<< "\$LINE"
OUTPUT="\${PAYLOAD%%|*}"
ARGS="\${PAYLOAD#*|}"
[[ -n "\$RUNNER" && -n "\$OUTPUT" && "\$ARGS" != "\$PAYLOAD" ]] || { printf 'Malformed manifest row: %s\n' "\$LINE" >&2; exit 2; }
if [[ -f "\$OUTPUT" && -f "\${OUTPUT%.npz}.config.json" ]]; then
    printf 'Skipping completed cell: %s\n' "\$OUTPUT"
    exit 0
fi
if [[ -e "\$OUTPUT" || -e "\${OUTPUT%.npz}.config.json" ]]; then
    printf 'Partial output exists; refusing ambiguous resume: %s\n' "\$OUTPUT" >&2
    exit 1
fi
nvidia-smi
eval "set -- \$ARGS"
printf 'runner=%s model=%s target=%s fold=%s seed=%s\n' "\$RUNNER" "\$MODEL" "\$TARGET" "\$FOLD" "\$SEED"
python "\$RUNNER" "\$@"
test -f "\$OUTPUT" || { printf 'Runner did not create expected NPZ: %s\n' "\$OUTPUT" >&2; exit 1; }
test -f "\${OUTPUT%.npz}.config.json" || { printf 'Runner did not create provenance sidecar: %s\n' "\${OUTPUT%.npz}.config.json" >&2; exit 1; }
EOF
    chmod +x "$pbs"
}

# Generate per-task PBS files for the pilot
PILOT_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/r3_pilot_${TASK_INDEX}.pbs"
    write_per_task_pbs "$PILOT_MANIFEST" "$TASK_INDEX" "regen_r3p_${TASK_INDEX}" "$task_pbs"
    PILOT_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$PILOT_MANIFEST"

# Generate per-task PBS files for the remainder
REMAINDER_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/r3_after_review_${TASK_INDEX}.pbs"
    write_per_task_pbs "$REMAINDER_MANIFEST" "$TASK_INDEX" "regen_r3r_${TASK_INDEX}" "$task_pbs"
    REMAINDER_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$REMAINDER_MANIFEST"

printf 'R3 cells: %s\n' $((4 * 2 * 9))
printf 'R3 runs: %s\n' "$TOTAL_RUNS"
printf 'Estimated GPU hours: ~%s\n' "$ESTIMATED_GPU_HOURS"
printf 'Pilot jobs: %s\n' "$PILOT_COUNT"
printf 'Post-review jobs: %s\n' "$REMAINDER_COUNT"
printf 'Pilot per-task PBS files: %s\n' "${#PILOT_PBS_LIST[@]}"
printf 'Post-review per-task PBS files: %s\n' "${#REMAINDER_PBS_LIST[@]}"
printf 'Fresh predictions: %s\n' "$PREDICTION_ROOT"
printf 'Fresh checkpoints: %s\n' "$CHECKPOINT_ROOT"
printf 'Sample pilot PBS: %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Sample post-review PBS: %s\n' "${REMAINDER_PBS_LIST[0]}"
printf 'No jobs submitted. Submit one pilot job to test: qsub %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Submit all pilots: for f in %s/r3_pilot_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
printf 'Submit all post-review: for f in %s/r3_after_review_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
printf 'Do not submit the post-review jobs until pilot sidecars pass the training/provenance check.\n'
