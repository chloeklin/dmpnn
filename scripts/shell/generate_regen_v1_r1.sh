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
ESTIMATED_GPU_HOURS=300
PILOT_COUNT=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/regen_v1/r1"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
FULL_MANIFEST="$MANIFEST_DIR/r1_all.manifest"
PILOT_MANIFEST="$MANIFEST_DIR/r1_pilot.manifest"
REMAINDER_MANIFEST="$MANIFEST_DIR/r1_after_review.manifest"
: > "$FULL_MANIFEST"

MODELS=(hpg_hier wdmpnn hpg_hier_octamer hpg_hier_junction hpg_hier_junction1)
TARGET_TOKENS=(EA_vs_SHE_eV IP_vs_SHE_eV)
TARGET_ARGS=("EA vs SHE (eV)" "IP vs SHE (eV)")
SEEDS=(42 43 44)
PREDICTION_ROOT="$PROJECT_DIR/predictions/regen_v1"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/regen_v1"

for fold in {0..8}; do
    for target_index in "${!TARGET_TOKENS[@]}"; do
        target_token="${TARGET_TOKENS[$target_index]}"
        target="${TARGET_ARGS[$target_index]}"
        for seed in "${SEEDS[@]}"; do
            for model in "${MODELS[@]}"; do
                output="$PREDICTION_ROOT/ea_ip_lomo/ea_ip__${target_token}__${model}__monomer_heldout__fold${fold}__s${seed}.npz"
                if [[ "$model" == "wdmpnn" ]]; then
                    runner="scripts/python/run_wdmpnn_generalization.py"
                    args="--split_types monomer_heldout --folds $fold --targets '$target' --seed $seed --split_seed 42 --patience 15 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/wdmpnn'"
                else
                    runner="scripts/python/run_hpg_generalization.py"
                    args="--split_types monomer_heldout --folds $fold --targets '$target' --models $model --stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --seed $seed --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/hpg'"
                fi
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "$model" "$target_token" "$fold" "$seed" "$output|$args" >> "$FULL_MANIFEST"
            done
        done
    done
done

TOTAL_RUNS="$(wc -l < "$FULL_MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((5 * 2 * 9 * 3))
[[ "$TOTAL_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s runs, generated %s\n' "$EXPECTED_RUNS" "$TOTAL_RUNS" >&2; exit 1; }
head -n "$PILOT_COUNT" "$FULL_MANIFEST" > "$PILOT_MANIFEST"
tail -n "+$((PILOT_COUNT + 1))" "$FULL_MANIFEST" > "$REMAINDER_MANIFEST"
REMAINDER_COUNT=$((TOTAL_RUNS - PILOT_COUNT))

write_pbs() {
    local manifest="$1"
    local task_count="$2"
    local name="$3"
    local pbs="$4"
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
#PBS -J 0-$((task_count - 1))

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
TASK_LOG_DIR="$PROJECT_DIR/logs/regen_v1/r1/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_ARRAY_INDEX}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/regen_v1/r1/manifests/$(basename "$manifest")"
LINE="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "\$MANIFEST")"
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

PILOT_PBS="$PBS_DIR/r1_pilot_10.pbs"
REMAINDER_PBS="$PBS_DIR/r1_after_review_260.pbs"
write_pbs "$PILOT_MANIFEST" "$PILOT_COUNT" "regen_r1p" "$PILOT_PBS"
write_pbs "$REMAINDER_MANIFEST" "$REMAINDER_COUNT" "regen_r1r" "$REMAINDER_PBS"

printf 'R1 cells: %s\n' $((5 * 2 * 9))
printf 'R1 runs: %s\n' "$TOTAL_RUNS"
printf 'Estimated GPU hours: ~%s\n' "$ESTIMATED_GPU_HOURS"
printf 'Pilot jobs: %s\n' "$PILOT_COUNT"
printf 'Post-review jobs: %s\n' "$REMAINDER_COUNT"
printf 'Fresh predictions: %s\n' "$PREDICTION_ROOT"
printf 'Fresh checkpoints: %s\n' "$CHECKPOINT_ROOT"
printf 'Pilot PBS: %s\n' "$PILOT_PBS"
printf 'Post-review PBS: %s\n' "$REMAINDER_PBS"
printf 'No jobs submitted. Submit only the pilot after review: qsub %s\n' "$PILOT_PBS"
printf 'Do not submit the post-review array until pilot sidecars pass the training/provenance check.\n'
