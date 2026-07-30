#!/bin/bash
# Generate per-task PBS files for the original-paper wDMPNN reproduction.
#
# Purpose: re-run wDMPNN under the configuration of Aldeghi & Coley,
# Chem. Sci. 2022, 13, 10486. The authors state in the SI (p. S4):
# "No hyperparameter optimization was performed for the D-MPNN and wD-MPNN
# models and the validation set was used only for early stopping", so the
# chemprop 1.4.0 CLI defaults are authoritative:
#   batch_size = 50  (polymer-chemprop/chemprop/args.py:93)
#   epochs     = 30  (polymer-chemprop/chemprop/args.py:382)
#   no early-stopping patience mechanism in chemprop 1.4.0
#
# We reproduce that here with --patience 30 against a 30-epoch cap, which
# means the EarlyStopping callback can never fire. This is intentional:
# chemprop 1.4.0 has no patience concept, so a smaller value would silently
# introduce a modern early-stopping halting mechanism that the original paper
# never used and would truncate the NoamLR schedule.

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
PILOT_COUNT=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/wdmpnn_original/r1_r3"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
rm -f "$PBS_DIR"/*.pbs
rm -f "$MANIFEST_DIR"/r1_r3_*
FULL_MANIFEST="$MANIFEST_DIR/r1_r3_all.manifest"
PILOT_MANIFEST="$MANIFEST_DIR/r1_r3_pilot.manifest"
REMAINDER_MANIFEST="$MANIFEST_DIR/r1_r3_after_review.manifest"
: > "$FULL_MANIFEST"

TARGET_TOKENS=(EA_vs_SHE_eV IP_vs_SHE_eV)
TARGET_ARGS=("EA vs SHE (eV)" "IP vs SHE (eV)")
SEEDS=(42 43 44)
SPLIT_TYPES=(monomer_heldout monomer_b_heldout_clustered)
SPLIT_SUBDIRS=(ea_ip_lomo ea_ip_lomo_b_clustered)
ORIG_TOKEN="__orig"
PREDICTION_ROOT="$PROJECT_DIR/predictions/wdmpnn_original"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/wdmpnn_original"

for split_index in "${!SPLIT_TYPES[@]}"; do
    SPLIT_TYPE="${SPLIT_TYPES[$split_index]}"
    SPLIT_SUBDIR="${SPLIT_SUBDIRS[$split_index]}"
    for target_index in "${!TARGET_TOKENS[@]}"; do
        target_token="${TARGET_TOKENS[$target_index]}"
        target="${TARGET_ARGS[$target_index]}"
        for seed in "${SEEDS[@]}"; do
            for fold in {0..8}; do
                output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${target_token}__wdmpnn__${SPLIT_TYPE}__fold${fold}__s${seed}${ORIG_TOKEN}.npz"
                runner="scripts/python/run_wdmpnn_generalization.py"
                args="--split_types $SPLIT_TYPE --folds $fold --targets '$target' --seed $seed --split_seed 42 --batch_size 50 --epochs 30 --patience 30 --frozen_protocol --protocol_variant original_paper --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/wdmpnn'"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "wdmpnn" "$target_token" "$fold" "$seed" "$output|$args" >> "$FULL_MANIFEST"
            done
        done
    done
done

TOTAL_RUNS="$(wc -l < "$FULL_MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((2 * 2 * 9 * 3))
[[ "$TOTAL_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s runs, generated %s\n' "$EXPECTED_RUNS" "$TOTAL_RUNS" >&2; exit 1; }

# Pre-flight: no planned output may already exist. If PREDICTION_ROOT's parent
# is not reachable on this host (e.g. the generator is being run locally, not
# on Gadi), skip the existence check and warn. Do not abort.
PRED_PARENT="$(dirname "$PREDICTION_ROOT")"
if [[ ! -d "$PRED_PARENT" ]]; then
    printf 'WARNING: %s is not reachable on this host; pre-flight existence check skipped. Re-run this generator on Gadi before submitting.\n' "$PRED_PARENT" >&2
else
    declare -a PREEXISTING=()
    while IFS=$'\t' read -r _runner _model _target_token _fold _seed payload; do
        output="${payload%%|*}"
        base_output="${output%${ORIG_TOKEN}.npz}.npz"
        base_config="${base_output%.npz}.config.json"
        if [[ -e "$output" || -e "${output%.npz}.config.json" || -e "$base_output" || -e "$base_config" ]]; then
            PREEXISTING+=("$output")
        fi
    done < "$FULL_MANIFEST"
    if [[ ${#PREEXISTING[@]} -gt 0 ]]; then
        printf 'ERROR: %s planned output(s) already exist:\n' "${#PREEXISTING[@]}" >&2
        printf '  %s\n' "${PREEXISTING[@]}" >&2
        exit 1
    fi
fi

# Pilot = one R1 EA seed 42 fold 0, plus one R3 EA seed 42 fold 0.
awk -F'\t' '$2 == "wdmpnn" && $3 == "EA_vs_SHE_eV" && $4 == 0 && $5 == 42' "$FULL_MANIFEST" > "$PILOT_MANIFEST"
awk -F'\t' '!($2 == "wdmpnn" && $3 == "EA_vs_SHE_eV" && $4 == 0 && $5 == 42)' "$FULL_MANIFEST" > "$REMAINDER_MANIFEST"

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
TASK_LOG_DIR="$PROJECT_DIR/logs/wdmpnn_original/r1_r3/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/wdmpnn_original/r1_r3/manifests/$(basename "$manifest")"
LINE="\$(sed -n "${line_num}p" "\$MANIFEST")"
IFS=\$'\t' read -r RUNNER MODEL TARGET FOLD SEED PAYLOAD <<< "\$LINE"
OUTPUT="\${PAYLOAD%%|*}"
ARGS="\${PAYLOAD#*|}"
# The runner writes the base filename (without the ${ORIG_TOKEN} token). We move
# the resulting .npz and sidecar into place after a successful run.
BASE_OUTPUT="\${OUTPUT%${ORIG_TOKEN}.npz}.npz"
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
# The original paper has no patience/early-stop concept. A 30-epoch cap with
# patience=30 means EarlyStopping can never fire, so every run must consume all
# 30 epochs.
epochs_actually_run=\$(python3 -c "import json; print(json.load(open('\${OUTPUT%.npz}.config.json'))['epochs_actually_run'])")
[[ "\$epochs_actually_run" -eq 30 ]] || { printf 'Run %s did not complete 30 epochs (got %s); possible silent early stop.\n' "\$OUTPUT" "\$epochs_actually_run" >&2; exit 1; }
EOF
    chmod +x "$pbs"
}

PILOT_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/wdmpnn_orig_p_${TASK_INDEX}.pbs"
    write_per_task_pbs "$PILOT_MANIFEST" "$TASK_INDEX" "wdmpnn_orig_p_${TASK_INDEX}" "$task_pbs"
    PILOT_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$PILOT_MANIFEST"

REMAINDER_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/wdmpnn_orig_r_${TASK_INDEX}.pbs"
    write_per_task_pbs "$REMAINDER_MANIFEST" "$TASK_INDEX" "wdmpnn_orig_r_${TASK_INDEX}" "$task_pbs"
    REMAINDER_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$REMAINDER_MANIFEST"

printf 'wDMPNN original cells (splits): %s\n' "${#SPLIT_TYPES[@]}"
printf 'wDMPNN original runs: %s\n' "$TOTAL_RUNS"
printf 'Pilot jobs: %s\n' "$PILOT_COUNT"
printf 'Post-review jobs: %s\n' "$REMAINDER_COUNT"
printf 'Charging project: %s\n' "$PROJECT"
printf 'Cost warning: per-run SU at batch_size 50 is unmeasured. The ~85 SU/run figure used elsewhere is for batch 512 and does not transfer: batch 50 means roughly 4.8x more optimiser steps per run but only 30 epochs instead of the regen_v1 median of ~63. Do not quote the regen cost for these runs.\n'
printf 'Pilot per-task PBS files: %s\n' "${#PILOT_PBS_LIST[@]}"
printf 'Post-review per-task PBS files: %s\n' "${#REMAINDER_PBS_LIST[@]}"
printf 'Fresh predictions: %s\n' "$PREDICTION_ROOT"
printf 'Fresh checkpoints: %s\n' "$CHECKPOINT_ROOT"
printf 'Sample pilot PBS: %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Sample post-review PBS: %s\n' "${REMAINDER_PBS_LIST[0]}"
printf 'No jobs submitted. Submit one pilot job to test: qsub %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Submit all pilots: for f in %s/wdmpnn_orig_p_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
printf 'Submit all post-review: for f in %s/wdmpnn_orig_r_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
printf 'Do not submit the post-review jobs until pilot sidecars pass the training/provenance check.\n'
