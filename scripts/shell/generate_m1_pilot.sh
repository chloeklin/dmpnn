#!/bin/bash
# Generate PBS scripts for the M1 monomer-level readout pilot and its
# published-config bridge.
#
# M1 isolates one architectural choice in the flat-graph wD-MPNN stack: whether
# the polymer vector is formed by pooling atoms directly to the polymer
# (wD-MPNN) or by first pooling atoms to monomer vectors and combining the two
# monomers with their mole fractions (M1).
#
# Pilot:  EA only, folds 0-8, seeds 42/43/44, our config (batch 64, 100 epochs,
#         patience 15) -> 27 jobs.
# Bridge: EA only, folds 0-8, seeds 42/43/44, published config (batch 50,
#         30 epochs, patience 30) -> 27 jobs.
#
# The two manifests are kept separate so the bridge can be submitted only after
# the pilot sidecars look sane.
#
# Usage: run locally (off Gadi) to generate PBS files.  It does not submit.
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
LOG_DIR="$LOCAL_PROJECT/logs/m1/pilot"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
rm -f "$PBS_DIR"/*.pbs

PILOT_MANIFEST="$MANIFEST_DIR/m1_pilot.manifest"
PUBLISHED_MANIFEST="$MANIFEST_DIR/m1_published.manifest"
: > "$PILOT_MANIFEST"
: > "$PUBLISHED_MANIFEST"

TARGET="EA vs SHE (eV)"
TARGET_TOKEN="EA_vs_SHE_eV"
SPLIT_TYPE="monomer_heldout"
SPLIT_SUBDIR="ea_ip_lomo"
PREDICTION_ROOT="$PROJECT_DIR/predictions/m1"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/m1"
LOCAL_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/m1"

# Comparator roots (local copy, for pre-flight checks)
HIER_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/regen_v1"
WDMPNN_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/regen_v1"

declare -a PREEXISTING_M1=()
declare -a MISSING_HIER=()
declare -a MISSING_WDMPNN=()

for fold in {0..8}; do
    for seed in 42 43 44; do
        # Pilot output token
        output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__wdmpnn_monomer_readout__${SPLIT_TYPE}__fold${fold}__s${seed}__m1.npz"
        local_output="$LOCAL_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__wdmpnn_monomer_readout__${SPLIT_TYPE}__fold${fold}__s${seed}__m1.npz"
        local_hier="$HIER_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__hpg_hier__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"
        local_wdmpnn="$WDMPNN_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__wdmpnn__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"

        if [[ -e "$local_output" || -e "${local_output%.npz}.config.json" ]]; then
            PREEXISTING_M1+=("$local_output")
        fi
        if [[ ! -f "$local_hier" ]]; then
            MISSING_HIER+=("$local_hier")
        fi
        if [[ ! -f "$local_wdmpnn" ]]; then
            MISSING_WDMPNN+=("$local_wdmpnn")
        fi

        runner="scripts/python/run_wdmpnn_generalization.py"
        args="--split_types $SPLIT_TYPE --folds $fold --targets '$TARGET' --monomer_readout --m1_variant ours --seed $seed --split_seed 42 --epochs 100 --patience 15 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT'"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "wdmpnn_monomer_readout" "$TARGET_TOKEN" "$fold" "$seed" "$output|$args" >> "$PILOT_MANIFEST"
    done
done

for fold in {0..8}; do
    for seed in 42 43 44; do
        # Published-config bridge output token
        output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__wdmpnn_monomer_readout__${SPLIT_TYPE}__fold${fold}__s${seed}__m1pub.npz"
        local_output="$LOCAL_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__wdmpnn_monomer_readout__${SPLIT_TYPE}__fold${fold}__s${seed}__m1pub.npz"
        local_hier="$HIER_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__hpg_hier__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"
        local_wdmpnn="$WDMPNN_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__wdmpnn__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"

        if [[ -e "$local_output" || -e "${local_output%.npz}.config.json" ]]; then
            PREEXISTING_M1+=("$local_output")
        fi
        # Only the published wD-MPNN comparator is strictly needed for the bridge,
        # but HPG-hier is checked too for consistency.
        if [[ ! -f "$local_hier" ]]; then
            MISSING_HIER+=("$local_hier")
        fi
        if [[ ! -f "$local_wdmpnn" ]]; then
            MISSING_WDMPNN+=("$local_wdmpnn")
        fi

        runner="scripts/python/run_wdmpnn_generalization.py"
        args="--split_types $SPLIT_TYPE --folds $fold --targets '$TARGET' --monomer_readout --m1_variant published --seed $seed --split_seed 42 --protocol_variant original_paper --batch_size 50 --epochs 30 --patience 30 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT'"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "wdmpnn_monomer_readout" "$TARGET_TOKEN" "$fold" "$seed" "$output|$args" >> "$PUBLISHED_MANIFEST"
    done
done

PILOT_RUNS="$(wc -l < "$PILOT_MANIFEST" | tr -d ' ')"
PUBLISHED_RUNS="$(wc -l < "$PUBLISHED_MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((9 * 1 * 3))
[[ "$PILOT_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s pilot runs, generated %s\n' "$EXPECTED_RUNS" "$PILOT_RUNS" >&2; exit 1; }
[[ "$PUBLISHED_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s published runs, generated %s\n' "$EXPECTED_RUNS" "$PUBLISHED_RUNS" >&2; exit 1; }

if [[ ${#PREEXISTING_M1[@]} -gt 0 ]]; then
    printf 'ERROR: %s planned M1 output(s) already exist:\n' "${#PREEXISTING_M1[@]}" >&2
    printf '  %s\n' "${PREEXISTING_M1[@]}" >&2
fi
if [[ ${#MISSING_HIER[@]} -gt 0 ]]; then
    printf 'ERROR: %s HPG-hier comparator(s) are missing:\n' "${#MISSING_HIER[@]}" >&2
    printf '  %s\n' "${MISSING_HIER[@]}" >&2
fi
if [[ ${#MISSING_WDMPNN[@]} -gt 0 ]]; then
    printf 'ERROR: %s wD-MPNN comparator(s) are missing:\n' "${#MISSING_WDMPNN[@]}" >&2
    printf '  %s\n' "${MISSING_WDMPNN[@]}" >&2
fi
if [[ ${#PREEXISTING_M1[@]} -gt 0 || ${#MISSING_HIER[@]} -gt 0 || ${#MISSING_WDMPNN[@]} -gt 0 ]]; then
    exit 1
fi

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
TASK_LOG_DIR="$PROJECT_DIR/logs/m1/pilot/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/m1/pilot/manifests/$(basename "$manifest")"
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

PILOT_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/m1_p_${TASK_INDEX}.pbs"
    write_per_task_pbs "$PILOT_MANIFEST" "$TASK_INDEX" "m1_p_${TASK_INDEX}" "$task_pbs"
    PILOT_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$PILOT_MANIFEST"

PUBLISHED_PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/m1_pub_${TASK_INDEX}.pbs"
    write_per_task_pbs "$PUBLISHED_MANIFEST" "$TASK_INDEX" "m1_pub_${TASK_INDEX}" "$task_pbs"
    PUBLISHED_PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$PUBLISHED_MANIFEST"

TOTAL_RUNS=$((PILOT_RUNS + PUBLISHED_RUNS))
ESTIMATED_SU=$(awk -v n="$TOTAL_RUNS" -v s="$SU_PER_RUN" 'BEGIN { printf "%.0f", n * s }')
ESTIMATED_KSU=$(awk -v su="$ESTIMATED_SU" 'BEGIN { printf "%.1f", su / 1000 }')

printf 'M1 pilot cells: %s\n' "$PILOT_RUNS"
printf 'M1 published-bridge cells: %s\n' "$PUBLISHED_RUNS"
printf 'Total runs: %s\n' "$TOTAL_RUNS"
printf 'Estimated cost: %s runs x %s SU ≈ %s kSU\n' "$TOTAL_RUNS" "$SU_PER_RUN" "$ESTIMATED_KSU"
printf 'Charging project: %s\n' "$PROJECT"
printf 'Pilot per-task PBS files: %s\n' "${#PILOT_PBS_LIST[@]}"
printf 'Bridge per-task PBS files: %s\n' "${#PUBLISHED_PBS_LIST[@]}"
printf 'Pilot manifest: %s\n' "$PILOT_MANIFEST"
printf 'Bridge manifest: %s\n' "$PUBLISHED_MANIFEST"
printf 'Fresh predictions: %s\n' "$PREDICTION_ROOT"
printf 'Fresh checkpoints: %s\n' "$CHECKPOINT_ROOT"
printf 'Sample pilot PBS: %s\n' "${PILOT_PBS_LIST[0]}"
printf 'Sample bridge PBS: %s\n' "${PUBLISHED_PBS_LIST[0]}"

printf '\n=== Submission runbook ===\n'
printf '1. Generator was just run from: %s\n' "$LOCAL_PROJECT"
printf '2. Check remaining SU: nci_account -P %s\n' "$PROJECT"
printf '3. Submit the M1 pilot one job at a time and inspect sidecars:\n'
printf '     %s\n' "${PILOT_PBS_LIST[0]}"
printf '     # Wait for it to finish, then verify sidecar has:\n'
printf '     #   resolved_config.monomer_readout == true\n'
printf '     #   resolved_config.m1_variant == "ours"\n'
printf '     #   resolved_config.batch_size == 64\n'
printf '     #   resolved_config.epochs == 100\n'
printf '     #   resolved_config.patience == 15\n'
printf '     #   resolved_config.frozen_protocol == true\n'
printf '     #   model == "wdmpnn_monomer_readout"\n'
printf '     #   best_epoch < 100 (no cap)\n'
printf '   Repeat for a second pilot if the first passes.\n'
printf '4. Submit the M1 pilot remainder only after the pilots pass by sending the per-task PBS files to the queue.\n'
printf '5. Submit the published-config bridge only after the pilot is read and looks sane.\n'
printf '6. Do not log in to Gadi manually; submit from local manifests only.\n'
