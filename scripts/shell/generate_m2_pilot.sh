#!/bin/bash
# Generate PBS scripts for the M2 pilot (hpg_hier_octamer_edges) on the A split (monomer_heldout).
#
# M2: 8-slot octamer chain + mean readout (via --stage2_readout stoich_weighted, which resolves
#     to OctamerEncoder mean pooling for stage2_mode=octamer_sequence, see HANDOFF §7) +
#     the 17-d junction edge feature restored into the path layers (--octamer_edge_features,
#     implied by the hpg_hier_octamer_edges model token).
#
# M2 isolates topology (M2 vs HPG-hier, both +features) and edge features (arm D vs M2, both
# 8-slot + mean). See analysis/model_diagnostics/PREREG_M2_2026-08-12.md.
#
# A pilot is all 9 folds x 1 target (EA) x 3 seeds = 27 runs. Do not pilot on a subset of folds.
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
SU_PER_RUN=36.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/m2/pilot"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"
rm -f "$PBS_DIR"/*.pbs

MANIFEST="$MANIFEST_DIR/m2_pilot.manifest"
: > "$MANIFEST"

SPLIT_TYPE="monomer_heldout"
SPLIT_SUBDIR="ea_ip_lomo"
TARGET_TOKEN="EA_vs_SHE_eV"
TARGET="EA vs SHE (eV)"
SEEDS=(42 43 44)
FOLDS=(0 1 2 3 4 5 6 7 8)

M2_MODEL="hpg_hier_octamer_edges"
M2_TOKEN="__m2"
M2_ARGS="--stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --stage2_readout stoich_weighted"

PREDICTION_ROOT="$PROJECT_DIR/predictions/m2"
CHECKPOINT_ROOT="$PROJECT_DIR/checkpoints/m2"
LOCAL_PREDICTION_ROOT="$LOCAL_PROJECT/predictions/m2"

# Comparator roots (local, for the pre-flight existence checks below).
HPG_HIER_ROOT="$LOCAL_PROJECT/predictions/regen_v1/$SPLIT_SUBDIR"
OCTAMER_ROOT="$LOCAL_PROJECT/predictions/regen_v1/$SPLIT_SUBDIR"
ARM_D_ROOT="$LOCAL_PROJECT/predictions/octamer_cd/$SPLIT_SUBDIR"

declare -a PREEXISTING=()
declare -a MISSING_HPG_HIER=()
declare -a MISSING_OCTAMER=()
declare -a FOLDS_MISSING_ARM_D=()

for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        hpg_hier_comparator="$HPG_HIER_ROOT/ea_ip__${TARGET_TOKEN}__hpg_hier__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"
        octamer_comparator="$OCTAMER_ROOT/ea_ip__${TARGET_TOKEN}__hpg_hier_octamer__${SPLIT_TYPE}__fold${fold}__s${seed}.npz"
        [[ -f "$hpg_hier_comparator" ]] || MISSING_HPG_HIER+=("$hpg_hier_comparator")
        [[ -f "$octamer_comparator" ]] || MISSING_OCTAMER+=("$octamer_comparator")
    done
    arm_d_comparator="$ARM_D_ROOT/ea_ip__${TARGET_TOKEN}__hpg_hier_octamer__${SPLIT_TYPE}__fold${fold}__s42__armD.npz"
    [[ -f "$arm_d_comparator" ]] || FOLDS_MISSING_ARM_D+=("$fold")
done

for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        output="$PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${M2_MODEL}__${SPLIT_TYPE}__fold${fold}__s${seed}${M2_TOKEN}.npz"
        local_output="$LOCAL_PREDICTION_ROOT/$SPLIT_SUBDIR/ea_ip__${TARGET_TOKEN}__${M2_MODEL}__${SPLIT_TYPE}__fold${fold}__s${seed}${M2_TOKEN}.npz"

        if [[ -e "$local_output" || -e "${local_output%.npz}.config.json" ]]; then
            PREEXISTING+=("$local_output")
        fi

        runner="scripts/python/run_hpg_generalization.py"
        args="--split_types $SPLIT_TYPE --folds $fold --targets '$TARGET' --models $M2_MODEL $M2_ARGS --seed $seed --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PREDICTION_ROOT' --checkpoint_dir '$CHECKPOINT_ROOT/hpg'"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$runner" "$M2_MODEL" "$TARGET_TOKEN" "$fold" "$seed" "$M2_TOKEN" "$output|$args" >> "$MANIFEST"
    done
done

TOTAL_RUNS="$(wc -l < "$MANIFEST" | tr -d ' ')"
EXPECTED_RUNS=$((9 * 1 * 3))
[[ "$TOTAL_RUNS" -eq "$EXPECTED_RUNS" ]] || { printf 'Expected %s runs, generated %s\n' "$EXPECTED_RUNS" "$TOTAL_RUNS" >&2; exit 1; }

if [[ ${#PREEXISTING[@]} -gt 0 ]]; then
    printf 'ERROR: %s planned output(s) already exist:\n' "${#PREEXISTING[@]}" >&2
    printf '  %s\n' "${PREEXISTING[@]}" >&2
fi
if [[ ${#MISSING_HPG_HIER[@]} -gt 0 ]]; then
    printf 'ERROR: %s hpg_hier comparator(s) are missing:\n' "${#MISSING_HPG_HIER[@]}" >&2
    printf '  %s\n' "${MISSING_HPG_HIER[@]}" >&2
fi
if [[ ${#MISSING_OCTAMER[@]} -gt 0 ]]; then
    printf 'ERROR: %s octamer comparator(s) are missing:\n' "${#MISSING_OCTAMER[@]}" >&2
    printf '  %s\n' "${MISSING_OCTAMER[@]}" >&2
fi
if [[ ${#PREEXISTING[@]} -gt 0 || ${#MISSING_HPG_HIER[@]} -gt 0 || ${#MISSING_OCTAMER[@]} -gt 0 ]]; then
    exit 1
fi

# Arm D is a report-only pre-flight: it does not block generation (arm D currently exists for
# folds 0 and 4 only; see PREREG_M2_2026-08-12.md §4). The M2 <-> arm D contrast is incomplete
# until the arm D pilot is extended, but M2 <-> HPG-hier and M2 <-> octamer are unaffected.
if [[ ${#FOLDS_MISSING_ARM_D[@]} -gt 0 ]]; then
    printf 'NOTE: arm D comparator missing for %s of %s folds (M2 <-> arm D contrast incomplete): %s\n' \
        "${#FOLDS_MISSING_ARM_D[@]}" "${#FOLDS[@]}" "${FOLDS_MISSING_ARM_D[*]}"
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
TASK_LOG_DIR="$PROJECT_DIR/logs/m2/pilot/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/m2/pilot/manifests/$(basename "$MANIFEST")"
LINE="\$(sed -n "${line_num}p" "\$MANIFEST")"
IFS=\$'\t' read -r RUNNER MODEL TARGET FOLD SEED TOKEN PAYLOAD <<< "\$LINE"
OUTPUT="\${PAYLOAD%%|*}"
ARGS="\${PAYLOAD#*|}"
BASE_OUTPUT="\${OUTPUT%\${TOKEN}.npz}.npz"
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
printf 'runner=%s model=%s target=%s fold=%s seed=%s token=%s\\n' "\$RUNNER" "\$MODEL" "\$TARGET" "\$FOLD" "\$SEED" "\$TOKEN"
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
    task_pbs="$PBS_DIR/m2_pilot_${TASK_INDEX}.pbs"
    write_per_task_pbs "$line_num" "m2_pilot_${TASK_INDEX}" "$task_pbs"
    PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$MANIFEST"

ESTIMATED_SU=$(awk -v n="$TOTAL_RUNS" -v s="$SU_PER_RUN" 'BEGIN { printf "%.0f", n * s }')
ESTIMATED_KSU=$(awk -v su="$ESTIMATED_SU" 'BEGIN { printf "%.1f", su / 1000 }')

printf 'Pilot runs: %s\n' "$TOTAL_RUNS"
printf 'Estimated cost: %s runs x %s SU ~ %s kSU\n' "$TOTAL_RUNS" "$SU_PER_RUN" "$ESTIMATED_KSU"
printf 'Charging project: %s\n' "$PROJECT"
printf 'Per-task PBS files: %s\n' "${#PBS_LIST[@]}"
printf 'PBS output directory: %s\n' "$PBS_DIR"
printf 'Manifest: %s\n' "$MANIFEST"
if [[ ${#FOLDS_MISSING_ARM_D[@]} -gt 0 ]]; then
    printf 'Folds lacking an arm D comparator (M2 <-> arm D incomplete): %s\n' "${FOLDS_MISSING_ARM_D[*]}"
else
    printf 'Arm D comparator present for all requested folds.\n'
fi
