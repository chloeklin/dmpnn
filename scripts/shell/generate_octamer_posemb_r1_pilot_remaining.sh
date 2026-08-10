#!/bin/bash
# Generate PBS files for the missing pilot seeds (43 and 44) of the R1
# octamer positional-embedding ablation.  Run this locally, then push the
# generated PBS files and manifest to Gadi and qsub them.
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
POSEMB_TOKEN="__noposemb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/octamer_posemb/r1"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$LOG_DIR/tasks"

FULL_MANIFEST="$MANIFEST_DIR/oct_posemb_r1_all.manifest"
[[ -f "$FULL_MANIFEST" ]] || { printf 'Run generate_octamer_posemb_r1.sh first to create %s\n' "$FULL_MANIFEST" >&2; exit 1; }

REMAINING_MANIFEST="$MANIFEST_DIR/oct_posemb_r1_pilot_remaining.manifest"
: > "$REMAINING_MANIFEST"

# Missing pilot seeds: EA folds 0 and 4, seeds 43 and 44.
awk -F'\t' '$2 == "hpg_hier_octamer" && $3 == "EA_vs_SHE_eV" && ($4 == 0 || $4 == 4) && ($5 == 43 || $5 == 44)' "$FULL_MANIFEST" >> "$REMAINING_MANIFEST"

REMAINING_COUNT="$(wc -l < "$REMAINING_MANIFEST" | tr -d ' ')"
[[ "$REMAINING_COUNT" -eq 4 ]] || { printf 'Expected 4 remaining pilot runs, found %s\n' "$REMAINING_COUNT" >&2; exit 1; }

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
TASK_LOG_DIR="$PROJECT_DIR/logs/octamer_posemb/r1/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/${name}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/octamer_posemb/r1/manifests/$(basename "$manifest")"
LINE="\$(sed -n "${line_num}p" "\$MANIFEST")"
IFS=\$'\t' read -r RUNNER MODEL TARGET FOLD SEED PAYLOAD <<< "\$LINE"
OUTPUT="\${PAYLOAD%%|*}"
ARGS="\${PAYLOAD#*|}"
BASE_OUTPUT="\${OUTPUT%${POSEMB_TOKEN}.npz}.npz"
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

PBS_LIST=()
TASK_INDEX=0
while IFS= read -r _line; do
    task_pbs="$PBS_DIR/oct_posemb_r1_p_rem_${TASK_INDEX}.pbs"
    write_per_task_pbs "$REMAINING_MANIFEST" "$TASK_INDEX" "oct_posemb_r1_p_rem_${TASK_INDEX}" "$task_pbs"
    PBS_LIST+=("$task_pbs")
    TASK_INDEX=$((TASK_INDEX + 1))
done < "$REMAINING_MANIFEST"

printf 'Generated %s remaining pilot PBS files:\n' "${#PBS_LIST[@]}"
printf '  %s\n' "${PBS_LIST[@]}"
printf 'Submit on Gadi with:\n'
printf '  for f in %s/oct_posemb_r1_p_rem_*.pbs; do qsub "$f"; done\n' "$PBS_DIR"
