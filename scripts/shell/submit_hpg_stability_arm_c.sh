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

DRY_RUN=false
NO_SUBMIT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run) DRY_RUN=true; shift ;;
        --no-submit) NO_SUBMIT=true; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/hpg_stability_fixes"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
TASK_LOG_DIR="$LOG_DIR/tasks"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$TASK_LOG_DIR"
MANIFEST="$MANIFEST_DIR/hpg_stability_arm_c.manifest"
PBS_SCRIPT="$PBS_DIR/hpg_stability_arm_c.pbs"
: > "$MANIFEST"

for fold in 0 1; do
    for repeat in 1 2 3; do
        output="$LOCAL_PROJECT/predictions/stability_fixes/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier__monomer_heldout__fold${fold}__s42__repeat${repeat}__arm_c.npz"
        [[ -f "$output" ]] && continue
        printf '%s\t%s\t%s\n' "$fold" "$repeat" "$output" >> "$MANIFEST"
    done
done

TASK_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
printf 'Pending Arm C jobs: %s\nManifest: %s\n' "$TASK_COUNT" "$MANIFEST"
if [[ "$TASK_COUNT" -eq 0 ]]; then
    printf 'All six Arm C artifacts already exist; nothing to submit.\n'
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
#PBS -N hpg_stab_c
#PBS -r y
#PBS -J 0-$((TASK_COUNT - 1))

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
test -f metadata/splits/monomer_heldout.json || { echo 'Missing metadata/splits/monomer_heldout.json' >&2; exit 1; }
TASK_LOG_DIR="$PROJECT_DIR/logs/hpg_stability_fixes/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/arm_c_\${PBS_ARRAY_INDEX}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/hpg_stability_fixes/manifests/$(basename "$MANIFEST")"
LINE="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "\$MANIFEST")"
IFS=\$'\t' read -r FOLD REPEAT OUTPUT <<< "\$LINE"
[[ -n "\$FOLD" && -n "\$REPEAT" ]] || { printf 'Missing manifest entry\n' >&2; exit 2; }
if [[ -f "\$OUTPUT" ]]; then
    printf 'Skipping completed artifact: %s\n' "\$OUTPUT"
    exit 0
fi
nvidia-smi
python scripts/python/run_hpg_generalization.py \
    --split_types monomer_heldout \
    --folds "\$FOLD" \
    --targets 'EA vs SHE (eV)' \
    --models hpg_hier \
    --stage1_pool sum \
    --stage2_depth 2 \
    --stage2_edge full \
    --stage2_readout stoich_weighted \
    --seed 42 \
    --split_seed 42 \
    --epochs 100 \
    --min_epochs 40 \
    --patience 30 \
    --batch_size 64 \
    --repeat "\$REPEAT" \
    --stability_fix arm_c \
    --prediction_dir "$PROJECT_DIR/predictions/stability_fixes"
EOF
chmod +x "$PBS_SCRIPT"
printf 'PBS script: %s\n' "$PBS_SCRIPT"
if [[ "$DRY_RUN" == true ]]; then
    nl -ba "$MANIFEST"
elif [[ "$NO_SUBMIT" == true ]]; then
    printf 'Generated only: qsub %s\n' "$PBS_SCRIPT"
else
    qsub "$PBS_SCRIPT"
fi
