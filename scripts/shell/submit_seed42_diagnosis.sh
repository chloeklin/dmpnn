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
WALLTIME="24:00:00"

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
LOG_DIR="$LOCAL_PROJECT/logs/seed42_diagnosis"
MANIFEST_DIR="$LOG_DIR/manifests"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/seed42_diagnosis.manifest"
PBS_SCRIPT="$LOG_DIR/seed42_diagnosis.pbs"

: > "$MANIFEST"
SPLITS=(group_disjoint pair_disjoint monomer_heldout)
TARGETS=("EA vs SHE (eV)" "IP vs SHE (eV)")
for split in "${SPLITS[@]}"; do
    if [[ "$split" == "monomer_heldout" ]]; then folds=(0 1 2 3 4 5 6 7 8); else folds=(0 1 2 3 4); fi
    for fold in "${folds[@]}"; do
        for target in "${TARGETS[@]}"; do
            printf '%s\n' "hpg --split_types $split --folds $fold --targets '$target' --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --seed 42" >> "$MANIFEST"
            printf '%s\n' "hpg --split_types $split --folds $fold --targets '$target' --models hpg_sum --seed 42" >> "$MANIFEST"
            printf '%s\n' "hpg --split_types $split --folds $fold --targets '$target' --models hpg_frac --seed 42" >> "$MANIFEST"
            printf '%s\n' "wdmpnn --split_types $split --folds $fold --targets '$target' --seed 42" >> "$MANIFEST"
            printf '%s\n' "stage2d --split_types $split --folds $fold --targets '$target' --models frac --seed 42" >> "$MANIFEST"
            printf '%s\n' "stage2d --split_types $split --folds $fold --targets '$target' --models 2d0_arch --seed 42" >> "$MANIFEST"
            printf '%s\n' "stage2d --split_types $split --folds $fold --targets '$target' --models 2d1_arch --seed 42" >> "$MANIFEST"
        done
    done
done
TASK_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
LAST_INDEX=$((TASK_COUNT - 1))

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
#PBS -N eaip_s42diag
#PBS -r y
#PBS -J 0-$LAST_INDEX

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR

TASK="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "$PROJECT_DIR/logs/seed42_diagnosis/manifests/$(basename "$MANIFEST")")"
read -r RUNNER TASK_ARGS <<< "\$TASK"
eval "set -- \$TASK_ARGS"
case "\$RUNNER" in
    hpg) python scripts/python/run_hpg_generalization.py "\$@" ;;
    wdmpnn) python scripts/python/run_wdmpnn_generalization.py "\$@" ;;
    stage2d) python scripts/python/run_stage2d_generalization.py "\$@" ;;
    *) printf 'Unknown runner: %s\\n' "\$RUNNER" >&2; exit 2 ;;
esac
EOF
chmod +x "$PBS_SCRIPT"

printf 'Manifest: %s\nTasks: %s\nPBS script: %s\n' "$MANIFEST" "$TASK_COUNT" "$PBS_SCRIPT"
if [[ "$DRY_RUN" == true ]]; then
    nl -ba "$MANIFEST"
elif [[ "$NO_SUBMIT" == true ]]; then
    printf 'Generated only: qsub %s\n' "$PBS_SCRIPT"
else
    qsub "$PBS_SCRIPT"
fi
