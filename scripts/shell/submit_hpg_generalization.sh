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
WALLTIME="12:30:00"

SEEDS="42"
DRY_RUN=false
NO_SUBMIT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=true; shift ;;
        --no-submit) NO_SUBMIT=true; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case ",$SEEDS," in
    *,42,*|*,43,*|*,44,*) ;;
    *) printf 'Use a non-empty subset of 42,43,44 for --seeds.\n' >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/hpg_generalization"
MANIFEST_DIR="$LOG_DIR/manifests"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/hpg_generalization_seeds_${SEEDS//,/_}.manifest"
PBS_SCRIPT="$LOG_DIR/hpg_generalization_seeds_${SEEDS//,/_}.pbs"

: > "$MANIFEST"
SPLITS=(group_disjoint pair_disjoint monomer_heldout)
TARGETS=("EA vs SHE (eV)" "IP vs SHE (eV)")
for seed in ${SEEDS//,/ }; do
    for split in "${SPLITS[@]}"; do
        if [[ "$split" == "monomer_heldout" ]]; then
            folds=(0 1 2 3 4 5 6 7 8)
        else
            folds=(0 1 2 3 4)
        fi
        for fold in "${folds[@]}"; do
            for target in "${TARGETS[@]}"; do
                printf '%s\n' "--split_types $split --folds $fold --targets '$target' --models hpg_sum --seed $seed" >> "$MANIFEST"
                printf '%s\n' "--split_types $split --folds $fold --targets '$target' --models hpg_frac --seed $seed" >> "$MANIFEST"
                printf '%s\n' "--split_types $split --folds $fold --targets '$target' --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --seed $seed" >> "$MANIFEST"
            done
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
#PBS -N hpg_gen
#PBS -J 0-$LAST_INDEX

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR

MANIFEST="$PROJECT_DIR/logs/hpg_generalization/manifests/$(basename "$MANIFEST")"
TASK_ARGS="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "\$MANIFEST")"
if [[ -z "\$TASK_ARGS" ]]; then
    printf 'No manifest entry for PBS_ARRAY_INDEX=%s\\n' "\$PBS_ARRAY_INDEX" >&2
    exit 2
fi
eval "set -- \$TASK_ARGS"
echo "Task \$PBS_ARRAY_INDEX: \$*"
python scripts/python/run_hpg_generalization.py "\$@"
EOF
chmod +x "$PBS_SCRIPT"

echo "Manifest: $MANIFEST"
echo "Tasks: $TASK_COUNT"
echo "PBS script: $PBS_SCRIPT"
echo "PBS header:"
sed -n '1,17p' "$PBS_SCRIPT"
if [[ "$DRY_RUN" == true ]]; then
    echo "Manifest contents:"
    nl -ba "$MANIFEST"
    echo "qsub $PBS_SCRIPT"
elif [[ "$NO_SUBMIT" == true ]]; then
    echo "Generated only: qsub $PBS_SCRIPT"
else
    qsub "$PBS_SCRIPT"
fi
