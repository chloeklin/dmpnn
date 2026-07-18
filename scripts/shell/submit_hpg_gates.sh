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
WALLTIME="00:40:00"

DRY_RUN=false
NO_SUBMIT=false
for arg in "$@"; do
    case "$arg" in
        --dry_run) DRY_RUN=true ;;
        --no-submit) NO_SUBMIT=true ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/hpg_gates"
MANIFEST_DIR="$LOG_DIR/manifests"
mkdir -p "$MANIFEST_DIR"
GATE1_MANIFEST="$MANIFEST_DIR/gate1.manifest"
GATE2_MANIFEST="$MANIFEST_DIR/gate2.manifest"
GATE1_PBS="$LOG_DIR/gate1.pbs"
GATE2_PBS="$LOG_DIR/gate2.pbs"
GATE1_REPORT_PBS="$LOG_DIR/gate1_report.pbs"
REPORT_PBS="$LOG_DIR/gate2_report.pbs"

printf '%s\n' "--split_types group_disjoint --folds 0 --targets 'EA vs SHE (eV)' --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --seed 42" > "$GATE1_MANIFEST"
printf '%s\n' "--split_types group_disjoint --folds 0 --targets 'EA vs SHE (eV)' --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --seed 43" > "$GATE2_MANIFEST"
printf '%s\n' "--split_types group_disjoint --folds 0 --targets 'EA vs SHE (eV)' --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --seed 44" >> "$GATE2_MANIFEST"

write_array_pbs() {
    local manifest="$1"
    local pbs="$2"
    local name="$3"
    local tasks
    tasks="$(wc -l < "$manifest" | tr -d ' ')"
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
#PBS -J 0-$((tasks - 1))

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
MANIFEST="$PROJECT_DIR/logs/hpg_gates/manifests/$(basename "$manifest")"
TASK_ARGS="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "\$MANIFEST")"
eval "set -- \$TASK_ARGS"
python scripts/python/run_hpg_generalization.py "\$@"
EOF
    chmod +x "$pbs"
}

write_array_pbs "$GATE1_MANIFEST" "$GATE1_PBS" "hpg_gate1"
write_array_pbs "$GATE2_MANIFEST" "$GATE2_PBS" "hpg_gate2"
cat > "$GATE1_REPORT_PBS" <<EOF
#!/bin/bash
#PBS -q normal
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N hpg_gate1r

set -euo pipefail
module load $MODULE_PYTHON
source $VENV_ACTIVATE
cd $PROJECT_DIR
PYTHONPATH=. python scripts/python/report_hpg_gate.py --models hpg_hier --seed 42
EOF
chmod +x "$GATE1_REPORT_PBS"
cat > "$REPORT_PBS" <<EOF
#!/bin/bash
#PBS -q normal
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N hpg_gate2r

set -euo pipefail
module load $MODULE_PYTHON
source $VENV_ACTIVATE
cd $PROJECT_DIR
PYTHONPATH=. python scripts/python/report_hpg_gate.py --models hpg_hier --seeds 43,44 --assert-nonzero-seed-std
EOF
chmod +x "$REPORT_PBS"

echo "Gate 1 manifest (2 tasks): $GATE1_MANIFEST"
nl -ba "$GATE1_MANIFEST"
echo "Gate 2 manifest (2 tasks): $GATE2_MANIFEST"
nl -ba "$GATE2_MANIFEST"
echo "Gate 1 PBS header:"
sed -n '1,16p' "$GATE1_PBS"
echo "Gate 1 report PBS header:"
sed -n '1,15p' "$GATE1_REPORT_PBS"
echo "Gate 2 report PBS header:"
sed -n '1,15p' "$REPORT_PBS"
if [[ "$DRY_RUN" == true ]]; then
    echo "qsub $GATE1_PBS"
    echo "qsub -W depend=afterok:<gate1-array-job-id> $GATE1_REPORT_PBS"
    echo "qsub $GATE2_PBS"
    echo "qsub -W depend=afterok:<gate2-array-job-id> $REPORT_PBS"
elif [[ "$NO_SUBMIT" == true ]]; then
    echo "Generated PBS scripts only."
else
    gate1_id="$(qsub "$GATE1_PBS")"
    gate1_report_id="$(qsub -W "depend=afterok:$gate1_id" "$GATE1_REPORT_PBS")"
    gate2_id="$(qsub "$GATE2_PBS")"
    report_id="$(qsub -W "depend=afterok:$gate2_id" "$REPORT_PBS")"
    echo "Submitted gate1=$gate1_id gate1_report=$gate1_report_id gate2=$gate2_id gate2_report=$report_id"
fi
