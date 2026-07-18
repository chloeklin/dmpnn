#!/bin/bash
set -euo pipefail

PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="normal"
NCPUS=12
MEM="100GB"
JOBFS="100GB"
MODULE_PYTHON="python3/3.12.1"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
WALLTIME="04:00:00"

SEEDS="42,43,44"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/seeded_diagnostics"
mkdir -p "$LOG_DIR"
PBS_SCRIPT="$LOG_DIR/seeded_diagnostics_${SEEDS//,/_}.pbs"

cat > "$PBS_SCRIPT" <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N seed_diag

set -euo pipefail
module load $MODULE_PYTHON
source $VENV_ACTIVATE
cd $PROJECT_DIR
python scripts/python/run_seeded_diagnostics.py --seeds $SEEDS --skip-aggregate
python scripts/python/aggregate_seeded_diagnostics.py --seeds $SEEDS
EOF
chmod +x "$PBS_SCRIPT"

echo "PBS script: $PBS_SCRIPT"
echo "PBS header:"
sed -n '1,16p' "$PBS_SCRIPT"
if [[ "$DRY_RUN" == true ]]; then
    echo "qsub $PBS_SCRIPT"
elif [[ "$NO_SUBMIT" == true ]]; then
    echo "Generated only: qsub $PBS_SCRIPT"
else
    qsub "$PBS_SCRIPT"
fi
