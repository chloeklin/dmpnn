#!/bin/bash
#
# wDMPNN LOMAO (Leave-One-Monomer-A-Out) Experiment - Cluster Submission
#
# Submits a single PBS job that trains wDMPNN on ea_ip using the
# a_held_out / leave_one_A_out (LOMAO) 9-fold cross-validation.
#
# Because all 9 folds are handled inside one train_graph.py run (unlike
# the group/pair scripts which submit one job per fold), this script
# submits one job only.
#
# Usage:
#   ./submit_wdmpnn_lomao.sh                  # Submit full 9-fold run
#   ./submit_wdmpnn_lomao.sh --dry_run        # Print command without submitting
#   ./submit_wdmpnn_lomao.sh --no-submit      # Generate PBS script only, do not qsub
#   ./submit_wdmpnn_lomao.sh --verify         # Run only the first fold (quick sanity check)

set -e

# ── Cluster configuration ──────────────────────────────────────────────
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
MODULE_PATH="/g/data/dk92/apps/Modules/modulefiles"

# PBS resources
WALLTIME="24:00:00"
NCPUS=12
MEM="100GB"
NGPUS=1
QUEUE="gpuvolta"
PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
JOBFS="100GB"

# ── Parse arguments ──────────────────────────────────────────────────────
DRY_RUN=false
NO_SUBMIT=false
VERIFY=false

for arg in "$@"; do
    case $arg in
        --dry_run)   DRY_RUN=true ;;
        --no-submit) NO_SUBMIT=true ;;
        --verify)    VERIFY=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/wdmpnn_lomao"
mkdir -p "$LOG_DIR"

# ── Build the python command ─────────────────────────────────────────────
BASE_CMD="python3 scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --model_name wDMPNN \
  --polymer_type copolymer \
  --split_type a_held_out \
  --a_held_out_protocol leave_one_A_out \
  --results_subdir HPG2Stage_LOMAO \
  --save_predictions"

if [[ "$VERIFY" == "true" ]]; then
    CMD="$BASE_CMD --max_folds 1"
    JOB_NAME="wdmpnn_lomao_verify"
    WALLTIME="03:00:00"
else
    CMD="$BASE_CMD"
    JOB_NAME="wdmpnn_lomao_full"
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "wDMPNN LOMAO Experiment - Job Submission"
echo "================================================================"
echo "  Mode:        $([ "$VERIFY" == "true" ] && echo "VERIFY (fold 0 only)" || echo "FULL (all 9 folds)")"
echo "  Job name:    $JOB_NAME"
echo "  Walltime:    $WALLTIME"
echo "  Project dir: $PROJECT_DIR"
echo "  Dry run:     $DRY_RUN"
echo "  No submit:   $NO_SUBMIT"
echo "  Command:     $CMD"
echo "================================================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [DRY] $JOB_NAME: $CMD"
    echo ""
    echo "================================================================"
    echo "Dry run complete. 1 job would be submitted."
    echo ""
    echo "To generate PBS script: ./submit_wdmpnn_lomao.sh --no-submit"
    echo "To submit full run:     ./submit_wdmpnn_lomao.sh"
    echo "To submit verify run:   ./submit_wdmpnn_lomao.sh --verify"
    echo "================================================================"
    exit 0
fi

PBS_SCRIPT="$LOG_DIR/${JOB_NAME}.pbs"
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
#PBS -N $JOB_NAME

module use $MODULE_PATH
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR

echo "Job: $JOB_NAME"
echo "Mode: $([ "$VERIFY" == "true" ] && echo "VERIFY fold 0 only" || echo "FULL 9-fold LOMAO")"
echo "Start: \$(date)"

$CMD

echo "End: \$(date)"
EOF

chmod +x "$PBS_SCRIPT"

if [[ "$NO_SUBMIT" == "true" ]]; then
    echo "  [GENERATED] $JOB_NAME -> $PBS_SCRIPT"
    echo ""
    echo "================================================================"
    echo "Generated PBS script: $PBS_SCRIPT"
    echo "To submit: qsub $PBS_SCRIPT"
    echo "================================================================"
else
    JOB_ID=$(qsub "$PBS_SCRIPT")
    echo "  [SUBMITTED] $JOB_NAME -> $JOB_ID"
    echo ""
    echo "================================================================"
    echo "Submitted 1 job."
    echo "Monitor with: qstat -u \$USER"
    echo ""
    echo "After completion, results will be in:"
    echo "  results/HPG2Stage_LOMAO/ea_ip__wDMPNN__a_held_out__target_*.csv"
    echo "================================================================"
fi
