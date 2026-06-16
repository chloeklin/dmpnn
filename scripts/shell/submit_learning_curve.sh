#!/bin/bash
#
# Stage 2D Learning Curve Experiment - Cluster Submission
#
# Submits individual PBS jobs for each (model × fold × fraction) combination.
# Each job trains one model with one fraction of training matched groups.
#
# Usage:
#   ./submit_learning_curve.sh              # Submit all 40 jobs
#   ./submit_learning_curve.sh --dry_run    # Print commands without submitting
#   ./submit_learning_curve.sh --no-submit  # Generate PBS scripts only

set -e

# ── Cluster configuration (matches generate_training_script.sh) ──
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
MODULE_PATH="/g/data/dk92/apps/Modules/modulefiles"

# PBS resources
WALLTIME="5:30:00"
NCPUS=12
MEM="100GB"
NGPUS=1
QUEUE="gpuvolta"
PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
JOBFS="100GB"

# Experiment configuration
MODELS=("2d0_arch" "2d1_arch")
FOLDS=(0 1 2 3 4)
FRACTIONS=(25 50 75 100)

# Parse arguments
DRY_RUN=false
NO_SUBMIT=false
for arg in "$@"; do
    case $arg in
        --dry_run)  DRY_RUN=true ;;
        --no-submit) NO_SUBMIT=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/learning_curve"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Stage 2D Learning Curve - Job Submission"
echo "================================================================"
echo "  Models: ${MODELS[*]}"
echo "  Folds: ${FOLDS[*]}"
echo "  Fractions: ${FRACTIONS[*]}"
echo "  Total jobs: $((${#MODELS[@]} * ${#FOLDS[@]} * ${#FRACTIONS[@]}))"
echo "  Walltime: $WALLTIME"
echo "  Project dir: $PROJECT_DIR"
echo "  Dry run: $DRY_RUN"
echo "  No submit: $NO_SUBMIT"
echo "================================================================"
echo ""

JOB_COUNT=0

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for FRAC in "${FRACTIONS[@]}"; do
            JOB_NAME="lc_${MODEL}_f${FOLD}_p${FRAC}"
            
            CMD="python3 scripts/python/run_stage2d_learning_curve.py --models $MODEL --folds $FOLD --fractions $FRAC"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  [DRY] $JOB_NAME: $CMD"
            else
                # Generate PBS script
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
echo "Model: $MODEL, Fold: $FOLD, Fraction: ${FRAC}%"
echo "Start: \$(date)"

$CMD

echo "End: \$(date)"
EOF
                
                chmod +x "$PBS_SCRIPT"
                
                if [[ "$NO_SUBMIT" == "true" ]]; then
                    echo "  [GENERATED] $JOB_NAME -> $PBS_SCRIPT"
                else
                    JOB_ID=$(qsub "$PBS_SCRIPT")
                    echo "  [SUBMITTED] $JOB_NAME -> $JOB_ID"
                fi
            fi
            
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "================================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. $JOB_COUNT jobs would be submitted."
    echo ""
    echo "To generate PBS scripts: ./submit_learning_curve.sh --no-submit"
    echo "To submit all jobs:      ./submit_learning_curve.sh"
elif [[ "$NO_SUBMIT" == "true" ]]; then
    echo "Generated $JOB_COUNT PBS scripts in: $LOG_DIR"
    echo "To submit all: for f in $LOG_DIR/lc_*.pbs; do qsub \$f; done"
else
    echo "Submitted $JOB_COUNT jobs."
    echo "Monitor with: qstat -u \$USER"
    echo ""
    echo "After completion, run analysis:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 experiments/hpg2stage/scripts/analyze_stage2d_learning_curve.py"
fi
echo "================================================================"
