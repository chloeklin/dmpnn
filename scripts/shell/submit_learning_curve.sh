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
#
# Prerequisites:
#   - Run dry_run first to generate group IDs:
#     python scripts/python/run_stage2d_learning_curve.py --dry_run

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/python/run_stage2d_learning_curve.py"
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"

MODELS=("2d0_arch" "2d1_arch")
FOLDS=(0 1 2 3 4)
FRACTIONS=(25 50 75 100)

WALLTIME="5:00:00"
NCPUS=4
MEM="32gb"
NGPUS=1
QUEUE="normal"

DRY_RUN=false
if [[ "$1" == "--dry_run" ]]; then
    DRY_RUN=true
fi

echo "================================================================"
echo "Stage 2D Learning Curve - Job Submission"
echo "================================================================"
echo "  Models: ${MODELS[*]}"
echo "  Folds: ${FOLDS[*]}"
echo "  Fractions: ${FRACTIONS[*]}"
echo "  Total jobs: $((${#MODELS[@]} * ${#FOLDS[@]} * ${#FRACTIONS[@]}))"
echo "  Walltime: $WALLTIME"
echo "  Dry run: $DRY_RUN"
echo "================================================================"
echo ""

# First ensure group IDs are generated
if [[ "$DRY_RUN" == "false" ]]; then
    echo "Step 1: Generating group subsampling (dry run)..."
    cd "$PROJECT_ROOT/scripts/python"
    python run_stage2d_learning_curve.py --dry_run
    echo ""
fi

JOB_COUNT=0

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for FRAC in "${FRACTIONS[@]}"; do
            JOB_NAME="lc_${MODEL}_f${FOLD}_p${FRAC}"
            LOG_DIR="$PROJECT_ROOT/logs/learning_curve"
            mkdir -p "$LOG_DIR"
            
            CMD="python $PYTHON_SCRIPT --models $MODEL --folds $FOLD --fractions $FRAC"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  [DRY] $JOB_NAME: $CMD"
            else
                # Generate PBS script
                PBS_SCRIPT="$LOG_DIR/${JOB_NAME}.pbs"
                cat > "$PBS_SCRIPT" <<EOF
#!/bin/bash
#PBS -N $JOB_NAME
#PBS -l walltime=$WALLTIME
#PBS -l ncpus=$NCPUS
#PBS -l mem=$MEM
#PBS -l ngpus=$NGPUS
#PBS -q $QUEUE
#PBS -o $LOG_DIR/${JOB_NAME}.out
#PBS -e $LOG_DIR/${JOB_NAME}.err
#PBS -l storage=gdata/vf+scratch/vf

cd $PROJECT_ROOT/scripts/python
source $VENV_ACTIVATE

echo "Job: $JOB_NAME"
echo "Model: $MODEL, Fold: $FOLD, Fraction: $FRAC%"
echo "Start: \$(date)"

$CMD

echo "End: \$(date)"
EOF
                
                # Submit
                JOB_ID=$(qsub "$PBS_SCRIPT")
                echo "  Submitted: $JOB_NAME -> $JOB_ID"
            fi
            
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "================================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. $JOB_COUNT jobs would be submitted."
else
    echo "Submitted $JOB_COUNT jobs."
    echo "Monitor with: qstat -u \$USER"
    echo "After completion, run analysis:"
    echo "  python analysis/results/hpg2stage/analyze_stage2d_learning_curve.py"
fi
echo "================================================================"
