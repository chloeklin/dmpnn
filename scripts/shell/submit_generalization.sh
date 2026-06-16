#!/bin/bash
#
# Stage 2D Generalization Experiments - Cluster Submission
#
# Submits individual PBS jobs for each (split_type × model × fold) combination.
# Each job trains one model variant with one split type on one fold.
#
# Experiments:
#   A) group_disjoint: Hold out entire (A,B,fracA) composition groups
#   B) pair_disjoint:  Hold out entire (A,B) monomer pairs
#
# Usage:
#   ./submit_generalization.sh              # Submit all 30 jobs
#   ./submit_generalization.sh --dry_run    # Print commands without submitting
#   ./submit_generalization.sh --no-submit  # Generate PBS scripts only

set -e

# ── Cluster configuration (matches generate_training_script.sh) ──
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
MODULE_PATH="/g/data/dk92/apps/Modules/modulefiles"

# PBS resources
WALLTIME="8:00:00"
NCPUS=12
MEM="100GB"
NGPUS=1
QUEUE="gpuvolta"
PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
JOBFS="100GB"

# Experiment configuration
SPLIT_TYPES=("group_disjoint" "pair_disjoint")
MODELS=("frac" "2d0_arch" "2d1_arch")
FOLDS=(0 1 2 3 4)

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
LOG_DIR="$LOCAL_PROJECT/logs/generalization"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Stage 2D Generalization Experiments - Job Submission"
echo "================================================================"
echo "  Split types: ${SPLIT_TYPES[*]}"
echo "  Models: ${MODELS[*]}"
echo "  Folds: ${FOLDS[*]}"
echo "  Total jobs: $((${#SPLIT_TYPES[@]} * ${#MODELS[@]} * ${#FOLDS[@]}))"
echo "  Walltime: $WALLTIME"
echo "  Project dir: $PROJECT_DIR"
echo "  Dry run: $DRY_RUN"
echo "  No submit: $NO_SUBMIT"
echo "================================================================"
echo ""

JOB_COUNT=0

for SPLIT in "${SPLIT_TYPES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for FOLD in "${FOLDS[@]}"; do
            # Short name for PBS (max 15 chars)
            SPLIT_SHORT="${SPLIT:0:3}"  # "gro" or "pai"
            JOB_NAME="gen_${SPLIT_SHORT}_${MODEL}_f${FOLD}"
            
            CMD="python3 scripts/python/run_stage2d_generalization.py --split_types $SPLIT --models $MODEL --folds $FOLD"
            
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
echo "Split: $SPLIT, Model: $MODEL, Fold: $FOLD"
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
    echo "To generate PBS scripts: ./submit_generalization.sh --no-submit"
    echo "To submit all jobs:      ./submit_generalization.sh"
elif [[ "$NO_SUBMIT" == "true" ]]; then
    echo "Generated $JOB_COUNT PBS scripts in: $LOG_DIR"
    echo "To submit all: for f in $LOG_DIR/gen_*.pbs; do qsub \$f; done"
else
    echo "Submitted $JOB_COUNT jobs."
    echo "Monitor with: qstat -u \$USER"
    echo ""
    echo "After completion, run analysis:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 experiments/hpg2stage/scripts/analyze_stage2d_generalization.py"
fi
echo "================================================================"
