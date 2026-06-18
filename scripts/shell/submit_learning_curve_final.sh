#!/bin/bash
#
# Stage 2D Learning Curve — Final Pipeline — Cluster Submission
#
# Uses the EXACT same training pipeline as the final Stage 2D models
# (EPOCHS=300, PATIENCE=30, filtered a_held_out splits).
#
# VERIFICATION PROTOCOL:
#   1. First submit only 100% fraction jobs (PHASE 1)
#   2. After completion, run evaluation script to confirm:
#        2D0-arch: EA R²(Δ) ≈ 0.84, IP R²(Δ) ≈ 0.91
#        2D1-arch: EA R²(Δ) ≈ 0.86, IP R²(Δ) ≈ 0.91
#   3. Only then uncomment and submit PHASE 2 (25/50/75%)
#
# Usage:
#   ./submit_learning_curve_final.sh              # Submit PHASE 1 only
#   ./submit_learning_curve_final.sh --all        # Submit all (after verification)
#   ./submit_learning_curve_final.sh --dry_run    # Print commands only
#   ./submit_learning_curve_final.sh --no-submit  # Generate PBS scripts only

set -e

# ── Cluster configuration (matches generate_training_script.sh) ──
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
MODULE_PATH="/g/data/dk92/apps/Modules/modulefiles"

# PBS resources — EPOCHS=300 needs more walltime than old LC (was 5:30:00)
WALLTIME="10:00:00"
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

# Parse arguments
DRY_RUN=false
NO_SUBMIT=false
RUN_ALL=false
for arg in "$@"; do
    case $arg in
        --dry_run)    DRY_RUN=true ;;
        --no-submit)  NO_SUBMIT=true ;;
        --all)        RUN_ALL=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/learning_curve_final"
mkdir -p "$LOG_DIR"

# Training script path (relative to PROJECT_DIR on cluster)
TRAIN_SCRIPT="experiments/hpg2stage/scripts/run_stage2d_learning_curve_final.py"

echo "================================================================"
echo "Stage 2D Learning Curve — Final Pipeline — Job Submission"
echo "================================================================"
echo "  Models    : ${MODELS[*]}"
echo "  Folds     : ${FOLDS[*]}"
echo "  Walltime  : $WALLTIME"
echo "  EPOCHS    : 300 (matches final Stage 2D)"
echo "  PATIENCE  : 30  (matches final Stage 2D)"
echo "  Project   : $PROJECT_DIR"
echo "  Run all   : $RUN_ALL"
echo "  Dry run   : $DRY_RUN"
echo "================================================================"
echo ""

JOB_COUNT=0

submit_job() {
    local MODEL=$1
    local FOLD=$2
    local FRAC=$3
    local PHASE=$4

    JOB_NAME="lcf_${MODEL}_f${FOLD}_p${FRAC}"
    CMD="python3 $TRAIN_SCRIPT --models $MODEL --folds $FOLD --fractions $FRAC"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [$PHASE] $JOB_NAME: $CMD"
    else
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
echo "EPOCHS=300, PATIENCE=30 (final pipeline)"
echo "Start: \$(date)"

$CMD

echo "End: \$(date)"
EOF
        chmod +x "$PBS_SCRIPT"

        if [[ "$NO_SUBMIT" == "true" ]]; then
            echo "  [$PHASE/GENERATED] $JOB_NAME -> $PBS_SCRIPT"
        else
            JOB_ID=$(qsub "$PBS_SCRIPT")
            echo "  [$PHASE/SUBMITTED] $JOB_NAME -> $JOB_ID"
        fi
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
}

# ════════════════════════════════════════════════════════════════════
# PHASE 1: 100% verification (ALWAYS submitted)
# ════════════════════════════════════════════════════════════════════
echo "── PHASE 1: 100% Verification ──"
for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        submit_job "$MODEL" "$FOLD" 100 "PHASE1"
    done
done
echo ""

# ════════════════════════════════════════════════════════════════════
# PHASE 2: 25/50/75% (only if --all)
# ════════════════════════════════════════════════════════════════════
if [[ "$RUN_ALL" == "true" ]]; then
    echo "── PHASE 2: 25/50/75% Fractions ──"
    for MODEL in "${MODELS[@]}"; do
        for FOLD in "${FOLDS[@]}"; do
            for FRAC in 25 50 75; do
                submit_job "$MODEL" "$FOLD" "$FRAC" "PHASE2"
            done
        done
    done
    echo ""
else
    echo "── PHASE 2: SKIPPED (pass --all after verifying 100%) ──"
    echo "  After PHASE 1 completes, verify with:"
    echo "    python3 experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py --fractions 100"
    echo "  Then rerun with --all to submit remaining fractions."
    echo ""
fi

echo "================================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. $JOB_COUNT jobs would be submitted."
elif [[ "$NO_SUBMIT" == "true" ]]; then
    echo "Generated $JOB_COUNT PBS scripts in: $LOG_DIR"
    echo "To submit all: for f in $LOG_DIR/lcf_*.pbs; do qsub \$f; done"
else
    echo "Submitted $JOB_COUNT jobs."
    echo "Monitor with: qstat -u \$USER"
fi
echo ""
echo "After completion, run evaluation:"
echo "  python3 experiments/hpg2stage/scripts/evaluate_stage2d_learning_curve_final.py"
echo "================================================================"
