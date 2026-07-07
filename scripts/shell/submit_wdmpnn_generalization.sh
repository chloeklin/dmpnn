#!/bin/bash
#
# wDMPNN Generalization Experiments - Cluster Submission
#
# Submits PBS jobs for wDMPNN with group_disjoint and pair_disjoint splits.
# Each job trains one (split_type × fold) combination.
#
# Experiments:
#   A) group_disjoint: Hold out entire (A,B,fracA) composition groups  [5 folds]
#   B) pair_disjoint:  Hold out entire (A,B) monomer pairs             [5 folds]
#
# Usage:
#   ./submit_wdmpnn_generalization.sh              # Submit all 20 jobs (2 splits × 5 folds × 2 targets)
#   ./submit_wdmpnn_generalization.sh --dry_run    # Print commands without submitting
#   ./submit_wdmpnn_generalization.sh --no-submit  # Generate PBS scripts only
#   ./submit_wdmpnn_generalization.sh --split group_disjoint   # Only one split type
#   ./submit_wdmpnn_generalization.sh --fold 0                 # Only one fold
#   ./submit_wdmpnn_generalization.sh --target EA              # Only EA target

set -e

# ── Cluster configuration ──────────────────────────────────────────────
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
MODULE_PATH="/g/data/dk92/apps/Modules/modulefiles"

# PBS resources
WALLTIME="12:30:00"
NCPUS=12
MEM="100GB"
NGPUS=1
QUEUE="gpuvolta"
PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
JOBFS="100GB"

# Experiment configuration
SPLIT_TYPES=("group_disjoint" "pair_disjoint")
FOLDS=(0 1 2 3 4)
TARGETS=("EA vs SHE (eV)" "IP vs SHE (eV)")
TARGET_KEYS=("EA" "IP")   # short keys for job names / file names

# Parse arguments
DRY_RUN=false
NO_SUBMIT=false
SPLIT_FILTER=""
FOLD_FILTER=""
TARGET_FILTER=""
for arg in "$@"; do
    case $arg in
        --dry_run)    DRY_RUN=true ;;
        --no-submit)  NO_SUBMIT=true ;;
        --split)      shift; SPLIT_FILTER="$1" ;;
        --fold)       shift; FOLD_FILTER="$1" ;;
        --target)     shift; TARGET_FILTER="$1" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/wdmpnn_generalization"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "wDMPNN Generalization Experiments - Job Submission"
echo "================================================================"
echo "  Split types: ${SPLIT_TYPES[*]}"
echo "  Targets:     ${TARGETS[*]}"
echo "  Folds:       ${FOLDS[*]}"
echo "  Total jobs:  $((${#SPLIT_TYPES[@]} * ${#FOLDS[@]} * ${#TARGETS[@]}))"
echo "  Walltime:    $WALLTIME"
echo "  Project dir: $PROJECT_DIR"
echo "  Dry run: $DRY_RUN"
echo "  No submit: $NO_SUBMIT"
echo "================================================================"
echo ""

JOB_COUNT=0

for SPLIT in "${SPLIT_TYPES[@]}"; do
    # Apply split filter if set
    if [[ -n "$SPLIT_FILTER" && "$SPLIT" != "$SPLIT_FILTER" ]]; then
        continue
    fi

    for FOLD in "${FOLDS[@]}"; do
        # Apply fold filter if set
        if [[ -n "$FOLD_FILTER" && "$FOLD" != "$FOLD_FILTER" ]]; then
            continue
        fi

        for TGT_IDX in "${!TARGETS[@]}"; do
            TARGET="${TARGETS[$TGT_IDX]}"
            TARGET_KEY="${TARGET_KEYS[$TGT_IDX]}"

            # Apply target filter if set
            if [[ -n "$TARGET_FILTER" && "$TARGET_KEY" != "$TARGET_FILTER" ]]; then
                continue
            fi

            SPLIT_SHORT="${SPLIT:0:3}"  # "gro" or "pai"
            JOB_NAME="wgen_${SPLIT_SHORT}_f${FOLD}_${TARGET_KEY}"

            CMD="python3 scripts/python/run_wdmpnn_generalization.py --split_types $SPLIT --folds $FOLD --targets \"$TARGET\""

            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  [DRY] $JOB_NAME: $CMD"
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
echo "Split: $SPLIT | Fold: $FOLD | Target: $TARGET"
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
    echo "To generate PBS scripts: ./submit_wdmpnn_generalization.sh --no-submit"
    echo "To submit all jobs:      ./submit_wdmpnn_generalization.sh"
elif [[ "$NO_SUBMIT" == "true" ]]; then
    echo "Generated $JOB_COUNT PBS scripts in: $LOG_DIR"
    echo "To submit all: for f in $LOG_DIR/wgen_*.pbs; do qsub \$f; done"
else
    echo "Submitted $JOB_COUNT jobs."
    echo "Monitor with: qstat -u \$USER"
    echo ""
    echo "After completion, run analysis:"
    echo "  python3 experiments/hpg2stage/scripts/analyze_stage2d_generalization.py"
fi
echo "================================================================"
