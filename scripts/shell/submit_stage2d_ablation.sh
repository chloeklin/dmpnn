#!/bin/bash
# Submit Stage 2D Ablation Study to PBS
# Generates one 12-hour PBS job per target + model combination.
# Wraps scripts/python/run_stage2d_ablation.py
#
# Usage:
#   ./submit_stage2d_ablation.sh [walltime=12:00:00] [folds=0,1,...,8] [models=2D1,2D1_FiLM,...] [targets=EA,IP] [--no-submit]
#
# Examples:
#   ./submit_stage2d_ablation.sh
#   ./submit_stage2d_ablation.sh walltime=24:00:00
#   ./submit_stage2d_ablation.sh folds=0,1,2
#   ./submit_stage2d_ablation.sh models=2D1_FiLM,2D1_NonlinearMix
#   ./submit_stage2d_ablation.sh targets=EA
#   ./submit_stage2d_ablation.sh --no-submit

WALLTIME="12:00:00"
FOLDS=""
MODELS=""
TARGETS="EA,IP"
SUBMIT_JOB=true

for arg in "$@"; do
  case $arg in
    walltime=*)   WALLTIME="${arg#walltime=}" ;;
    folds=*)      FOLDS="${arg#folds=}" ;;
    models=*)     MODELS="${arg#models=}" ;;
    targets=*)     TARGETS="${arg#targets=}" ;;
    --no-submit)  SUBMIT_JOB=false ;;
    *)
      echo "Warning: Unknown argument '$arg' ignored"
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default model list (must match keys in run_stage2d_ablation.py)
if [ -z "$MODELS" ]; then
  MODEL_LIST=(2D1 2D1_FiLM 2D1_NonlinearMix 2D1_FiLM_NonlinearMix)
else
  IFS=',' read -ra MODEL_LIST <<< "$MODELS"
fi

# Target list
IFS=',' read -ra TARGET_LIST <<< "$TARGETS"

# Optional folds flag
FOLDS_FLAG=""
[ -n "$FOLDS" ] && FOLDS_FLAG="--folds $FOLDS"

# If qsub is not available locally, force --no-submit mode
if [ "$SUBMIT_JOB" = true ] && ! command -v qsub &> /dev/null; then
  echo ""
  echo "Note: qsub not found on this machine. Forcing --no-submit mode."
  echo "Submit the generated scripts from Gadi with:"
  SUBMIT_JOB=false
fi

SUBMITTED_JOBS=()

for TARGET in "${TARGET_LIST[@]}"; do
  for MODEL in "${MODEL_LIST[@]}"; do
    JOB_NAME="S2DA_${TARGET}_${MODEL}"
    OUTPUT_SCRIPT="${SCRIPT_DIR}/run_stage2d_ablation_${TARGET}_${MODEL}.sh"

    cat > "$OUTPUT_SCRIPT" << EOF
#!/bin/bash
#PBS -q gpuvolta
#PBS -P ng76
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=$WALLTIME
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N ${JOB_NAME}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Stage 2D Ablation Study: ${TARGET} / ${MODEL}
python3 scripts/python/run_stage2d_ablation.py --target ${TARGET} --models ${MODEL} ${FOLDS_FLAG}
EOF

    chmod +x "$OUTPUT_SCRIPT"
    echo "Generated: $OUTPUT_SCRIPT"

    if [ "$SUBMIT_JOB" = true ]; then
      JOB_ID=$(qsub "$OUTPUT_SCRIPT")
      if [ $? -eq 0 ]; then
        echo "  Submitted: $JOB_ID (${JOB_NAME})"
        SUBMITTED_JOBS+=("$JOB_ID")
      else
        echo "  Error: Failed to submit ${JOB_NAME}"
        exit 1
      fi
    fi
  done
done

echo ""
if [ "$SUBMIT_JOB" = true ]; then
  echo "All jobs submitted successfully."
  echo "Monitor with: qstat -u \$USER"
  echo "Job IDs: ${SUBMITTED_JOBS[*]}"
else
  echo "Scripts generated but not submitted (--no-submit or qsub unavailable)."
  echo "Submit from Gadi with:"
  for TARGET in "${TARGET_LIST[@]}"; do
    for MODEL in "${MODEL_LIST[@]}"; do
      echo "  qsub ${SCRIPT_DIR}/run_stage2d_ablation_${TARGET}_${MODEL}.sh"
    done
  done
fi
