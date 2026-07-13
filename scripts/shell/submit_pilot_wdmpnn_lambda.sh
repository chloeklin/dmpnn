#!/usr/bin/env bash
# =============================================================================
# submit_pilot_wdmpnn_lambda.sh
#
# Pilot λ_within sweep for wDMPNN on the EA target (group_disjoint, fold 0).
#
# Lambda values : 0.0  0.03  0.1  0.3
# Target        : EA vs SHE (eV)
# Split         : group_disjoint
# Fold          : 0
# Seed          : 42  (hard-coded in run_wdmpnn_generalization.py)
# Epochs        : 300 / patience 30  (hard-coded)
# Batch size    : 512               (hard-coded)
# Results subdir: wDMPNN_Pilot_Lambda
#
# Usage
# -----
#   bash submit_pilot_wdmpnn_lambda.sh [--dry-run] [--no-submit]
#
#   --dry-run    : print job scripts to stdout; do not write files or submit
#   --no-submit  : write PBS files but do NOT call qsub
#
# PBS resources : 12 CPUs, 1 GPU (volta), 100GB RAM, 12:30 h walltime
# Project       : ng76
# Queue         : gpuvolta
# =============================================================================

set -euo pipefail

# ── Cluster / environment constants ──────────────────────────────────────────
PROJECT_CODE="ng76"
QUEUE="gpuvolta"
WALLTIME="12:30:00"
NCPUS=12
NGPUS=1
MEM="100GB"
STORAGE="scratch/um09+gdata/dk92"
JOBFS="100GB"
MODULE_PATH="/g/data/dk92/apps/Modules/modulefiles"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
WORKDIR="/scratch/um09/hl4138/dmpnn"

# ── Experiment parameters ─────────────────────────────────────────────────────
TARGET="EA vs SHE (eV)"
SPLIT="group_disjoint"
FOLD=0
RESULTS_SUBDIR="wDMPNN_Pilot_Lambda"
LAMBDAS=(0.0 0.03 0.1 0.3)

# ── Parse flags ───────────────────────────────────────────────────────────────
DRY_RUN=false
NO_SUBMIT=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --no-submit)  NO_SUBMIT=true ;;
        *)            echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_DIR="${SCRIPT_DIR}/pbs_pilot_lambda"
if ! $DRY_RUN; then
    mkdir -p "$PBS_DIR"
fi

echo "=================================================="
echo " wDMPNN Pilot Lambda Sweep"
echo "=================================================="
echo "  Target       : ${TARGET}"
echo "  Split        : ${SPLIT}"
echo "  Fold         : ${FOLD}"
echo "  Results dir  : ${RESULTS_SUBDIR}"
echo "  Lambda values: ${LAMBDAS[*]}"
echo "  Dry run      : ${DRY_RUN}"
echo "  No-submit    : ${NO_SUBMIT}"
echo "=================================================="

# ── Generate one PBS script per lambda ───────────────────────────────────────
for LW in "${LAMBDAS[@]}"; do
    # Filesystem-safe lambda tag (e.g. 0.03 → lw0p03)
    LW_TAG="lw$(echo "$LW" | tr '.' 'p')"
    JOB_NAME="wdmpnn_pilot_${LW_TAG}_${SPLIT}_fold${FOLD}"
    PBS_FILE="${PBS_DIR}/${JOB_NAME}.pbs"

    # ── PBS script body ───────────────────────────────────────────────────────
    PBS_BODY=$(cat <<PBSEOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -P ${PROJECT_CODE}
#PBS -q ${QUEUE}
#PBS -l walltime=${WALLTIME}
#PBS -l ncpus=${NCPUS}
#PBS -l ngpus=${NGPUS}
#PBS -l mem=${MEM}
#PBS -l storage=${STORAGE}
#PBS -l jobfs=${JOBFS}

# ── Environment ──────────────────────────────────────────────────────────────
module use ${MODULE_PATH}
module load ${MODULE_PYTHON} ${MODULE_CUDA}

source ${VENV_ACTIVATE}
cd ${WORKDIR}

echo "=============================="
echo "Job   : ${JOB_NAME}"
echo "Node  : \$(hostname)"
echo "Time  : \$(date)"
echo "Lambda: ${LW}"
echo "=============================="

# ── Training command ──────────────────────────────────────────────────────────
python3 scripts/python/run_wdmpnn_generalization.py \
    --folds ${FOLD} \
    --split_types ${SPLIT} \
    --targets "${TARGET}" \
    --lambda_within ${LW} \
    --results_subdir ${RESULTS_SUBDIR}

echo "=============================="
echo "Done: \$(date)"
echo "=============================="
PBSEOF
)

    if $DRY_RUN; then
        echo ""
        echo "──────────────────────────────────────────────────"
        echo "  [DRY RUN] Would write: ${PBS_FILE}"
        echo "──────────────────────────────────────────────────"
        echo "$PBS_BODY"
    else
        echo "$PBS_BODY" > "$PBS_FILE"
        echo "  Wrote: ${PBS_FILE}"

        if ! $NO_SUBMIT; then
            JOB_ID=$(qsub "$PBS_FILE")
            echo "  Submitted: ${JOB_ID}  (lambda=${LW})"
        else
            echo "  [NO-SUBMIT] Would qsub: ${PBS_FILE}"
        fi
    fi
done

echo ""
echo "All done."
if ! $DRY_RUN && ! $NO_SUBMIT; then
    echo "Monitor with: qstat -u \$USER"
fi
