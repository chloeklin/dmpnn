#!/bin/bash

# wDMPNN Stage 2D Baseline Training — PBS Job Script
# ═══════════════════════════════════════════════════════════════════════
# Generates and submits PBS jobs for wDMPNN on ea_ip dataset.
# Uses the batch system pattern (like batch_generate_scripts.sh).
#
# Usage:
#   ./run_wdmpnn_stage2d_baselines.sh              # Generate all 3 split jobs
#   ./run_wdmpnn_stage2d_baselines.sh a_held_out   # Generate only a_held_out job
#   ./run_wdmpnn_stage2d_baselines.sh --no-submit  # Generate but don't submit
#
# Splits:
#   1. a_held_out       (via train_graph.py --split_type a_held_out)
#   2. group_disjoint   (via run_wdmpnn_generalization.py)
#   3. pair_disjoint    (via run_wdmpnn_generalization.py)
#
# Each produces per-fold .npz with: y_true, y_pred, test_indices,
# smiles_A, smiles_B, fracA, fracB, poly_type
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────
NO_SUBMIT=false
SPLIT_FILTER=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-submit) NO_SUBMIT=true; shift ;;
    a_held_out|group_disjoint|pair_disjoint) SPLIT_FILTER="$1"; shift ;;
    -h|--help)
      echo "Usage: $0 [split_type] [--no-submit]"
      echo ""
      echo "Arguments:"
      echo "  split_type     One of: a_held_out, group_disjoint, pair_disjoint"
      echo "                 (omit to generate all 3)"
      echo "  --no-submit    Generate scripts but don't qsub"
      echo ""
      echo "Examples:"
      echo "  $0                           # Generate all 3 jobs"
      echo "  $0 a_held_out                # Generate only a_held_out"
      echo "  $0 --no-submit               # Generate all, don't submit"
      echo "  $0 group_disjoint --no-submit # Generate group_disjoint only"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── PBS Configuration ────────────────────────────────────────────────
WALLTIME="12:30:00"
NCPUS=12
NGPUS=1
MEM="48GB"
QUEUE="gpuvolta"
PROJECT="um09"
STORAGE="scratch/um09+gdata/um09"

# ── Python Environment ───────────────────────────────────────────────
# Match the pattern from generate_training_script.sh (Gadi cluster)
MODULE_LOAD='
module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate'

TRAIN_GRAPH="scripts/python/train_graph.py"
WDMPNN_GEN="scripts/python/run_wdmpnn_generalization.py"

# Ensure scripts exist
if [ ! -f "$PROJECT_ROOT/$WDMPNN_GEN" ]; then
    echo "Error: Script not found: $WDMPNN_GEN"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo " wDMPNN Stage 2D Baseline Job Generator"
echo " Project root: $PROJECT_ROOT"
echo " Generate time: $(date)"
echo " Filter: ${SPLIT_FILTER:-all}"
echo " Auto-submit: $([ "$NO_SUBMIT" = true ] && echo "NO" || echo "YES")"
echo "═══════════════════════════════════════════════════════════════"

# ── Define split configurations ─────────────────────────────────────
declare -A SPLIT_CONFIGS
declare -A SPLIT_CMDS

# a_held_out: uses train_graph.py with --split_type a_held_out
SPLIT_CONFIGS[a_held_out]="ea_ip__wDMPNN__a_held_out"
SPLIT_CMDS[a_held_out]='
# Train EA target
python3 '"$TRAIN_GRAPH"' \
    --dataset_name ea_ip \
    --model_name wDMPNN \
    --task_type reg \
    --polymer_type copolymer \
    --split_type a_held_out \
    --save_predictions \
    --target "EA vs SHE (eV)"

# Train IP target  
python3 '"$TRAIN_GRAPH"' \
    --dataset_name ea_ip \
    --model_name wDMPNN \
    --task_type reg \
    --polymer_type copolymer \
    --split_type a_held_out \
    --save_predictions \
    --target "IP vs SHE (eV)"'

# group_disjoint: uses custom generalization script
SPLIT_CONFIGS[group_disjoint]="ea_ip__wDMPNN__group_disjoint"
SPLIT_CMDS[group_disjoint]='
python3 '"$WDMPNN_GEN"' \
    --split_types group_disjoint \
    --folds 0,1,2,3,4 \
    --targets "EA vs SHE (eV),IP vs SHE (eV)"'

# pair_disjoint: uses custom generalization script
SPLIT_CONFIGS[pair_disjoint]="ea_ip__wDMPNN__pair_disjoint"
SPLIT_CMDS[pair_disjoint]='
python3 '"$WDMPNN_GEN"' \
    --split_types pair_disjoint \
    --folds 0,1,2,3,4 \
    --targets "EA vs SHE (eV),IP vs SHE (eV)"'

# ── Generate jobs ───────────────────────────────────────────────────
generate_job() {
    local split_name="$1"
    local job_name="${SPLIT_CONFIGS[$split_name]}"
    local commands="${SPLIT_CMDS[$split_name]}"
    local script_file="${SCRIPT_DIR}/train_wdmpnn_${split_name}.sh"

    cat > "$script_file" << EOJOB
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -N $job_name
#PBS -j oe

$MODULE_LOAD

# Set working directory
cd "$PROJECT_ROOT"

# Verify environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found after module load/activate"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo " wDMPNN Training: $split_name"
echo " Start: \$(date)"
echo " Host: \$(hostname)"
echo " CUDA: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Python: \$(which python3)"
echo "═══════════════════════════════════════════════════════════════"

# Run commands (python3 is available after module load + activate)
$commands

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Complete: \$(date)"
echo " Predictions:"
echo "   a_held_out:     predictions/wDMPNN/"
echo "   group_disjoint: predictions/wDMPNN_Gen/"
echo "   pair_disjoint:  predictions/wDMPNN_Gen/"
echo "═══════════════════════════════════════════════════════════════"
EOJOB

    chmod +x "$script_file"
    echo "Generated: $script_file"

    if [ "$NO_SUBMIT" = false ]; then
        echo "  Submitting to PBS..."
        JOB_ID=$(qsub "$script_file")
        if [ $? -eq 0 ]; then
            echo "  ✓ Job submitted: $JOB_ID"
        else
            echo "  ✗ Submission failed"
            return 1
        fi
    else
        echo "  (not submitted -- use qsub $script_file to submit)"
    fi
    echo ""
}

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Generate selected splits
SPLITS_TO_RUN=(a_held_out group_disjoint pair_disjoint)
if [ -n "$SPLIT_FILTER" ]; then
    SPLITS_TO_RUN=("$SPLIT_FILTER")
fi

for split in "${SPLITS_TO_RUN[@]}"; do
    if [ -n "${SPLIT_CONFIGS[$split]:-}" ]; then
        generate_job "$split"
    else
        echo "Warning: Unknown split '$split'"
    fi
done

echo "═══════════════════════════════════════════════════════════════"
echo " Job generation complete"
echo "═══════════════════════════════════════════════════════════════"
