#!/bin/bash
#PBS -P um09
#PBS -q gpuvolta
#PBS -l walltime=12:30:00
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l ngpus=1
#PBS -l storage=scratch/um09+gdata/um09
#PBS -l wd
#PBS -j oe

# ═══════════════════════════════════════════════════════════════════════
# wDMPNN Stage 2D Baseline Training
# ═══════════════════════════════════════════════════════════════════════
# Trains wDMPNN on ea_ip dataset for all three split types:
#   1. a_held_out       (via train_graph.py --split_type a_held_out)
#   2. group_disjoint   (via run_wdmpnn_generalization.py)
#   3. pair_disjoint    (via run_wdmpnn_generalization.py)
#
# Each produces per-fold .npz with: y_true, y_pred, test_indices,
# smiles_A, smiles_B, fracA, fracB, poly_type
#
# Submit one job per block (a_held_out, group_disjoint, pair_disjoint)
# or run all sequentially on a single long job.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────
module load python3/3.11.7
module load cuda/12.3.2

PROJECT_ROOT="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="./venv/bin/python3"
TRAIN_GRAPH="scripts/python/train_graph.py"
WDMPNN_GEN="scripts/python/run_wdmpnn_generalization.py"

echo "═══════════════════════════════════════════════════════════════"
echo " wDMPNN Stage 2D Baselines"
echo " Project root: $PROJECT_ROOT"
echo " Start time:   $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ── Select which split(s) to run ─────────────────────────────────────
# Override via environment variable, e.g.:
#   SPLIT_MODE=a_held_out qsub scripts/run_wdmpnn_stage2d_baselines.sh
# Default: run all three
SPLIT_MODE="${SPLIT_MODE:-all}"

# ─────────────────────────────────────────────────────────────────────
# 1. A-HELD-OUT (uses train_graph.py native split)
# ─────────────────────────────────────────────────────────────────────
if [[ "$SPLIT_MODE" == "all" || "$SPLIT_MODE" == "a_held_out" ]]; then
    echo ""
    echo "━━━ [1/3] wDMPNN — a_held_out ━━━"
    echo ""

    for TARGET in "EA vs SHE (eV)" "IP vs SHE (eV)"; do
        echo "  Target: $TARGET"
        $PYTHON "$TRAIN_GRAPH" \
            --dataset_name ea_ip \
            --model_name wDMPNN \
            --task_type reg \
            --polymer_type copolymer \
            --split_type a_held_out \
            --save_predictions \
            --target "$TARGET"
    done

    echo "  ✓ a_held_out complete"
fi

# ─────────────────────────────────────────────────────────────────────
# 2. GROUP-DISJOINT (uses custom script)
# ─────────────────────────────────────────────────────────────────────
if [[ "$SPLIT_MODE" == "all" || "$SPLIT_MODE" == "group_disjoint" ]]; then
    echo ""
    echo "━━━ [2/3] wDMPNN — group_disjoint ━━━"
    echo ""

    $PYTHON "$WDMPNN_GEN" \
        --split_types group_disjoint \
        --folds 0,1,2,3,4

    echo "  ✓ group_disjoint complete"
fi

# ─────────────────────────────────────────────────────────────────────
# 3. PAIR-DISJOINT (uses custom script)
# ─────────────────────────────────────────────────────────────────────
if [[ "$SPLIT_MODE" == "all" || "$SPLIT_MODE" == "pair_disjoint" ]]; then
    echo ""
    echo "━━━ [3/3] wDMPNN — pair_disjoint ━━━"
    echo ""

    $PYTHON "$WDMPNN_GEN" \
        --split_types pair_disjoint \
        --folds 0,1,2,3,4

    echo "  ✓ pair_disjoint complete"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " DONE: $(date)"
echo " Predictions:"
echo "   a_held_out:     predictions/wDMPNN/"
echo "   group_disjoint: predictions/wDMPNN_Gen/"
echo "   pair_disjoint:  predictions/wDMPNN_Gen/"
echo "═══════════════════════════════════════════════════════════════"
