#!/bin/bash
# Backfill the 5 specific missing regen_v1 cells listed in
# analysis/model_diagnostics/HANDOFF_2026-07-29.md §12:
#
#   HPG-hier          R1 (A-heldout)  EA  fold2  s43
#   HPG-hier-octamer  R3 (B-cluster)  IP  fold7  s44
#   HPG-hier-junction R1 (A-heldout)  EA  fold2  s42
#   HPG-hier-junction R1 (A-heldout)  EA  fold8  s42
#   HPG-hier-junction1 R1 (A-heldout) IP  fold4  s44
#
# Usage:
#   scripts/shell/backfill_regen_v1_missing_5.sh          # generate PBS only
#   scripts/shell/backfill_regen_v1_missing_5.sh --submit # generate and qsub
#
# The script is safe to re-run: it skips cells that already have both
# .npz and .config.json and refuses partial outputs.

set -euo pipefail

PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="gpuvolta"
NCPUS=12
NGPUS=1
MEM="100GB"
JOBFS="100GB"
WALLTIME="06:00:00"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"

SUBMIT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) SUBMIT=true; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/backfill_regen_v1_missing_5"
MANIFEST_DIR="$LOG_DIR/manifests"
PBS_DIR="$LOG_DIR/pbs"
TASK_LOG_DIR="$LOG_DIR/tasks"
mkdir -p "$MANIFEST_DIR" "$PBS_DIR" "$TASK_LOG_DIR"

MANIFEST="$MANIFEST_DIR/missing_5.manifest"
: > "$MANIFEST"

PRED_ROOT="$PROJECT_DIR/predictions/regen_v1"
CKPT_ROOT="$PROJECT_DIR/checkpoints/regen_v1/hpg"
COMMON_HPG_ARGS="--stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --split_seed 42 --epochs 100 --patience 15 --min_epochs 1 --batch_size 64 --frozen_protocol --prediction_dir '$PRED_ROOT' --checkpoint_dir '$CKPT_ROOT'"

add() {
    local split=$1 target_arg=$2 model=$3 fold=$4 seed=$5
    local subdir target_token target_str

    case "$split" in
        monomer_heldout) subdir="ea_ip_lomo" ;;
        monomer_b_heldout_clustered) subdir="ea_ip_lomo_b_clustered" ;;
        *) printf 'Unknown split: %s\n' "$split" >&2; exit 2 ;;
    esac

    case "$target_arg" in
        EA) target_token="EA_vs_SHE_eV"; target_str="EA vs SHE (eV)" ;;
        IP) target_token="IP_vs_SHE_eV"; target_str="IP vs SHE (eV)" ;;
        *) printf 'Unknown target: %s\n' "$target_arg" >&2; exit 2 ;;
    esac

    local output="$PRED_ROOT/$subdir/ea_ip__${target_token}__${model}__${split}__fold${fold}__s${seed}.npz"
    local args="--split_types $split --folds $fold --targets '$target_str' --models $model $COMMON_HPG_ARGS --seed $seed"
    printf '%s\t%s\t%s\t%s\n' "scripts/python/run_hpg_generalization.py" "$model" "$output" "$args" >> "$MANIFEST"
}

add monomer_heldout EA hpg_hier 2 43
add monomer_b_heldout_clustered IP hpg_hier_octamer 7 44
add monomer_heldout EA hpg_hier_junction 2 42
add monomer_heldout EA hpg_hier_junction 8 42
add monomer_heldout IP hpg_hier_junction1 4 44

TASK_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
PBS_SCRIPT="$PBS_DIR/backfill_regen_v1_missing_5.pbs"

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
#PBS -N backfill_regen_v1_m5
#PBS -r y
#PBS -J 0-$((TASK_COUNT - 1))

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
TASK_LOG_DIR="$PROJECT_DIR/logs/backfill_regen_v1_missing_5/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/m5_\${PBS_ARRAY_INDEX}_\${PBS_JOBID}.log") 2>&1
MANIFEST="$PROJECT_DIR/logs/backfill_regen_v1_missing_5/manifests/$(basename "$MANIFEST")"
LINE="\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "\$MANIFEST")"
IFS=\$'\\t' read -r RUNNER MODEL OUTPUT ARGS <<< "\$LINE"
[[ -n "\$RUNNER" ]] || { printf 'Missing manifest entry\n' >&2; exit 2; }
if [[ -f "\$OUTPUT" && -f "\${OUTPUT%.npz}.config.json" ]]; then
    printf 'Skipping completed cell: %s\n' "\$OUTPUT"
    exit 0
fi
if [[ -e "\$OUTPUT" || -e "\${OUTPUT%.npz}.config.json" ]]; then
    printf 'Partial output exists; refusing ambiguous resume: %s\n' "\$OUTPUT" >&2
    exit 1
fi
eval "set -- \$ARGS"
printf 'runner=%s model=%s output=%s\n' "\$RUNNER" "\$MODEL" "\$OUTPUT"
python "\$RUNNER" "\$@"
EOF
chmod +x "$PBS_SCRIPT"

printf 'Manifest: %s\n' "$MANIFEST"
printf 'PBS script: %s\n' "$PBS_SCRIPT"

if [[ "$SUBMIT" == true ]]; then
    JOB_ID=$(qsub "$PBS_SCRIPT")
    printf 'Submitted: %s\n' "$JOB_ID"
else
    printf 'PBS script generated but NOT submitted.\n'
    printf 'To submit: qsub %s\n' "$PBS_SCRIPT"
fi
