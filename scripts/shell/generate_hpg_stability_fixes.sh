#!/bin/bash
set -euo pipefail

PROJECT="um09"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="gpuvolta"
NCPUS=12
NGPUS=1
MEM="100GB"
JOBFS="100GB"
WALLTIME="04:00:00"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$LOCAL_PROJECT/logs/hpg_stability_fixes"
PBS_DIR="$OUT/pbs"
PREDICTION_DIR="$PROJECT_DIR/predictions/stability_fixes"
mkdir -p "$PBS_DIR"
rm -f "$PBS_DIR"/*.pbs

index=0
for fix in best_checkpoint row_val_best; do
    for fold in 0 1; do
        for repeat in 1 2 3; do
            epochs=100
            pbs="$PBS_DIR/stability_$(printf '%02d' "$index")_${fix}_fold${fold}_repeat${repeat}.pbs"
            cat > "$pbs" <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N stab_${fix:0:5}_f${fold}_r${repeat}
#PBS -r y
set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
test -f metadata/splits/monomer_heldout.json || { echo 'Missing metadata/splits/monomer_heldout.json' >&2; exit 1; }
nvidia-smi
python scripts/python/run_hpg_generalization.py --split_types monomer_heldout --folds $fold --targets 'EA vs SHE (eV)' --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --stage2_readout stoich_weighted --seed 42 --split_seed 42 --epochs $epochs --patience 15 --batch_size 64 --repeat $repeat --stability_fix $fix --prediction_dir "$PREDICTION_DIR"
EOF
            chmod +x "$pbs"
            index=$((index + 1))
        done
    done
done

[[ "$index" -eq 12 ]] || { printf 'Expected 12 jobs, generated %s\n' "$index" >&2; exit 1; }
printf 'Generated %s stability-test PBS jobs under project %s. No jobs submitted.\n' "$index" "$PROJECT"
printf 'PBS directory: %s\n' "$PBS_DIR"
