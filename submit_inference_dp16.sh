#!/bin/bash
#PBS -N wd_infer_dp16
#PBS -P ng76
#PBS -l storage=scratch/um09
#PBS -l walltime=00:30:00
#PBS -l ncpus=4
#PBS -l mem=16GB
#PBS -l wd
#PBS -o /scratch/um09/hl4138/dmpnn/logs/
#PBS -e /scratch/um09/hl4138/dmpnn/logs/

set -euo pipefail
module load python3/3.12.1
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/
CODE_ROOT="${POLYMER_CHEMPROP_ROOT:-$(python3 -c 'from run_inference_dp8 import POLYMER_CHEMPROP; print(POLYMER_CHEMPROP)')}"
if [[ -n "${POLYMER_CHEMPROP_ACTIVATE:-}" ]]; then
    source "$POLYMER_CHEMPROP_ACTIVATE"
elif [[ -f "$CODE_ROOT/.venv/bin/activate" ]]; then
    source "$CODE_ROOT/.venv/bin/activate"
elif [[ -f "$CODE_ROOT/venv/bin/activate" ]]; then
    source "$CODE_ROOT/venv/bin/activate"
else
    echo "No polymer-chemprop environment found. Export POLYMER_CHEMPROP_ACTIVATE to its activation script with qsub -v." >&2
    exit 2
fi
export OMP_NUM_THREADS="${PBS_NCPUS:-4}"
export MKL_NUM_THREADS="${PBS_NCPUS:-4}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-4}"
mkdir -p inference_results
python -u run_inference_dp16.py --code-root "$CODE_ROOT" --output inference_results/inference_dp16.csv
