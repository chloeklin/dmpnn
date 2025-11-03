#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=4:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-htpmd-DMPNN

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: htpmd
# Model: DMPNN
# Descriptors: false
# RDKit: false
# Batch Norm: false
# Train Size: full
# Expected Result: results/DMPNN/htpmd_baseline.csv

echo "Starting evaluation..."
echo "Model: DMPNN"
echo "Dataset: htpmd"
echo "Configuration: desc=false, rdkit=false, batch_norm=false"

python3 scripts/python/evaluate_model.py \
    --model_name DMPNN --dataset_name htpmd --checkpoint_path /Users/u6788552/Desktop/experiments/dmpnn/checkpoints/DMPNN/htpmd__Conductivity__batch_norm__rep0

echo "Evaluation complete!"
echo "Results saved to: results/DMPNN/htpmd_baseline.csv"
