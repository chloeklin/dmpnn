#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-tc-DMPNN_DiffPool

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: tc
# Model: DMPNN_DiffPool
# Configuration: Auto-detected from checkpoint path
# Expected Result: results/DMPNN_DiffPool/tc_baseline.csv

echo "Starting evaluation..."
echo "Model: DMPNN_DiffPool"
echo "Dataset: tc"
echo "Configuration: Auto-detected from checkpoint path"

python3 scripts/python/evaluate_model.py \
    --model_name DMPNN_DiffPool --dataset_name tc --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/DMPNN_DiffPool/tc__TC__rep0/logs/checkpoints/epoch=35-step=1728.ckpt"

echo "Evaluation complete!"
echo "Results saved to: results/DMPNN_DiffPool/tc_baseline.csv"
