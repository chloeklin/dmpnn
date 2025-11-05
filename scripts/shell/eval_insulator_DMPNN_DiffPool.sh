#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-insulator-DMPNN_DiffPool

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: insulator
# Model: DMPNN_DiffPool
# Configuration: Auto-detected from checkpoint path
# Expected Result: results/DMPNN_DiffPool/insulator_baseline.csv

echo "Starting evaluation..."
echo "Model: DMPNN_DiffPool"
echo "Dataset: insulator"
echo "Configuration: Auto-detected from checkpoint path"

python3 scripts/python/evaluate_model.py \
    --model_name DMPNN_DiffPool --dataset_name insulator --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/DMPNN_DiffPool/insulator__bandgap_chain__batch_norm__rep0/logs/checkpoints/epoch=28-step=6003.ckpt"

echo "Evaluation complete!"
echo "Results saved to: results/DMPNN_DiffPool/insulator_baseline.csv"
