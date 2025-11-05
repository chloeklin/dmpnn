#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-tc-wDMPNN

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: tc
# Model: wDMPNN
# Configuration: Auto-detected from checkpoint path
# Expected Result: results/wDMPNN/tc_rdkit_baseline.csv

echo "Starting evaluation..."
echo "Model: wDMPNN"
echo "Dataset: tc"
echo "Configuration: Auto-detected from checkpoint path"

python3 scripts/python/evaluate_model.py \
    --model_name wDMPNN --dataset_name tc --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/wDMPNN/tc__TC__rdkit__rep0/logs/checkpoints/epoch=38-step=1872.ckpt" --preprocessing_path "/scratch/um09/hl4138/dmpnn/preprocessing/tc__TC__rdkit__rep0"

echo "Evaluation complete!"
echo "Results saved to: results/wDMPNN/tc_rdkit_baseline.csv"
