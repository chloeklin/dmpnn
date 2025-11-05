#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-opv-DMPNN

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: opv
# Model: DMPNN
# Configuration: Auto-detected from checkpoint path
# Expected Result: results/DMPNN/opv_rdkit_baseline.csv

echo "Starting evaluation..."
echo "Model: DMPNN"
echo "Dataset: opv"
echo "Configuration: Auto-detected from checkpoint path"

python3 scripts/python/evaluate_model.py \
    --model_name DMPNN --dataset_name opv --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/DMPNN/opv_camb3lyp__delta_homo__rdkit__rep0/logs/checkpoints/epoch=90-step=81081.ckpt" --preprocessing_path "/scratch/um09/hl4138/dmpnn/preprocessing/opv_camb3lyp__delta_homo__rdkit__rep0"

echo "Evaluation complete!"
echo "Results saved to: results/DMPNN/opv_rdkit_baseline.csv"
