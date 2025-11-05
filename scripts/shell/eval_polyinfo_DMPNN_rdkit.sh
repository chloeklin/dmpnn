#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-polyinfo-DMPNN

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: polyinfo
# Model: DMPNN
# Configuration: Auto-detected from checkpoint path
# Expected Result: results/DMPNN/polyinfo_rdkit_baseline.csv

echo "Starting evaluation..."
echo "Model: DMPNN"
echo "Dataset: polyinfo"
echo "Configuration: Auto-detected from checkpoint path"

python3 scripts/python/evaluate_model.py \
    --model_name DMPNN --dataset_name polyinfo --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/DMPNN/polyinfo__Class__rdkit__batch_norm__rep0/logs/checkpoints/epoch=41-step=25578.ckpt" --preprocessing_path "/scratch/um09/hl4138/dmpnn/preprocessing/polyinfo__Class__rdkit__batch_norm__rep0"

echo "Evaluation complete!"
echo "Results saved to: results/DMPNN/polyinfo_rdkit_baseline.csv"
