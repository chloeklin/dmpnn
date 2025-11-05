#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=4:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-polyinfo-DMPNN_DiffPool

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: polyinfo
# Model: DMPNN_DiffPool
# Descriptors: false
# RDKit: false
# Batch Norm: false
# Train Size: full
# Expected Result: results/DMPNN_DiffPool/polyinfo_baseline.csv

echo "Starting evaluation..."
echo "Model: DMPNN_DiffPool"
echo "Dataset: polyinfo"
echo "Configuration: desc=false, rdkit=false, batch_norm=false"

python3 scripts/python/evaluate_model.py \
    --model_name DMPNN_DiffPool --dataset_name polyinfo --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/DMPNN_DiffPool/polyinfo__Class__rep0/logs/checkpoints/epoch=89-step=54810.ckpt"

echo "Evaluation complete!"
echo "Results saved to: results/DMPNN_DiffPool/polyinfo_baseline.csv"
