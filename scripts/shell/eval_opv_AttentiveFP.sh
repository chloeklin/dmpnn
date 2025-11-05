#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N eval-opv-AttentiveFP

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Evaluation Configuration
# Dataset: opv
# Model: AttentiveFP
# Descriptors: false
# RDKit: false
# Batch Norm: false
# Train Size: full
# Expected Result: results/AttentiveFP/opv_baseline.csv

echo "Starting evaluation..."
echo "Model: AttentiveFP"
echo "Dataset: opv"
echo "Configuration: desc=false, rdkit=false, batch_norm=false"

python3 scripts/python/evaluate_model.py \
    --model_name AttentiveFP --dataset_name opv --checkpoint_path "/scratch/um09/hl4138/dmpnn/checkpoints/AttentiveFP/opv_camb3lyp__delta_optical_lumo__rep0/best.pt"

echo "Evaluation complete!"
echo "Results saved to: results/AttentiveFP/opv_baseline.csv"
