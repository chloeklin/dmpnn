#!/bin/bash
#PBS -P um09
#PBS -q gpuvolta
#PBS -l walltime=4:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l jobfs=100GB
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -N eval_polyinfo_wDMPNN

# Change to project directory
cd /Users/u6788552/Desktop/experiments/dmpnn

# Activate conda environment
source ~/.bashrc
conda activate chemprop

# Run evaluation
python3 scripts/python/evaluate_model.py \
    --model_name wDMPNN --dataset_name polyinfo

echo "Evaluation complete!"
echo "Results saved to: results/wDMPNN/polyinfo_baseline.csv"
