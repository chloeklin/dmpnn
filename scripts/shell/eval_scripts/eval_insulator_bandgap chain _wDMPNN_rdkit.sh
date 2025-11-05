#!/bin/bash
#PBS -N eval_insulator_bandgap chain _wDMPNN
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M ${USER}@student.unsw.edu.au
#PBS -m abe

# Auto-generated evaluation script
# Model: wDMPNN
# Dataset: insulator
# Target: bandgap chain 
# Generated: Wed Nov  5 22:56:47 AEDT 2025

cd $PBS_O_WORKDIR

# Load environment
module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

# Run evaluation
python3 /Users/u6788552/Desktop/experiments/dmpnn/scripts/python/evaluate_model.py \
    --model_name wDMPNN \
    --dataset_name insulator \
    --target bandgap chain  \
    --checkpoint_path "/Users/u6788552/Desktop/experiments/dmpnn/checkpoints/wDMPNN/insulator__bandgap_chain__rdkit__rep0/logs/checkpoints/epoch=58-step=3068.ckpt" \
     \
    --incl_rdkit

echo "Evaluation completed for wDMPNN on insulator::bandgap chain "
