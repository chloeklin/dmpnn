#!/bin/bash
#PBS -N eval_htpmd_Conductivity_AttentiveFP
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M ${USER}@student.unsw.edu.au
#PBS -m abe

# Auto-generated evaluation script
# Model: AttentiveFP
# Dataset: htpmd
# Target: Conductivity
# Generated: Wed Nov  5 22:56:41 AEDT 2025

cd $PBS_O_WORKDIR

# Load environment
module load python3/3.12.1
module load intel-mkl/2023.2.0
source ~/dmpnn-venv/bin/activate

# Run evaluation
python3 /Users/u6788552/Desktop/experiments/dmpnn/scripts/python/evaluate_model.py \
    --model_name AttentiveFP \
    --dataset_name htpmd \
    --target Conductivity \
    --checkpoint_path "/Users/u6788552/Desktop/experiments/dmpnn/checkpoints/AttentiveFP/htpmd__Conductivity__rep0/best.pt" \
     \
    

echo "Evaluation completed for AttentiveFP on htpmd::Conductivity"
