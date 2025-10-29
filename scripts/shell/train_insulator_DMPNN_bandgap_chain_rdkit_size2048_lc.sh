#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N DMPNN_insulator_bandgap_chain_rdkit_size2048_lc

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# DMPNN learning curve training for insulator/bandgap_chain, train_size: 2048
python3 scripts/python/train_graph.py \
    --dataset_name insulator \
    --model_name DMPNN \
    --target bandgap_chain \
    --train_size 2048 \
    --save_predictions \
    --export_embeddings --incl_rdkit


##TODO

# Add additional experiments here as needed

