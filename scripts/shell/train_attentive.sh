#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=00:40:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N attentivefp

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.9.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# wDMPNN training
python3 scripts/python/train_attentivefp.py --dataset_name insulator --export_embeddings
# python3 scripts/python/train_attentivefp.py --dataset_name htpmd --export_embeddings
# python3 scripts/python/train_attentivefp.py --dataset_name opv_camb3lyp --export_embeddings

##TODO

# Add additional experiments here as needed

