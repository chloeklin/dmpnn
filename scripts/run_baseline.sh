#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=01:30:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N baseline_htpmd_desc_rdkit

cd /scratch/um09/hl4138
module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source venvs/dmpnn-venv/bin/activate

cd dmpnn

##Completed
python3 scripts/evaluate_model.py --dataset_name htpmd --descriptor --incl_rdkit
# python3 scripts/evaluate_model.py --dataset_name htpmd --descriptor
# python3 scripts/evaluate_model.py --dataset_name htpmd 

# python3 scripts/evaluate_model.py --dataset_name insulator --model_name DMPNN

 