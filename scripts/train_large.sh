#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N dmpnn_large_insulator


cd /scratch/um09/hl4138
module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source venvs/dmpnn-venv/bin/activate

cd dmpnn
python3 scripts/train_large.py --dataset_name insulator
python3 scripts/train_large.py --dataset_name polyinfo --task_type multi --n_classes 21
python3 scripts/train_large.py --dataset_name htpmd --descriptor_columns Molality Monomer Monomer_Molecular_Weight DoP Density
python3 scripts/train_large.py --dataset_name opv_b3lyp
python3 scripts/train_large.py --dataset_name opv_camb3lyp 