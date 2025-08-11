#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=00:20:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N wdmpnn_insulator

cd /scratch/um09/hl4138
module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source venvs/dmpnn-venv/bin/activate

cd dmpnn
python3 scripts/train_large_wdmpnn.py --dataset_name insulator > output.txt
# python3 scripts/train_large_wdmpnn.py --dataset_name polyinfo --task_type multi --n_classes 21 > polyinfo_output.txt

