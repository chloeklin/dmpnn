#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N AttentiveFP_opv_all_size250_lc

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# AttentiveFP learning curve training for OPV CAM-B3LYP dataset, train_size: 250
python3 scripts/python/train_attentivefp.py \
    --dataset_name opv_camb3lyp \
    --model_name AttentiveFP \
    --task_type reg \
    --train_size 250 \
    --polymer_type homo \
    --save_predictions \
    --export_embeddings

# Add additional experiments here as needed

