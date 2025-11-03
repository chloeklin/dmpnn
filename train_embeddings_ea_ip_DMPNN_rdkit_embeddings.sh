#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N embeddings_DMPNN_ea_ip_DMPNN_rdkit_embeddings

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

echo "=== Starting Embeddings-Only Training ==="
echo "Dataset: ea_ip"
echo "Model: DMPNN"
echo "Task Type: reg"
echo "Target: 'All targets'"
echo "RDKit: --incl_rdkit"
echo "Descriptors: 'No'"
echo "Batch Norm: 'No'"
echo "========================================"

# Run training with embeddings export, then stop
if [[ "DMPNN" == "AttentiveFP" ]]; then
    python3 scripts/python/train_attentivefp.py --dataset_name ea_ip --model_name DMPNN --export_embeddings --incl_rdkit
else
    python3 scripts/python/train_graph.py --dataset_name ea_ip --model_name DMPNN --export_embeddings --incl_rdkit
fi

echo "=== Embeddings Extraction Complete ==="
echo "Embeddings saved to results/embeddings/"
echo "Job finished successfully!"

