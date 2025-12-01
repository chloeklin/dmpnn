#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=00:10:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N graphormer-train

module use /g/data/dk92/apps/Modules/modulefiles
module load module load NCI-ai-ml/24.08 python3/3.10.4 gcc/12.2.0
source /scratch/um09/hl4138/graphormer-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# Dataset paths
DATA_DIR="/scratch/um09/hl4138/dmpnn/data"

# ============================================================================
# OPV CAM-B3LYP Dataset
# ============================================================================
# echo "Training Graphormer on opv_camb3lyp dataset..."
# python3 Graphormer/main.py \
#     --csv_path "${DATA_DIR}/opv_camb3lyp.csv" \
#     --smiles_col "smiles" \
#     --task_type "reg"

# ============================================================================
# Insulator Dataset
# ============================================================================
echo "Training Graphormer on insulator dataset..."
python3 Graphormer/main.py \
    --csv_path "${DATA_DIR}/insulator.csv" \
    --smiles_col "smiles" \
    --target_cols "bandgap_chain" \
    --task_type "reg"

# ============================================================================
# HTPMD Dataset (with descriptors and specific target)
# ============================================================================
# echo "Training Graphormer on htpmd dataset..."
# python3 Graphormer/main.py \
#     --csv_path "${DATA_DIR}/htpmd.csv" \
#     --smiles_col "smiles" \
#     --target_cols "Conductivity,TFSI Diffusivity,Li Diffusivity,Poly Diffusivity,Transference Number" \
#     --task_type "reg"

# ============================================================================
# Notes:
# ============================================================================
# Column Selection Logic:
# -----------------------
# 1. --smiles_col: Specifies the SMILES column (required)
# 2. --descriptor_cols: Global molecular descriptors (optional, comma-separated)
#    Example: "Mn,Mw,PDI" for polymer datasets
# 3. --target_cols: Target columns to predict (optional, comma-separated)
#    - If NOT specified: ALL remaining columns (after SMILES & descriptors) become targets
#    - If specified: ONLY these columns are used as targets (ignores others)
#    Example: "bandgap_chain" (ignores WDMPNN_Input column)
#
# To ignore columns: Explicitly specify --target_cols with only the columns you want
#
# Default Training Parameters (from main.py):
# -------------------------------------------
# - batch_size: 16
# - num_epochs: 16
# - lr: 2e-4
# - num_workers: 4
# - seed: 1
# - replicates: 5
# - n_splits: 1 (holdout validation)
# - num_layers: 12 (Graphormer-Small)
# - hidden_dim: 768
# - num_heads: 32
# - dropout: 0.1
# - results_dir: results
#
# To override any parameter, add it to the command, e.g.:
#   --batch_size 32 --num_epochs 100 --num_workers 8
