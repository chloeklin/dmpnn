#!/bin/bash

# Script Generator for Embeddings-Only Training Jobs
# Creates PBS job scripts that run train_graph.py or train_attentivefp.py until embedding extraction, then stop
# Takes dataset name, model type, and optional flags
#
# Usage:
#   ./generate_embeddings_script.sh <dataset> <model> <walltime> [incl_rdkit] [incl_desc] [incl_ab] [task_type] [--no-submit]
#
# Examples:
#   ./generate_embeddings_script.sh insulator DMPNN 2:00:00
#   ./generate_embeddings_script.sh htpmd wDMPNN 4:00:00 incl_rdkit incl_desc
#   ./generate_embeddings_script.sh polyinfo AttentiveFP 3:00:00 incl_rdkit multi

# Check if dataset name, model, and walltime are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <dataset> <model> <walltime> [incl_rdkit] [incl_desc] [incl_ab] [task_type]"
    echo ""
    echo "Available models: DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP, PPG"
    echo "Walltime format: HH:MM:SS (e.g., 2:00:00 for 2 hours)"
    echo "Optional flags: incl_rdkit, incl_desc, incl_ab, batch_norm, binary, multi, pretrain_monomer"
    echo ""
    echo "Examples:"
    echo "  $0 insulator DMPNN 2:00:00"
    echo "  $0 htpmd wDMPNN 4:00:00 incl_rdkit incl_desc"
    exit 1
fi

# Parse arguments
DATASET="$1"
MODEL="$2"
WALLTIME="$3"
INCL_RDKIT=""
INCL_DESC=""
INCL_AB=""
BATCH_NORM=""
PRETRAIN_MONOMER=""
TASK_TYPE="reg"
TARGET=""

# Validate model type (must be a graph model that supports embeddings)
case $MODEL in
    DMPNN|wDMPNN|DMPNN_DiffPool|AttentiveFP|PPG)
        ;;
    *)
        echo "Error: Invalid model '$MODEL'. Available models for embeddings: DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP, PPG"
        exit 1
        ;;
esac

# Parse optional arguments
SUBMIT_JOB=true

for arg in "${@:4}"; do
    case $arg in
        incl_rdkit)
            INCL_RDKIT="--incl_rdkit"
            ;;
        incl_desc)
            INCL_DESC="--incl_desc"
            ;;
        incl_ab)
            INCL_AB="--incl_ab"
            ;;
        batch_norm)
            BATCH_NORM="--batch_norm"
            ;;
        pretrain_monomer)
            PRETRAIN_MONOMER="--pretrain_monomer"
            ;;
        binary|multi)
            TASK_TYPE=$arg
            ;;
        target=*)
            TARGET="${arg#target=}"
            ;;
        --no-submit)
            SUBMIT_JOB=false
            ;;
        *)
            echo "Warning: Unknown argument '$arg' ignored"
            ;;
    esac
done

# Build suffix for filenames
SUFFIX="_${MODEL}"
if [ -n "$INCL_DESC" ]; then
    SUFFIX="${SUFFIX}_desc"
fi
if [ -n "$INCL_RDKIT" ]; then
    SUFFIX="${SUFFIX}_rdkit"
fi
if [ -n "$INCL_AB" ]; then
    SUFFIX="${SUFFIX}_ab"
fi
if [ -n "$BATCH_NORM" ]; then
    SUFFIX="${SUFFIX}_batch_norm"
fi
if [ -n "$PRETRAIN_MONOMER" ]; then
    SUFFIX="${SUFFIX}_pretrain"
fi
if [ "$TASK_TYPE" != "reg" ]; then
    SUFFIX="${SUFFIX}_${TASK_TYPE}"
fi
if [ -n "$TARGET" ]; then
    SUFFIX="${SUFFIX}_${TARGET}"
fi

# Add embeddings-only identifier
SUFFIX="${SUFFIX}_embeddings"

# Build command arguments
ARGS="--dataset_name $DATASET --model_name $MODEL --export_embeddings"

if [ "$TASK_TYPE" != "reg" ]; then
    ARGS="$ARGS --task_type $TASK_TYPE"
fi
if [ -n "$INCL_DESC" ]; then
    ARGS="$ARGS $INCL_DESC"
fi
if [ -n "$INCL_RDKIT" ]; then
    ARGS="$ARGS $INCL_RDKIT"
fi
if [ -n "$INCL_AB" ]; then
    ARGS="$ARGS $INCL_AB"
fi
if [ -n "$BATCH_NORM" ]; then
    ARGS="$ARGS $BATCH_NORM"
fi
if [ -n "$PRETRAIN_MONOMER" ]; then
    ARGS="$ARGS $PRETRAIN_MONOMER"
fi
if [ -n "$TARGET" ]; then
    ARGS="$ARGS --target $TARGET"
fi

# Output script filename
OUTPUT_SCRIPT="train_embeddings_${DATASET}${SUFFIX}.sh"

# Create output directory if it doesn't exist
mkdir -p scripts/shell

# Generate the PBS script
cat > "$OUTPUT_SCRIPT" << EOF
#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=$WALLTIME
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N embeddings_${MODEL}_${DATASET}${SUFFIX}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

echo "=== Starting Embeddings-Only Training ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Task Type: $TASK_TYPE"
echo "Target: ${TARGET:-'All targets'}"
echo "RDKit: ${INCL_RDKIT:-'No'}"
echo "Descriptors: ${INCL_DESC:-'No'}"
echo "Batch Norm: ${BATCH_NORM:-'No'}"
echo "========================================"

# Run training with embeddings export, then stop
if [[ "$MODEL" == "AttentiveFP" ]]; then
    python3 scripts/python/train_attentivefp.py $ARGS
else
    python3 scripts/python/train_graph.py $ARGS
fi

echo "=== Embeddings Extraction Complete ==="
echo "Embeddings saved to results/embeddings/"
echo "Job finished successfully!"

EOF

# Make the generated script executable
chmod +x "$OUTPUT_SCRIPT"

echo "Generated embeddings-only training script: $OUTPUT_SCRIPT"

# Submit job automatically unless --no-submit flag is used
if [ "$SUBMIT_JOB" = true ]; then
    echo "Submitting job to PBS queue..."
    JOB_ID=$(qsub "$OUTPUT_SCRIPT")
    if [ $? -eq 0 ]; then
        echo "Job submitted successfully: $JOB_ID"
        echo "Monitor with: qstat -u $USER"
    else
        echo "Error: Failed to submit job"
        exit 1
    fi
else
    echo "Script created but not submitted (--no-submit flag used)"
    echo "To submit manually, run: qsub $OUTPUT_SCRIPT"
fi

echo ""
echo "Generated script content:"
echo "------------------------"
cat "$OUTPUT_SCRIPT"
