#!/bin/bash

# Script Generator for Training Jobs
# Creates PBS job scripts following the train_large.sh format for training.
# Takes dataset name, model type, and optional flags for RDKit descriptors and additional descriptors.
#
# Usage:
#   ./generate_training_script.sh <dataset> <model> <walltime> [incl_rdkit] [incl_desc] [incl_ab] [task_type] [--no-submit]
#
# Examples:
#   ./generate_training_script.sh insulator DMPNN 2:00:00
#   ./generate_training_script.sh insulator tabular 1:30:00 incl_rdkit incl_ab
#   ./generate_training_script.sh htpmd wDMPNN 4:00:00 incl_rdkit incl_desc
#   ./generate_training_script.sh polyinfo DMPNN 3:00:00 incl_rdkit incl_desc multi
#   ./generate_training_script.sh insulator tabular 2:00:00 incl_ab incl_desc incl_rdkit
#   ./generate_training_script.sh insulator DMPNN 2:00:00 --no-submit  # Create script only, don't submit

# Check if dataset name, model, and walltime are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <dataset> <model> <walltime> [incl_rdkit] [incl_desc] [incl_ab] [task_type]"
    echo ""
    echo "Available models: tabular, DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP"
    echo "Walltime format: HH:MM:SS (e.g., 2:00:00 for 2 hours)"
    echo "Optional flags: incl_rdkit, incl_desc, incl_ab, batch_norm, binary, multi, pretrain_monomer, save_checkpoint, save_predictions, export_embeddings"
    echo ""
    echo "Examples:"
    echo "  $0 insulator DMPNN 2:00:00"
    echo "  $0 insulator tabular 1:30:00 incl_rdkit incl_ab"
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
SAVE_CHECKPOINT=""
SAVE_PREDICTIONS=""
EXPORT_EMBEDDINGS=""
TASK_TYPE="reg"

# Validate model type
case $MODEL in
    tabular|DMPNN|wDMPNN|DMPNN_DiffPool|AttentiveFP)
        ;;
    *)
        echo "Error: Invalid model '$MODEL'. Available models: tabular, DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP"
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
        save_checkpoint)
            SAVE_CHECKPOINT="--save_checkpoint"
            ;;
        save_predictions)
            SAVE_PREDICTIONS="--save_predictions"
            ;;
        export_embeddings)
            EXPORT_EMBEDDINGS="--export_embeddings"
            ;;
        binary|multi)
            TASK_TYPE=$arg
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

# Build command arguments based on model type
if [ "$MODEL" = "tabular" ]; then
    ARGS="--dataset_name $DATASET"
    SCRIPT_NAME="train_tabular.py"
    OUTPUT_PREFIX="tabular"
elif [ "$MODEL" = "AttentiveFP" ]; then
    ARGS="--dataset_name $DATASET"
    SCRIPT_NAME="train_attentivefp.py"
    OUTPUT_PREFIX="AttentiveFP"
else
    # For DMPNN, wDMPNN, DMPNN_DiffPool
    ARGS="--dataset_name $DATASET --model_name $MODEL"
    SCRIPT_NAME="train_graph.py"
    OUTPUT_PREFIX="$MODEL"
fi

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
if [ -n "$SAVE_CHECKPOINT" ]; then
    ARGS="$ARGS $SAVE_CHECKPOINT"
fi
if [ -n "$SAVE_PREDICTIONS" ]; then
    ARGS="$ARGS $SAVE_PREDICTIONS"
fi
if [ -n "$EXPORT_EMBEDDINGS" ]; then
    ARGS="$ARGS $EXPORT_EMBEDDINGS"
fi

# Output script filename
OUTPUT_SCRIPT="train_${DATASET}${SUFFIX}.sh"

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
#PBS -N ${OUTPUT_PREFIX}_${DATASET}${SUFFIX}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# ${MODEL} training
python3 scripts/python/$SCRIPT_NAME $ARGS 


##TODO

# Add additional experiments here as needed

EOF

# Make the generated script executable
chmod +x "$OUTPUT_SCRIPT"

echo "Generated training script: $OUTPUT_SCRIPT"

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
