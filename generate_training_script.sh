#!/bin/bash

# Script Generator for Training Jobs
# Creates PBS job scripts following the train_large.sh format for training.
# Takes dataset name, model type, and optional flags for RDKit descriptors and additional descriptors.
#
# Usage:
#   ./generate_training_script.sh <dataset> <model> [incl_rdkit] [descriptor] [task_type]
#
# Examples:
#   ./generate_training_script.sh insulator DMPNN
#   ./generate_training_script.sh insulator tabular incl_rdkit
#   ./generate_training_script.sh htpmd wDMPNN incl_rdkit descriptor
#   ./generate_training_script.sh polyinfo DMPNN incl_rdkit descriptor multi

# Check if dataset name and model are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <model> [incl_rdkit] [descriptor] [task_type]"
    echo ""
    echo "Available models: tabular, DMPNN, wDMPNN"
    echo ""
    echo "Examples:"
    echo "  $0 insulator DMPNN"
    echo "  $0 insulator tabular incl_rdkit"
    echo "  $0 htpmd wDMPNN incl_rdkit descriptor"
    echo "  $0 polyinfo DMPNN incl_rdkit descriptor multi"
    exit 1
fi

# Parse arguments
DATASET=$1
MODEL=$2
INCL_RDKIT=""
DESCRIPTOR=""
TASK_TYPE="reg"

# Validate model type
case $MODEL in
    tabular|DMPNN|wDMPNN)
        ;;
    *)
        echo "Error: Invalid model '$MODEL'. Available models: tabular, DMPNN, wDMPNN"
        exit 1
        ;;
esac

# Check for optional flags (skip first two arguments)
for arg in "${@:3}"; do
    case $arg in
        incl_rdkit)
            INCL_RDKIT="--incl_rdkit"
            ;;
        descriptor)
            DESCRIPTOR="--descriptor"
            ;;
        binary|multi)
            TASK_TYPE=$arg
            ;;
    esac
done

# Build suffix for filenames
SUFFIX="_${MODEL}"
if [ -n "$DESCRIPTOR" ]; then
    SUFFIX="${SUFFIX}_desc"
fi
if [ -n "$INCL_RDKIT" ]; then
    SUFFIX="${SUFFIX}_rdkit"
fi
if [ "$TASK_TYPE" != "reg" ]; then
    SUFFIX="${SUFFIX}_${TASK_TYPE}"
fi

# Build command arguments based on model type
if [ "$MODEL" = "tabular" ]; then
    ARGS="--dataset_name $DATASET"
    SCRIPT_NAME="train_tabular.py"
    OUTPUT_PREFIX="tabular"
else
    # For DMPNN and wDMPNN
    ARGS="--dataset_name $DATASET --model_name $MODEL"
    SCRIPT_NAME="train_graph.py"
    OUTPUT_PREFIX="$MODEL"
fi

if [ "$TASK_TYPE" != "reg" ]; then
    ARGS="$ARGS --task_type $TASK_TYPE"
fi
if [ -n "$DESCRIPTOR" ]; then
    ARGS="$ARGS $DESCRIPTOR"
fi
if [ -n "$INCL_RDKIT" ]; then
    ARGS="$ARGS $INCL_RDKIT"
fi

# Output script filename
OUTPUT_SCRIPT="scripts/shell/train_${DATASET}${SUFFIX}.sh"

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
#PBS -l walltime=2:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N ${OUTPUT_PREFIX}_${DATASET}${SUFFIX}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# ${MODEL} training
python3 scripts/python/$SCRIPT_NAME $ARGS > ${OUTPUT_PREFIX}_${DATASET}${SUFFIX}.txt


##TODO

# Add additional experiments here as needed

EOF

# Make the generated script executable
chmod +x "$OUTPUT_SCRIPT"

echo "Generated training script: $OUTPUT_SCRIPT"
echo "To submit the job, run: qsub $OUTPUT_SCRIPT"
echo ""
echo "Generated script content:"
echo "------------------------"
cat "$OUTPUT_SCRIPT"
