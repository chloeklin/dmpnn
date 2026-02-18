#!/bin/bash

# Script Generator for Training Jobs
# Creates PBS job scripts following the train_large.sh format for training.
# Takes dataset name, model type, and optional flags for RDKit descriptors and additional descriptors.
#
# Usage:
#   ./generate_training_script.sh <dataset> <model> <walltime> [incl_rdkit] [incl_desc] [incl_ab] [task_type] [--no-submit] [key=value...]
#
# Examples:
#   ./generate_training_script.sh insulator DMPNN 2:00:00
#   ./generate_training_script.sh insulator tabular 1:30:00 incl_rdkit incl_ab
#   ./generate_training_script.sh htpmd wDMPNN 4:00:00 incl_rdkit incl_desc
#   ./generate_training_script.sh polyinfo DMPNN 3:00:00 incl_rdkit incl_desc multi
#   ./generate_training_script.sh opv_camb3lyp AttentiveFP 12:30:00 export_embeddings target=gap
#   ./generate_training_script.sh opv_camb3lyp AttentiveFP 12:30:00 export_embeddings target=gap --no-submit
#
# Recognized key=value extras:
#   target=<name>          # single prediction target/column
#   train_size=<float>     # e.g., 0.8
#   polymer_type=<string>  # e.g., copolymer

# Check args
if [ $# -lt 3 ]; then
    echo "Usage: $0 <dataset> <model> <walltime> [incl_rdkit] [incl_desc] [incl_ab] [task_type] [--no-submit] [key=value...]"
    echo ""
    echo "Available models: tabular, DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP, PPG, GAT, GATv2, GIN, GIN0, GINE"
    echo "Optional flags: incl_rdkit, incl_desc, incl_ab, batch_norm, binary, multi, pretrain_monomer, save_checkpoint, save_predictions, export_embeddings"
    echo "Extra key=value: target=..., train_size=..., polymer_type=..."
    exit 1
fi

# Parse required
DATASET="$1"
MODEL="$2"
WALLTIME="$3"

# Option flags
INCL_RDKIT=""
INCL_DESC=""
INCL_AB=""
BATCH_NORM=""
PRETRAIN_MONOMER=""
SAVE_CHECKPOINT=""
SAVE_PREDICTIONS=""
EXPORT_EMBEDDINGS=""
TASK_TYPE="reg"
SUBMIT_JOB=true

# Extra key=value
TARGET=""
TRAIN_SIZE=""
POLYMER_TYPE=""

# Validate model
case $MODEL in
  tabular|DMPNN|wDMPNN|DMPNN_DiffPool|AttentiveFP|PPG|GAT|GATv2|GIN|GIN0|GINE) ;;
  *)
    echo "Error: Invalid model '$MODEL'. Available: tabular, DMPNN, wDMPNN, DMPNN_DiffPool, AttentiveFP, PPG, GAT, GATv2, GIN, GIN0, GINE"
    exit 1
    ;;
esac

# Parse optional args
for arg in "${@:4}"; do
  case $arg in
    incl_rdkit)         INCL_RDKIT="--incl_rdkit" ;;
    incl_desc)          INCL_DESC="--incl_desc" ;;
    incl_ab)            INCL_AB="--incl_ab" ;;
    batch_norm)         BATCH_NORM="--batch_norm" ;;
    pretrain_monomer)   PRETRAIN_MONOMER="--pretrain_monomer" ;;
    save_checkpoint)    SAVE_CHECKPOINT="--save_checkpoint" ;;
    save_predictions)   SAVE_PREDICTIONS="--save_predictions" ;;
    export_embeddings)  EXPORT_EMBEDDINGS="--export_embeddings" ;;
    binary|multi)       TASK_TYPE=$arg ;;
    --no-submit)        SUBMIT_JOB=false ;;
    target=*)           TARGET="${arg#target=}" ;;
    train_size=*)       TRAIN_SIZE="${arg#train_size=}" ;;
    polymer_type=*)     POLYMER_TYPE="${arg#polymer_type=}" ;;
    *)
      echo "Warning: Unknown argument '$arg' ignored"
      ;;
  esac
done

# Base ARGS and script mapping
if [ "$MODEL" = "tabular" ]; then
  ARGS="--dataset_name $DATASET"
  SCRIPT_NAME="train_tabular.py"
  OUTPUT_PREFIX="tabular"
elif [ "$MODEL" = "AttentiveFP" ]; then
  ARGS="--dataset_name $DATASET"
  SCRIPT_NAME="train_attentivefp.py"
  OUTPUT_PREFIX="AttentiveFP"
else
  # DMPNN, wDMPNN, DMPNN_DiffPool, PPG
  ARGS="--dataset_name $DATASET --model_name $MODEL"
  SCRIPT_NAME="train_graph.py"
  OUTPUT_PREFIX="$MODEL"
fi

# Task type
if [ "$TASK_TYPE" != "reg" ]; then
  ARGS="$ARGS --task_type $TASK_TYPE"
fi

# Append flags to ARGS
[ -n "$INCL_DESC" ]          && ARGS="$ARGS $INCL_DESC"
[ -n "$INCL_RDKIT" ]         && ARGS="$ARGS $INCL_RDKIT"
[ -n "$INCL_AB" ]            && ARGS="$ARGS $INCL_AB"
[ -n "$BATCH_NORM" ]         && ARGS="$ARGS $BATCH_NORM"
[ -n "$PRETRAIN_MONOMER" ]   && ARGS="$ARGS $PRETRAIN_MONOMER"
[ -n "$SAVE_CHECKPOINT" ]    && ARGS="$ARGS $SAVE_CHECKPOINT"
[ -n "$SAVE_PREDICTIONS" ]   && ARGS="$ARGS $SAVE_PREDICTIONS"
[ -n "$EXPORT_EMBEDDINGS" ]  && ARGS="$ARGS $EXPORT_EMBEDDINGS"
[ -n "$POLYMER_TYPE" ]       && ARGS="$ARGS --polymer_type $POLYMER_TYPE"
[ -n "$TRAIN_SIZE" ]         && ARGS="$ARGS --train_size $TRAIN_SIZE"
[ -n "$TARGET" ]             && ARGS="$ARGS --target $TARGET"

# Filename/jobname suffix
SUFFIX="_${MODEL}"
[ -n "$INCL_DESC" ]        && SUFFIX="${SUFFIX}_desc"
[ -n "$INCL_RDKIT" ]       && SUFFIX="${SUFFIX}_rdkit"
[ -n "$INCL_AB" ]          && SUFFIX="${SUFFIX}_ab"
[ -n "$BATCH_NORM" ]       && SUFFIX="${SUFFIX}_batch_norm"
[ -n "$PRETRAIN_MONOMER" ] && SUFFIX="${SUFFIX}_pretrain"
[ "$TASK_TYPE" != "reg" ]  && SUFFIX="${SUFFIX}_${TASK_TYPE}"
[ -n "$POLYMER_TYPE" ]     && SUFFIX="${SUFFIX}_$POLYMER_TYPE"
[ -n "$TRAIN_SIZE" ]       && SUFFIX="${SUFFIX}_ts${TRAIN_SIZE}"
# Replace spaces with underscores in target name for job name compatibility
[ -n "$TARGET" ]           && SUFFIX="${SUFFIX}_${TARGET// /_}"

OUTPUT_SCRIPT="train_${DATASET}${SUFFIX}.sh"

# Ensure output dir
mkdir -p scripts/shell

# Emit PBS script
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
EOF

chmod +x "$OUTPUT_SCRIPT"

echo "Generated training script: $OUTPUT_SCRIPT"

# Submit
if [ "$SUBMIT_JOB" = true ]; then
  echo "Submitting job to PBS queue..."
  JOB_ID=$(qsub "$OUTPUT_SCRIPT")
  if [ $? -eq 0 ]; then
    echo "Job submitted successfully: $JOB_ID"
    echo "Monitor with: qstat -u \$USER"
  else
    echo "Error: Failed to submit job"
    exit 1
  fi
else
  echo "Script created but not submitted (--no-submit used)"
  echo "To submit manually, run: qsub $OUTPUT_SCRIPT"
fi

echo ""
echo "Generated script content:"
echo "------------------------"
cat "$OUTPUT_SCRIPT"
