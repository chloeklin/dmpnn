#!/bin/bash

# Generate training scripts for opv_camb3lyp dataset with learning curve analysis using AttentiveFP
# Creates individual PBS scripts for each train_size combination
# Features: --save_predictions on, --export_embeddings on

set -e

# OPV CAM-B3LYP targets
TARGETS=(
    "optical_lumo" "gap" "homo" "lumo" "spectral_overlap"
    "delta_optical_lumo"
    "homo_extrapolated" "gap_extrapolated"
)

# Train sizes for learning curve analysis
TRAIN_SIZES=("250" "500" "1000" "2000" "3500" "5000" "8000" "12000")

# Default settings
WALLTIME="10:00:00"
OUTPUT_DIR="./"
DRY_RUN=false
SUBMIT_JOBS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --walltime)
            WALLTIME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --submit)
            SUBMIT_JOBS=true
            shift
            ;;
        --targets)
            shift
            TARGETS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TARGETS+=("$1")
                shift
            done
            ;;
        --train-sizes)
            shift
            TRAIN_SIZES=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TRAIN_SIZES+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --walltime TIME       PBS walltime (default: 04:00:00)"
            echo "  --output-dir DIR      Output directory (default: ./)"
            echo "  --dry-run             Show what would be generated"
            echo "  --targets TARGET...   Specific targets to generate"
            echo "  --train-sizes SIZE... Specific train sizes to use"
            echo "  --submit              Automatically submit generated jobs to PBS queue"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to generate a PBS script
generate_script() {
    local target="$1"
    local train_size="$2"
    
    local size_suffix=""
    if [[ "$train_size" != "full" ]]; then
        size_suffix="_size${train_size}"
    fi
    
    local filename="train_opv_camb3lyp_AttentiveFP_${target}${size_suffix}_lc.sh"
    local filepath="${OUTPUT_DIR}/${filename}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would generate: ${filename} (size=${train_size})"
        return 0
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Generate the PBS script
    cat > "$filepath" << EOF
#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime=${WALLTIME}
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N AttentiveFP_opv_${target}${size_suffix}_lc

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/

# AttentiveFP learning curve training for OPV CAM-B3LYP dataset, train_size: ${train_size}
python3 scripts/python/train_attentivefp.py \\
    --dataset_name opv_camb3lyp \\
    --model_name AttentiveFP \\
    --task_type reg \\
    --train_size ${train_size} \\
    --polymer_type homo \\
    --save_predictions \\
    --export_embeddings

# Add additional experiments here as needed

EOF
    
    # Make executable
    chmod +x "$filepath"
    echo "✅ Generated: ${filename} (size=${train_size})"
}

# Main generation
echo "🚀 Generating AttentiveFP learning curve training scripts for opv_camb3lyp"
echo "📁 Output directory: $OUTPUT_DIR"
echo "⏰ Walltime: $WALLTIME"
echo "📝 Note: Will process all targets in the dataset"
echo "📊 Train sizes: ${TRAIN_SIZES[*]}"

# Generate scripts for each train size
for size in "${TRAIN_SIZES[@]}"; do
    generate_script "all" "$size"
done


echo "✨ Script generation complete!"

# If submit flag is set, submit all generated jobs
if [[ "$SUBMIT_JOBS" == "true" ]]; then
    echo "🚀 Submitting all generated jobs to PBS..."
    for script in "${OUTPUT_DIR}"/train_opv_camb3lyp_AttentiveFP_*_lc.sh; do
        if [[ -f "$script" ]]; then
            echo "Submitting $script..."
            qsub "$script"
        fi
    done
    echo "✅ All jobs submitted!"
fi
