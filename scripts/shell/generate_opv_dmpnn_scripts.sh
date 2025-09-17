#!/bin/bash

# Generate training scripts for opv_camb3lyp dataset with DMPNN model
# Creates individual PBS scripts for each target with and without RDKit descriptors

set -e

# OPV CAM-B3LYP targets
# TARGETS=(
#     "optical_lumo" "gap" "homo" "lumo" "spectral_overlap"
#     "delta_homo" "delta_lumo" "delta_optical_lumo"
#     "homo_extrapolated" "lumo_extrapolated" "gap_extrapolated"
#     "optical_lumo_extrapolated"
# )

TARGETS=(
    "optical_lumo" "gap" "homo" "lumo" "spectral_overlap"
    "delta_optical_lumo"
    "homo_extrapolated" "gap_extrapolated"
)

# Default settings
WALLTIME="04:00:00"
OUTPUT_DIR="./"
DRY_RUN=false
RDKIT_ONLY=false
NO_RDKIT_ONLY=false
SUBMIT_JOBS=false
BATCH_NORM=false

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
        --rdkit-only)
            RDKIT_ONLY=true
            shift
            ;;
        --no-rdkit-only)
            NO_RDKIT_ONLY=true
            shift
            ;;
        --submit)
            SUBMIT_JOBS=true
            shift
            ;;
        --batch-norm)
            BATCH_NORM=true
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --walltime TIME     PBS walltime (default: 04:00:00)"
            echo "  --output-dir DIR    Output directory (default: scripts/shell)"
            echo "  --dry-run           Show what would be generated"
            echo "  --rdkit-only        Only generate RDKit variants"
            echo "  --no-rdkit-only     Only generate non-RDKit variants"
            echo "  --targets TARGET... Specific targets to generate"
            echo "  --submit            Automatically submit generated jobs to PBS queue"
            echo "  --batch-norm        Enable batch normalization in the model"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate conflicting options
if [[ "$RDKIT_ONLY" == "true" && "$NO_RDKIT_ONLY" == "true" ]]; then
    echo "Error: Cannot specify both --rdkit-only and --no-rdkit-only"
    exit 1
fi

# Function to generate a PBS script
generate_script() {
    local target="$1"
    local use_rdkit="$2"
    
    local rdkit_suffix=""
    local rdkit_flag=""
    
    if [[ "$use_rdkit" == "true" ]]; then
        rdkit_suffix="_rdkit"
        rdkit_flag=" --incl_rdkit"
    fi
    
    local batch_norm_suffix=""
    local batch_norm_flag=""
    
    if [[ "$BATCH_NORM" == "true" ]]; then
        batch_norm_suffix="_batch_norm"
        batch_norm_flag=" --batch_norm"
    fi
    
    local filename="train_opv_camb3lyp_DMPNN_${target}${rdkit_suffix}${batch_norm_suffix}.sh"
    local filepath="${OUTPUT_DIR}/${filename}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        local variant_name="without RDKit"
        if [[ "$use_rdkit" == "true" ]]; then
            variant_name="with RDKit"
        fi
        echo "[DRY RUN] Would generate: ${filename} (${variant_name})"
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
#PBS -N DMPNN_opv_camb3lyp_${target}${rdkit_suffix}${batch_norm_suffix}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# DMPNN training for target: ${target}
python3 scripts/python/train_graph.py --dataset_name opv_camb3lyp --model_name DMPNN --target ${target}${rdkit_flag}${batch_norm_flag}


##TODO

# Add additional experiments here as needed

EOF
    
    # Make executable
    chmod +x "$filepath"
    
    local variant_name="without RDKit"
    if [[ "$use_rdkit" == "true" ]]; then
        variant_name="with RDKit"
    fi
    echo "✅ Generated: ${filename} (${variant_name})"
}

# Main generation
echo "🚀 Generating DMPNN training scripts for opv_camb3lyp"
echo "📁 Output directory: $OUTPUT_DIR"
echo "⏰ Walltime: $WALLTIME"
echo "🎯 Targets: ${#TARGETS[@]} targets"
echo "🧠 Batch Norm: $BATCH_NORM"
echo "============================================================"

generated_count=0

for target in "${TARGETS[@]}"; do
    # Determine which variants to generate
    variants=()
    
    if [[ "$RDKIT_ONLY" == "true" ]]; then
        variants=("true")
    elif [[ "$NO_RDKIT_ONLY" == "true" ]]; then
        variants=("false")
    else
        variants=("false" "true")  # Both variants
    fi
    
    for use_rdkit in "${variants[@]}"; do
        generate_script "$target" "$use_rdkit"
        ((++generated_count))
    done
done

echo "============================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 DRY RUN: Would generate $generated_count scripts"
else
    echo "✅ Successfully generated $generated_count training scripts"
    echo "📂 Scripts saved to: $OUTPUT_DIR"
    
    if [[ $generated_count -gt 0 ]]; then
        if [[ "$SUBMIT_JOBS" == "true" ]]; then
            echo ""
            echo "🚀 Submitting jobs to PBS queue..."
            cd "$OUTPUT_DIR"
            submitted_count=0
            for target in "${TARGETS[@]}"; do
                if [[ "$RDKIT_ONLY" != "true" ]]; then
                    script_name="train_opv_camb3lyp_DMPNN_${target}.sh"
                    if [[ -f "$script_name" ]]; then
                        job_id=$(qsub "$script_name")
                        echo "✅ Submitted: $script_name -> $job_id"
                        ((++submitted_count))
                    fi
                fi
                if [[ "$NO_RDKIT_ONLY" != "true" ]]; then
                    script_name="train_opv_camb3lyp_DMPNN_${target}_rdkit.sh"
                    if [[ -f "$script_name" ]]; then
                        job_id=$(qsub "$script_name")
                        echo "✅ Submitted: $script_name -> $job_id"
                        ((++submitted_count))
                    fi
                fi
            done
            echo "📊 Total jobs submitted: $submitted_count"
            echo "📈 Monitor with: qstat -u $USER"
        else
            echo ""
            echo "🚀 To submit all jobs, run:"
            echo "cd $OUTPUT_DIR"
            for target in "${TARGETS[@]}"; do
                if [[ "$RDKIT_ONLY" != "true" ]]; then
                    echo "qsub train_opv_camb3lyp_DMPNN_${target}.sh"
                fi
                if [[ "$NO_RDKIT_ONLY" != "true" ]]; then
                    echo "qsub train_opv_camb3lyp_DMPNN_${target}_rdkit.sh"
                fi
            done
        fi
    fi
fi
