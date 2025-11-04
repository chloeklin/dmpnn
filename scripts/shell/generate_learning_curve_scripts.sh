#!/bin/bash

# Generate learning curve scripts from YAML configuration
# Reads dataset-specific settings from learning_curve_config.yaml

set -e

# Default settings
CONFIG_FILE="./learning_curve_config.yaml"
OUTPUT_DIR="./"
DRY_RUN=false
SUBMIT_JOBS=false
DATASETS=()
FILTER_MODELS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
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
        --datasets)
            shift
            DATASETS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --models)
            shift
            FILTER_MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FILTER_MODELS+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Generate learning curve training scripts from YAML configuration."
            echo ""
            echo "Options:"
            echo "  --config FILE         Path to YAML config file (default: scripts/shell/learning_curve_config.yaml)"
            echo "  --output-dir DIR      Output directory (default: ./)"
            echo "  --datasets DS...      Only generate for specific datasets (default: all)"
            echo "  --models MODEL...     Only generate for specific models (default: all)"
            echo "  --dry-run             Show what would be generated"
            echo "  --submit              Automatically submit generated jobs to PBS queue"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  # Generate all scripts from config"
            echo "  $0"
            echo ""
            echo "  # Generate only for OPV dataset"
            echo "  $0 --datasets opv_camb3lyp"
            echo ""
            echo "  # Generate only DMPNN scripts (not wDMPNN)"
            echo "  $0 --models DMPNN"
            echo ""
            echo "  # Dry run to preview"
            echo "  $0 --dry-run"
            echo ""
            echo "  # Generate and submit"
            echo "  $0 --datasets insulator --submit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if Python is available (for YAML parsing)
if ! command -v python3 &> /dev/null; then
    echo "Error: 'python3' command not found. Please install Python 3."
    exit 1
fi

# Define yaml parser function (replacement for yq)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_PARSER="$SCRIPT_DIR/yaml_parser.py"

if [[ ! -f "$YAML_PARSER" ]]; then
    echo "Error: yaml_parser.py not found at: $YAML_PARSER"
    exit 1
fi

# Function to parse YAML (replacement for yq eval)
yq() {
    if [[ "$1" == "eval" ]]; then
        shift
        python3 "$YAML_PARSER" "$2" "$1"
    else
        python3 "$YAML_PARSER" "$@"
    fi
}

# Function to generate a PBS script
generate_script() {
    local dataset="$1"
    local model="$2"
    local target="$3"
    local train_size="$4"
    local use_rdkit="$5"
    local use_batch_norm="$6"
    local use_desc="$7"
    local walltime="$8"
    local script_type="$9"
    
    local rdkit_suffix=""
    local rdkit_flag=""
    
    if [[ "$use_rdkit" == "true" ]]; then
        rdkit_suffix="_rdkit"
        rdkit_flag=" --incl_rdkit"
    fi
    
    local batch_norm_suffix=""
    local batch_norm_flag=""
    
    if [[ "$use_batch_norm" == "true" ]]; then
        batch_norm_suffix="_batch_norm"
        batch_norm_flag=" --batch_norm"
    fi
    
    local desc_suffix=""
    local desc_flag=""
    
    if [[ "$use_desc" == "true" ]]; then
        desc_suffix="_desc"
        desc_flag=" --incl_desc"
    fi
    
    local size_suffix=""
    if [[ "$train_size" != "full" ]]; then
        size_suffix="_size${train_size}"
    fi
    
    local script_name
    if [[ "$script_type" == "tabular" ]]; then
        script_name="train_${dataset}_tabular_${target}${desc_suffix}${rdkit_suffix}${batch_norm_suffix}${size_suffix}_lc.sh"
    else
        script_name="train_${dataset}_${model}_${target}${rdkit_suffix}${batch_norm_suffix}${size_suffix}_lc.sh"
    fi
    
    local filepath="${OUTPUT_DIR}/${script_name}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        local variant_desc="${model}"
        [[ "$use_rdkit" == "true" ]] && variant_desc="${variant_desc}+RDKit"
        [[ "$use_batch_norm" == "true" ]] && variant_desc="${variant_desc}+BatchNorm"
        echo "[DRY RUN] ${script_name} (${variant_desc}, size=${train_size})"
        return 0
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Read global settings from YAML
    local queue=$(yq eval '.global.queue' "$CONFIG_FILE")
    local project=$(yq eval '.global.project' "$CONFIG_FILE")
    local ncpus=$(yq eval '.global.ncpus' "$CONFIG_FILE")
    local ngpus=$(yq eval '.global.ngpus' "$CONFIG_FILE")
    local mem=$(yq eval '.global.mem' "$CONFIG_FILE")
    local storage=$(yq eval '.global.storage' "$CONFIG_FILE")
    local jobfs=$(yq eval '.global.jobfs' "$CONFIG_FILE")
    local module_path=$(yq eval '.global.module_path' "$CONFIG_FILE")
    local python_module=$(yq eval '.global.python_module' "$CONFIG_FILE")
    local cuda_module=$(yq eval '.global.cuda_module' "$CONFIG_FILE")
    local venv_path=$(yq eval '.global.venv_path' "$CONFIG_FILE")
    local work_dir=$(yq eval '.global.work_dir' "$CONFIG_FILE")
    
    # Determine training script and build command
    local train_script
    local train_command
    local job_name
    local comment
    
    if [[ "$script_type" == "tabular" ]]; then
        train_script="scripts/python/train_tabular.py"
        job_name="tabular_${dataset}_${target}${desc_suffix}${rdkit_suffix}${batch_norm_suffix}${size_suffix}_lc"
        comment="Tabular learning curve training for ${dataset}/${target}, train_size: ${train_size}"
        
        # Tabular training uses --dataset_name, --target, --train_size, and feature flags
        train_command="python3 ${train_script} \\
    --dataset_name ${dataset} \\
    --target ${target} \\
    --train_size ${train_size}${desc_flag}${rdkit_flag}${batch_norm_flag}"
    else
        # AttentiveFP uses its own training script
        if [[ "$model" == "AttentiveFP" ]]; then
            train_script="scripts/python/train_attentivefp.py"
        else
            train_script="scripts/python/train_graph.py"
        fi
        
        job_name="${model}_${dataset}_${target}${rdkit_suffix}${batch_norm_suffix}${size_suffix}_lc"
        comment="${model} learning curve training for ${dataset}/${target}, train_size: ${train_size}"
        
        # Graph training uses --model_name, --save_predictions, and --export_embeddings
        train_command="python3 ${train_script} \\
    --dataset_name ${dataset} \\
    --model_name ${model} \\
    --target ${target} \\
    --train_size ${train_size} \\
    --save_predictions \\
    --export_embeddings${rdkit_flag}${batch_norm_flag}"
    fi
    
    # Generate the PBS script
    cat > "$filepath" << EOF
#!/bin/bash

#PBS -q ${queue}
#PBS -P ${project}
#PBS -l ncpus=${ncpus}
#PBS -l ngpus=${ngpus}
#PBS -l mem=${mem}
#PBS -l walltime=${walltime}
#PBS -l storage=${storage}
#PBS -l jobfs=${jobfs}
#PBS -N ${job_name}

module use ${module_path}
module load ${python_module} ${cuda_module}
source ${venv_path}
cd ${work_dir}


# ${comment}
${train_command}


##TODO

# Add additional experiments here as needed

EOF
    
    # Make executable
    chmod +x "$filepath"
    
    local variant_desc="${model}"
    [[ "$use_rdkit" == "true" ]] && variant_desc="${variant_desc}+RDKit"
    [[ "$use_batch_norm" == "true" ]] && variant_desc="${variant_desc}+BatchNorm"
    echo "✅ ${script_name} (${variant_desc}, size=${train_size})"
}

# Main generation
echo "🚀 Generating learning curve training scripts from config"
echo "📁 Config file: $CONFIG_FILE"
echo "📂 Output directory: $OUTPUT_DIR"
echo "============================================================"

# Get list of datasets from config
all_datasets=$(yq eval '.datasets | keys | .[]' "$CONFIG_FILE")

# Filter datasets if specified
if [[ ${#DATASETS[@]} -gt 0 ]]; then
    datasets_to_process=("${DATASETS[@]}")
else
    # Read datasets line by line into array (compatible with older bash)
    datasets_to_process=()
    while IFS= read -r dataset; do
        datasets_to_process+=("$dataset")
    done <<< "$all_datasets"
fi

generated_count=0

for dataset in "${datasets_to_process[@]}"; do
    # Check if dataset exists in config
    dataset_check=$(yq eval ".datasets.${dataset}" "$CONFIG_FILE")
    if [[ "$dataset_check" == "null" ]]; then
        echo "⚠️  Warning: Dataset '${dataset}' not found in config, skipping..."
        continue
    fi
    
    echo ""
    echo "📊 Processing dataset: ${dataset}"
    
    # Read dataset-specific settings
    targets=()
    while IFS= read -r target; do
        targets+=("$target")
    done < <(yq eval ".datasets.${dataset}.targets[]" "$CONFIG_FILE")
    
    train_sizes=()
    while IFS= read -r size; do
        train_sizes+=("$size")
    done < <(yq eval ".datasets.${dataset}.train_sizes[]" "$CONFIG_FILE")
    
    models=()
    while IFS= read -r model; do
        models+=("$model")
    done < <(yq eval ".datasets.${dataset}.models[]" "$CONFIG_FILE")
    walltime=$(yq eval ".datasets.${dataset}.walltime" "$CONFIG_FILE")
    
    # Read variant settings
    gen_rdkit=$(yq eval '.global.variants.rdkit' "$CONFIG_FILE")
    gen_no_rdkit=$(yq eval '.global.variants.no_rdkit' "$CONFIG_FILE")
    gen_batch_norm=$(yq eval '.global.variants.batch_norm' "$CONFIG_FILE")
    
    # Read dataset-specific descriptor settings
    has_descriptors=$(yq eval ".datasets.${dataset}.has_descriptors" "$CONFIG_FILE")
    
    # Set default for has_descriptors if null or empty
    [[ "$has_descriptors" == "null" || -z "$has_descriptors" ]] && has_descriptors="false"
    
    # Only read descriptor variants if dataset has descriptors
    if [[ "$has_descriptors" == "true" ]]; then
        gen_desc=$(yq eval ".datasets.${dataset}.descriptor_variants.use_descriptors" "$CONFIG_FILE")
        gen_no_desc=$(yq eval ".datasets.${dataset}.descriptor_variants.no_descriptors" "$CONFIG_FILE")
        
        # Set defaults for descriptor variants if null or empty
        [[ "$gen_desc" == "null" || -z "$gen_desc" ]] && gen_desc="false"
        [[ "$gen_no_desc" == "null" || -z "$gen_no_desc" ]] && gen_no_desc="true"
    else
        gen_desc="false"
        gen_no_desc="true"
    fi
    
    echo "  Targets: ${#targets[@]} (${targets[*]})"
    echo "  Train sizes: ${#train_sizes[@]} (${train_sizes[*]})"
    echo "  Models: ${#models[@]} (${models[*]})"
    
    # Show descriptor settings for tabular models
    has_tabular=false
    for model in "${models[@]}"; do
        if [[ "$model" == "tabular" ]]; then
            has_tabular=true
            break
        fi
    done
    
    if [[ "$has_tabular" == "true" ]]; then
        echo "  Has descriptors: ${has_descriptors}"
        if [[ "$has_descriptors" == "true" ]]; then
            echo "  Descriptor variants: use=${gen_desc}, no_use=${gen_no_desc}"
        fi
    fi
    
    # Filter models if specified
    models_to_use=()
    if [[ ${#FILTER_MODELS[@]} -gt 0 ]]; then
        for model in "${models[@]}"; do
            for filter_model in "${FILTER_MODELS[@]}"; do
                if [[ "$model" == "$filter_model" ]]; then
                    models_to_use+=("$model")
                    break
                fi
            done
        done
    else
        models_to_use=("${models[@]}")
    fi
    
    if [[ ${#models_to_use[@]} -eq 0 ]]; then
        echo "  ⚠️  No models match filter, skipping..."
        continue
    fi
    
    # Function to determine script type based on model
    get_script_type() {
        local model="$1"
        case "$model" in
            "tabular")
                echo "tabular"
                ;;
            *)
                echo "graph"
                ;;
        esac
    }
    
    # Generate scripts for each combination
    for model in "${models_to_use[@]}"; do
        # Determine script type based on model
        script_type=$(get_script_type "$model")
        
        for target in "${targets[@]}"; do
            for train_size in "${train_sizes[@]}"; do
                # Determine which RDKit variants to generate
                rdkit_variants=()
                if [[ "$gen_no_rdkit" == "true" ]]; then
                    rdkit_variants+=("false")
                fi
                if [[ "$gen_rdkit" == "true" ]]; then
                    rdkit_variants+=("true")
                fi
                
                # Determine batch norm variants
                batch_norm_variants=("false")
                if [[ "$gen_batch_norm" == "true" ]]; then
                    batch_norm_variants+=("true")
                fi
                
                # Determine descriptor variants (only for tabular models that have descriptors)
                desc_variants=("false")
                if [[ "$script_type" == "tabular" && "$has_descriptors" == "true" ]]; then
                    desc_variants=()
                    if [[ "$gen_no_desc" == "true" ]]; then
                        desc_variants+=("false")
                    fi
                    if [[ "$gen_desc" == "true" ]]; then
                        desc_variants+=("true")
                    fi
                fi
                
                for use_rdkit in "${rdkit_variants[@]}"; do
                    for use_batch_norm in "${batch_norm_variants[@]}"; do
                        if [[ "$script_type" == "tabular" ]]; then
                            # For tabular models, iterate over descriptor variants
                            for use_desc in "${desc_variants[@]}"; do
                                generate_script "$dataset" "$model" "$target" "$train_size" "$use_rdkit" "$use_batch_norm" "$use_desc" "$walltime" "$script_type"
                                ((++generated_count))
                            done
                        else
                            # For graph models, use false for descriptors (not applicable)
                            generate_script "$dataset" "$model" "$target" "$train_size" "$use_rdkit" "$use_batch_norm" "false" "$walltime" "$script_type"
                            ((++generated_count))
                        fi
                    done
                done
            done
        done
    done
done

echo "============================================================"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 DRY RUN: Would generate $generated_count scripts"
else
    echo "✅ Successfully generated $generated_count training scripts"
    echo "📂 Scripts saved to: $OUTPUT_DIR"
    
    if [[ $generated_count -gt 0 && "$SUBMIT_JOBS" == "true" ]]; then
        echo ""
        echo "🚀 Submitting jobs to PBS queue..."
        cd "$OUTPUT_DIR"
        submitted_count=0
        for script in train_*_lc.sh; do
            if [[ -f "$script" ]]; then
                job_id=$(qsub "$script")
                echo "✅ Submitted: $script -> $job_id"
                ((++submitted_count))
            fi
        done
        echo "📊 Total jobs submitted: $submitted_count"
        echo "📈 Monitor with: qstat -u $USER"
    elif [[ $generated_count -gt 0 ]]; then
        echo ""
        echo "🚀 To submit all jobs, run:"
        echo "cd $OUTPUT_DIR && for script in train_*_lc.sh; do qsub \"\$script\"; done"
    fi
fi
