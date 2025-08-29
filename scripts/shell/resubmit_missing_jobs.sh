#!/bin/bash

# Script to resubmit training jobs that don't have corresponding result CSV files
# Usage: ./resubmit_missing_jobs.sh

SCRIPT_DIR="scripts/shell"
RESULTS_DIR="results"

echo "Checking for missing result CSV files and resubmitting jobs..."
echo "=================================================="

# Function to get expected CSV name from script
get_expected_csv() {
    local script_name="$1"
    local base_name=$(basename "$script_name" .sh)
    
    # Parse script to determine dataset, model, and flags
    local dataset=$(echo "$base_name" | sed 's/train_\([^_]*\)_.*/\1/')
    
    if [[ "$base_name" == *"DMPNN"* ]]; then
        # Graph training - aggregate results format (no target in filename)
        local suffix=""
        if [[ "$base_name" == *"desc"* ]] && [[ "$base_name" == *"rdkit"* ]]; then
            suffix="_descriptors_rdkit"
        elif [[ "$base_name" == *"rdkit"* ]]; then
            suffix="_rdkit"
        elif [[ "$base_name" == *"desc"* ]]; then
            suffix="_descriptors"
        fi
        echo "${dataset}${suffix}_DMPNN_results.csv"
    elif [[ "$base_name" == *"wDMPNN"* ]]; then
        # wDMPNN training - aggregate results format
        local suffix=""
        if [[ "$base_name" == *"desc"* ]] && [[ "$base_name" == *"rdkit"* ]]; then
            suffix="_descriptors_rdkit"
        elif [[ "$base_name" == *"rdkit"* ]]; then
            suffix="_rdkit"
        elif [[ "$base_name" == *"desc"* ]]; then
            suffix="_descriptors"
        fi
        echo "${dataset}${suffix}_wDMPNN_results.csv"
    else
        # Tabular training
        local suffix=""
        if [[ "$base_name" == *"desc"* ]] && [[ "$base_name" == *"rdkit"* ]]; then
            suffix="_descriptors_rdkit"
        elif [[ "$base_name" == *"rdkit"* ]]; then
            suffix="_rdkit"
        elif [[ "$base_name" == *"desc"* ]]; then
            suffix="_descriptors"
        elif [[ "$base_name" == *"_ab"* ]]; then
            suffix="${suffix}_ab"
        fi
        
        echo "${dataset}_tabular${suffix}.csv"
    fi
}

# Function to submit job
submit_job() {
    local script_path="$1"
    local expected_csv="$2"
    
    echo "Missing: $expected_csv"
    echo "Submitting: $script_path"
    
    # Check if we're on a PBS system (like NCI Gadi)
    if command -v qsub &> /dev/null; then
        qsub "$script_path"
        echo "Job submitted via qsub"
    else
        # Fallback to direct execution
        echo "PBS not available, running directly..."
        bash "$script_path"
    fi
    echo ""
}

# Main loop through all training scripts
for script in ${SCRIPT_DIR}/train*.sh; do
    if [[ -f "$script" ]]; then
        script_name=$(basename "$script")
        expected_csv=$(get_expected_csv "$script_name")
        csv_path="${RESULTS_DIR}/${expected_csv}"
        
        echo "Checking: $script_name -> $expected_csv"
        
        if [[ ! -f "$csv_path" ]]; then
            submit_job "$script" "$expected_csv"
        else
            echo "Found: $expected_csv (skipping)"
            echo ""
        fi
    fi
done

echo "=================================================="
echo "Resubmission check complete!"
