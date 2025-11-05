#!/bin/bash

# Analyze Missing Preprocessing Script
# This script analyzes job output logs to identify missing preprocessing files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default log directory (PBS usually puts logs in the job submission directory)
LOG_DIR="$PROJECT_ROOT"
OUTPUT_FILE="missing_preprocessing_report.txt"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Analyze job output logs to identify missing preprocessing files"
            echo ""
            echo "Options:"
            echo "  --log-dir DIR     Directory containing job output logs (default: project root)"
            echo "  --output FILE     Output report file (default: missing_preprocessing_report.txt)"
            echo "  --verbose, -v     Show verbose output"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Analyze logs in project root"
            echo "  $0 --log-dir /path/to/logs           # Analyze logs in specific directory"
            echo "  $0 --verbose --output report.txt     # Verbose output to custom file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🔍 Analyzing Missing Preprocessing Files"
echo "========================================"
echo "Log directory: $LOG_DIR"
echo "Output file: $OUTPUT_FILE"
echo ""

# Initialize counters
total_logs=0
failed_logs=0
missing_configs=()

# Create/clear output file
> "$OUTPUT_FILE"

echo "MISSING PREPROCESSING ANALYSIS REPORT" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "Log directory: $LOG_DIR" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find all job output files (common PBS patterns)
log_files=()
for pattern in "*.o[0-9]*" "*.out" "eval-*.o[0-9]*" "eval_*.o[0-9]*"; do
    while IFS= read -r -d '' file; do
        log_files+=("$file")
    done < <(find "$LOG_DIR" -name "$pattern" -type f -print0 2>/dev/null)
done

if [ ${#log_files[@]} -eq 0 ]; then
    echo "❌ No job output log files found in $LOG_DIR"
    echo "   Looking for patterns: *.o[0-9]*, *.out, eval-*.o[0-9]*, eval_*.o[0-9]*"
    echo ""
    echo "Try specifying a different log directory with --log-dir"
    exit 1
fi

echo "📂 Found ${#log_files[@]} log files to analyze"
echo ""

# Analyze each log file
for log_file in "${log_files[@]}"; do
    total_logs=$((total_logs + 1))
    
    if [ "$VERBOSE" = true ]; then
        echo "🔍 Analyzing: $(basename "$log_file")"
    fi
    
    # Check if this log contains preprocessing errors
    if grep -q "Preprocessing files incomplete" "$log_file" && grep -q "Skipping split.*due to missing metadata" "$log_file"; then
        failed_logs=$((failed_logs + 1))
        
        echo "❌ FAILED: $(basename "$log_file")" >> "$OUTPUT_FILE"
        echo "   File: $log_file" >> "$OUTPUT_FILE"
        
        # Extract dataset, model, and configuration info
        dataset=$(grep -o "Dataset.*: [a-zA-Z0-9_]*" "$log_file" | head -1 | cut -d: -f2 | tr -d ' ')
        model=$(grep -o "Model.*: [a-zA-Z0-9_]*" "$log_file" | head -1 | cut -d: -f2 | tr -d ' ')
        
        # Extract configuration flags
        desc_enabled=$(grep -o "Descriptors.*: [A-Za-z]*" "$log_file" | head -1 | cut -d: -f2 | tr -d ' ')
        rdkit_enabled=$(grep -o "RDKit desc\..*: [A-Za-z]*" "$log_file" | head -1 | cut -d: -f2 | tr -d ' ')
        batch_norm_enabled=$(grep -o "Batch norm.*: [A-Za-z]*" "$log_file" | head -1 | cut -d: -f2 | tr -d ' ')
        
        echo "   Dataset: $dataset" >> "$OUTPUT_FILE"
        echo "   Model: $model" >> "$OUTPUT_FILE"
        echo "   Configuration: desc=$desc_enabled, rdkit=$rdkit_enabled, batch_norm=$batch_norm_enabled" >> "$OUTPUT_FILE"
        
        # Extract the missing preprocessing paths
        echo "   Missing preprocessing paths:" >> "$OUTPUT_FILE"
        grep "Full preprocessing path with suffixes:" "$log_file" | while read -r line; do
            path=$(echo "$line" | cut -d: -f4- | tr -d ' ')
            echo "     $path" >> "$OUTPUT_FILE"
        done
        
        # Extract the specific missing files
        echo "   Missing files:" >> "$OUTPUT_FILE"
        grep "No preprocessing metadata found at" "$log_file" | while read -r line; do
            missing_file=$(echo "$line" | grep -o "/[^[:space:]]*\.json")
            echo "     $missing_file" >> "$OUTPUT_FILE"
        done
        
        echo "" >> "$OUTPUT_FILE"
        
        # Build configuration key for summary
        config_key="${model}|${dataset}|${desc_enabled}|${rdkit_enabled}|${batch_norm_enabled}"
        missing_configs+=("$config_key")
        
        if [ "$VERBOSE" = true ]; then
            echo "   ❌ Missing preprocessing: $dataset ($model)"
        fi
    else
        if [ "$VERBOSE" = true ]; then
            echo "   ✅ OK"
        fi
    fi
done

# Generate summary
echo "" >> "$OUTPUT_FILE"
echo "SUMMARY" >> "$OUTPUT_FILE"
echo "=======" >> "$OUTPUT_FILE"
echo "Total log files analyzed: $total_logs" >> "$OUTPUT_FILE"
echo "Failed jobs (missing preprocessing): $failed_logs" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ $failed_logs -gt 0 ]; then
    echo "UNIQUE MISSING CONFIGURATIONS:" >> "$OUTPUT_FILE"
    printf '%s\n' "${missing_configs[@]}" | sort -u | while read -r config; do
        IFS='|' read -r model dataset desc rdkit batch_norm <<< "$config"
        echo "  - $model on $dataset (desc=$desc, rdkit=$rdkit, batch_norm=$batch_norm)" >> "$OUTPUT_FILE"
    done
    echo "" >> "$OUTPUT_FILE"
    
    echo "RECOMMENDED ACTIONS:" >> "$OUTPUT_FILE"
    echo "1. Regenerate preprocessing files by running training scripts for missing configurations" >> "$OUTPUT_FILE"
    echo "2. Or use --skip-batch-norm flag if you don't need batch normalization" >> "$OUTPUT_FILE"
    echo "3. Or remove --preprocessing_path argument to auto-generate correct paths" >> "$OUTPUT_FILE"
fi

# Display summary to console
echo "📊 ANALYSIS COMPLETE"
echo "===================="
echo "Total log files: $total_logs"
echo "Failed jobs: $failed_logs"

if [ $failed_logs -gt 0 ]; then
    echo ""
    echo "❌ Found $failed_logs jobs with missing preprocessing files"
    echo ""
    echo "📋 Unique missing configurations:"
    printf '%s\n' "${missing_configs[@]}" | sort -u | while read -r config; do
        IFS='|' read -r model dataset desc rdkit batch_norm <<< "$config"
        echo "  • $model on $dataset (desc=$desc, rdkit=$rdkit, batch_norm=$batch_norm)"
    done
    echo ""
    echo "💡 SOLUTIONS:"
    echo "1. 🔄 Regenerate evaluation scripts without batch norm:"
    echo "   ./generate_evaluation_scripts.sh --skip-batch-norm --force"
    echo ""
    echo "2. 🏃 Run training to generate missing preprocessing:"
    echo "   python train_graph.py --dataset_name DATASET --model_name MODEL [flags]"
    echo ""
    echo "3. 🚫 Remove --preprocessing_path from evaluation commands"
    echo ""
else
    echo "✅ All jobs completed successfully - no missing preprocessing files!"
fi

echo ""
echo "📄 Detailed report saved to: $OUTPUT_FILE"
