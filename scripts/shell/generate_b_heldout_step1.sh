#!/bin/bash
set -euo pipefail

PROJECT="um09"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="gpuvolta"
NCPUS=12
NGPUS=1
MEM="100GB"
JOBFS="100GB"
WALLTIME="24:00:00"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$LOCAL_PROJECT/logs/b_heldout_step1"
MANIFEST="$OUT/manifest.tsv"
PBS_DIR="$OUT/pbs"
mkdir -p "$PBS_DIR"
: > "$MANIFEST"
rm -f "$PBS_DIR"/*.pbs

TARGETS=("EA vs SHE (eV)" "IP vs SHE (eV)")
TARGET_TOKENS=(EA_vs_SHE_eV IP_vs_SHE_eV)
MODELS=(hpg_hier wdmpnn hpg_hier_octamer hpg_hier_junction)

for fold in {0..8}; do
    for target_index in "${!TARGETS[@]}"; do
        target="${TARGETS[$target_index]}"
        target_token="${TARGET_TOKENS[$target_index]}"
        for model in "${MODELS[@]}"; do
            printf '%s\t%s\t%s\t%s\n' "$fold" "$target_token" "$model" "$target" >> "$MANIFEST"
        done
    done
done

TASK_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
[[ "$TASK_COUNT" -eq 72 ]] || { printf 'Expected 72 jobs, found %s\n' "$TASK_COUNT" >&2; exit 1; }

index=0
while IFS=$'\t' read -r fold target_token model target; do
    pbs="$PBS_DIR/step1_$(printf '%02d' "$index")_${model}_f${fold}_${target_token}.pbs"
    if [[ "$model" == "wdmpnn" ]]; then
        command="python scripts/python/run_wdmpnn_generalization.py --split_types monomer_b_heldout --folds $fold --targets '$target' --seed 42 --split_seed 42"
    else
        command="python scripts/python/run_hpg_generalization.py --split_types monomer_b_heldout --folds $fold --targets '$target' --models $model --stage1_pool sum --stage2_depth 2 --stage2_edge full --octamer_len 8 --n_random_samples 16 --n_coupling_steps 2 --seed 42 --split_seed 42"
    fi
    cat > "$pbs" <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N b1_${model}_f${fold}
#PBS -r y
set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR
$command
EOF
    chmod +x "$pbs"
    index=$((index + 1))
done < "$MANIFEST"

printf 'Jobs generated: %s\n' "$TASK_COUNT"
printf 'Requested walltime per job: %s GPU-hours\n' "24"
printf 'Conservative maximum GPU-hours: %s\n' "$((TASK_COUNT * 24))"
printf 'Manifest: %s\n' "$MANIFEST"
printf 'PBS directory: %s\n' "$PBS_DIR"
printf 'No jobs submitted.\n'
