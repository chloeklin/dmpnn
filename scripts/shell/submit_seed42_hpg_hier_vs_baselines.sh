#!/bin/bash
set -euo pipefail

PROJECT="ng76"
STORAGE="scratch/um09+gdata/dk92"
QUEUE="gpuvolta"
NCPUS=12
NGPUS=1
MEM="100GB"
JOBFS="100GB"
MODULE_PYTHON="python3/3.12.1"
MODULE_CUDA="cuda/12.0.0"
VENV_ACTIVATE="/home/659/hl4138/dmpnn-venv/bin/activate"
PROJECT_DIR="/scratch/um09/hl4138/dmpnn"
WALLTIME="24:00:00"

DRY_RUN=false
NO_SUBMIT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run) DRY_RUN=true; shift ;;
        --no-submit) NO_SUBMIT=true; shift ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$LOCAL_PROJECT/logs/seed42_hpg_hier_vs_baselines"
MANIFEST_DIR="$LOG_DIR/manifests"
TASK_LOG_DIR="$LOG_DIR/tasks"
mkdir -p "$MANIFEST_DIR" "$TASK_LOG_DIR"
MANIFEST="$MANIFEST_DIR/seed42_hpg_hier_vs_baselines.manifest"
PBS_SCRIPT_DIR="$LOG_DIR/pbs"

: > "$MANIFEST"
SPLITS=(group_disjoint pair_disjoint monomer_heldout)
TARGETS=("EA vs SHE (eV)" "IP vs SHE (eV)")
TARGET_TOKENS=(EA_vs_SHE_eV IP_vs_SHE_eV)
for split in "${SPLITS[@]}"; do
    if [[ "$split" == "monomer_heldout" ]]; then
        folds=(0 1 2 3 4 5 6 7 8)
    else
        folds=(0 1 2 3 4)
    fi
    for fold in "${folds[@]}"; do
        for target_index in "${!TARGETS[@]}"; do
            target="${TARGETS[$target_index]}"
            target_token="${TARGET_TOKENS[$target_index]}"
            printf 'hpg\t%s\t%s\t%s\t%s\t--split_types %s --folds %s --targets %q --models hpg_hier --stage1_pool sum --stage2_depth 2 --stage2_edge full --seed 42\n' "$target_token" hpg_hier "$split" "$fold" "$split" "$fold" "$target" >> "$MANIFEST"
            printf 'wdmpnn\t%s\t%s\t%s\t%s\t--split_types %s --folds %s --targets %q --seed 42\n' "$target_token" wdmpnn "$split" "$fold" "$split" "$fold" "$target" >> "$MANIFEST"
            printf 'stage2d\t%s\t%s\t%s\t%s\t--split_types %s --folds %s --targets %q --models 2d1_arch --seed 42\n' "$target_token" chemarch "$split" "$fold" "$split" "$fold" "$target" >> "$MANIFEST"
        done
    done
done

TASK_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
if [[ "$TASK_COUNT" -ne 114 ]]; then
    printf 'Expected 114 tasks, found %s\n' "$TASK_COUNT" >&2
    exit 1
fi

mkdir -p "$PBS_SCRIPT_DIR"
rm -f "$PBS_SCRIPT_DIR"/seed42_hvb_*.pbs
PBS_SCRIPTS=()
task_index=0
while IFS=$'\t' read -r runner target_token model_token split fold task_args; do
    case "$split" in
        group_disjoint) prediction_subdir="ea_ip_group" ;;
        pair_disjoint) prediction_subdir="ea_ip_pair" ;;
        monomer_heldout) prediction_subdir="ea_ip_lomo" ;;
        *) printf 'Unknown split: %s\n' "$split" >&2; exit 2 ;;
    esac
    pbs_script="$PBS_SCRIPT_DIR/seed42_hvb_$(printf '%03d' "$task_index")_${model_token}_${split}_f${fold}_${target_token}.pbs"
    cat > "$pbs_script" <<EOF
#!/bin/bash
#PBS -q $QUEUE
#PBS -P $PROJECT
#PBS -l ncpus=$NCPUS
#PBS -l ngpus=$NGPUS
#PBS -l mem=$MEM
#PBS -l walltime=$WALLTIME
#PBS -l storage=$STORAGE
#PBS -l jobfs=$JOBFS
#PBS -N s42_${model_token}_f${fold}
#PBS -r y

set -euo pipefail
module load $MODULE_PYTHON $MODULE_CUDA
source $VENV_ACTIVATE
cd $PROJECT_DIR

TASK_LOG_DIR="$PROJECT_DIR/logs/seed42_hpg_hier_vs_baselines/tasks"
mkdir -p "\$TASK_LOG_DIR"
exec > >(tee -a "\$TASK_LOG_DIR/task_${task_index}_\${PBS_JOBID}.log") 2>&1

PREDICTION_PATH="$PROJECT_DIR/predictions/$prediction_subdir/ea_ip__${target_token}__${model_token}__${split}__fold${fold}__s42.npz"
if [[ -f "\$PREDICTION_PATH" ]]; then
    printf 'Skipping existing prediction: %s\\n' "\$PREDICTION_PATH"
    exit 0
fi

set -- $task_args
printf 'Task ${task_index}: runner=${runner} model=${model_token} split=${split} fold=${fold} target=${target_token}\\n'
case "$runner" in
    hpg) python scripts/python/run_hpg_generalization.py "\$@" ;;
    wdmpnn) python scripts/python/run_wdmpnn_generalization.py "\$@" ;;
    stage2d) python scripts/python/run_stage2d_generalization.py "\$@" ;;
    *) printf 'Unknown runner: %s\\n' "$runner" >&2; exit 2 ;;
esac
EOF
    chmod +x "$pbs_script"
    PBS_SCRIPTS+=("$pbs_script")
    task_index=$((task_index + 1))
done < "$MANIFEST"

printf 'Manifest: %s\nTasks: %s\nPBS jobs: %s\nTask logs: %s\n' "$MANIFEST" "$TASK_COUNT" "${#PBS_SCRIPTS[@]}" "$TASK_LOG_DIR"
if [[ "$DRY_RUN" == true ]]; then
    for pbs_script in "${PBS_SCRIPTS[@]}"; do
        printf 'qsub %s\n' "$pbs_script"
    done
elif [[ "$NO_SUBMIT" == true ]]; then
    for pbs_script in "${PBS_SCRIPTS[@]}"; do
        printf 'Generated only: qsub %s\n' "$pbs_script"
    done
else
    for pbs_script in "${PBS_SCRIPTS[@]}"; do
        qsub "$pbs_script"
    done
fi
