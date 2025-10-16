# Scripts Documentation

This directory contains all training, evaluation, and utility scripts for the polymer property prediction project.

## Directory Structure

- `python/` - Python scripts for model training, evaluation, and data processing
- `shell/` - Bash scripts for batch job submission and experiment generation

---

## Python Scripts (`python/`)

### Training Scripts

#### `train_graph.py`
Main training script for graph neural network models (D-MPNN, wDMPNN, PPG).

**Key Features:**
- Supports regression, binary, and multi-class classification
- Handles dataset-specific and RDKit descriptors
- Implements train_size subsampling for learning curves
- Automatic preprocessing with caching (imputation, correlation filtering, scaling)
- Cross-validation or holdout validation based on dataset size
- Checkpoint saving and resumption
- Prediction export for analysis

**Usage:**
```bash
python train_graph.py \
    --dataset_name opv_camb3lyp \
    --model_name DMPNN \
    --task_type reg \
    --incl_desc \
    --incl_rdkit \
    --train_size 500 \
    --save_predictions \
    --save_checkpoint
```

**Key Arguments:**
- `--dataset_name` - Dataset file name (without .csv)
- `--model_name` - Model type: DMPNN, wDMPNN, or PPG
- `--task_type` - Task: reg, binary, multi, or mixed-reg-multi
- `--target` - Specific target to train (trains all if not specified)
- `--polymer_type` - homo or copolymer
- `--incl_desc` - Include dataset-specific descriptors
- `--incl_rdkit` - Include RDKit 2D descriptors
- `--batch_norm` - Enable batch normalization
- `--train_size` - Training samples: integer (e.g., 500) or "full"
- `--save_predictions` - Save y_true and y_pred for each split
- `--save_checkpoint` - Save model checkpoints
- `--export_embeddings` - Export GNN embeddings
- `--pretrain_monomer` - Train multitask model on pooled homopolymer data

**Outputs:**
- Results: `results/{ModelName}/{dataset}{config}_results_{target}.csv`
- Checkpoints: `checkpoints/{experiment_name}/`
- Preprocessing: `preprocessing/{ModelName}/{experiment_name}/`
- Predictions: `predictions/{experiment_name}/` (if enabled)

---

#### `train_attentivefp.py`
Training script for AttentiveFP graph attention network.

**Key Features:**
- AttentiveFP-specific atom and bond features (39-dim atoms, 10-dim bonds)
- Compatible with D-MPNN preprocessing pipeline
- Supports same arguments as train_graph.py
- Per-split checkpoint saving and loading
- Embedding export capability

**Usage:**
```bash
python train_attentivefp.py \
    --dataset_name opv_camb3lyp \
    --model_name AttentiveFP \
    --task_type reg \
    --train_size 1000 \
    --save_predictions
```

**Key Arguments:**
- Same as `train_graph.py` plus:
- `--hidden` - Hidden layer size (default: 300)
- `--lr` - Learning rate (default: 1e-3)
- `--epochs` - Max epochs (default: 300)
- `--patience` - Early stopping patience (default: 30)
- `--batch_size` - Batch size (default: 64)
- `--device` - cuda or cpu

**Note:** AttentiveFP was missing train_size subsampling logic until recently fixed. Ensure you're using the latest version.

---

#### `train_tabular.py`
Training script for tabular baseline models (Linear, Random Forest, XGBoost).

**Key Features:**
- Three baseline models trained simultaneously
- Supports RDKit descriptors, dataset descriptors, and atom/bond pooled features
- Per-split preprocessing with proper train/test isolation
- Handles homopolymer and copolymer data
- Feature scaling per model (Linear/Logistic require scaling, tree models don't)
- Incremental result saving (skips completed experiments)

**Usage:**
```bash
python train_tabular.py \
    --dataset_name opv_camb3lyp \
    --task_type reg \
    --incl_rdkit \
    --incl_desc \
    --incl_ab \
    --train_size 500
```

**Key Arguments:**
- `--dataset_name` - Dataset file name
- `--task_type` - reg, binary, or multi
- `--incl_desc` - Include dataset-specific descriptors
- `--incl_rdkit` - Include RDKit 2D descriptors
- `--incl_ab` - Include atom/bond pooled features
- `--polymer_type` - homo or copolymer
- `--train_size` - Training samples or "full"

**Models Trained:**
- **Linear/Logistic** - Ridge regression or logistic regression with feature scaling
- **RF** - Random Forest (500 trees, max_depth=20)
- **XGB** - XGBoost with early stopping on validation set

**Outputs:**
- Results: `results/tabular/{dataset}{config}.csv`
- Preprocessing: `out/{target}/` (per-split metadata and objects)

---

#### `train_pae_tg.py`
Training script for Polymer Atom Encoder (PAE) with PyTorch Geometric.

**Key Features:**
- Specialized for polymer representation learning
- Uses BigSMILES notation for polymer structures
- Custom graph construction for polymer chains

**Usage:**
```bash
python train_pae_tg.py \
    --dataset_name block \
    --task_type reg
```

---

### Evaluation Scripts

#### `evaluate_model.py`
Comprehensive model evaluation script for trained models.

**Key Features:**
- Loads trained checkpoints and evaluates on test sets
- Supports all model types (DMPNN, AttentiveFP, tabular)
- Applies exact same preprocessing as training
- Handles multiple targets and splits
- Exports predictions and embeddings

**Usage:**
```bash
python evaluate_model.py \
    --dataset_name opv_camb3lyp \
    --model_name DMPNN \
    --checkpoint_dir checkpoints/opv_camb3lyp__homo__desc__rdkit__rep0 \
    --target homo \
    --split 0
```

**Key Arguments:**
- `--checkpoint_dir` - Path to checkpoint directory
- `--target` - Target property to evaluate
- `--split` - Split index to evaluate
- `--export_embeddings` - Export model embeddings
- `--save_predictions` - Save predictions to file

---

#### `batch_evaluate_all.py`
Batch evaluation script for multiple experiments.

**Key Features:**
- Evaluates all checkpoints matching a pattern
- Parallel evaluation support
- Aggregates results across experiments
- Useful for systematic evaluation of learning curve experiments

**Usage:**
```bash
python batch_evaluate_all.py \
    --checkpoint_pattern "checkpoints/opv_*" \
    --output_dir evaluation_results
```

---

#### `run_all_evaluations.py`
Orchestrates evaluation across multiple models and configurations.

**Key Features:**
- Reads evaluation configuration from YAML
- Submits evaluation jobs for all specified experiments
- Tracks completion status
- Generates summary reports

**Usage:**
```bash
python run_all_evaluations.py --config evaluation_config.yaml
```

---

### Data Processing Scripts

#### `combine_datasets.py`
Combines multiple datasets into a single unified dataset.

**Key Features:**
- Merges datasets with different targets
- Handles missing values and column alignment
- Adds dataset source tracking
- Useful for multi-task learning and transfer learning

**Usage:**
```bash
python combine_datasets.py \
    --input_files data/dataset1.csv data/dataset2.csv \
    --output_file data/combined.csv \
    --add_source_column
```

---

#### `combine_target_results.py`
Aggregates results across multiple targets into summary tables.

**Key Features:**
- Combines per-target result files
- Calculates aggregate statistics
- Generates comparison tables
- Exports to CSV and formatted tables

**Usage:**
```bash
python combine_target_results.py \
    --results_dir results/DMPNN \
    --output_file results/DMPNN_summary.csv
```

---

#### `prepare_pae_inputs.py`
Prepares input data for Polymer Atom Encoder training.

**Key Features:**
- Converts SMILES to BigSMILES notation
- Validates polymer structures
- Creates train/val/test splits
- Exports in PAE-compatible format

**Usage:**
```bash
python prepare_pae_inputs.py \
    --input_file data/polymers.csv \
    --output_dir data/pae_inputs
```

---

#### `deduplicte_rdkit_desc.py`
Removes duplicate RDKit descriptors from feature sets.

**Key Features:**
- Identifies identical descriptors with different names
- Removes redundant features
- Preserves feature name mapping
- Reduces feature dimensionality

**Usage:**
```bash
python deduplicte_rdkit_desc.py \
    --input_file data/features.csv \
    --output_file data/features_deduplicated.csv
```

---

### Utility Scripts

#### `utils.py`
Core utility functions used across all scripts.

**Key Functions:**
- `setup_training_environment()` - Loads config, sets up paths
- `load_and_preprocess_data()` - Data loading with validation
- `generate_data_splits()` - Creates train/val/test splits
- `build_experiment_paths()` - Constructs experiment directory paths
- `manage_preprocessing_cache()` - Handles preprocessing artifact saving/loading
- `validate_checkpoint_compatibility()` - Checks checkpoint compatibility
- `save_aggregate_results()` - Saves results with proper naming
- `build_sklearn_models()` - Creates tabular baseline models
- `set_seed()` - Sets random seeds for reproducibility

---

#### `tabular_utils.py`
Utilities specific to tabular model training.

**Key Functions:**
- `build_features()` - Assembles feature blocks (AB, RDKit, descriptors)
- `preprocess_descriptor_data()` - Preprocessing pipeline (imputation, correlation filtering)
- `save_preprocessing_objects()` - Saves preprocessing metadata and objects
- `compute_rdkit_desc()` - Computes RDKit 2D descriptors
- `atom_bond_block_from_smiles()` - Extracts atom/bond pooled features
- `eval_regression()`, `eval_binary()`, `eval_multi()` - Evaluation metrics

---

#### `visualize_combined_results.py`
Creates visualizations from combined result tables.

**Key Features:**
- Bar plots comparing models
- Heatmaps of performance across targets
- Learning curve plots
- Statistical significance testing

**Usage:**
```bash
python visualize_combined_results.py \
    --results_file results/combined_results.csv \
    --output_dir plots/comparisons
```

---

#### `check_missing_baselines.py`
Identifies missing experiments in a batch run.

**Key Features:**
- Scans result directories for expected files
- Lists incomplete experiments
- Generates resubmission scripts
- Useful for debugging failed jobs

**Usage:**
```bash
python check_missing_baselines.py \
    --results_dir results \
    --expected_targets homo lumo gap \
    --expected_splits 5
```

---

### Configuration Files

#### `train_config.yaml`
Central configuration file for training experiments. See [`PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md) for details.

---

## Shell Scripts (`shell/`)

### Script Generation

#### `generate_training_script.sh`
Generates PBS job submission scripts from experiment configuration.

**Usage:**
```bash
bash generate_training_script.sh experiments.yaml
```

**Outputs:** Individual `.sh` files for each experiment configuration.

---

#### `generate_opv_dmpnn_scripts.sh`
Generates D-MPNN training scripts for OPV dataset experiments.

**Features:**
- Creates scripts for all target properties
- Multiple descriptor configurations
- Batch normalization variants

---

#### `generate_opv_learning_curve_scripts.sh`
Generates learning curve experiment scripts for D-MPNN.

**Features:**
- Multiple training sizes (250, 500, 1000, 2000, 3500, 5000, 8000, 12000, full)
- All targets and configurations
- Organized by training size

---

#### `generate_opv_attentivefp_learning_curve_scripts.sh`
Generates AttentiveFP learning curve scripts.

**Features:**
- Same training sizes as D-MPNN
- AttentiveFP-specific hyperparameters
- Compatible with D-MPNN preprocessing

---

#### `generate_opv_tabular_learning_curve_scripts.sh`
Generates tabular baseline learning curve scripts.

**Features:**
- All three baseline models (Linear, RF, XGB)
- Multiple feature configurations
- Same training sizes as graph models

---

#### `generate_opv_evaluation_scripts.sh`
Generates evaluation scripts for trained models.

**Features:**
- Evaluates all checkpoints
- Exports predictions and embeddings
- Organized by model and configuration

---

#### `batch_generate_scripts.sh`
Batch script generator that calls multiple generation scripts.

**Usage:**
```bash
bash batch_generate_scripts.sh
```

---

### Job Submission and Management

#### `train_*.sh` (Individual Training Scripts)
PBS job submission scripts for specific experiments. Generated by script generation tools.

**Example:** `train_opv_camb3lyp_DMPNN_all_size500_lc.sh`

**PBS Headers:**
- Queue: gpuvolta (for GPU jobs)
- Resources: CPUs, GPUs, memory, walltime
- Storage: scratch and gdata mounts

---

#### `batch_evaluate_all.sh`
Submits evaluation jobs for all trained models.

**Usage:**
```bash
bash batch_evaluate_all.sh
```

---

#### `resubmit_missing_jobs.sh`
Identifies and resubmits failed or missing jobs.

**Features:**
- Scans for incomplete results
- Generates resubmission scripts
- Tracks job dependencies

**Usage:**
```bash
bash resubmit_missing_jobs.sh
```

---

#### `delete_sh.sh`
Utility to clean up generated shell scripts.

**Usage:**
```bash
bash delete_sh.sh
```

---

### Template Scripts

#### `train_attentive.sh`
Template for AttentiveFP training jobs.

#### `run_baseline.sh`
Template for tabular baseline training jobs.

#### `evaluate.sh`
Template for model evaluation jobs.

---

## Workflow Examples

### 1. Training a Single Model
```bash
python python/train_graph.py \
    --dataset_name opv_camb3lyp \
    --model_name DMPNN \
    --task_type reg \
    --incl_rdkit \
    --train_size 1000 \
    --save_predictions
```

### 2. Generating Learning Curve Experiments
```bash
# Generate all scripts
bash shell/generate_opv_learning_curve_scripts.sh

# Submit jobs
for script in shell/train_opv_camb3lyp_DMPNN_*_lc.sh; do
    qsub "$script"
done
```

### 3. Training Tabular Baselines
```bash
python python/train_tabular.py \
    --dataset_name opv_camb3lyp \
    --task_type reg \
    --incl_rdkit \
    --incl_ab \
    --train_size 500
```

### 4. Evaluating Trained Models
```bash
python python/evaluate_model.py \
    --dataset_name opv_camb3lyp \
    --model_name DMPNN \
    --checkpoint_dir checkpoints/opv_camb3lyp__homo__rdkit__size1000__rep0 \
    --target homo \
    --split 0 \
    --export_embeddings
```

### 5. Batch Evaluation
```bash
python python/run_all_evaluations.py --config shell/evaluation_config.yaml
```

---

## Notes

- All Python scripts support `--help` for detailed argument descriptions
- Shell scripts are configured for PBS job scheduler (NCI Gadi HPC)
- Modify PBS headers in shell scripts for different HPC environments
- Use `train_config.yaml` for consistent configuration across experiments
- Preprocessing artifacts are cached and reused when compatible
- Train_size subsampling uses per-split RNGs for reproducibility
