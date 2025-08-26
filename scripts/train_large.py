import argparse
import re
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd


from chemprop import data, featurizers, nn
from utils import (set_seed, process_data, make_repeated_splits, 
                  load_drop_indices, create_all_data, build_model_and_trainer, 
                  get_metric_list, load_config, filter_insulator_data)



# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')
parser.add_argument('--descriptor', action='store_true',
                    help='Use dataset-specific descriptors')
parser.add_argument('--incl_rdkit', action='store_true',
                    help='Include RDKit descriptors')
parser.add_argument('--model_name', type=str, default="DMPNN",choices=["DMPNN", "wDMPNN","periodic"],
                    help='Name of the model to use')
args = parser.parse_args()


# Load configuration
config = load_config()

# Set up paths and parameters
chemprop_dir = Path.cwd()

# Get paths from config with defaults
paths = config.get('PATHS', {})
data_dir = chemprop_dir / paths.get('data_dir', 'data')
checkpoint_dir = chemprop_dir / paths.get('checkpoint_dir', 'checkpoints') / args.model_name
results_dir = chemprop_dir / paths.get('results_dir', 'results') / args.model_name

# Create necessary directories
checkpoint_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Set up file paths
input_path = chemprop_dir / data_dir / f"{args.dataset_name}.csv"

# Get model-specific configuration
model_config = config['MODELS'].get(args.model_name, {})
if not model_config:
    print(f"Warning: No configuration found for model '{args.model_name}'. Using defaults.")

# Set parameters from config with defaults
GLOBAL_CONFIG = config.get('GLOBAL', {})
SEED = GLOBAL_CONFIG.get('SEED', 42)
REPLICATES = GLOBAL_CONFIG.get('REPLICATES', 5)
EPOCHS = GLOBAL_CONFIG.get('EPOCHS', 300)
num_workers = min(
    GLOBAL_CONFIG.get('NUM_WORKERS', 8),
    os.cpu_count() or 1
)

# Model-specific parameters
smiles_column = model_config.get('smiles_column', 'smiles')
ignore_columns = model_config.get('ignore_columns', [])
MODEL_NAME = args.model_name

# Get dataset descriptors from config
DATASET_DESCRIPTORS = config.get('DATASET_DESCRIPTORS', {}).get(args.dataset_name, [])
descriptor_columns = DATASET_DESCRIPTORS if args.descriptor else []

# === Set Random Seed ===
set_seed(SEED)

# === Load Data ===
df_input = pd.read_csv(input_path)

# Apply insulator dataset filtering if needed
if args.dataset_name == "insulator" and args.model_name == "wDMPNN":
    df_input = filter_insulator_data(df_input, smiles_column)

# Read the saved exclusions from the wDMPNN preprocessing step
if args.model_name == "DMPNN":
    drop_idx, excluded_smis = load_drop_indices(chemprop_dir, args.dataset_name)
    if drop_idx:
        print(f"Dropping {len(drop_idx)} rows from {args.dataset_name} due to exclusions.")
        df_input = df_input.drop(index=drop_idx, errors="ignore").reset_index(drop=True)


# Automatically detect target columns (all columns except ignored ones)
target_columns = [c for c in df_input.columns
                 if c not in ([smiles_column] + 
                            DATASET_DESCRIPTORS + 
                            ignore_columns)]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")



print("\n=== Training Configuration ===")
print(f"Dataset          : {args.dataset_name}")
print(f"Task type        : {args.task_type}")
print(f"Model            : {args.model_name}")
print(f"SMILES column    : {smiles_column}")
print(f"Descriptor cols  : {descriptor_columns}")
print(f"Ignore columns   : {ignore_columns}")
print(f"Descriptors      : {'Enabled' if args.descriptor else 'Disabled'}")
print(f"RDKit desc.      : {'Enabled' if args.incl_rdkit else 'Disabled'}")
print(f"Epochs           : {EPOCHS}")
print(f"Replicates       : {REPLICATES}")
print(f"Workers          : {num_workers}")
print(f"Random seed      : {SEED}")
print("================================\n")


smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() if args.model_name == "DMPNN" else featurizers.PolymerMolGraphFeaturizer()
      

for target in target_columns:
    # Extract target values
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1) # reshaping target to be 2D
    all_data = create_all_data(smis, ys, combined_descriptor_data, MODEL_NAME)


    # === Split via Random/Stratified Split with 5 Repetitions ===
    if args.task_type in ['binary', 'multi']:
        y_class = df_input[target].to_numpy(dtype=int, copy=False)
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=REPLICATES,
            seed=SEED,
            y_class=y_class
            
        )
    else:
        train_indices, val_indices, test_indices =  make_repeated_splits(
            task_type=args.task_type,
            replicates=REPLICATES,
            seed=SEED,
            mols=[d.mol for d in all_data]
        )
    


    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # === Train ===
    results_all = []

    for i in range(REPLICATES):
        # Get train/val/test for current repetition
        if MODEL_NAME == "DMPNN":
            train, val, test = data.MoleculeDataset(train_data[i], featurizer), data.MoleculeDataset(val_data[i], featurizer), data.MoleculeDataset(test_data[i], featurizer) 
        else:
            data.PolymerDataset(train_data[i], featurizer), data.PolymerDataset(val_data[i], featurizer), data.PolymerDataset(test_data[i], featurizer) 
        
        # normalise targets
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
        # normalise descriptors (if any)
        X_d_transform = None
        if combined_descriptor_data is not None:
            descriptor_scaler = train.normalize_inputs("X_d")
            val.normalize_inputs("X_d", descriptor_scaler)
            X_d_transform = nn.ScaleTransform.from_standard_scaler(descriptor_scaler)
        
        # Create dataloaders
        train_loader = data.build_dataloader(train, num_workers=num_workers)
        val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)

        # Modular metric selection
        n_classes_arg = n_classes_per_target[target] if args.task_type == 'multi' else None
        metric_list = get_metric_list(
            args.task_type,
            target=target,
            n_classes=n_classes_arg,
            df_input=df_input
        )
        batch_norm = False
        # Create checkpoint directory structure
        checkpoint_path = (
            checkpoint_dir / 
            f"{args.dataset_name}__{target}"
            f"{'__desc' if descriptor_columns else ''}"
            f"{'__rdkit' if args.incl_rdkit else ''}"
            f"__rep{i}"
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        scaler_arg = scaler if args.task_type == 'reg' else None
        mpnn, trainer = build_model_and_trainer(
            args=args,
            combined_descriptor_data=combined_descriptor_data,
            n_classes=n_classes_arg,
            scaler=scaler_arg,
            X_d_transform=X_d_transform,
            checkpoint_path=checkpoint_path,
            batch_norm=batch_norm,
            metric_list=metric_list
        )
        last_ckpt = None
        if os.path.exists(checkpoint_path):
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
            if ckpt_files:
                last_ckpt = str(Path(checkpoint_path) / sorted(ckpt_files)[-1])

        # Train
        trainer.fit(mpnn, train_loader, val_loader, ckpt_path=last_ckpt)
        results = trainer.test(dataloaders=test_loader)
        test_metrics = results[0]
        results_all.append(test_metrics)
    

    # Convert to DataFrame
    results_df = pd.DataFrame(results_all)
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()

    print(f"\n[{target}] Mean across {REPLICATES} splits:\n{mean_metrics}")
    print(f"\n[{target}] Std across {REPLICATES} splits:\n{std_metrics}")


    # Optional: save to file
    results_df.to_csv(results_dir / f"{args.dataset_name}_{target}{"_descriptors" if descriptor_columns is not None else ""}{"_rdkit" if args.incl_rdkit else ""}_{MODEL_NAME}_results.csv", index=False)
