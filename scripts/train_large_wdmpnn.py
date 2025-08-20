import argparse
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
from utils import *

from chemprop import data, featurizers, nn

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
parser.add_argument('--model_name', type=str, default="wDMPNN",
                    help='Name of the model to use')

args = parser.parse_args()

# Set up paths and parameters
chemprop_dir = Path.cwd()
input_path = chemprop_dir / "data" / f"{args.dataset_name}.csv"
num_workers = min(8, os.cpu_count() or 1)
smiles_column = 'WDMPNN_Input'  # name of the column containing SMILES strings
SEED = 42
REPLICATES = 5
EPOCHS = 300
MODEL_NAME = args.model_name


# === Set Random Seed ===
set_seed(SEED)

# === Load Data ===
df_input = pd.read_csv(input_path)

if args.dataset_name == "insulator":
   
    # Copy of original data for index reference
    df_orig = df_input.copy()

    # Identify rows without '*' in the SMILES string
    missing_star_mask = ~df_orig[smiles_column].astype(str).str.contains(r"\*", regex=True)
    missing_star_indices = df_orig[missing_star_mask].index.tolist()

    # Identify rows with specific problematic SMILES
    excluded_smis = {"[*:1]CC(C)/C=C\[*:2]C|1.0|<1-2:0.5:0.5", "[*:1]CC/C=C\[*:2]Cl|1.0|<1-2:0.5:0.5", "[*:1]C1CCC(C=[*:2])C1|1.0|<1-2:0.5:0.5"}
    problem_smiles_mask = df_orig[smiles_column].isin(excluded_smis)
    problem_smiles_indices = df_orig[problem_smiles_mask].index.tolist()

    # Save missing '*' indices
    with open(f"{args.dataset_name}_skipped_indices.txt", "w") as f:
        for idx in missing_star_indices:
            f.write(f"{idx}\n")

    # Save excluded problematic SMILES (index + SMILES string)
    with open(f"{args.dataset_name}_excluded_problematic_smiles.txt", "w") as f:
        for idx in problem_smiles_indices:
            smi = df_orig.loc[idx, smiles_column]
            f.write(f"{idx},{smi}\n")

    # === Apply filtering on original DataFrame ===
    # Remove rows without '*' or with the specific problematic SMILES
    final_mask = ~(missing_star_mask | problem_smiles_mask)
    df_input = df_orig[final_mask].reset_index(drop=True)


# Read descriptor columns from args
descriptor_columns = args.descriptor_columns or []
ignore_columns = ['smiles']
# Automatically detect target columns (all columns except 'smiles')
target_columns = [c for c in df_input.columns
                  if c not in ([smiles_column] + DATASET_DESCRIPTORS.get(args.dataset_name, []) + descriptor_columns + ignore_columns)]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")

smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.PolymerMolGraphFeaturizer()

for target in target_columns:
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1)
    
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
        train, val, test = data.PolymerDataset(train_data[i], featurizer), data.PolymerDataset(val_data[i], featurizer), data.PolymerDataset(test_data[i], featurizer)
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
        X_d_transform = None
        if combined_descriptor_data is not None:
            descriptor_scaler = train.normalize_inputs("X_d")
            val.normalize_inputs("X_d", descriptor_scaler)
            X_d_transform = nn.ScaleTransform.from_standard_scaler(descriptor_scaler)
        

        train_loader = data.build_dataloader(train, num_workers=num_workers)
        val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)

        
        # Get the number of tasks from the training data
        n_tasks = len(target_columns)
        scaler_arg = scaler if args.task_type == 'reg' else None
        n_classes_arg = n_classes_per_target[target] if args.task_type == 'multi' else None
        metric_list = get_metric_list(
            args.task_type,
            target=target,
            n_classes=n_classes_arg,
            df_input=df_input
        )
        
        batch_norm = False
        checkpoint_path = (
            f"checkpoints/{MODEL_NAME}/"
            f"{args.dataset_name}__{target}"
            f"{'__desc' if descriptor_columns else ''}"
            f"{'__rdkit' if args.incl_rdkit else ''}"
            f"__rep{i}/"
            )
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
                # Optionally sort to get the latest/best checkpoint
                last_ckpt = str(Path(checkpoint_path) / sorted(ckpt_files)[-1])


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
    Path(chemprop_dir / f"results/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f"{chemprop_dir}/results/{MODEL_NAME}/{args.dataset_name}_{target}{"_descriptors" if descriptor_columns is not None else ""}{"_rdkit" if args.incl_rdkit else ""}_{MODEL_NAME}_results.csv", index=False)
