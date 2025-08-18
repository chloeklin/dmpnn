import argparse
from pathlib import Path
import os
import torch
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.model_selection import KFold
from utils import *
from chemprop import data, featurizers, models, nn

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')
parser.add_argument('--n_classes', type=int, default=3,
                    help='Number of classes for multi-class classification')
parser.add_argument('--descriptor_columns', type=str, nargs='+', default=[],
                    help='List of extra descriptor column names to use as global features')
parser.add_argument('--model_name', type=str, choices=['DMPNN', 'wDMPNN'], default="DMPNN",
                    help='Name of the model to use')

args = parser.parse_args()

# Set up paths and parameters
chemprop_dir = Path.cwd()
input_path = chemprop_dir / "data" / f"{args.dataset_name}.csv"
num_workers = 0  # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles' if args.model_name == "DMPNN" else 'WDMPNN_Input'  # name of the column containing SMILES strings
SEED = 42
REPLICATES = 5
EPOCHS = 300

# === Set Random Seed ===
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Load Data ===
df_input = pd.read_csv(input_path)

# Read descriptor columns from args
descriptor_columns = args.descriptor_columns or []
ignore_columns = ['WDMPNN_Input'] if args.model_name == "wDMPNN" else ['smiles']
# Automatically detect target columns (all columns except 'smiles')
target_columns = [c for c in df_input.columns
                  if c not in ([smiles_column] + descriptor_columns + ignore_columns)]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")

smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()


for target in target_columns:
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
    


    _, _, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # === Evaluate on test set ===
    results_all = []
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    for i in range(REPLICATES):
        test = data.MoleculeDataset(test_data[i], featurizer)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)
        
        # Load best ckpt
        if descriptor_data is not None:
            checkpoint_path = f"checkpoints/{args.dataset_name}_descriptors/rep_{i}/"
        else:
            checkpoint_path = f"checkpoints/{args.dataset_name}/rep_{i}/"
        last_ckpt = None
        if os.path.exists(checkpoint_path):
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
            if ckpt_files:
                # Optionally sort to get the latest/best checkpoint
                last_ckpt = str(Path(checkpoint_path) / sorted(ckpt_files)[-1])

        mpnn = models.MPNN.load_from_checkpoint(last_ckpt)
        mpnn.eval()

        with torch.no_grad():
            fingerprints = [
                mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)
                for batch in test_loader
            ]
            fingerprints = torch.cat(fingerprints, 0)
        

    # Convert to DataFrame
    results_df = pd.DataFrame(results_all)
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()

    print("Mean performance across 5 splits:")
    print(mean_metrics)

    print("\nStandard deviation across 5 splits:")
    print(std_metrics)

    # Optional: save to file
    Path(chemprop_dir / f"results/{MODEL_NAME}/{args.dataset_name}").mkdir(parents=True, exist_ok=True)
    if descriptor_data is not None:
        results_df.to_csv(f"{chemprop_dir}/results/{args.dataset_name}_dmpnn_descriptors_results.csv", index=False)
    else:
        results_df.to_csv(f"{chemprop_dir}/results/{args.dataset_name}_dmpnn_results.csv", index=False)
