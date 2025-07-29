import argparse
from pathlib import Path
import os
import torch
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.model_selection import KFold

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


args = parser.parse_args()

# Set up paths and parameters
chemprop_dir = Path.cwd()
input_path = chemprop_dir / "data" / f"{args.dataset_name}.csv"
num_workers = 0  # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles'  # name of the column containing SMILES strings
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

# Automatically detect target columns (all columns except 'smiles')
target_columns = [c for c in df_input.columns
                  if c not in ([smiles_column] + descriptor_columns)]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")

# === Process Data ===
smis = df_input.loc[:, smiles_column].values
for i, smi in enumerate(smis):
    if not isinstance(smi, str):
        print(f"Non-string SMILES at index {i}: {smi} (type: {type(smi)})")




ys = df_input.loc[:, target_columns].values
descriptor_data = df_input[descriptor_columns].values if descriptor_columns else None
if descriptor_data is not None:
    all_data = [data.MoleculeDatapoint.from_smi(smi, y, x_d=descriptors) for smi, y, descriptors in zip(smis, ys, descriptor_data)]
else:
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]


# === Split via Random/Stratified Split with 5 Repetitions ===
from sklearn.model_selection import StratifiedShuffleSplit

if args.task_type in ['binary', 'multi']:
    y_class = df_input[target_columns[0]].values  # assumes a single task for stratification
    train_indices, val_indices, test_indices = [], [], []

    for i in range(REPLICATES):
        sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + i)
        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=SEED + i)  # 10% val from 90% train

        idx = np.arange(len(y_class))
        train_val_idx, test_idx = next(sss_outer.split(idx, y_class))
        train_idx, val_idx = next(sss_inner.split(train_val_idx, y_class[train_val_idx]))

        train_indices.append(idx[train_val_idx][train_idx].tolist())
        val_indices.append(idx[train_val_idx][val_idx].tolist())
        test_indices.append(idx[test_idx].tolist())
else:
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, "RANDOM", (0.8, 0.1, 0.1), seed=SEED, num_replicates=REPLICATES
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
if descriptor_data is not None:
    results_df.to_csv(f"{chemprop_dir}/results/{args.dataset_name}_dmpnn_descriptors_results.csv", index=False)
else:
    results_df.to_csv(f"{chemprop_dir}/results/{args.dataset_name}_dmpnn_results.csv", index=False)
