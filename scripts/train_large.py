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
ys = df_input.loc[:, target_columns].values
descriptor_data = df_input[descriptor_columns].values if descriptor_columns else None
all_data = [data.MoleculeDatapoint.from_smi(smi, y, descriptor_data=descriptors) for smi, y, descriptors in zip(smis, ys, descriptor_data)]


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




train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)


# Create necessary directories
Path(chemprop_dir / "results").mkdir(parents=True, exist_ok=True)

# === Train ===
results_all = []
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
for i in range(REPLICATES):
    train, val, test = data.MoleculeDataset(train_data[i], featurizer), data.MoleculeDataset(val_data[i], featurizer), data.MoleculeDataset(test_data[i], featurizer)
    scaler = train.normalize_targets()
    descriptor_scaler = train.normalize_inputs("descriptor")
    val.normalize_targets(scaler)
    val.normalize_inputs("descriptor", descriptor_scaler)
    

    train_loader = data.build_dataloader(train, num_workers=num_workers)
    val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn_input_dim = mp.output_dim + descriptor_data.shape[1] if descriptor_data is not None else mp.output_dim
    
    # Get the number of tasks from the training data
    n_tasks = len(target_columns)
    
    if args.task_type == 'reg':
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(output_transform=output_transform, n_tasks=n_tasks, input_dim=ffn_input_dim)
        metric_list = [nn.metrics.MAE(), nn.metrics.RMSE(), nn.metrics.R2Score()]
    elif args.task_type == 'binary':
        ffn = nn.BinaryClassificationFFN()
        metric_list = [nn.metrics.BinaryAccuracy(), nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score()]
    elif args.task_type == 'multi':
        ffn = nn.MulticlassClassificationFFN()
        metric_list = [nn.metrics.Accuracy(), nn.metrics.F1Score(), nn.metrics.AUROC()]
    
    batch_norm = False
    X_d_transform = nn.ScaleTransform.from_standard_scaler(descriptor_scaler) if descriptor_data is not None else None

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform)

    checkpoint_path = f"checkpoints/{args.dataset_name}/rep_{i}/"
    last_ckpt = None
    if os.path.exists(checkpoint_path):
        ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]
        if ckpt_files:
            # Optionally sort to get the latest/best checkpoint
            last_ckpt = str(Path(checkpoint_path) / sorted(ckpt_files)[-1])


    # Configure model checkpointing
    Path(f"checkpoints/{args.dataset_name}/rep_{i}/").mkdir(parents=True, exist_ok=True)
    checkpointing = ModelCheckpoint(
        dirpath=f"checkpoints/{args.dataset_name}/rep_{i}/",  # Unique folder for each split
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",     # Metric to monitor
        patience=30,            # Number of epochs with no improvement after which training stops
        mode="min",             # We're minimizing val_loss
        verbose=True
        )


    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=EPOCHS, # number of epochs to train for
        callbacks=[early_stop, checkpointing], # Use the configured checkpoint callback
    )

    trainer.fit(mpnn, train_loader, val_loader, ckpt_path=last_ckpt)
    results = trainer.test(dataloaders=test_loader)
    test_metrics = results[0]
    results_all.append(test_metrics)
    

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
