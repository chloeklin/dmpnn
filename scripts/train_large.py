import argparse
from pathlib import Path

import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

from chemprop import data, featurizers, models, nn

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')

args = parser.parse_args()

# Set up paths and parameters
chemprop_dir = Path.cwd().parent
input_path = chemprop_dir / "data" / f"{args.dataset_name}.csv"
num_workers = 0  # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles'  # name of the column containing SMILES strings
SEED = 42
REPLICATES = 5
EPOCHS = 300

# === Load Data ===
df_input = pd.read_csv(input_path)

# Automatically detect target columns (all columns except 'smiles')
target_columns = [col for col in df_input.columns if col != smiles_column]
if not target_columns:
    raise ValueError(f"No target columns found. Expected at least one column other than '{smiles_column}'")

# === Process Data ===
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values
all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]


# === Split via Random with 5 Repetitions ===
mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(mols, "RANDOM", (0.8, 0.1, 0.1), seed=SEED, num_replicates=REPLICATES)  # unpack the tuple into three separate lists
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

# Create necessary directories
(chemprop_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(chemprop_dir / "results").mkdir(parents=True, exist_ok=True)

# === Train ===
results_all = []
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
for i in range(REPLICATES):
    train, val, test = data.MoleculeDataset(train_data[i], featurizer), data.MoleculeDataset(val_data[i], featurizer), data.MoleculeDataset(test_data[i], featurizer)
    scaler = train.normalize_targets()
    val.normalize_targets(scaler)

    train_loader = data.build_dataloader(train, num_workers=num_workers)
    val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    
    if args.task_type == 'reg':
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(output_transform=output_transform)
        metric_list = [nn.metrics.MAE(), nn.metrics.RMSE(), nn.metrics.R2Score()]
    elif args.task_type == 'binary':
        ffn = nn.BinaryClassificationFFN()
        metric_list = [nn.metrics.BinaryAccuracy(), nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score()]
    elif args.task_type == 'multi':
        ffn = nn.MulticlassClassificationFFN()
        metric_list = [nn.metrics.Accuracy(), nn.metrics.F1Score(), nn.metrics.AUROC()]
    
    batch_norm = False
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
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

    trainer.fit(mpnn, train_loader, val_loader)
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
results_df.to_csv(f"{chemprop_dir}/results/{args.dataset_name}_dmpnn_results.csv", index=False)

        
