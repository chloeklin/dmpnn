import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from utils import (
    process_data,
    create_all_data,
    make_repeated_splits,
)

from chemprop import data, featurizers, models, nn

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)


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
parser.add_argument("--baselines", type=str, nargs="+", default=None,
                        help="Which baselines to train (Regression: Linear RF XGB; Classification: LogReg RF XGB). Omit for all.")

args = parser.parse_args()

# Set up paths and parameters
chemprop_dir = Path.cwd()
input_path = chemprop_dir / "data" / f"{args.dataset_name}.csv"
num_workers = 0  # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles' if args.model_name == "DMPNN" else 'WDMPNN_Input'  # name of the column containing SMILES strings
SEED = 42
REPLICATES = 5

# Seed / defaults
set_seed(SEED)

# defaults for baselines if not given
if args.baselines is None:
    args.baselines = ["Linear", "RF", "XGB"] if args.task_type == "reg" else ["LogReg", "RF", "XGB"]


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
# Which variant are we evaluating?
use_desc = bool(descriptor_columns)
use_rdkit = bool(args.incl_rdkit)

variant_tokens = []
if use_desc:
    variant_tokens.append("desc")
if use_rdkit:
    variant_tokens.append("rdkit")

variant_label = "original" if not variant_tokens else "+".join(variant_tokens)
variant_tag = "" if variant_label == "original" else "_" + variant_label.replace("+", "_")


smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

 # Prepare wide-format rows: a dict per (replicate, model)
rep_model_to_row = {}
# ensure pre-create rows for all replicate×model combos
# (we'll fill target-specific metric keys as we compute them)
models_dict = build_sklearn_models(args.task_type, n_classes=args.n_classes, baselines=args.baselines)
model_names_example = list(models_dict.keys())
models_dict = build_sklearn_models(args.task_type, n_classes=args.n_classes, baselines=args.baselines)
model_names_example = list(models_dict.keys())

# Common metadata for this run
base_meta = {
    "dataset": args.dataset_name,
    "encoder": args.model_name,   # "DMPNN" or "wDMPNN"
    "variant": variant_label,     # "original", "desc", "rdkit", or "desc+rdkit"
}

    
# Iterate per target
for target in target_columns:
    # Prepare data
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1) # reshaping target to be 2D
    all_data = create_all_data(smis, ys, combined_descriptor_data, args.model_name)


    # Build replicates of splits
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
    
    # Split to datasets for each replicate
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )


    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    for i in range(REPLICATES):
        # Ensure pre-create rows for all replicate×model combos
        for mname in model_names_example:
            rep_model_to_row[(i, mname)] = {
                **base_meta,
                "replicate": i,
                "model": mname,
            }

        # Prepare datasets
        train, val, test = data.MoleculeDataset(train_data[i], featurizer), data.MoleculeDataset(val_data[i], featurizer), data.MoleculeDataset(test_data[i], featurizer)
        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
        # Normalize input descriptors (to match training’s scaling, if used)
        X_d_transform = None
        if combined_descriptor_data is not None:
            descriptor_scaler = train.normalize_inputs("X_d")
            val.normalize_inputs("X_d", descriptor_scaler)
            X_d_transform = nn.ScaleTransform.from_standard_scaler(descriptor_scaler)

        train_loader = data.build_dataloader(train, num_workers=num_workers)
        val_loader = data.build_dataloader(val, num_workers=num_workers, shuffle=False)
        test_loader = data.build_dataloader(test, num_workers=num_workers, shuffle=False)
        
        # Load best ckpt
        checkpoint_path = (
            f"checkpoints/{args.model_name}/"
            f"{args.dataset_name}__{target}"
            f"{'__desc' if descriptor_columns else ''}"
            f"{'__rdkit' if args.incl_rdkit else ''}"
            f"__rep{i}/"
            )
        last_ckpt = load_best_checkpoint(Path(checkpoint_path))
        if last_ckpt is None:
            # no checkpoint → skip this replicate (leave row without this target's metrics)
            print(f"WARNING: No checkpoint found at {ckpt_dir}; skipping rep {rep} for target {target}.", file=sys.stderr)
            continue

        # Load encoder and make fingerprints
        mpnn = models.MPNN.load_from_checkpoint(str(last_ckpt))
        mpnn.eval()
        X_train = get_encodings_from_loader(mpnn, train_loader)
        X_val = get_encodings_from_loader(mpnn, val_loader)
        X_test = get_encodings_from_loader(mpnn, test_loader)

        # Combine train+val to fit baselines
        X_fit = np.concatenate([X_train, X_val], axis=0)
        y_fit = np.concatenate([
            df_proc.loc[train_indices[i], target].to_numpy(),
            df_proc.loc[val_indices[i],   target].to_numpy()
        ], axis=0)

        y_test = df_proc.loc[test_indices[i], target].to_numpy()

        # Build requested baselines
        models_dict = build_sklearn_models(args.task_type, n_classes=args.n_classes, baselines=args.baselines)

        # Fit baselines on train+val, compute metrics on test only
        for name, model in models_dict.items():
            model.fit(X_fit, y_fit)

            if args.task_type == "reg":
                y_pred = model.predict(X_test)
                r2   = r2_score(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rep_model_to_row[(rep, name)][f"{target}_R2"] = r2; rep_model_to_row[(rep, name)][f"{target}_MAE"] = mae; rep_model_to_row[(rep, name)][f"{target}_RMSE"] = rmse

            else:
                y_pred = model.predict(X_te)
                acc = accuracy_score(y_test, y_pred)
                avg = "macro" if args.task_type == "multi" else "binary"
                f1  = f1_score(y_test, y_pred, average=avg)
                rep_model_to_row[(rep, name)][f"{target}_ACC"] = acc; row[f"{target}_F1"] = f1

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_te)
                    try:
                        if args.task_type == "binary":
                            auc = roc_auc_score(y_test, proba[:, 1])
                        else:
                            from sklearn.preprocessing import label_binarize
                            y_bin = label_binarize(y_test, classes=list(range(args.n_classes)))
                            auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                        rep_model_to_row[(rep, name)][f"{target}_ROC_AUC"] = auc
                    except Exception:
                        pass
        

    # finalize wide dataframe
    wide_rows = list(rep_model_to_row.values())
    results_df = pd.DataFrame(wide_rows).sort_values(["replicate", "model"]).reset_index(drop=True)

    # write
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"{args.dataset_name}_{args.model_name}_fp.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Wrote wide results to: {out_csv}")