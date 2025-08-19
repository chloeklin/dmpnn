from typing import Optional, List
import numpy as np
from pathlib import Path
import pandas as pd

DATASET_DESCRIPTORS = {
    "htpmd": ["Molality", "Monomer_Molecular_Weight", "DoP", "Density"],
}

def set_seed(seed: int = 42):
    import os
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def combine_descriptors(rdkit_data, descriptor_data):
    import numpy as np
    if rdkit_data is not None and descriptor_data is not None:
        return np.asarray([np.concatenate([r, d]) for r, d in zip(rdkit_data, descriptor_data)], dtype=np.float32)
    elif rdkit_data is not None:
        return np.asarray(rdkit_data, dtype=np.float32)
    elif descriptor_data is not None:
        return np.asarray(descriptor_data, dtype=np.float32)
    else:
        return None

def preprocess_classification_labels(df_input, target_columns, task_type):
    import numpy as np
    
    for tcol in target_columns:
        # Map strings to ints if needed
        if df_input[tcol].dtype.kind in {'U', 'S', 'O'}:
            classes = sorted(df_input[tcol].dropna().unique().tolist())
            class_to_idx = {c: i for i, c in enumerate(classes)}
            df_input[tcol] = df_input[tcol].map(class_to_idx)

        # Forbid negative labels
        if (df_input[tcol].dropna() < 0).any():
            raise ValueError(
                f"Found negative class labels in {tcol}. Replace missing labels with NaN, not -1."
            )

        # Ensure int dtype
        df_input[tcol] = df_input[tcol].astype(int)

        uniq = np.sort(df_input[tcol].dropna().unique())

        if task_type == 'multi':
            # If you want per-column n_classes:
            # Check if labels are contiguous 0..C-1
            expected_max = uniq.size - 1
            if uniq.min() != 0 or uniq.max() != expected_max:
                remap = {c: i for i, c in enumerate(uniq)}
                df_input[tcol] = df_input[tcol].map(remap)
                uniq = np.sort(df_input[tcol].dropna().unique())
            print(f"[multi] {tcol}: classes={uniq} (n={len(uniq)})")
        else:  # binary
            if not np.array_equal(uniq, [0, 1]) and not np.array_equal(uniq, [0]) and not np.array_equal(uniq, [1]):
                raise ValueError(
                    f"Binary labels in {tcol} must be 0/1. Found {uniq}."
                )
    return df_input

def process_data(df_input, smiles_column, descriptor_columns, target_columns, args):
    from chemprop import featurizers
    from chemprop.utils import make_mol
    # === Process Data ===
    smis = df_input.loc[:, smiles_column].values
    for i, smi in enumerate(smis):
        if not isinstance(smi, str):
            print(f"Non-string SMILES at index {i}: {smi} (type: {type(smi)})")

    # --- enforce clean labels for classification ---
    if args.task_type in ['binary', 'multi']:
        df_input = preprocess_classification_labels(df_input, target_columns, args.task_type)

    n_classes_per_target = {}
    if args.task_type == 'multi':
        for tcol in target_columns:
            n_classes_per_target[tcol] = int(df_input[tcol].dropna().nunique())


    descriptor_data = df_input[descriptor_columns].values if descriptor_columns else None
    if args.incl_rdkit:
        feat = featurizers.RDKit2DFeaturizer()
        if args.model_name == "wDMPNN":
            original_smis = df_input.loc[:, "smiles"].values
            mols = [make_mol(smi, keep_h=False, add_h=False, ignore_stereo=False) for smi in original_smis]
        else:
            mols = [make_mol(smi, keep_h=False, add_h=False, ignore_stereo=False) for smi in smis]
            
        rdkit_data = [feat(mol) for mol in mols]
    else:
        rdkit_data = None
    combined_descriptor_data = combine_descriptors(rdkit_data, descriptor_data)

    if combined_descriptor_data is not None and combined_descriptor_data.ndim != 2:
        raise ValueError(f"X_d must be 2D, got {combined_descriptor_data.shape}")

    return smis, df_input, combined_descriptor_data, n_classes_per_target

def create_all_data(smis, ys, combined_descriptor_data, model_name):
    from chemprop import data
    if model_name == "DMPNN":
        if combined_descriptor_data is not None:
            return [data.MoleculeDatapoint.from_smi(smi, y, x_d=desc) for smi, y, desc in zip(smis, ys, combined_descriptor_data)]
        else:
            return [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    else:
        if combined_descriptor_data is not None:
            return [data.PolymerDatapoint.from_smi(smi, y, x_d=desc) for smi, y, desc in zip(smis, ys, combined_descriptor_data)]
        else:
            return [data.PolymerDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]


        


def get_metric_list(task_type, target=None, n_classes=None, df_input=None):
    from chemprop import nn
    import numpy as np
    if task_type == 'reg':
        return [nn.metrics.MAE(), nn.metrics.RMSE(), nn.metrics.R2Score()]
    elif task_type == 'binary':
        metrics = [nn.metrics.BinaryAccuracy(), nn.metrics.BinaryF1Score()]
        # Add AUROC only if both classes appear
        if df_input is not None and target is not None and len(np.unique(df_input[target].dropna())) > 1:
            metrics.append(nn.metrics.BinaryAUROC())
        return metrics
    elif task_type == 'multi':
        return [
            nn.metrics.MulticlassAccuracy(num_classes=n_classes, average='macro'),
            nn.metrics.MulticlassF1Score(num_classes=n_classes, average='macro'),
            nn.metrics.MulticlassAUROC(num_classes=n_classes, average='macro')
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def build_model_and_trainer(args, combined_descriptor_data, n_classes, scaler, X_d_transform, checkpoint_path, batch_norm, metric_list):
    from chemprop import nn, models
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    import lightning.pytorch as pl

    # Select Message Passing Scheme
    if args.model_name == "wDMPNN":
        mp = nn.WeightedBondMessagePassing(d_v=72, d_e=86)
    elif args.model_name == "DMPNN":
        mp = nn.BondMessagePassing()
    input_dim = mp.output_dim + combined_descriptor_data.shape[1] if combined_descriptor_data is not None else mp.output_dim
    
    # Model selection
    if args.task_type == 'reg':
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler) if scaler is not None else None
        ffn = nn.RegressionFFN(output_transform=output_transform, n_tasks=1, input_dim=input_dim)
    elif args.task_type == 'binary':
        ffn = nn.BinaryClassificationFFN(input_dim=input_dim)
    elif args.task_type == 'multi':
        ffn = nn.MulticlassClassificationFFN(n_classes=n_classes, input_dim=input_dim)

    agg = nn.MeanAggregation()
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform)
    # Checkpointing
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpointing = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=30,
        mode="min",
        verbose=True
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=300,
        callbacks=[early_stop, checkpointing],
    )
    return mpnn, trainer

def make_repeated_splits(
    task_type: str,
    replicates: int,
    seed: int,
    *,
    y_class: Optional[np.ndarray] = None,   # required for classification
    mols: Optional[List] = None,            # required for regression (Chemprop)
) :
    from chemprop import data
    from sklearn.model_selection import StratifiedShuffleSplit  
    # === Split via Random/Stratified Split with 5 Repetitions ===
    if task_type in ['binary', 'multi']:
        train_indices, val_indices, test_indices = [], [], []

        for i in range(replicates):
            sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed + i)
            sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=seed + i)  # 10% val from 90% train

            idx = np.arange(len(y_class))
            train_val_idx, test_idx = next(sss_outer.split(idx, y_class))
            train_idx, val_idx = next(sss_inner.split(train_val_idx, y_class[train_val_idx]))

            train_indices.append(idx[train_val_idx][train_idx].tolist())
            val_indices.append(idx[train_val_idx][val_idx].tolist())
            test_indices.append(idx[test_idx].tolist())
    else:
        train_indices, val_indices, test_indices = data.make_split_indices(
            mols, "RANDOM", (0.8, 0.1, 0.1), seed=seed, num_replicates=replicates
        )
    
    return train_indices, val_indices, test_indices



def build_sklearn_models(task_type, n_classes=None, baselines=["Linear", "RF", "XGB"]):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  
    from xgboost import XGBRegressor, XGBClassifier
    models_dict = {}
    if task_type == "reg":
        for baseline in baselines:
            if baseline == "Linear":
                models_dict["Linear"] = LinearRegression()
            elif baseline == "RF":
                models_dict["RF"] = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
            elif baseline == "XGB":
                models_dict["XGB"] = XGBRegressor(
                    n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
            )
    else:
        multi = (task_type == "multi")
        for baseline in baselines:
            if baseline == "LogReg":
                models_dict["LogReg"] = LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=-1 if hasattr(LogisticRegression(), "n_jobs") else None,
                    multi_class="multinomial" if multi else "auto"
                )
            elif baseline == "RF":
                models_dict["RF"] = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
            elif baseline == "XGB":
                models_dict["XGB"] = XGBClassifier(
                    n_estimators=500,
                    max_depth=10,
                    objective="multi:softprob" if multi else "binary:logistic",
                    num_class=n_classes if multi else None,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss" if multi else "logloss"
            )
    return models_dict


def load_best_checkpoint(ckpt_dir: Path):
    import os
    if not ckpt_dir.exists():
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
    ckpts.sort()
    return ckpt_dir / ckpts[-1]


# def get_encodings_from_loader(model, loader):
#     import torch
#     encs = []
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         for batch in loader:
#             # 1) Get the graph object
#             bmg = getattr(batch, "bmg", None)
#             if bmg is None:
#                 raise ValueError(f"Batch has no 'bmg': {batch}")

#             # 2) Move it to the right device
#             if hasattr(bmg, "to") and callable(getattr(bmg, "to")):
#                 # Many Chemprop BatchMolGraph versions implement .to(device)
#                 bmg = bmg.to(device)
#             else:
#                 # Fallback: move known tensor fields manually
#                 for name in ("V", "E", "edge_index", "batch"):
#                     t = getattr(bmg, name, None)
#                     if isinstance(t, torch.Tensor):
#                         setattr(bmg, name, t.to(device, non_blocking=True))

#             # 3) Move optional descriptor tensors
#             V_d = getattr(batch, "V_d", None)
#             if isinstance(V_d, torch.Tensor):
#                 V_d = V_d.to(device, non_blocking=True)

#             X_d = getattr(batch, "X_d", None)
#             if isinstance(X_d, torch.Tensor):
#                 X_d = X_d.to(device, non_blocking=True)
            
#             enc = model.fingerprint(bmg,V_d,X_d)
#             encs.append(enc)
#     return torch.cat(encs, dim=0).cpu().numpy()
def get_encodings_from_loader(model, loader):
    import torch
    encs = []
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(loader):
            # 1) Graph
            bmg = getattr(batch, "bmg", None)
            if bmg is None:
                raise ValueError(f"[batch {j}] has no 'bmg'")

            # 2) Move to device without losing the object if .to() is in-place
            if hasattr(bmg, "to") and callable(bmg.to):
                ret = bmg.to(device)          # may return None (in-place)
                if ret is not None:
                    bmg = ret
            # Ensure key tensors are on device
            bmg.V = bmg.V.to(device, non_blocking=True)
            bmg.edge_index = bmg.edge_index.to(device, non_blocking=True)
            if getattr(bmg, "batch", None) is not None:
                bmg.batch = bmg.batch.to(device, non_blocking=True)
            # Provide empty edge features if missing (concat expects E)
            if getattr(bmg, "E", None) is None:
                n_edges = bmg.edge_index.size(1)
                bmg.E = bmg.V.new_empty((n_edges, 0))
            else:
                bmg.E = bmg.E.to(device, non_blocking=True)

            # 3) Optional extras
            V_d = getattr(batch, "V_d", None)
            if isinstance(V_d, torch.Tensor):
                V_d = V_d.to(device, non_blocking=True)
            else:
                V_d = None

            X_d = getattr(batch, "X_d", None)
            if isinstance(X_d, torch.Tensor):
                X_d = X_d.to(device, non_blocking=True)
            else:
                X_d = None

            # 4) Encode
            enc = model.fingerprint(bmg, V_d, X_d)
            encs.append(enc)

    return torch.cat(encs, dim=0).cpu().numpy()

    

def load_drop_indices(root_dir, dataset_name: str):
    import pandas as pd 
    """Reads <dataset>_skipped_indices.txt and <dataset>_excluded_problematic_smiles.txt.
    Returns (sorted_indices, excluded_smiles_list)."""
    skip_path = Path(root_dir / f"{dataset_name}_skipped_indices.txt")
    prob_path = Path(root_dir / f"{dataset_name}_excluded_problematic_smiles.txt")

    idxs = set()
    excluded_smis = []

    # one index per line
    if skip_path.exists():
        with skip_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        idxs.add(int(line))
                    except ValueError:
                        pass  # ignore malformed lines

    # "idx,smiles" per line
    if prob_path.exists():
        with prob_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split only on first comma (SMILES can sometimes contain commas if quoted, but your file format uses a single comma)
                parts = line.split(",", 1)
                if len(parts) == 2:
                    idx_str, smi = parts
                    excluded_smis.append(smi)
                    try:
                        idxs.add(int(idx_str))
                    except ValueError:
                        pass

    return sorted(idxs), excluded_smis

def upsert_csv(out_csv: Path, new_df: pd.DataFrame, key_cols: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        old = pd.read_csv(out_csv)

        # Union of columns, keeping key cols first
        all_cols = key_cols + [c for c in (set(old.columns) | set(new_df.columns)) if c not in key_cols]
        old = old.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)

        # Index on keys
        old_i = old.set_index(key_cols)
        new_i = new_df.set_index(key_cols)

        # Start with old, add any new rows/cols
        combined = old_i.combine_first(new_i)

        # Overwrite with new where provided
        combined.update(new_i)

        combined.reset_index().to_csv(out_csv, index=False)
    else:
        new_df.to_csv(out_csv, index=False)

