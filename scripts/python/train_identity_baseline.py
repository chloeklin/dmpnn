#!/usr/bin/env python3
"""
Identity-Embedding Baseline for Copolymer Property Prediction
=============================================================

Tests whether monomer *identity* (categorical ID) plus composition fractions
is sufficient for prediction, **without any chemistry-aware graph encoding**.

Each unique monomer SMILES is mapped to an integer ID.  A shared
``nn.Embedding(num_monomers, embed_dim)`` table produces embeddings
``e_A`` and ``e_B`` for the two monomers.

Three modes are supported (``--copolymer_mode``):

* **mix**:      ``z = fracA * e_A + fracB * e_B``
                head input = ``[z]`` or ``[z || meta]`` if descriptors present.
* **mean**:     ``z = (e_A + e_B) / 2``
                head input = ``[z]`` or ``[z || meta]`` if descriptors present.
* **interact**: head input = ``[e_A || e_B || |e_A-e_B| || e_A*e_B || fracA || fracB || meta]``

The MLP prediction head, training loop (early stopping, 5-fold CV),
results CSV, and logging style mirror the existing graph pipeline.
"""

import argparse
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from utils import (
    set_seed, setup_training_environment, load_and_preprocess_data,
    determine_split_strategy, save_model_results,
    generate_a_held_out_splits, save_fold_assignments,
)
from tabular_utils import group_splits

# polymer_input integration — optional structured copolymer validation
# The identity baseline uses categorical monomer IDs (not graph featurizers),
# but polymer_input can still validate the copolymer schema and extract
# scalar features from PolymerSpec objects.  See polymer_input/README.md.
try:
    from polymer_input import (
        PolymerParser, SchemaMapping, validate_polymer_spec,
        extract_scalar_features, collect_scalar_keys,
    )
    _HAS_POLYMER_INPUT = True
except ImportError:
    _HAS_POLYMER_INPUT = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================== MODEL DEFINITION ===============================

class IdentityBaselineModel(nn.Module):
    """MLP with shared monomer-identity embeddings."""

    def __init__(self, num_monomers, embed_dim, mode, meta_dim=0,
                 hidden_dims=(256, 128), task_type="reg", n_classes=None,
                 dropout=0.1):
        super().__init__()
        self.mode = mode
        self.task_type = task_type
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        # Shared embedding table (index 0 reserved for unknown/padding)
        self.embedding = nn.Embedding(num_monomers + 1, embed_dim, padding_idx=0)

        # Compute input dim for the MLP head
        if mode == "mix":
            head_input_dim = embed_dim + meta_dim
        elif mode == "mean":
            head_input_dim = embed_dim + meta_dim
        elif mode == "interact":
            # e_A, e_B, |e_A-e_B|, e_A*e_B  → 4*embed_dim
            # + fracA, fracB → +2
            # + meta descriptors → +meta_dim
            head_input_dim = 4 * embed_dim + 2 + meta_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build MLP
        layers = []
        in_dim = head_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        # Output head
        if task_type == "reg":
            layers.append(nn.Linear(in_dim, 1))
        elif task_type == "binary":
            layers.append(nn.Linear(in_dim, 1))
        elif task_type == "multi":
            assert n_classes is not None and n_classes >= 2
            layers.append(nn.Linear(in_dim, n_classes))
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        self.mlp = nn.Sequential(*layers)

    def forward(self, id_A, id_B, fracA, fracB, meta=None):
        """
        Parameters
        ----------
        id_A, id_B : LongTensor [B]
        fracA, fracB : FloatTensor [B]
        meta : FloatTensor [B, meta_dim] or None
        """
        e_A = self.embedding(id_A)   # [B, D]
        e_B = self.embedding(id_B)   # [B, D]

        if self.mode == "mix":
            z = fracA.unsqueeze(1) * e_A + fracB.unsqueeze(1) * e_B  # [B, D]
            if meta is not None:
                z = torch.cat([z, meta], dim=1)
        elif self.mode == "mean":
            z = (e_A + e_B) / 2.0  # [B, D]
            if meta is not None:
                z = torch.cat([z, meta], dim=1)
        elif self.mode == "interact":
            diff = torch.abs(e_A - e_B)
            prod = e_A * e_B
            z = torch.cat([
                e_A, e_B, diff, prod,
                fracA.unsqueeze(1), fracB.unsqueeze(1),
            ], dim=1)
            if meta is not None:
                z = torch.cat([z, meta], dim=1)

        return self.mlp(z)


# ========================== EVALUATION =====================================

def eval_regression(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def eval_classification(y_true, y_pred, y_proba, task_type, all_labels=None):
    """Evaluate classification metrics matching train_graph.py conventions."""
    out = {"acc": accuracy_score(y_true, y_pred)}
    if task_type == "binary":
        out["f1_macro"] = f1_score(y_true, y_pred, zero_division=0)  # Use f1_macro for consistency
        if y_proba is not None:
            try:
                out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception as e:
                logger.warning(f"Failed to calculate roc_auc: {e}")
                out["roc_auc"] = float('nan')
            try:
                out["logloss"] = log_loss(y_true, y_proba, labels=all_labels)
            except Exception as e:
                logger.warning(f"Failed to calculate logloss: {e}")
                out["logloss"] = float('nan')
        else:
            out["roc_auc"] = float('nan')
            out["logloss"] = float('nan')
    else:  # multiclass
        out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if y_proba is not None:
            try:
                out["logloss"] = log_loss(y_true, y_proba, labels=all_labels)
            except Exception as e:
                logger.warning(f"Failed to calculate logloss: {e}")
                out["logloss"] = float('nan')
            try:
                # Multi-class ROC-AUC using one-vs-rest
                out["roc_auc"] = roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
            except Exception as e:
                logger.warning(f"Failed to calculate roc_auc: {e}")
                out["roc_auc"] = float('nan')
        else:
            out["roc_auc"] = float('nan')
            out["logloss"] = float('nan')
    return out


# ========================== TRAINING LOOP ==================================

def train_one_split(
    model, train_loader, val_loader, test_loader,
    task_type, lr, epochs, patience, device,
    target_scaler=None, n_classes=None, all_labels=None,
):
    """Train for one split, return test metrics dict."""
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if task_type == "reg":
        criterion = nn.MSELoss()
    elif task_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multi":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            id_A, id_B, fA, fB, meta, y = [b.to(device) if b is not None else None for b in batch]
            optimizer.zero_grad()
            out = model(id_A, id_B, fA, fB, meta)

            if task_type == "reg":
                loss = criterion(out.squeeze(-1), y.float())
            elif task_type == "binary":
                loss = criterion(out.squeeze(-1), y.float())
            else:
                loss = criterion(out, y.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            n_train += len(y)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                id_A, id_B, fA, fB, meta, y = [b.to(device) if b is not None else None for b in batch]
                out = model(id_A, id_B, fA, fB, meta)

                if task_type == "reg":
                    loss = criterion(out.squeeze(-1), y.float())
                elif task_type == "binary":
                    loss = criterion(out.squeeze(-1), y.float())
                else:
                    loss = criterion(out, y.long())

                val_loss += loss.item() * len(y)
                n_val += len(y)

        val_loss /= max(n_val, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # ---- Test evaluation ----
    all_preds = []
    all_proba = []
    all_y = []
    with torch.no_grad():
        for batch in test_loader:
            id_A, id_B, fA, fB, meta, y = [b.to(device) if b is not None else None for b in batch]
            out = model(id_A, id_B, fA, fB, meta)

            if task_type == "reg":
                preds = out.squeeze(-1).cpu().numpy()
            elif task_type == "binary":
                proba = torch.sigmoid(out.squeeze(-1)).cpu().numpy()
                preds = (proba >= 0.5).astype(int)
                all_proba.append(np.stack([1 - proba, proba], axis=1))
            else:
                proba = torch.softmax(out, dim=1).cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_proba.append(proba)

            all_preds.append(preds)
            all_y.append(y.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba) if all_proba else None

    if task_type == "reg":
        # Inverse-transform predictions if target was scaled
        if target_scaler is not None:
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        metrics = eval_regression(y_true, y_pred)
    else:
        metrics = eval_classification(y_true, y_pred, y_proba, task_type, all_labels=all_labels)

    return metrics, y_true, y_pred


# ========================== DATASET / DATALOADER ===========================

class CopolymerIdentityDataset(torch.utils.data.Dataset):
    """Simple dataset returning (id_A, id_B, fracA, fracB, meta, y)."""

    def __init__(self, id_A, id_B, fracA, fracB, meta, y):
        self.id_A = torch.LongTensor(id_A)
        self.id_B = torch.LongTensor(id_B)
        self.fracA = torch.FloatTensor(fracA)
        self.fracB = torch.FloatTensor(fracB)
        self.meta = torch.FloatTensor(meta) if meta is not None else None
        self.y = torch.FloatTensor(y) if y.dtype in [np.float64, np.float32] else torch.LongTensor(y)

    def __len__(self):
        return len(self.id_A)

    def __getitem__(self, idx):
        meta_item = self.meta[idx] if self.meta is not None else None
        return self.id_A[idx], self.id_B[idx], self.fracA[idx], self.fracB[idx], meta_item, self.y[idx]


def collate_fn(batch):
    """Custom collate to handle optional meta=None."""
    id_A = torch.stack([b[0] for b in batch])
    id_B = torch.stack([b[1] for b in batch])
    fA = torch.stack([b[2] for b in batch])
    fB = torch.stack([b[3] for b in batch])
    meta = torch.stack([b[4] for b in batch]) if batch[0][4] is not None else None
    y = torch.stack([b[5] for b in batch])
    return id_A, id_B, fA, fB, meta, y


# ========================== VOCABULARY =====================================

def build_vocab(smiles_A_series, smiles_B_series):
    """Build monomer vocabulary from all unique SMILES in A and B columns.
    
    Returns
    -------
    vocab : dict  {smiles_string: int_id}
        ID 0 is reserved for unknown/padding, so IDs start at 1.
    """
    all_smiles = set(smiles_A_series.dropna().unique()) | set(smiles_B_series.dropna().unique())
    all_smiles = sorted(all_smiles)  # deterministic ordering
    vocab = {smi: i + 1 for i, smi in enumerate(all_smiles)}
    return vocab


def encode_smiles(series, vocab):
    """Map a pandas Series of SMILES strings to integer IDs via vocab."""
    return np.array([vocab.get(str(s), 0) for s in series], dtype=np.int64)


# ========================== ARGUMENT PARSER ================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Identity-embedding baseline for copolymer prediction (no chemistry features)."
    )
    # Dataset
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset file (without .csv)")
    parser.add_argument("--task_type", type=str, choices=["reg", "binary", "multi"],
                        default="reg", help="Task type")
    parser.add_argument("--targets", type=str, nargs='+', default=None,
                        help="Target column names to train on")

    # Polymer
    parser.add_argument("--polymer_type", type=str, default="copolymer",
                        choices=["homo", "copolymer"],
                        help="Must be copolymer for this baseline")

    # Descriptors (meta features)
    parser.add_argument("--incl_desc", action="store_true",
                        help="Include dataset-specific descriptor columns as meta features")
    parser.add_argument("--incl_poly_type", action="store_true",
                        help="Include one-hot encoding of poly_type column (ea_ip only)")

    # Mode (use --copolymer_mode to match other graph models)
    parser.add_argument("--copolymer_mode", type=str, choices=["mix", "mean", "interact"], default="mix",
                        help="Composition mode: mix (weighted), mean (unweighted average), or interact")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Dimensionality of monomer identity embeddings")

    # MLP architecture
    parser.add_argument("--hidden_dims", type=str, default="256,128",
                        help="Comma-separated hidden layer sizes for MLP head")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in MLP")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    # Training
    parser.add_argument("--train_size", type=str, default=None,
                        help="Subsample training set to this size")

    # Split type
    parser.add_argument("--split_type", type=str, choices=["random", "a_held_out"],
                        default="random",
                        help="Split strategy: random (default) or a_held_out (group by smiles_A)")

    # Compatibility flags (silently accepted for batch script compatibility)
    parser.add_argument("--incl_rdkit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--incl_ab", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch_norm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export_embeddings", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save_predictions", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save_checkpoint", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fusion_mode", type=str, default="late_concat", help=argparse.SUPPRESS)
    parser.add_argument("--aux_task", type=str, default="off", help=argparse.SUPPRESS)
    parser.add_argument("--aux_descriptor_cols", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_aux", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--film_layers", type=str, default="all", help=argparse.SUPPRESS)
    parser.add_argument("--film_hidden_dim", type=int, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


# ========================== MAIN ===========================================

def main():
    args = parse_args()

    if args.polymer_type != "copolymer":
        raise ValueError("This baseline is designed for copolymer datasets only.")

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    # ---- Setup using shared utilities ----
    setup_info = setup_training_environment(args, model_type="tabular")
    SEED = setup_info["SEED"]
    REPLICATES = setup_info["REPLICATES"]
    results_dir = setup_info["results_dir"]
    descriptor_columns = setup_info["descriptor_columns"]

    # Use graph-style epochs/patience from config
    config = setup_info["config"]
    GLOBAL_CONFIG = config.get("GLOBAL", {})
    EPOCHS = GLOBAL_CONFIG.get("EPOCHS", 300)
    PATIENCE = GLOBAL_CONFIG.get("PATIENCE", 30)

    set_seed(SEED)

    if _HAS_POLYMER_INPUT:
        logger.info("polymer_input package available — PolymerSpec validation enabled")
    else:
        logger.debug("polymer_input package not installed — using standard copolymer flow only")

    # ---- Load and preprocess data (reuses copolymer handling) ----
    # If --incl_poly_type, temporarily remove poly_type from dataset_ignore so it
    # survives load_and_preprocess_data; we'll handle exclusion from targets manually.
    if args.incl_poly_type:
        ds_ignore = config.get('DATASET_IGNORE', {}).get(args.dataset_name, []) or []
        if "poly_type" in ds_ignore:
            ds_ignore_filtered = [c for c in ds_ignore if c != "poly_type"]
            config['DATASET_IGNORE'][args.dataset_name] = ds_ignore_filtered

    df_input, target_columns = load_and_preprocess_data(args, setup_info)

    # Remove poly_type from target columns if it survived (identity baseline uses it as a feature)
    if args.incl_poly_type and "poly_type" in target_columns:
        target_columns = [c for c in target_columns if c != "poly_type"]

    # ---- Filter targets if --targets is specified ----
    if args.targets:
        requested = [t.strip() for t in args.targets]
        valid = [t for t in requested if t in target_columns]
        if not valid:
            raise ValueError(f"None of the requested targets {requested} found in dataset. Available: {target_columns}")
        target_columns = valid
        logger.info(f"Training on specified targets: {target_columns}")
    else:
        logger.info(f"Training on all detected targets: {target_columns}")

    # ---- Build monomer vocabulary ----
    vocab = build_vocab(df_input["smilesA"], df_input["smilesB"])
    num_monomers = len(vocab)
    logger.info(f"Monomer vocabulary size: {num_monomers}")

    # Encode SMILES to integer IDs
    all_id_A = encode_smiles(df_input["smilesA"], vocab)
    all_id_B = encode_smiles(df_input["smilesB"], vocab)
    all_fracA = df_input["fracA"].values.astype(np.float32)
    all_fracB = df_input["fracB"].values.astype(np.float32)

    # ---- Meta features (descriptors) ----
    if descriptor_columns:
        meta_raw = df_input[descriptor_columns].values.astype(np.float32)
        # Impute NaNs with column means
        col_means = np.nanmean(meta_raw, axis=0)
        for c in range(meta_raw.shape[1]):
            mask = np.isnan(meta_raw[:, c])
            meta_raw[mask, c] = col_means[c]
        meta_dim = meta_raw.shape[1]
        logger.info(f"Using {meta_dim} descriptor columns as meta features: {descriptor_columns}")
    else:
        meta_raw = None
        meta_dim = 0
        logger.info("No descriptor (meta) features.")

    # ---- One-hot encoding of poly_type (ea_ip only) ----
    poly_type_classes = None
    if args.incl_poly_type:
        poly_type_vals = df_input["poly_type"].astype(str).values
        poly_type_classes = sorted(set(poly_type_vals))
        pt_to_idx = {c: i for i, c in enumerate(poly_type_classes)}
        n_poly_types = len(poly_type_classes)
        poly_type_onehot = np.zeros((len(df_input), n_poly_types), dtype=np.float32)
        for row_i, pt in enumerate(poly_type_vals):
            poly_type_onehot[row_i, pt_to_idx[pt]] = 1.0
        logger.info(f"One-hot encoded poly_type: {n_poly_types} classes {poly_type_classes}")

        # Concatenate to meta features
        if meta_raw is not None:
            meta_raw = np.concatenate([meta_raw, poly_type_onehot], axis=1)
        else:
            meta_raw = poly_type_onehot
        meta_dim = meta_raw.shape[1]
        logger.info(f"Meta features after poly_type concat: {meta_dim} dims")

    # ---- Save vocab ----
    out_base = results_dir / "IdentityBaseline"
    out_base.mkdir(parents=True, exist_ok=True)
    vocab_path = out_base / f"{args.dataset_name}_monomer_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"Saved monomer vocabulary to {vocab_path}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Per-target training ----
    all_results = []

    for target in target_columns:
        logger.info(f"\n{'='*60}")
        logger.info(f"Target: {target}")
        logger.info(f"{'='*60}")

        y_all = df_input[target].values

        # Filter valid samples (non-NaN target)
        if args.task_type == "reg":
            valid_mask = ~np.isnan(y_all.astype(float))
        else:
            valid_mask = ~pd.isna(y_all)

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            logger.warning(f"No valid samples for target {target}. Skipping.")
            continue

        y_valid = y_all[valid_idx]
        idA_valid = all_id_A[valid_idx]
        idB_valid = all_id_B[valid_idx]
        fA_valid = all_fracA[valid_idx]
        fB_valid = all_fracB[valid_idx]
        meta_valid = meta_raw[valid_idx] if meta_raw is not None else None
        df_valid = df_input.iloc[valid_idx].reset_index(drop=True)

        # n_classes for classification
        n_classes = None
        all_labels = None
        if args.task_type == "multi":
            all_labels = np.sort(np.unique(y_valid))
            n_classes = len(all_labels)
            # Remap labels to 0..n_classes-1 for CrossEntropyLoss
            label_map = {lab: i for i, lab in enumerate(all_labels)}
            y_valid = np.array([label_map[v] for v in y_valid])
            logger.info(f"[{target}] {n_classes} classes, labels: {all_labels.tolist()}")
        elif args.task_type == "binary":
            n_classes = 2
            all_labels = np.array([0, 1])
            y_valid = y_valid.astype(np.float32)
        else:
            y_valid = y_valid.astype(np.float32)

        # ---- Splits ----
        n_splits, local_reps = determine_split_strategy(len(y_valid), REPLICATES)

        train_indices, val_indices, test_indices = [], [], []
        if args.split_type == "a_held_out":
            # A-held-out: GroupKFold by canonicalized smiles_A
            sA_col = "smilesA" if "smilesA" in df_valid.columns else "smiles_A"
            valid_smiles_A = df_valid[sA_col].astype(str).values
            n_splits = 5  # enforce 5-fold for a_held_out
            train_indices, val_indices, test_indices = generate_a_held_out_splits(
                valid_smiles_A, len(y_valid), SEED, n_splits=n_splits, logger=logger,
            )
            save_fold_assignments(
                train_indices, val_indices, test_indices,
                valid_smiles_A, args.dataset_name, SEED, results_dir, logger=logger,
            )
        else:
            for r in range(local_reps):
                tr, va, te = group_splits(df_valid, y_valid, args.task_type, n_splits, SEED + r)
                train_indices.extend(tr)
                val_indices.extend(va)
                test_indices.extend(te)

        num_splits = len(train_indices)
        logger.info(f"[{target}] {num_splits} splits (n_splits={n_splits}, reps={local_reps})")

        target_results = []

        for i in range(num_splits):
            tr = train_indices[i]
            va = val_indices[i]
            te = test_indices[i]

            # Apply train_size subsampling
            if args.train_size is not None and args.train_size.lower() != "full":
                target_size = int(args.train_size)
                if target_size < len(tr):
                    rng = np.random.default_rng(SEED + i)
                    tr = rng.choice(tr, size=target_size, replace=False)
                    logger.info(f"  Split {i}: Subsampled training set to {target_size}")

            # ---- Scale meta features ----
            meta_tr = meta_valid[tr] if meta_valid is not None else None
            meta_va = meta_valid[va] if meta_valid is not None else None
            meta_te = meta_valid[te] if meta_valid is not None else None

            if meta_tr is not None:
                meta_scaler = StandardScaler()
                meta_tr = meta_scaler.fit_transform(meta_tr)
                meta_va = meta_scaler.transform(meta_va)
                meta_te = meta_scaler.transform(meta_te)

            # ---- Scale targets for regression ----
            target_scaler = None
            y_tr = y_valid[tr].copy()
            y_va = y_valid[va].copy()
            y_te_orig = y_valid[te].copy()

            if args.task_type == "reg":
                target_scaler = StandardScaler()
                y_tr = target_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
                y_va = target_scaler.transform(y_va.reshape(-1, 1)).flatten()
                # y_te stays in original scale for evaluation
                y_te = y_te_orig
            else:
                y_te = y_te_orig

            # ---- Build datasets and loaders ----
            train_ds = CopolymerIdentityDataset(idA_valid[tr], idB_valid[tr], fA_valid[tr], fB_valid[tr], meta_tr, y_tr)
            val_ds = CopolymerIdentityDataset(idA_valid[va], idB_valid[va], fA_valid[va], fB_valid[va], meta_va, y_va)
            test_ds = CopolymerIdentityDataset(idA_valid[te], idB_valid[te], fA_valid[te], fB_valid[te], meta_te, y_te)

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

            # ---- Build model ----
            model = IdentityBaselineModel(
                num_monomers=num_monomers,
                embed_dim=args.embed_dim,
                mode=args.copolymer_mode,
                meta_dim=meta_dim,
                hidden_dims=hidden_dims,
                task_type=args.task_type,
                n_classes=n_classes,
                dropout=args.dropout,
            )

            logger.info(f"  Split {i}: train={len(tr)}, val={len(va)}, test={len(te)}, "
                        f"params={sum(p.numel() for p in model.parameters()):,}")

            # ---- Train and evaluate ----
            # For classification, pass original labels for log_loss
            cls_labels = all_labels if args.task_type != "reg" else None

            metrics, y_true_split, y_pred_split = train_one_split(
                model, train_loader, val_loader, test_loader,
                task_type=args.task_type,
                lr=args.lr,
                epochs=EPOCHS,
                patience=PATIENCE,
                device=device,
                target_scaler=target_scaler,
                n_classes=n_classes,
                all_labels=cls_labels,
            )
            metrics["split"] = i
            target_results.append(metrics)

            # Save predictions if requested
            if args.save_predictions:
                pred_out_dir = results_dir.parent / "predictions" / "IdentityBaseline"
                pred_out_dir.mkdir(parents=True, exist_ok=True)
                mode_sfx = f"__copoly_{args.copolymer_mode}"
                pt_sfx = "__poly_type" if args.incl_poly_type else ""
                split_sfx = f"__{args.split_type}" if args.split_type != "random" else ""
                fname = f"{args.dataset_name}__{target}{mode_sfx}{pt_sfx}{split_sfx}__split{i}.npz"
                np.savez_compressed(
                    pred_out_dir / fname,
                    y_true=y_true_split,
                    y_pred=y_pred_split,
                    test_indices=np.array(te),
                )
                logger.info(f"Saved predictions → {pred_out_dir / fname}")

            # Log split results
            metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items() if k != "split"]
            logger.info(f"  Split {i}: {', '.join(metric_strs)}")

        # Aggregate and save results for this target
        if target_results:
            results_df = pd.DataFrame(target_results)
            numeric_cols = [c for c in results_df.columns if c != "split"]
            mean_metrics = results_df[numeric_cols].mean()
            std_metrics = results_df[numeric_cols].std()
            logger.info(f"\n[{target}] Mean across {num_splits} splits:")
            for c in numeric_cols:
                logger.info(f"  {c}: {mean_metrics[c]:.4f} ± {std_metrics[c]:.4f}")
            results_df["target"] = target
            
            # Save separate file per target (matches graph model behavior)
            model_name = "IdentityBaseline"
            filename_parts = [args.dataset_name]
            if args.incl_desc:
                filename_parts.append("desc")
            # Use copoly_{mode} to match other graph models, not identity_{mode}
            filename_parts.append(f"copoly_{args.copolymer_mode}")
            if args.incl_poly_type:
                filename_parts.append("poly_type")
            if args.split_type != "random":
                filename_parts.append(args.split_type)
            if args.train_size and args.train_size.lower() != "full":
                filename_parts.append(f"size{args.train_size}")
            # Add target suffix for per-target files
            filename_parts.append(f"target_{target}")
            filename = "__".join(filename_parts) + "_results.csv"

            out_dir = results_dir / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            results_df.to_csv(out_path, index=False)
            logger.info(f"✓ Saved results to {out_path}")
            
            all_results.append(results_df)

    # ---- Save combined results (optional, for convenience) ----
    if all_results and len(target_columns) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        
        model_name = "IdentityBaseline"
        filename_parts = [args.dataset_name]
        if args.incl_desc:
            filename_parts.append("desc")
        filename_parts.append(f"copoly_{args.copolymer_mode}")
        if args.incl_poly_type:
            filename_parts.append("poly_type")
        if args.split_type != "random":
            filename_parts.append(args.split_type)
        if args.train_size and args.train_size.lower() != "full":
            filename_parts.append(f"size{args.train_size}")
        filename = "__".join(filename_parts) + "_results.csv"

        out_dir = results_dir / model_name
        out_path = out_dir / filename
        combined.to_csv(out_path, index=False)
        logger.info(f"\n✓ Saved combined results to {out_path}")

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("Training Complete - Summary Statistics")
        logger.info(f"{'='*70}")
        for col in combined.columns:
            if col not in ["split", "target"]:
                try:
                    logger.info(f"  {col}: {combined[col].mean():.4f} ± {combined[col].std():.4f}")
                except (TypeError, ValueError):
                    pass
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
