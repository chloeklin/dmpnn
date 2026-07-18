"""
HPG Generalization Experiments — 2×2 grid launcher
====================================================
Runs all combinations of {hpg_variant} × {chain_edge_mode} by calling
train_graph.py as a subprocess for each cell:

  hpg_variant    : baseline (sum pooling), frac (fraction-weighted pooling)
  chain_edge_mode: degree (bidirectional, weight=1), stochastic (Markov transitions)

Usage
-----
    python run_hpg_generalization.py --dataset_name ea_ip \\
        --split_type a_held_out \\
        --targets "EA vs SHE (eV)" "IP vs SHE (eV)" \\
        [--pooling_types baseline frac] \\
        [--chain_edge_modes degree stochastic] \\
        [--dry_run]

Additional train_graph.py flags can be passed through --extra_args, e.g.:
    python run_hpg_generalization.py --dataset_name ea_ip \\
        --extra_args "--incl_desc" "--n_splits 5"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from chemprop.data.hpg import HPGDatapoint, HPGDataset, hpg_collate_fn
from chemprop.data.hpg_hier import TwoStageHPGDatapoint, TwoStageHPGDataset, two_stage_hpg_collate_fn
from chemprop.featurizers.molgraph.hpg import HPG_ATOM_FDIM, HPGMolGraphFeaturizer
from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGFeaturizer
from chemprop.models.hpg import HPGMPNN
from chemprop.models.hpg_hier import HPGHierMPNN
from chemprop.nn.transforms import UnscaleTransform
from evaluation.naming import make_prediction_filename, split_subdir, standard_model_name, standard_split_name, standard_target_token
from run_stage2d_generalization import build_group_keys, build_pair_keys, generate_group_disjoint_splits, generate_pair_disjoint_splits, verify_no_leakage, verify_pair_disjoint_extra
from utils import generate_a_held_out_splits, set_seed

DATA_PATH = ROOT_DIR / "data" / "ea_ip.csv"
META_PATH = ROOT_DIR / "metadata" / "splits" / "monomer_heldout.json"
PREDICTIONS_DIR = ROOT_DIR / "predictions"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "HPG_Gen"
TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
MODEL_TO_POOLING = {
    "hpg_sum": "sum",
    "hpg_frac": "frac_weighted",
    "hpg_hier": "hpg_hier",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPG generalization experiments")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--folds", default=None)
    parser.add_argument("--split_types", default="group_disjoint,pair_disjoint,monomer_heldout")
    parser.add_argument("--models", default="hpg_sum,hpg_frac")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--stage1_pool", choices=("sum", "mean", "attention"), default="sum")
    parser.add_argument("--stage2_depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--stage2_edge", choices=("full", "transition_only", "junction_only"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_lomao_splits(df: pd.DataFrame):
    train, val, test, _ = generate_a_held_out_splits(
        df["smiles_A"].astype(str).values, len(df), seed=42,
        protocol="leave_one_A_out", logger=logger,
    )
    metadata = json.loads(META_PATH.read_text())["folds"]
    if len(test) != len(metadata):
        raise AssertionError(f"Expected {len(metadata)} monomer-heldout folds, got {len(test)}")
    for fold, indices in enumerate(test):
        expected = np.asarray(metadata[fold]["global_test_indices"], dtype=int)
        if not np.array_equal(indices, expected):
            raise AssertionError(f"monomer_heldout fold {fold} test indices differ from metadata")
    return train, val, test


def _build_splits(df: pd.DataFrame, split_type: str):
    if split_type == "group_disjoint":
        keys = build_group_keys(df)
        splits = generate_group_disjoint_splits(df, n_splits=5, seed=42)
    elif split_type == "pair_disjoint":
        keys = build_pair_keys(df)
        splits = generate_pair_disjoint_splits(df, n_splits=5, seed=42)
    elif split_type == "monomer_heldout":
        return _load_lomao_splits(df)
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    train, val, test = splits
    for fold in range(len(train)):
        verify_no_leakage(train[fold], val[fold], test[fold], keys, fold, split_type)
        if split_type == "pair_disjoint":
            verify_pair_disjoint_extra(train[fold], val[fold], test[fold], keys, fold)
    return train, val, test


def _build_graphs(df: pd.DataFrame):
    featurizer = HPGMolGraphFeaturizer()
    graphs = [
        featurizer(
            [str(row.smiles_A), str(row.smiles_B)], [(0, 1, 1.0)],
            frag_fracs=np.asarray([row.fracA, row.fracB], dtype=np.float32),
        )
        for row in df.itertuples(index=False)
    ]
    if featurizer.d_v != HPG_ATOM_FDIM:
        raise AssertionError(f"Featurizer d_v={featurizer.d_v}, expected {HPG_ATOM_FDIM}")
    return graphs


def _dataset(graphs, values: np.ndarray, indices: np.ndarray) -> HPGDataset:
    return HPGDataset([
        HPGDatapoint(mg=graphs[i], y=np.asarray([values[i]], dtype=np.float32))
        for i in indices
    ])


def _build_hier_graphs(df: pd.DataFrame, stage2_edge: str):
    if "WDMPNN_Input" not in df:
        raise ValueError("hpg_hier requires the WDMPNN_Input column")
    featurizer = TwoStageHPGFeaturizer()
    return [featurizer(value, stage2_edge=stage2_edge) for value in df["WDMPNN_Input"].astype(str)]


def _train_hier_fold(graphs, values, train_idx, val_idx, test_idx, target, split_type, fold, args):
    set_seed(args.seed + fold)
    build_dataset = lambda indices: TwoStageHPGDataset([
        TwoStageHPGDatapoint(graphs[index], np.asarray([values[index]], dtype=np.float32))
        for index in indices
    ])
    train_ds, val_ds, test_ds = build_dataset(train_idx), build_dataset(val_idx), build_dataset(test_idx)
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    loaders = [
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=two_stage_hpg_collate_fn, num_workers=args.num_workers),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=two_stage_hpg_collate_fn, num_workers=args.num_workers),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=two_stage_hpg_collate_fn, num_workers=args.num_workers),
    ]
    checkpoint_path = CHECKPOINT_DIR / f"ea_ip__{standard_target_token(target)}__hpg_hier__{split_type}__fold{fold}__s{args.seed}"
    model = HPGHierMPNN(atom_fdim=75, bond_fdim=graphs[0].monomer_graphs[0].E.shape[1], d_h=128,
                         stage1_pool=args.stage1_pool, stage2_depth=args.stage2_depth)
    model._output_transform = UnscaleTransform.from_standard_scaler(scaler)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices=1, logger=False,
                         default_root_dir=str(checkpoint_path), enable_model_summary=False,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
                                    ModelCheckpoint(dirpath=str(checkpoint_path), monitor="val_loss", mode="min", save_top_k=1, save_last=True)])
    trainer.fit(model, loaders[0], loaders[1])
    batches = trainer.predict(model=model, dataloaders=loaders[2])
    return torch.cat([batch.detach().cpu() for batch in batches]).numpy().reshape(-1)


def _train_fold(graphs, values, train_idx, val_idx, test_idx, pooling_type, target, split_type, fold, args):
    set_seed(args.seed + fold)
    train_ds = _dataset(graphs, values, train_idx)
    val_ds = _dataset(graphs, values, val_idx)
    test_ds = _dataset(graphs, values, test_idx)
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=hpg_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=hpg_collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=hpg_collate_fn, num_workers=args.num_workers)
    checkpoint_path = CHECKPOINT_DIR / f"ea_ip__{standard_target_token(target)}__{pooling_type}__{split_type}__fold{fold}__s{args.seed}"
    model = HPGMPNN(d_v=HPG_ATOM_FDIM, d_e=1, d_h=128, d_ffn=64, depth=6, num_heads=8,
                    dropout_mp=0.0, dropout_ffn=0.2, n_tasks=1, pooling_type=pooling_type,
                    task_type="regression")
    model._output_transform = UnscaleTransform.from_standard_scaler(scaler)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1, logger=False,
        default_root_dir=str(checkpoint_path), enable_model_summary=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
            ModelCheckpoint(dirpath=str(checkpoint_path), monitor="val_loss", mode="min", save_top_k=1, save_last=True),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    batches = trainer.predict(model=model, dataloaders=test_loader)
    return torch.cat([batch.detach().cpu() for batch in batches]).numpy().reshape(-1)


def main() -> None:
    args = parse_args()
    split_types = [item.strip() for item in args.split_types.split(",")]
    models = [item.strip() for item in args.models.split(",")]
    targets = TARGETS if args.targets is None else [item.strip() for item in args.targets.split(",")]
    invalid = set(models) - set(MODEL_TO_POOLING)
    if invalid:
        raise ValueError(f"Unknown models: {sorted(invalid)}")
    requested_paths = []
    for split_type in split_types:
        n_folds = 9 if split_type == "monomer_heldout" else 5
        folds = list(range(n_folds)) if args.folds is None else [int(item) for item in args.folds.split(",")]
        if any(fold < 0 or fold >= n_folds for fold in folds):
            raise ValueError(f"Invalid fold requested for {split_type}: {folds}")
        for target in targets:
            for model_token in models:
                requested_paths.extend(
                    PREDICTIONS_DIR / split_subdir(split_type) /
                    make_prediction_filename(target, model_token, split_type, fold, seed=args.seed)
                    for fold in folds
                )
    if not args.force and all(path.exists() for path in requested_paths):
        logger.info("All requested predictions already exist; exiting without loading data.")
        return
    df = pd.read_csv(DATA_PATH)
    standard_graphs = _build_graphs(df) if any(model != "hpg_hier" for model in models) else None
    hier_graphs = _build_hier_graphs(df, args.stage2_edge) if "hpg_hier" in models else None
    split_sets = {split_type: _build_splits(df, split_type) for split_type in split_types}
    for split_type, (trains, vals, tests) in split_sets.items():
        folds = list(range(len(trains))) if args.folds is None else [int(item) for item in args.folds.split(",")]
        for target in targets:
            values = df[target].to_numpy(dtype=np.float32)
            for model_token in models:
                for fold in folds:
                    prediction_dir = PREDICTIONS_DIR / split_subdir(split_type)
                    prediction_path = prediction_dir / make_prediction_filename(target, model_token, split_type, fold, seed=args.seed)
                    if prediction_path.exists() and not args.force:
                        logger.info("Skipping existing prediction: %s", prediction_path)
                        continue
                    if args.dry_run:
                        logger.info("Dry run: %s %s %s fold=%d", model_token, target, split_type, fold)
                        continue
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    y_pred = (
                        _train_hier_fold(hier_graphs, values, trains[fold], vals[fold], tests[fold], target, split_type, fold, args)
                        if model_token == "hpg_hier" else
                        _train_fold(standard_graphs, values, trains[fold], vals[fold], tests[fold], MODEL_TO_POOLING[model_token], target, split_type, fold, args)
                    )
                    y_true = values[tests[fold]].astype(np.float64)
                    if y_pred.shape != y_true.shape:
                        raise AssertionError(f"Prediction shape {y_pred.shape} != target shape {y_true.shape}")
                    np.savez_compressed(
                        prediction_path, y_true=y_true, y_pred=y_pred.astype(np.float64), test_indices=tests[fold],
                        split_type=standard_split_name(split_type), model=standard_model_name(model_token),
                        target=standard_target_token(target), fold=fold, seed=args.seed,
                        n_train=len(trains[fold]), n_val=len(vals[fold]), n_test=len(tests[fold]),
                        prediction_scale="physical_units", smiles_A=df.iloc[tests[fold]]["smiles_A"].to_numpy(),
                        smiles_B=df.iloc[tests[fold]]["smiles_B"].to_numpy(), fracA=df.iloc[tests[fold]]["fracA"].to_numpy(),
                        fracB=df.iloc[tests[fold]]["fracB"].to_numpy(), poly_type=df.iloc[tests[fold]]["poly_type"].to_numpy(),
                    )
                    logger.info("Saved: %s", prediction_path)


if __name__ == "__main__":
    main()
