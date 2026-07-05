"""
wDMPNN Generalization Experiments
==================================
Trains wDMPNN on ea_ip with group_disjoint and pair_disjoint splits.

These splits are not natively supported by train_graph.py, so this script
handles split generation internally and trains wDMPNN through the standard
homopolymer path (PolymerMolGraphFeaturizer + WDMPNN_Input column).

Saves per-fold:
  - y_true, y_pred (real scale)
  - test_indices (row indices into ea_ip.csv)
  - Sample metadata: smiles_A, smiles_B, fracA, fracB, poly_type

Output:
  predictions/wDMPNN_Gen/ea_ip__{target}__wDMPNN__{split_type}__fold{i}.npz

Usage:
    python run_wdmpnn_generalization.py [--dry_run] [--folds 0,1,2,3,4]
        [--split_types group_disjoint,pair_disjoint]
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from chemprop import data, featurizers, nn
from chemprop import models
from chemprop.data import PolymerDatapoint, PolymerDataset, build_dataloader
from utils import (
    set_seed, get_metric_list, pick_best_checkpoint,
    canonicalize_smiles,
)
from run_stage2d_generalization import (
    generate_group_disjoint_splits, generate_pair_disjoint_splits,
    build_group_keys, build_pair_keys,
    verify_no_leakage, verify_pair_disjoint_extra,
)

# ── Configuration ────────────────────────────────────────────────────
DATA_PATH = ROOT_DIR / 'data' / 'ea_ip.csv'
PREDICTIONS_DIR = ROOT_DIR / 'predictions' / 'wDMPNN_Gen'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints' / 'wDMPNN_Gen'

TARGETS = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
N_FOLDS = 5
SEED = 42
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def build_wdmpnn_model(n_targets=1):
    """Build a wDMPNN model with standard configuration."""
    mp = nn.WeightedBondMessagePassing()
    agg = nn.WeightedMeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim)
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)
    return mpnn


def train_wdmpnn_fold(df, smis_wdmpnn, target, train_idx, val_idx, test_idx,
                      fold_idx, split_type):
    """Train wDMPNN for a single fold and return predictions + metadata."""

    # Extract target values
    ys = df[target].astype(float).values.reshape(-1, 1)

    # Create datapoints using WDMPNN_Input SMILES
    # wDMPNN uses PolymerDatapoint.from_smi() which parses "SMILES|w1|w2|...<edge1<edge2..."
    # and PolymerDataset with PolymerMolGraphFeaturizer
    polymer_featurizer = featurizers.PolymerMolGraphFeaturizer()
    all_datapoints = [PolymerDatapoint.from_smi(smi, y) for smi, y in zip(smis_wdmpnn, ys)]

    # Split data
    train_data = [all_datapoints[i] for i in train_idx]
    val_data = [all_datapoints[i] for i in val_idx]
    test_data = [all_datapoints[i] for i in test_idx]

    train_ds = PolymerDataset(train_data, polymer_featurizer)
    val_ds = PolymerDataset(val_data, polymer_featurizer)
    test_ds = PolymerDataset(test_data, polymer_featurizer)

    # Normalize targets (fit on train)
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)

    # Build checkpoint path
    target_short = target.replace(' ', '_').replace('(', '').replace(')', '')
    ckpt_path = CHECKPOINT_DIR / f'ea_ip__{target_short}__wDMPNN__{split_type}__fold{fold_idx}'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Metrics
    metric_list = get_metric_list('reg', target=target)

    # Build model (wDMPNN uses weighted message passing and aggregation)
    mp = nn.WeightedBondMessagePassing()
    agg = nn.WeightedMeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, output_transform=output_transform)
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)

    # Trainer
    from lightning import pytorch as pl
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        enable_progress_bar=True,
        accelerator='auto',
        devices=1,
        default_root_dir=str(ckpt_path / 'logs'),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min'),
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_path / 'logs' / 'checkpoints'),
                monitor='val_loss', mode='min', save_top_k=1,
            ),
        ],
        logger=pl.loggers.TensorBoardLogger(str(ckpt_path), name='logs'),
    )

    # Dataloaders
    train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    val_loader = build_dataloader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
    test_loader = build_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)

    # Check for existing training
    done_flag = ckpt_path / "TRAINING_COMPLETE"

    if done_flag.exists():
        best_ckpt_path, _ = pick_best_checkpoint(ckpt_path)
        if best_ckpt_path is not None:
            logger.info(f"  Skipping training (COMPLETE): {ckpt_path.name}")
            map_location = None if torch.cuda.is_available() else torch.device("cpu")
            checkpoint = torch.load(best_ckpt_path, map_location=map_location, weights_only=False)
            mpnn.load_state_dict(checkpoint['state_dict'])
            if torch.cuda.is_available():
                mpnn = mpnn.to(torch.device("cuda"))
            mpnn.eval()
        else:
            logger.warning(f"  TRAINING_COMPLETE but no checkpoint. Retraining.")
            done_flag.unlink()

    if not done_flag.exists():
        inprog_flag = ckpt_path / "TRAINING_IN_PROGRESS"
        inprog_flag.touch(exist_ok=True)
        try:
            trainer.fit(mpnn, train_loader, val_loader)
            best_ckpt_path, best_val_loss = pick_best_checkpoint(ckpt_path)
            if best_ckpt_path:
                with open(ckpt_path / "best.json", "w") as f:
                    json.dump({"best_ckpt": str(best_ckpt_path), "best_val_loss": best_val_loss}, f, indent=2)
                done_flag.touch()
        finally:
            if inprog_flag.exists():
                inprog_flag.unlink(missing_ok=True)

    # Get predictions (real scale via output_transform)
    y_pred = trainer.predict(model=mpnn, dataloaders=test_loader)
    y_true = np.array([test_ds[j].y for j in range(len(test_ds))], dtype=float)

    if isinstance(y_pred, list):
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    elif hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()

    return y_true.flatten(), y_pred.flatten()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='wDMPNN Generalization Experiments')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only generate and verify splits, do not train')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated fold indices (default: all)')
    parser.add_argument('--split_types', type=str, default='group_disjoint,pair_disjoint',
                       help='Comma-separated split types')
    parser.add_argument('--targets', type=str, default=None,
                       help='Comma-separated target names (default: all)')
    cli_args = parser.parse_args()

    folds_to_run = [int(x) for x in cli_args.folds.split(',')]
    split_types = [x.strip() for x in cli_args.split_types.split(',')]
    targets = TARGETS if cli_args.targets is None else [x.strip() for x in cli_args.targets.split(',')]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("wDMPNN GENERALIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"  Split types: {split_types}")
    print(f"  Targets: {targets}")
    print(f"  Folds: {folds_to_run}")
    print(f"  Dry run: {cli_args.dry_run}")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows")

    smis_wdmpnn = df['WDMPNN_Input'].astype(str).values
    group_keys = build_group_keys(df)
    pair_keys = build_pair_keys(df)

    # ── Generate and verify splits ────────────────────────────────────
    splits = {}
    for split_type in split_types:
        print(f"\n{'─'*70}")
        print(f"SPLIT TYPE: {split_type}")
        print(f"{'─'*70}")

        if split_type == 'group_disjoint':
            train_indices, val_indices, test_indices = generate_group_disjoint_splits(
                df, n_splits=N_FOLDS, seed=SEED
            )
            verify_keys = group_keys
        elif split_type == 'pair_disjoint':
            train_indices, val_indices, test_indices = generate_pair_disjoint_splits(
                df, n_splits=N_FOLDS, seed=SEED
            )
            verify_keys = pair_keys
        else:
            raise ValueError(f"Unknown split type: {split_type}. Use group_disjoint or pair_disjoint.")

        for fold_i in range(N_FOLDS):
            verify_no_leakage(
                train_indices[fold_i], val_indices[fold_i], test_indices[fold_i],
                verify_keys, fold_i, split_type
            )
            if split_type == 'pair_disjoint':
                verify_pair_disjoint_extra(
                    train_indices[fold_i], val_indices[fold_i], test_indices[fold_i],
                    pair_keys, fold_i
                )

        splits[split_type] = (train_indices, val_indices, test_indices)
        print(f"  ✓ All {N_FOLDS} folds verified for {split_type}")

    if cli_args.dry_run:
        print("\n[DRY RUN] Split verification complete. Exiting without training.")
        return

    # ── Training loop ────────────────────────────────────────────────
    for target in targets:
        for split_type in split_types:
            train_indices, val_indices, test_indices = splits[split_type]

            for fold_idx in folds_to_run:
                tr = train_indices[fold_idx]
                va = val_indices[fold_idx]
                te = test_indices[fold_idx]

                logger.info(
                    f"\n  Training: wDMPNN | {split_type} | {target} | fold={fold_idx} | "
                    f"n_train={len(tr)} | n_val={len(va)} | n_test={len(te)}"
                )

                y_true, y_pred = train_wdmpnn_fold(
                    df, smis_wdmpnn, target,
                    tr, va, te, fold_idx, split_type,
                )

                # Save predictions with full metadata
                pred_file = (PREDICTIONS_DIR /
                            f'ea_ip__{target}__wDMPNN__{split_type}__fold{fold_idx}.npz')
                np.savez_compressed(
                    pred_file,
                    y_true=y_true,
                    y_pred=y_pred,
                    test_indices=te,
                    split_type=split_type,
                    fold=fold_idx,
                    n_train=len(tr),
                    n_val=len(va),
                    n_test=len(te),
                    # Metadata for Stage 2D analysis
                    smiles_A=df.iloc[te]['smiles_A'].values,
                    smiles_B=df.iloc[te]['smiles_B'].values,
                    fracA=df.iloc[te]['fracA'].values,
                    fracB=df.iloc[te]['fracB'].values,
                    poly_type=df.iloc[te]['poly_type'].values,
                )
                logger.info(f"    Saved: {pred_file.name}")

    print("\n" + "=" * 70)
    print("wDMPNN GENERALIZATION EXPERIMENTS COMPLETE")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    set_seed(SEED)
    main()
