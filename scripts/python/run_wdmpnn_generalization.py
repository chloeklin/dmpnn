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
from chemprop.data.collate import collate_polymer_batch
from chemprop.data.samplers import GroupAwareSampler
from chemprop.nn.within_group_loss import within_group_residual_loss
from utils import (
    set_seed, get_metric_list, pick_best_checkpoint,
    canonicalize_smiles,
)
from run_stage2d_generalization import (
    generate_group_disjoint_splits, generate_pair_disjoint_splits,
    build_group_keys, build_pair_keys,
    verify_no_leakage, verify_pair_disjoint_extra,
)
from evaluation.naming import (
    make_prediction_filename, standard_model_name, standard_split_name,
    standard_target_token, split_subdir,
)

# ── Configuration ────────────────────────────────────────────────────
DATA_PATH = ROOT_DIR / 'data' / 'ea_ip.csv'
PREDICTIONS_DIR = ROOT_DIR / 'predictions'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints' / 'wDMPNN_Gen'

TARGETS = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
N_FOLDS = 5
SEED = 42
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 512

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _lambda_tag(lw: float) -> str:
    """Convert lambda_within to a filesystem-safe string (e.g. 0.03 → '0p03')."""
    return str(lw).replace('.', 'p')


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

class WDMPNNWithinGroupLoss(models.MPNN):
    """wDMPNN with normalized within-group residual loss.

    When lambda_within > 0 the total training loss is:
        L_total = L_overall + lambda_within * L_within

    Group IDs are injected into the last column of X_d in the training batch
    (by setting x_d = [group_id] on each training PolymerDatapoint before
    constructing the dataset).  They are stripped before the forward pass so
    the FFN always sees X_d=None, preserving exact backward compatibility with
    the original wDMPNN architecture.

    Validation and test dataloaders are unchanged (x_d=None).
    """

    def __init__(self, *args, lambda_within: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_within = lambda_within

    def training_step(self, batch, batch_idx):
        if self.lambda_within == 0.0:
            return super().training_step(batch, batch_idx)

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        targets_orig = targets.clone()

        group_ids = None
        X_d_model = None
        if X_d is not None:
            group_ids = X_d[:, -1].long()
            X_d_model = X_d[:, :-1] if X_d.shape[1] > 1 else None

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d_model)
        main_targets, aux_targets, main_mask, aux_mask, main_lt, main_gt = \
            self._split_targets(targets, mask, lt_mask, gt_mask)
        preds = self.predictor.train_step(Z)

        l_overall = self.criterion(preds, main_targets, main_mask, weights, main_lt, main_gt)

        l_within = torch.zeros(1, device=l_overall.device).squeeze()
        if group_ids is not None:
            l_within, _ = within_group_residual_loss(targets_orig, preds, group_ids)

        l_total = l_overall + self.lambda_within * l_within

        self.log("train_loss", self.criterion, batch_size=batch_size,
                 prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_loss_overall", l_overall.detach(), batch_size=batch_size,
                 prog_bar=False, on_epoch=True, on_step=False)
        self.log("train_loss_within", l_within.detach(), batch_size=batch_size,
                 prog_bar=False, on_epoch=True, on_step=False)

        return l_total


class _GroupBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that yields complete group-aware batches each epoch.

    Wraps :class:`GroupAwareSampler._make_batches` so the DataLoader calls
    ``next(iter(batch_sampler))`` to get a *list of indices* per batch, rather
    than a scalar index.  This prevents PyTorch's default :class:`BatchSampler`
    from slicing across GroupAwareSampler's internal batch boundaries.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        batch_size: int,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        self._impl = GroupAwareSampler(group_ids, batch_size, seed, drop_last)

    def __iter__(self):
        return iter(self._impl._make_batches())

    def __len__(self) -> int:
        return (
            self._impl._n_samples + self._impl.batch_size - 1
        ) // self._impl.batch_size


def build_wdmpnn_model(n_targets=1):
    """Build a wDMPNN model with standard configuration."""
    mp = nn.WeightedBondMessagePassing()
    agg = nn.WeightedMeanAggregation()
    ffn = nn.RegressionFFN(input_dim=mp.output_dim)
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)
    return mpnn


def train_wdmpnn_fold(df, smis_wdmpnn, target, train_idx, val_idx, test_idx,
                      fold_idx, split_type, lambda_within=0.0,
                      all_group_ids=None, results_subdir=None):
    """Train wDMPNN for a single fold and return predictions + metadata.

    Parameters
    ----------
    lambda_within : float
        Weight for the normalized within-group residual loss.  0.0 reproduces
        the original wDMPNN pipeline exactly.
    all_group_ids : np.ndarray | None
        Integer group IDs for all dataset rows (length = len(df)).
        Required when lambda_within > 0.
    results_subdir : str | None
        Override the checkpoint/predictions subdirectory name.  When None,
        uses the module-level CHECKPOINT_DIR / PREDICTIONS_DIR.
    """

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

    # Pre-featurize all graphs once (avoids re-running RDKit on every __getitem__ per epoch)
    train_ds.cache = True
    val_ds.cache = True
    test_ds.cache = True

    # Normalize targets (fit on train)
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)

    # Build checkpoint path (include lambda tag for non-default runs)
    target_short = target.replace(' ', '_').replace('(', '').replace(')', '')
    include_tag = (lambda_within > 0.0) or (results_subdir is not None)
    lambda_suffix = f"__lw{_lambda_tag(lambda_within)}" if include_tag else ""
    ckpt_base = (ROOT_DIR / 'checkpoints' / results_subdir) if results_subdir else CHECKPOINT_DIR
    ckpt_path = ckpt_base / f'ea_ip__{target_short}__wDMPNN__{split_type}__fold{fold_idx}{lambda_suffix}'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Metrics
    metric_list = get_metric_list('reg', target=target)

    # Set group_id on training datapoints for GroupAwareSampler + L_within
    train_group_ids = None
    if lambda_within > 0.0:
        if all_group_ids is None:
            raise ValueError("lambda_within > 0 requires all_group_ids to be provided")
        train_group_ids = all_group_ids[train_idx].astype(np.int64)
        for i, dp in enumerate(train_data):
            dp.x_d = np.array([float(train_group_ids[i])], dtype=np.float32)
        # PolymerDataset caches X_d during __post_init__; refresh it via the setter
        # so __getitem__ returns the updated group_id column in every batch.
        train_ds.X_d = np.array([dp.x_d for dp in train_data])

    # Build model (wDMPNN uses weighted message passing and aggregation)
    mp = nn.WeightedBondMessagePassing()
    agg = nn.WeightedMeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, output_transform=output_transform)
    if lambda_within > 0.0:
        mpnn = WDMPNNWithinGroupLoss(mp, agg, ffn, batch_norm=False, lambda_within=lambda_within)
    else:
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
    if lambda_within > 0.0 and train_group_ids is not None:
        from torch.utils.data import DataLoader as TorchDataLoader
        drop_last = (len(train_ds) % BATCH_SIZE == 1)
        batch_sampler = _GroupBatchSampler(train_group_ids, batch_size=BATCH_SIZE,
                                           seed=SEED, drop_last=drop_last)
        train_loader = TorchDataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=4, pin_memory=True, collate_fn=collate_polymer_batch,
        )
    else:
        train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    val_loader = build_dataloader(val_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)
    test_loader = build_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

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
                    json.dump(
                        {"best_ckpt": str(best_ckpt_path), "best_val_loss": best_val_loss,
                         "lambda_within": lambda_within},
                        f, indent=2,
                    )
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
    parser.add_argument('--lambda_within', type=float, default=0.0,
                       help='Weight for normalized within-group residual loss (default: 0.0)')
    parser.add_argument('--results_subdir', type=str, default=None,
                       help='Override checkpoint/predictions subdirectory name')
    cli_args = parser.parse_args()

    folds_to_run = [int(x) for x in cli_args.folds.split(',')]
    split_types = [x.strip() for x in cli_args.split_types.split(',')]
    targets = TARGETS if cli_args.targets is None else [x.strip() for x in cli_args.targets.split(',')]
    lambda_within = cli_args.lambda_within
    results_subdir = cli_args.results_subdir

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("wDMPNN GENERALIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"  Split types: {split_types}")
    print(f"  Targets: {targets}")
    print(f"  Folds: {folds_to_run}")
    print(f"  Lambda within: {lambda_within}")
    print(f"  Results subdir: {results_subdir or '(default)'}")
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

    # Build integer group IDs (needed when lambda_within > 0)
    all_group_ids = None
    if lambda_within > 0.0:
        _, inverse = np.unique(group_keys, return_inverse=True)
        all_group_ids = inverse.astype(np.int64)
        logger.info(f"Built {len(np.unique(all_group_ids))} unique group IDs for lambda_within={lambda_within}")

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
                    lambda_within=lambda_within,
                    all_group_ids=all_group_ids,
                    results_subdir=results_subdir,
                )

                # Save predictions using canonical naming convention.
                # Always use 'wdmpnn' as the canonical model token (avoids
                # registering every lambda variant in naming.py).  The lambda
                # value is encoded as a '__lw{tag}' suffix on the filename stem
                # and stored verbatim in the npz 'lambda_within' field.
                lw_file_tag = f"__lw{_lambda_tag(lambda_within)}" if lambda_within > 0.0 else ""
                pred_base = (ROOT_DIR / 'predictions' / results_subdir) if results_subdir \
                    else PREDICTIONS_DIR
                _subdir = pred_base / split_subdir(split_type)
                _subdir.mkdir(parents=True, exist_ok=True)
                base_name = make_prediction_filename(target, 'wdmpnn', split_type, fold_idx)
                pred_file = _subdir / (base_name[:-4] + lw_file_tag + '.npz')
                np.savez_compressed(
                    pred_file,
                    y_true=y_true,
                    y_pred=y_pred,
                    test_indices=te,
                    split_type=standard_split_name(split_type),
                    model='wdmpnn',
                    target=standard_target_token(target),
                    fold=fold_idx,
                    n_train=len(tr),
                    n_val=len(va),
                    n_test=len(te),
                    lambda_within=lambda_within,
                    prediction_scale="physical_units",
                    smiles_A=df.iloc[te]['smiles_A'].values,
                    smiles_B=df.iloc[te]['smiles_B'].values,
                    fracA=df.iloc[te]['fracA'].values,
                    fracB=df.iloc[te]['fracB'].values,
                    poly_type=df.iloc[te]['poly_type'].values,
                )
                logger.info(f"    Saved: {pred_file.name}")

    print("\n" + "=" * 70)
    print("wDMPNN GENERALIZATION EXPERIMENTS COMPLETE")
    print(f"Predictions: {PREDICTIONS_DIR} (canonical subdirs)")
    print("=" * 70)


if __name__ == '__main__':
    set_seed(SEED)
    main()
