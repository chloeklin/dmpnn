"""
Stage 2D Generalization Experiments
====================================
Tests whether architecture effects learned by 2D1 are transferable to unseen
chemistry, or if 2D1 exploits chemistry-specific patterns.

Two evaluation protocols:

  A) GROUP-DISJOINT: Hold out entire (A, B, fracA) composition groups.
     All architecture variants of a group go to the same fold.
     Tests: can architecture effects transfer to unseen compositions
     of known chemistry pairs?

  B) PAIR-DISJOINT: Hold out entire (A, B) monomer pairs.
     All compositions and architecture variants of a pair go together.
     Tests: can architecture effects transfer to completely unseen
     chemistry pairs?

Models: frac, 2d0_arch, 2d1_arch
Splits: 5-fold GroupKFold
Hyperparams: Same as original HPG2Stage experiments.

Usage:
    python run_stage2d_generalization.py [--dry_run] [--folds 0,1,2,3,4]
        [--split_types group_disjoint,pair_disjoint]
        [--models frac,2d0_arch,2d1_arch]

Output:
    predictions/HPG2Stage_Gen/  (prediction .npz files)
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import torch
from pathlib import Path
from collections import defaultdict

# Ensure project root and scripts dir are importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from chemprop import data, featurizers
from chemprop.nn.stage2d import ARCH_LABEL_MAP
from utils import (
    set_seed, build_copolymer_model_and_trainer, get_metric_list,
    create_copolymer_data, canonicalize_smiles, pick_best_checkpoint,
)
sys.path.insert(0, str(ROOT_DIR))
from evaluation.naming import (
    make_prediction_filename, standard_model_name, standard_split_name,
    standard_target_token, split_subdir, DATASET_NAME as _CANONICAL_DATASET,
)

# ── Configuration ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'ea_ip.csv'
PREDICTIONS_DIR = ROOT / 'predictions'

MODELS = ['frac', '2d0_arch', '2d1_arch']
N_FOLDS = 5
SEED = 42
TARGETS = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
DATASET_NAME = 'ea_ip'

# Training hyperparameters (match original HPG2Stage runs)
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SPLIT GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def build_group_keys(df):
    """Build group key = (smiles_A, smiles_B, fracA) for each row.
    
    This defines matched groups: rows that share the same composition
    but differ only in architecture (poly_type).
    """
    return (df['smiles_A'].astype(str) + '||' +
            df['smiles_B'].astype(str) + '||' +
            df['fracA'].astype(str)).values


def build_pair_keys(df):
    """Build pair key = (smiles_A, smiles_B) for each row.
    
    This groups all compositions and architectures of a monomer pair.
    """
    return (df['smiles_A'].astype(str) + '||' +
            df['smiles_B'].astype(str)).values


def generate_group_disjoint_splits(df, n_splits=5, seed=42):
    """Generate group-disjoint 5-fold CV splits.
    
    Grouping key: (smiles_A, smiles_B, fracA).
    All architecture variants of a composition group stay together.
    Ensures no composition leakage between train/val/test.
    
    Returns: (train_indices, val_indices, test_indices) — lists of arrays.
    """
    from sklearn.model_selection import GroupKFold

    group_keys = build_group_keys(df)
    idx_all = np.arange(len(df))
    
    gkf = GroupKFold(n_splits=n_splits)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for fold_i, (train_val_idx, test_idx) in enumerate(gkf.split(idx_all, groups=group_keys)):
        # Sub-split train_val into train/val BY GROUP
        tv_groups = group_keys[train_val_idx]
        tv_unique = np.unique(tv_groups)
        
        rng = np.random.default_rng(seed + fold_i)
        n_val_groups = max(1, int(round(0.1 * len(tv_unique))))
        val_group_set = set(rng.choice(tv_unique, size=n_val_groups, replace=False))
        
        val_mask = np.array([g in val_group_set for g in tv_groups])
        va_idx = train_val_idx[val_mask]
        tr_idx = train_val_idx[~val_mask]
        
        train_indices.append(tr_idx)
        val_indices.append(va_idx)
        test_indices.append(test_idx)
    
    return train_indices, val_indices, test_indices


def generate_pair_disjoint_splits(df, n_splits=5, seed=42):
    """Generate pair-disjoint 5-fold CV splits.
    
    Grouping key: (smiles_A, smiles_B).
    All compositions and architectures of a monomer pair stay together.
    Ensures no chemistry-pair leakage between train/val/test.
    
    Returns: (train_indices, val_indices, test_indices) — lists of arrays.
    """
    from sklearn.model_selection import GroupKFold

    pair_keys = build_pair_keys(df)
    idx_all = np.arange(len(df))
    
    gkf = GroupKFold(n_splits=n_splits)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for fold_i, (train_val_idx, test_idx) in enumerate(gkf.split(idx_all, groups=pair_keys)):
        # Sub-split train_val into train/val BY PAIR
        tv_pairs = pair_keys[train_val_idx]
        tv_unique = np.unique(tv_pairs)
        
        rng = np.random.default_rng(seed + fold_i)
        n_val_pairs = max(1, int(round(0.1 * len(tv_unique))))
        val_pair_set = set(rng.choice(tv_unique, size=n_val_pairs, replace=False))
        
        val_mask = np.array([p in val_pair_set for p in tv_pairs])
        va_idx = train_val_idx[val_mask]
        tr_idx = train_val_idx[~val_mask]
        
        train_indices.append(tr_idx)
        val_indices.append(va_idx)
        test_indices.append(test_idx)
    
    return train_indices, val_indices, test_indices


# ═══════════════════════════════════════════════════════════════════════
# LEAKAGE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_no_leakage(train_idx, val_idx, test_idx, keys, fold_i, split_type):
    """Verify no group/pair leakage between splits.
    
    Prints group counts and asserts disjointness.
    """
    train_keys = set(keys[train_idx])
    val_keys = set(keys[val_idx])
    test_keys = set(keys[test_idx])
    
    key_type = "groups" if split_type == "group_disjoint" else "pairs"
    
    print(f"  Fold {fold_i}: train {key_type}={len(train_keys)}, "
          f"val {key_type}={len(val_keys)}, test {key_type}={len(test_keys)}")
    print(f"           train samples={len(train_idx)}, "
          f"val samples={len(val_idx)}, test samples={len(test_idx)}")
    
    # Assert disjointness
    overlap_tr_te = train_keys & test_keys
    overlap_va_te = val_keys & test_keys
    overlap_tr_va = train_keys & val_keys
    
    assert not overlap_tr_te, \
        f"Fold {fold_i}: train∩test leakage! {len(overlap_tr_te)} {key_type} overlap"
    assert not overlap_va_te, \
        f"Fold {fold_i}: val∩test leakage! {len(overlap_va_te)} {key_type} overlap"
    assert not overlap_tr_va, \
        f"Fold {fold_i}: train∩val leakage! {len(overlap_tr_va)} {key_type} overlap"
    
    print(f"           ✓ No leakage (all splits disjoint)")
    
    return train_keys, val_keys, test_keys


def verify_pair_disjoint_extra(train_idx, val_idx, test_idx, pair_keys, fold_i):
    """For pair-disjoint: verify pairs are also disjoint (stronger condition)."""
    train_pairs = set(pair_keys[train_idx])
    val_pairs = set(pair_keys[val_idx])
    test_pairs = set(pair_keys[test_idx])
    
    assert not (train_pairs & test_pairs), \
        f"Fold {fold_i}: train/test PAIR overlap!"
    assert not (val_pairs & test_pairs), \
        f"Fold {fold_i}: val/test PAIR overlap!"
    assert not (train_pairs & val_pairs), \
        f"Fold {fold_i}: train/val PAIR overlap!"
    
    print(f"           ✓ Pair disjointness verified")


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_single_run(df_input, data_A, data_B, fA, fB, orig_Xd,
                     train_idx, val_idx, test_idx,
                     variant, target, fold_idx, split_type,
                     featurizer, args_template, group_keys=None):
    """Train a single model run and return predictions."""
    
    from chemprop.data import CopolymerDataset, build_dataloader
    
    copolymer_mode = f'stage2d_{variant}'
    lambda_within = getattr(args_template, 'lambda_within', 0.0)
    
    # Build integer group_ids if group_keys provided (needed for lambda_within > 0)
    if group_keys is not None and lambda_within > 0.0:
        unique_keys, inverse = np.unique(group_keys, return_inverse=True)
        all_group_ids = inverse.astype(np.int64)
        train_group_ids = all_group_ids[train_idx]
    else:
        train_group_ids = None

    # When lambda_within > 0, append group_id as column 1 of X_d so the
    # training step can extract it.  Column 0 stays arch_idx.
    def make_Xd_with_groups(idx_subset, base_Xd, gids):
        """Append group_id column to X_d for the given subset."""
        if gids is None:
            return base_Xd[idx_subset]
        return np.concatenate(
            [base_Xd[idx_subset], gids.reshape(-1, 1).astype(np.float32)], axis=1
        )

    train_Xd = make_Xd_with_groups(train_idx, orig_Xd, train_group_ids)
    val_Xd   = orig_Xd[val_idx]    # group_id not needed outside training
    test_Xd  = orig_Xd[test_idx]

    # Build datasets
    train_dA = [data_A[j] for j in train_idx]
    train_dB = [data_B[j] for j in train_idx]
    val_dA = [data_A[j] for j in val_idx]
    val_dB = [data_B[j] for j in val_idx]
    test_dA = [data_A[j] for j in test_idx]
    test_dB = [data_B[j] for j in test_idx]

    # Patch X_d on data_A copies so CopolymerDataset picks up the right columns
    for i, dp in enumerate(train_dA):
        dp.x_d = train_Xd[i]
    for i, dp in enumerate(val_dA):
        dp.x_d = val_Xd[i]
    for i, dp in enumerate(test_dA):
        dp.x_d = test_Xd[i]

    train_ds = CopolymerDataset(
        train_dA, train_dB, fA[train_idx], fB[train_idx], featurizer,
        group_ids=train_group_ids,
    )
    val_ds = CopolymerDataset(val_dA, val_dB, fA[val_idx], fB[val_idx], featurizer)
    test_ds = CopolymerDataset(test_dA, test_dB, fA[test_idx], fB[test_idx], featurizer)
    
    # Normalize targets
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    
    # Build checkpoint path
    checkpoint_path = (ROOT / 'checkpoints' / 'HPG2Stage_Gen' /
                      f'{DATASET_NAME}__{target}__{copolymer_mode}__{split_type}__fold{fold_idx}')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics
    metric_list = get_metric_list('reg', target=target)
    
    # Build model
    args_template.copolymer_mode = copolymer_mode
    mpnn, trainer = build_copolymer_model_and_trainer(
        args=args_template,
        combined_descriptor_data=train_Xd,
        scaler=scaler,
        checkpoint_path=checkpoint_path,
        copolymer_mode=copolymer_mode,
        batch_norm=False,
        metric_list=metric_list,
        early_stopping_patience=PATIENCE,
        max_epochs=EPOCHS,
        save_checkpoint=True,
        lambda_within=lambda_within,
    )
    
    # Dataloaders
    train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, seed=SEED)
    val_loader = build_dataloader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
    test_loader = build_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
    
    # Check for existing training
    done_flag = checkpoint_path / "TRAINING_COMPLETE"
    inprog_flag = checkpoint_path / "TRAINING_IN_PROGRESS"
    
    if done_flag.exists():
        best_ckpt_path, _ = pick_best_checkpoint(checkpoint_path)
        if best_ckpt_path is not None:
            logger.info(f"  Skipping training (COMPLETE): {checkpoint_path.name}")
            use_cuda = torch.cuda.is_available()
            map_location = None if use_cuda else torch.device("cpu")
            mpnn_fresh, _ = build_copolymer_model_and_trainer(
                args=args_template,
                combined_descriptor_data=orig_Xd,
                scaler=scaler,
                checkpoint_path=checkpoint_path,
                copolymer_mode=copolymer_mode,
                batch_norm=False,
                metric_list=metric_list,
                early_stopping_patience=PATIENCE,
                max_epochs=EPOCHS,
                save_checkpoint=True,
            )
            checkpoint = torch.load(best_ckpt_path, map_location=map_location, weights_only=False)
            mpnn_fresh.load_state_dict(checkpoint['state_dict'])
            mpnn = mpnn_fresh
            if use_cuda:
                mpnn = mpnn.to(torch.device("cuda"))
            mpnn.eval()
        else:
            logger.warning(f"  TRAINING_COMPLETE but no checkpoint found. Retraining.")
            done_flag.unlink()
    
    if not done_flag.exists():
        inprog_flag.touch(exist_ok=True)
        try:
            trainer.fit(mpnn, train_loader, val_loader)
            best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
            if best_ckpt_path:
                with open(checkpoint_path / "best.json", "w") as f:
                    json.dump({"best_ckpt": str(best_ckpt_path), "best_val_loss": best_val_loss}, f, indent=2)
                done_flag.touch()
        finally:
            if inprog_flag.exists():
                inprog_flag.unlink(missing_ok=True)
    
    # Get predictions
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
    parser = argparse.ArgumentParser(description='Stage 2D Generalization Experiments')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only generate and verify splits, do not train')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated fold indices to run (default: all)')
    parser.add_argument('--split_types', type=str, default='group_disjoint,pair_disjoint',
                       help='Comma-separated split types (default: group_disjoint,pair_disjoint)')
    parser.add_argument('--models', type=str, default='frac,2d0_arch,2d1_arch',
                       help='Comma-separated model variants (default: frac,2d0_arch,2d1_arch)')
    cli_args = parser.parse_args()
    
    folds_to_run = [int(x) for x in cli_args.folds.split(',')]
    split_types = [x.strip() for x in cli_args.split_types.split(',')]
    models_to_run = [x.strip() for x in cli_args.models.split(',')]
    
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STAGE 2D GENERALIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"  Models: {models_to_run}")
    print(f"  Split types: {split_types}")
    print(f"  Folds: {folds_to_run}")
    print(f"  Dry run: {cli_args.dry_run}")
    print(f"  Predictions dir: {PREDICTIONS_DIR}")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows")
    
    # Build keys
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
            raise ValueError(f"Unknown split type: {split_type}")
        
        # Verify no leakage for each fold
        for fold_i in range(N_FOLDS):
            verify_no_leakage(
                train_indices[fold_i], val_indices[fold_i], test_indices[fold_i],
                verify_keys, fold_i, split_type
            )
            
            # Extra pair-disjoint verification
            if split_type == 'pair_disjoint':
                verify_pair_disjoint_extra(
                    train_indices[fold_i], val_indices[fold_i], test_indices[fold_i],
                    pair_keys, fold_i
                )
            
            # For group_disjoint: also verify that groups are disjoint
            if split_type == 'group_disjoint':
                # Check that no (A,B,fracA) group leaks
                verify_no_leakage(
                    train_indices[fold_i], val_indices[fold_i], test_indices[fold_i],
                    group_keys, fold_i, split_type
                )
        
        splits[split_type] = (train_indices, val_indices, test_indices)
        print(f"  ✓ All {N_FOLDS} folds verified for {split_type}")
    
    if cli_args.dry_run:
        print("\n[DRY RUN] Split generation and verification complete. Exiting without training.")
        return
    
    # ── Prepare data for training ────────────────────────────────────
    train_args = argparse.Namespace(
        model_name='DMPNN',
        dataset_name=DATASET_NAME,
        polymer_type='copolymer',
        copolymer_mode='stage2d_frac',  # will be overwritten per model
        split_type='group_disjoint',
        task_type='reg',
        batch_norm=False,
        incl_desc=False,
        incl_rdkit=False,
        fusion_mode='late_concat',
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        save_predictions=True,
        export_embeddings=False,
        train_size=None,
        results_subdir='HPG2Stage_Gen',
        aux_task='off',
        _aux_cols=[],
        _n_aux_targets=0,
        lambda_within=0.0,
    )
    
    # Prepare copolymer data
    smis_A = df['smiles_A'].astype(str).values
    smis_B = df['smiles_B'].astype(str).values
    fracA_arr = df['fracA'].astype(float).values
    fracB_arr = df['fracB'].astype(float).values
    
    # Stage2D ordinal architecture encoding
    arch_ordinal = df['poly_type'].astype(str).str.lower().str.strip().map(ARCH_LABEL_MAP)
    orig_Xd = arch_ordinal.values.astype(np.float32).reshape(-1, 1)
    
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    
    # ── Training loop ────────────────────────────────────────────────
    for target in TARGETS:
        logger.info(f"\n{'='*60}\nTarget: {target}\n{'='*60}")
        
        ys = df[target].astype(float).values.reshape(-1, 1)
        
        # Create datapoints once (reused across folds/splits)
        data_A, data_B, fA, fB = create_copolymer_data(
            smis_A, smis_B, fracA_arr, fracB_arr, ys,
            orig_Xd, 'DMPNN',
        )
        logger.info(f"  Created {len(data_A)} copolymer datapoints for {target}")
        
        for split_type in split_types:
            train_indices, val_indices, test_indices = splits[split_type]
            
            for fold_idx in folds_to_run:
                tr = train_indices[fold_idx]
                va = val_indices[fold_idx]
                te = test_indices[fold_idx]
                
                for variant in models_to_run:
                    logger.info(
                        f"\n  Training: {variant} | {split_type} | fold={fold_idx} | "
                        f"n_train={len(tr)} | n_val={len(va)} | n_test={len(te)}"
                    )
                    
                    # Train
                    y_true, y_pred = train_single_run(
                        df, data_A, data_B, fA, fB, orig_Xd,
                        tr, va, te,
                        variant, target, fold_idx, split_type,
                        featurizer, train_args,
                        group_keys=group_keys,
                    )
                    
                    # Save predictions using canonical naming convention
                    _model_internal = f'stage2d_{variant}'
                    _subdir = PREDICTIONS_DIR / split_subdir(split_type)
                    _subdir.mkdir(parents=True, exist_ok=True)
                    pred_file = _subdir / make_prediction_filename(
                        target, _model_internal, split_type, fold_idx
                    )
                    np.savez_compressed(
                        pred_file,
                        y_true=y_true,
                        y_pred=y_pred,
                        test_indices=te,
                        split_type=standard_split_name(split_type),
                        model=standard_model_name(_model_internal),
                        target=standard_target_token(target),
                        fold=fold_idx,
                        n_train=len(tr),
                        n_val=len(va),
                        n_test=len(te),
                        prediction_scale="physical_units",
                    )
                    logger.info(f"    Saved: {pred_file.name}")
    
    print("\n" + "=" * 70)
    print("GENERALIZATION EXPERIMENTS COMPLETE")
    print(f"Predictions: {PREDICTIONS_DIR} (canonical subdirs)")
    print("=" * 70)


if __name__ == '__main__':
    set_seed(SEED)
    main()
