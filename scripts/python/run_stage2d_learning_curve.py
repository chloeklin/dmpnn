"""
Stage 2D Learning Curve Experiment
===================================
Train 2D0-arch and 2D1-arch on subsampled matched groups to determine
whether performance is data-limited.

Matched group = (smiles_A, smiles_B, fracA, fracB)
Entire groups are selected together to prevent architecture information leak.

Training fractions: 25%, 50%, 75%, 100% of training matched groups.
Val/test splits remain unchanged.

Usage (from scripts/python/):
    python run_stage2d_learning_curve.py [--dry_run] [--folds 0,1,2,3,4] [--fractions 25,50,75,100]

Output:
    predictions/HPG2Stage_LC/  (prediction .npz files)
    analysis/results/hpg2stage/learning_curve_output/  (group_ids, metadata)
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
from dataclasses import replace

# Ensure project root and scripts dir are importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from chemprop import data, featurizers
from chemprop.nn.stage2d import ARCH_LABEL_MAP
from utils import (
    set_seed, build_copolymer_model_and_trainer, get_metric_list,
    create_copolymer_data, generate_a_held_out_splits,
    canonicalize_smiles, pick_best_checkpoint,
    create_base_argument_parser, add_model_specific_args,
    setup_model_environment, build_experiment_paths, manage_preprocessing_cache,
)

# ── Configuration ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'ea_ip.csv'
PREDICTIONS_DIR = ROOT / 'predictions' / 'HPG2Stage_LC'
LC_OUTPUT_DIR = ROOT / 'analysis' / 'results' / 'hpg2stage' / 'learning_curve_output'

MODELS = ['2d0_arch', '2d1_arch']
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
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
# MATCHED-GROUP SUBSAMPLING
# ═══════════════════════════════════════════════════════════════════════

def build_matched_groups(df):
    """Build matched group key for each row.
    Group = (smiles_A, smiles_B, fracA, fracB)."""
    return (df['smiles_A'].astype(str) + '||' +
            df['smiles_B'].astype(str) + '||' +
            df['fracA'].astype(str) + '||' +
            df['fracB'].astype(str)).values


def subsample_training_groups(train_indices, group_keys, fraction, seed, fold_idx):
    """Subsample training matched groups at given fraction.
    
    Returns subsampled training indices (entire groups included/excluded).
    """
    if fraction >= 1.0:
        return train_indices
    
    # Get unique groups in training set
    train_groups = group_keys[train_indices]
    unique_groups = np.unique(train_groups)
    
    # Deterministic subsample
    rng = np.random.default_rng(seed + fold_idx * 1000 + int(fraction * 100))
    n_select = max(1, int(len(unique_groups) * fraction))
    selected_groups = set(rng.choice(unique_groups, size=n_select, replace=False))
    
    # Filter training indices to only those in selected groups
    mask = np.array([g in selected_groups for g in train_groups])
    subsampled_indices = train_indices[mask]
    
    return subsampled_indices, selected_groups


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train_single_run(df_input, data_A, data_B, fA, fB, orig_Xd,
                     train_idx, val_idx, test_idx,
                     variant, target, fold_idx, fraction,
                     featurizer, args_template):
    """Train a single model run and return predictions."""
    
    from chemprop.data import CopolymerDataset, build_dataloader
    from sklearn.impute import SimpleImputer
    
    copolymer_mode = f'stage2d_{variant}'
    
    # Build datasets
    train_dA = [data_A[j] for j in train_idx]
    train_dB = [data_B[j] for j in train_idx]
    val_dA = [data_A[j] for j in val_idx]
    val_dB = [data_B[j] for j in val_idx]
    test_dA = [data_A[j] for j in test_idx]
    test_dB = [data_B[j] for j in test_idx]
    
    train_ds = CopolymerDataset(train_dA, train_dB, fA[train_idx], fB[train_idx], featurizer)
    val_ds = CopolymerDataset(val_dA, val_dB, fA[val_idx], fB[val_idx], featurizer)
    test_ds = CopolymerDataset(test_dA, test_dB, fA[test_idx], fB[test_idx], featurizer)
    
    # Normalize targets
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    
    # Build checkpoint path
    frac_str = f"frac{int(fraction*100)}"
    checkpoint_path = (ROOT / 'checkpoints' / 'HPG2Stage_LC' /
                      f'{DATASET_NAME}__{target}__{copolymer_mode}__fold{fold_idx}__{frac_str}')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics
    metric_list = get_metric_list('reg', target=target)
    
    # Build model
    args_template.copolymer_mode = copolymer_mode
    mpnn, trainer = build_copolymer_model_and_trainer(
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
    
    # Dataloaders
    train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    val_loader = build_dataloader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
    test_loader = build_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
    
    # Check for existing training
    done_flag = checkpoint_path / "TRAINING_COMPLETE"
    inprog_flag = checkpoint_path / "TRAINING_IN_PROGRESS"
    
    if done_flag.exists():
        best_ckpt_path, _ = pick_best_checkpoint(checkpoint_path)
        if best_ckpt_path is not None:
            logger.info(f"  Skipping training (COMPLETE): {checkpoint_path.name}")
            # Load checkpoint
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
            logger.warning(f"  TRAINING_COMPLETE exists but no checkpoint found. Retraining.")
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
    parser = argparse.ArgumentParser(description='Stage 2D Learning Curve Experiment')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only generate group subsampling, do not train')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated fold indices to run (default: all)')
    parser.add_argument('--fractions', type=str, default='25,50,75,100',
                       help='Comma-separated training fractions as percentages (default: 25,50,75,100)')
    parser.add_argument('--models', type=str, default='2d0_arch,2d1_arch',
                       help='Comma-separated model variants (default: 2d0_arch,2d1_arch)')
    cli_args = parser.parse_args()
    
    folds_to_run = [int(x) for x in cli_args.folds.split(',')]
    fractions_to_run = [int(x) / 100.0 for x in cli_args.fractions.split(',')]
    models_to_run = [x.strip() for x in cli_args.models.split(',')]
    
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STAGE 2D LEARNING CURVE EXPERIMENT")
    print("=" * 70)
    print(f"  Models: {models_to_run}")
    print(f"  Folds: {folds_to_run}")
    print(f"  Fractions: {[f'{int(f*100)}%' for f in fractions_to_run]}")
    print(f"  Dry run: {cli_args.dry_run}")
    print(f"  Predictions dir: {PREDICTIONS_DIR}")
    print(f"  Output dir: {LC_OUTPUT_DIR}")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows")
    
    # Build matched group keys
    group_keys = build_matched_groups(df)
    
    # Build smiles_A array for split generation
    smiles_A = df['smiles_A'].astype(str).values
    n_datapoints = len(df)
    
    # Generate a_held_out splits (reproduces exact same splits as training)
    logger.info("Generating a_held_out splits...")
    train_indices, val_indices, test_indices = generate_a_held_out_splits(
        smiles_A, n_datapoints, SEED, n_splits=N_FOLDS, logger=logger
    )
    
    # Generate group subsampling for each fold × fraction
    subsampling_metadata = {}
    
    for fold_idx in folds_to_run:
        tr = train_indices[fold_idx]
        
        # Get unique training groups
        train_groups = group_keys[tr]
        unique_train_groups = np.unique(train_groups)
        logger.info(f"Fold {fold_idx}: {len(tr)} training samples, "
                   f"{len(unique_train_groups)} unique training matched groups")
        
        fold_meta = {}
        for frac in fractions_to_run:
            if frac >= 1.0:
                sub_indices = tr
                selected_groups = set(unique_train_groups)
            else:
                sub_indices, selected_groups = subsample_training_groups(
                    tr, group_keys, frac, SEED, fold_idx
                )
            
            frac_key = f"fold{fold_idx}_frac{int(frac*100)}"
            fold_meta[frac_key] = {
                'fold': fold_idx,
                'fraction': frac,
                'n_original_train': len(tr),
                'n_subsampled_train': len(sub_indices),
                'n_original_groups': len(unique_train_groups),
                'n_selected_groups': len(selected_groups),
                'n_val': len(val_indices[fold_idx]),
                'n_test': len(test_indices[fold_idx]),
            }
            
            # Save group IDs for reproducibility
            group_ids_file = LC_OUTPUT_DIR / f'selected_groups_{frac_key}.json'
            with open(group_ids_file, 'w') as f:
                json.dump(sorted(selected_groups), f)
            
            logger.info(f"  {frac_key}: {len(sub_indices)} samples "
                       f"({len(selected_groups)}/{len(unique_train_groups)} groups)")
            
            subsampling_metadata[frac_key] = fold_meta[frac_key]
    
    # Save subsampling metadata
    meta_file = LC_OUTPUT_DIR / 'subsampling_metadata.json'
    with open(meta_file, 'w') as f:
        json.dump(subsampling_metadata, f, indent=2)
    logger.info(f"Saved subsampling metadata to {meta_file}")
    
    if cli_args.dry_run:
        print("\n[DRY RUN] Group subsampling complete. Exiting without training.")
        print(f"Metadata saved to: {LC_OUTPUT_DIR}")
        return
    
    # ── Prepare data for training ────────────────────────────────────
    # Create args-like object for model building
    train_args = argparse.Namespace(
        model_name='DMPNN',
        dataset_name=DATASET_NAME,
        polymer_type='copolymer',
        copolymer_mode='stage2d_2d0_arch',  # will be overwritten per model
        split_type='a_held_out',
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
        results_subdir='HPG2Stage_LC',
        aux_task='off',
        _aux_cols=[],
        _n_aux_targets=0,
    )
    
    # Prepare copolymer data (same for all runs)
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
        
        # Create datapoints once (reused across folds/fractions)
        data_A, data_B, fA, fB = create_copolymer_data(
            smis_A, smis_B, fracA_arr, fracB_arr, ys,
            orig_Xd, 'DMPNN',
        )
        logger.info(f"  Created {len(data_A)} copolymer datapoints for {target}")
        
        for fold_idx in folds_to_run:
            tr_full = train_indices[fold_idx]
            va = val_indices[fold_idx]
            te = test_indices[fold_idx]
            
            for frac in fractions_to_run:
                # Get subsampled training indices
                if frac >= 1.0:
                    tr_sub = tr_full
                else:
                    frac_key = f"fold{fold_idx}_frac{int(frac*100)}"
                    group_ids_file = LC_OUTPUT_DIR / f'selected_groups_{frac_key}.json'
                    with open(group_ids_file, 'r') as f:
                        selected_groups = set(json.load(f))
                    train_groups = group_keys[tr_full]
                    mask = np.array([g in selected_groups for g in train_groups])
                    tr_sub = tr_full[mask]
                
                for variant in models_to_run:
                    frac_pct = int(frac * 100)
                    logger.info(f"\n  Training: {variant} | fold={fold_idx} | "
                               f"frac={frac_pct}% | n_train={len(tr_sub)}")
                    
                    # Train
                    y_true, y_pred = train_single_run(
                        df, data_A, data_B, fA, fB, orig_Xd,
                        tr_sub, va, te,
                        variant, target, fold_idx, frac,
                        featurizer, train_args,
                    )
                    
                    # Save predictions with test indices for direct row matching
                    pred_file = (PREDICTIONS_DIR /
                                f'{DATASET_NAME}__{target}__stage2d_{variant}__'
                                f'fold{fold_idx}__frac{frac_pct}__split{fold_idx}.npz')
                    np.savez_compressed(
                        pred_file,
                        y_true=y_true,
                        y_pred=y_pred,
                        test_indices=te,
                        n_train=len(tr_sub),
                        fold=fold_idx,
                        fraction=frac,
                    )
                    logger.info(f"    Saved: {pred_file.name}")
    
    print("\n" + "=" * 70)
    print("LEARNING CURVE TRAINING COMPLETE")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print(f"Metadata: {LC_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    set_seed(SEED)
    main()
