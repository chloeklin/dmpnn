"""
Stage 2D Learning Curve — Final Pipeline
==========================================
Replicates the EXACT train_graph.py pipeline for Stage 2D models, adding only
matched-group subsampling of the training set.

CRITICAL: The 100 % fraction MUST reproduce the final Stage 2D results.
Any discrepancy indicates a pipeline mismatch that must be debugged first.

Differences from old run_stage2d_learning_curve.py:
  1. EPOCHS = 300, PATIENCE = 30  (matches train_config.yaml / train_graph.py)
  2. Split generation on *filtered* valid datapoints (matches train_graph.py)
  3. Saves original-DF row indices alongside predictions for evaluation

Matched group  = (smiles_A, smiles_B, fracA, fracB)
Entire groups are selected together to prevent architecture information leak.
Val/test sets remain identical to the final Stage 2D split at every fraction.

Usage (from project root):
    python experiments/hpg2stage/scripts/run_stage2d_learning_curve_final.py \\
        [--dry_run] [--folds 0,1,2,3,4] [--fractions 25,50,75,100]
        [--models 2d0_arch,2d1_arch] [--config <yaml>]

Output:
    predictions/HPG2Stage_LC_Final/   (prediction .npz files)
    experiments/hpg2stage/output/learning_curve_final/  (group_ids, metadata)
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

# Ensure project root and scripts dir are importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SCRIPTS_PY = PROJECT_ROOT / 'scripts' / 'python'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_PY))

from chemprop import data as cdata, featurizers
from chemprop.nn.stage2d import ARCH_LABEL_MAP
from utils import (
    set_seed, build_copolymer_model_and_trainer, get_metric_list,
    create_copolymer_data, generate_a_held_out_splits,
    canonicalize_smiles, pick_best_checkpoint,
)

# ── Configuration ────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / 'data' / 'ea_ip.csv'
PREDICTIONS_DIR = PROJECT_ROOT / 'predictions' / 'HPG2Stage_LC_Final'
LC_OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'hpg2stage' / 'output' / 'learning_curve_final'

MODELS = ['2d0_arch', '2d1_arch']
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
N_FOLDS = 5
SEED = 42
TARGETS = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
DATASET_NAME = 'ea_ip'

# Training hyperparameters — MUST match train_config.yaml / train_graph.py
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING  (mirrors train_graph.py copolymer branch exactly)
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare_data():
    """Load ea_ip.csv and prepare copolymer data exactly as train_graph.py does.

    Replicates load_and_preprocess_data() from utils.py:
      1. Pair canonicalization: swap A↔B if B < A alphabetically
      2. Fraction normalization: fracA + fracB = 1.0 exactly
      3. Stage2D ordinal architecture encoding
      4. group_key from canonicalized pair + fractions

    Returns
    -------
    df : DataFrame  (preprocessed — used for group_key / poly_type lookups)
    smis_A, smis_B : lists of str  (canonicalized)
    fracA_arr, fracB_arr : np.ndarray
    orig_Xd : np.ndarray  (ordinal arch labels, shape (n_rows, 1))
    """
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows")

    # ── Pair canonicalization (matches load_and_preprocess_data) ───────
    # Ensures A/B ordering is deterministic: alphabetically smaller = A
    def _canon_pair(a, b, wa, wb):
        a = "" if pd.isna(a) else str(a)
        b = "" if pd.isna(b) else str(b)
        if b < a:
            return b, a, wb, wa
        return a, b, wa, wb

    raw_A = df['smiles_A'].astype(str).tolist()
    raw_B = df['smiles_B'].astype(str).tolist()
    raw_fA = df['fracA'].values.astype(float)
    raw_fB = df['fracB'].values.astype(float)

    canA, canB, fA_list, fB_list = [], [], [], []
    for a, b, wa, wb in zip(raw_A, raw_B, raw_fA, raw_fB):
        a2, b2, wa2, wb2 = _canon_pair(a, b, wa, wb)
        canA.append(a2)
        canB.append(b2)
        fA_list.append(wa2)
        fB_list.append(wb2)

    # ── Fraction normalization (matches load_and_preprocess_data) ─────
    fracA_arr = np.array(fA_list, dtype=float)
    fracB_arr = np.array(fB_list, dtype=float)
    fsum = fracA_arr + fracB_arr
    fracA_arr = fracA_arr / fsum
    fracB_arr = 1.0 - fracA_arr

    smis_A = canA
    smis_B = canB

    # Write back to df for consistent group_key / downstream lookups
    df['smilesA'] = canA
    df['smilesB'] = canB
    df['fracA'] = fracA_arr
    df['fracB'] = fracB_arr

    # Stage2D ordinal architecture encoding
    arch_ordinal = (df['poly_type'].astype(str).str.lower().str.strip()
                    .map(ARCH_LABEL_MAP))
    orig_Xd = arch_ordinal.values.astype(np.float32).reshape(-1, 1)

    # Build group_key on canonicalized pair + normalized fractions
    def _round6(x):
        try:
            return round(float(x), 6)
        except Exception:
            return x

    df['group_key'] = [
        f"{a}||{b}||{_round6(fa)}||{_round6(fb)}"
        for a, b, fa, fb in zip(canA, canB, fracA_arr, fracB_arr)
    ]

    logger.info(f"  Pair-canonicalized: {sum(1 for a, b in zip(raw_A, canA) if a != b)} rows swapped A↔B")

    return df, smis_A, smis_B, fracA_arr, fracB_arr, orig_Xd


def create_datapoints_and_splits(df, smis_A, smis_B, fracA_arr, fracB_arr,
                                 orig_Xd, target):
    """Create datapoints for one target and generate a_held_out splits.

    Mirrors train_graph.py lines 826-868 exactly:
      1. create_copolymer_data → filters invalid rows
      2. build valid_smiles_A aligned with filtered datapoints
      3. generate_a_held_out_splits on valid_smiles_A

    Returns
    -------
    data_A, data_B, fA, fB, n_datapoints,
    train_indices, val_indices, test_indices,
    valid_orig_indices  (original df row indices for each valid datapoint)
    """
    ys = df[target].astype(float).values.reshape(-1, 1)

    # create_copolymer_data drops rows with empty SMILES or all-NaN targets
    data_A, data_B, fA, fB = create_copolymer_data(
        smis_A, smis_B, fracA_arr, fracB_arr, ys,
        orig_Xd, 'DMPNN',
    )
    n_datapoints = len(data_A)
    logger.info(f"[{target}] Created {n_datapoints} copolymer datapoints "
                f"(from {len(df)} raw rows)")

    # Build valid_smiles_A aligned with filtered datapoints — exact copy of
    # train_graph.py lines 848-863
    valid_smiles_A = []
    valid_orig_indices = []
    for idx in range(len(df)):
        y_val = ys[idx]
        has_A = bool(smis_A[idx])
        has_B = bool(smis_B[idx])
        sA_val = smis_A[idx] if has_A else ""
        if has_A and has_B and pd.notna(y_val).any():
            valid_smiles_A.append(sA_val)
            valid_orig_indices.append(idx)
    valid_smiles_A = np.array(valid_smiles_A, dtype=str)
    valid_orig_indices = np.array(valid_orig_indices, dtype=int)
    assert len(valid_smiles_A) == n_datapoints, \
        f"smiles_A length ({len(valid_smiles_A)}) != n_datapoints ({n_datapoints})"

    # Generate a_held_out splits — same function, same args as train_graph.py
    train_indices, val_indices, test_indices = generate_a_held_out_splits(
        valid_smiles_A, n_datapoints, SEED, n_splits=N_FOLDS, logger=logger
    )

    return (data_A, data_B, fA, fB, n_datapoints,
            train_indices, val_indices, test_indices,
            valid_orig_indices)


# ═══════════════════════════════════════════════════════════════════════
# MATCHED-GROUP SUBSAMPLING
# ═══════════════════════════════════════════════════════════════════════

def subsample_training_groups(train_idx, group_keys_valid, fraction,
                              seed, fold_idx):
    """Subsample training matched groups at given fraction.

    Parameters
    ----------
    train_idx : np.ndarray   indices into the *valid* datapoint arrays
    group_keys_valid : np.ndarray  group key for each valid datapoint
    fraction : float in (0, 1]
    seed, fold_idx : ints for deterministic RNG

    Returns
    -------
    sub_train_idx : np.ndarray  subsampled training indices
    selected_groups : set of str  selected group keys
    """
    if fraction >= 1.0:
        train_groups = group_keys_valid[train_idx]
        return train_idx, set(np.unique(train_groups))

    train_groups = group_keys_valid[train_idx]
    unique_groups = np.unique(train_groups)

    rng = np.random.default_rng(seed + fold_idx * 1000 + int(fraction * 100))
    n_select = max(1, int(len(unique_groups) * fraction))
    selected_groups = set(rng.choice(unique_groups, size=n_select, replace=False))

    mask = np.array([g in selected_groups for g in train_groups])
    return train_idx[mask], selected_groups


# ═══════════════════════════════════════════════════════════════════════
# TRAINING  (mirrors train_graph.py lines 1072-1260 for Stage2D)
# ═══════════════════════════════════════════════════════════════════════

def train_single_run(data_A, data_B, fA, fB, orig_Xd,
                     train_idx, val_idx, test_idx,
                     variant, target, fold_idx, fraction,
                     featurizer, args_template):
    """Train a single model run and return predictions.

    Exactly replicates train_graph.py copolymer branch:
      - CopolymerDataset construction
      - Target normalization (train only, val gets same scaler)
      - No descriptor normalization for Stage2D (ordinal arch labels)
      - build_copolymer_model_and_trainer with EPOCHS/PATIENCE
      - Checkpoint skip logic with TRAINING_COMPLETE flag
      - trainer.predict for raw predictions
    """
    from chemprop.data import CopolymerDataset, build_dataloader

    copolymer_mode = f'stage2d_{variant}'

    # Build datasets
    train_dA = [data_A[j] for j in train_idx]
    train_dB = [data_B[j] for j in train_idx]
    val_dA   = [data_A[j] for j in val_idx]
    val_dB   = [data_B[j] for j in val_idx]
    test_dA  = [data_A[j] for j in test_idx]
    test_dB  = [data_B[j] for j in test_idx]

    train_ds = CopolymerDataset(train_dA, train_dB, fA[train_idx], fB[train_idx], featurizer)
    val_ds   = CopolymerDataset(val_dA, val_dB, fA[val_idx], fB[val_idx], featurizer)
    test_ds  = CopolymerDataset(test_dA, test_dB, fA[test_idx], fB[test_idx], featurizer)

    # Normalize targets (regression)
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    # NOTE: Stage2D skips descriptor normalization (ordinal arch label)

    # Build checkpoint path
    frac_pct = int(fraction * 100)
    checkpoint_path = (PROJECT_ROOT / 'checkpoints' / 'HPG2Stage_LC_Final' /
                       f'{DATASET_NAME}__{target}__stage2d_{variant}__'
                       f'a_held_out__fold{fold_idx}__frac{frac_pct}')
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Metrics
    metric_list = get_metric_list('reg', target=target)

    # Build model — matches train_graph.py line 1141
    args_template.copolymer_mode = copolymer_mode
    mpnn, trainer = build_copolymer_model_and_trainer(
        args=args_template,
        combined_descriptor_data=orig_Xd,
        scaler=scaler,
        checkpoint_path=checkpoint_path,
        copolymer_mode=copolymer_mode,
        batch_norm=False,               # matches batch_experiments.yaml
        metric_list=metric_list,
        early_stopping_patience=PATIENCE,
        max_epochs=EPOCHS,
        save_checkpoint=True,
    )

    # Dataloaders
    train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE,
                                    num_workers=0, pin_memory=True)
    val_loader   = build_dataloader(val_ds, batch_size=BATCH_SIZE,
                                    num_workers=0, shuffle=False, pin_memory=True)
    test_loader  = build_dataloader(test_ds, batch_size=BATCH_SIZE,
                                    num_workers=0, shuffle=False, pin_memory=True)

    # Skip-training logic (exact copy of train_graph.py)
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
            checkpoint = torch.load(best_ckpt_path, map_location=map_location,
                                    weights_only=False)
            mpnn_fresh.load_state_dict(checkpoint['state_dict'])
            mpnn = mpnn_fresh
            if use_cuda:
                mpnn = mpnn.to(torch.device("cuda"))
            mpnn.eval()
        else:
            logger.warning("  TRAINING_COMPLETE but no checkpoint found. Retraining.")
            done_flag.unlink()

    if not done_flag.exists():
        inprog_flag.touch(exist_ok=True)
        try:
            trainer.fit(mpnn, train_loader, val_loader)
            best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
            if best_ckpt_path:
                with open(checkpoint_path / "best.json", "w") as f:
                    json.dump({"best_ckpt": str(best_ckpt_path),
                               "best_val_loss": best_val_loss}, f, indent=2)
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
    parser = argparse.ArgumentParser(
        description='Stage 2D Learning Curve — Final Pipeline')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only generate group subsampling, do not train')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                        help='Comma-separated fold indices (default: all)')
    parser.add_argument('--fractions', type=str, default='25,50,75,100',
                        help='Training fractions as %% (default: 25,50,75,100)')
    parser.add_argument('--models', type=str, default='2d0_arch,2d1_arch',
                        help='Model variants (default: 2d0_arch,2d1_arch)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config override')
    cli = parser.parse_args()

    folds_to_run = [int(x) for x in cli.folds.split(',')]
    fractions_to_run = [int(x) / 100.0 for x in cli.fractions.split(',')]
    models_to_run = [x.strip() for x in cli.models.split(',')]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2D LEARNING CURVE — FINAL PIPELINE")
    print("=" * 70)
    print(f"  Models     : {models_to_run}")
    print(f"  Folds      : {folds_to_run}")
    print(f"  Fractions  : {[f'{int(f*100)}%' for f in fractions_to_run]}")
    print(f"  EPOCHS     : {EPOCHS}")
    print(f"  PATIENCE   : {PATIENCE}")
    print(f"  BATCH_SIZE : {BATCH_SIZE}")
    print(f"  SEED       : {SEED}")
    print(f"  Dry run    : {cli.dry_run}")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print(f"  Output     : {LC_OUTPUT_DIR}")
    print("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────
    df, smis_A, smis_B, fracA_arr, fracB_arr, orig_Xd = load_and_prepare_data()

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # args template — matches batch_experiments.yaml eaip_base + stage2d entries
    train_args = argparse.Namespace(
        model_name='DMPNN',
        dataset_name=DATASET_NAME,
        polymer_type='copolymer',
        copolymer_mode='stage2d_2d0_arch',    # overwritten per model
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
        results_subdir='HPG2Stage_LC_Final',
        aux_task='off',
        _aux_cols=[],
        _n_aux_targets=0,
    )

    # ── Per-target loop ───────────────────────────────────────────────
    for target in TARGETS:
        logger.info(f"\n{'='*60}\nTarget: {target}\n{'='*60}")

        # Create datapoints + splits (EXACT train_graph.py logic)
        (data_A, data_B, fA, fB, n_dp,
         train_indices, val_indices, test_indices,
         valid_orig_indices) = create_datapoints_and_splits(
            df, smis_A, smis_B, fracA_arr, fracB_arr, orig_Xd, target
        )

        # Build group keys for valid datapoints (using original df indices)
        group_keys_valid = df['group_key'].values[valid_orig_indices]

        # ── Generate / save group subsampling metadata ────────────────
        for fold_idx in folds_to_run:
            tr_full = train_indices[fold_idx]
            va = val_indices[fold_idx]
            te = test_indices[fold_idx]

            train_groups_full = group_keys_valid[tr_full]
            unique_train_groups = np.unique(train_groups_full)
            logger.info(
                f"Fold {fold_idx}: {len(tr_full)} train, {len(va)} val, "
                f"{len(te)} test | {len(unique_train_groups)} training groups"
            )

            for frac in fractions_to_run:
                tr_sub, selected_groups = subsample_training_groups(
                    tr_full, group_keys_valid, frac, SEED, fold_idx
                )
                frac_pct = int(frac * 100)
                frac_key = f"fold{fold_idx}_frac{frac_pct}"

                # Metadata
                meta = {
                    'fold': fold_idx,
                    'fraction': frac,
                    'target': target,
                    'n_original_train': int(len(tr_full)),
                    'n_subsampled_train': int(len(tr_sub)),
                    'n_original_groups': int(len(unique_train_groups)),
                    'n_selected_groups': int(len(selected_groups)),
                    'n_val': int(len(va)),
                    'n_test': int(len(te)),
                    'n_valid_datapoints': int(n_dp),
                    'epochs': EPOCHS,
                    'patience': PATIENCE,
                    'seed': SEED,
                }

                # Save selected group keys
                group_file = LC_OUTPUT_DIR / f'selected_groups_{frac_key}.json'
                if not group_file.exists():
                    # Only write once (first target); groups are target-independent
                    with open(group_file, 'w') as f:
                        json.dump({
                            'selected_groups': sorted(selected_groups),
                            'n_groups': len(selected_groups),
                            'n_samples': int(len(tr_sub)),
                            'n_unique_A': len(set(
                                g.split('||')[0] for g in selected_groups)),
                            'n_unique_B': len(set(
                                g.split('||')[1] for g in selected_groups)),
                            'n_unique_pairs': len(set(
                                g.split('||')[0] + '||' + g.split('||')[1]
                                for g in selected_groups)),
                        }, f, indent=2)

                # Save metadata
                meta_file = LC_OUTPUT_DIR / f'metadata_{frac_key}.json'
                if not meta_file.exists():
                    with open(meta_file, 'w') as f:
                        json.dump(meta, f, indent=2)

                logger.info(
                    f"  {frac_key}: {len(tr_sub)} samples "
                    f"({len(selected_groups)}/{len(unique_train_groups)} groups)"
                )

        if cli.dry_run:
            continue

        # ── Training loop ─────────────────────────────────────────────
        for fold_idx in folds_to_run:
            tr_full = train_indices[fold_idx]
            va = val_indices[fold_idx]
            te = test_indices[fold_idx]

            for frac in fractions_to_run:
                frac_pct = int(frac * 100)
                frac_key = f"fold{fold_idx}_frac{frac_pct}"

                # Reload selected groups from saved file
                if frac >= 1.0:
                    tr_sub = tr_full
                else:
                    group_file = LC_OUTPUT_DIR / f'selected_groups_{frac_key}.json'
                    with open(group_file, 'r') as f:
                        info = json.load(f)
                    selected_groups = set(info['selected_groups'])
                    train_groups = group_keys_valid[tr_full]
                    mask = np.array([g in selected_groups for g in train_groups])
                    tr_sub = tr_full[mask]

                for variant in models_to_run:
                    logger.info(
                        f"\n  Training: {variant} | fold={fold_idx} | "
                        f"frac={frac_pct}% | n_train={len(tr_sub)}"
                    )

                    y_true, y_pred = train_single_run(
                        data_A, data_B, fA, fB, orig_Xd,
                        tr_sub, va, te,
                        variant, target, fold_idx, frac,
                        featurizer, train_args,
                    )

                    # Save predictions with original df indices
                    orig_test_indices = valid_orig_indices[te]
                    pred_file = (
                        PREDICTIONS_DIR /
                        f'{DATASET_NAME}__{target}__stage2d_{variant}__'
                        f'a_held_out__fold{fold_idx}__frac{frac_pct}.npz'
                    )
                    np.savez_compressed(
                        pred_file,
                        y_true=y_true,
                        y_pred=y_pred,
                        test_indices=orig_test_indices,
                        fold=fold_idx,
                        fraction=frac,
                        n_train=len(tr_sub),
                        variant=variant,
                        target=target,
                    )
                    logger.info(f"    Saved: {pred_file.name}")

    if cli.dry_run:
        print("\n[DRY RUN] Group subsampling complete. No training.")
        print(f"  Metadata: {LC_OUTPUT_DIR}")
        return

    print("\n" + "=" * 70)
    print("LEARNING CURVE TRAINING COMPLETE")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print(f"  Metadata   : {LC_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    set_seed(SEED)
    import lightning.pytorch as pl
    pl.seed_everything(SEED, workers=True)
    main()
