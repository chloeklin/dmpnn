"""
Stage 2D Ablation Study
========================
Controlled ablation comparing alternative chemistry–architecture fusion
mechanisms against the 2D1 baseline:

  1. 2D1         — additive residual (baseline)
  2. 2D1_FiLM    — FiLM architecture modulation
  3. 2D1_NLMix   — nonlinear composition pooling
  4. 2D1_FiLM_NLMix — both modifications combined

Evaluation:
  - Leave-One-Monomer-A-Out (LOMAO) split → 9 folds
  - Metrics: EA/IP R², MAE  +  ΔEA/ΔIP R², MAE (architecture recovery)

Outputs:
  - predictions/HPG2Stage_Ablation/*.npz
  - output/stage2d_ablation/results_ablation.csv
  - output/stage2d_ablation/ablation_summary.md

Usage:
    python run_stage2d_ablation.py [--dry_run] [--folds 0,1,2,3,4,5,6,7,8]
        [--models 2d1_arch,2d1_film,2d1_nlmix,2d1_film_nlmix]
"""

import logging
import time
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error

# ── Path setup ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from chemprop import data, featurizers
from chemprop.nn.stage2d import ARCH_LABEL_MAP, Stage2Aggregator
from utils import (
    set_seed, build_copolymer_model_and_trainer, get_metric_list,
    create_copolymer_data, canonicalize_smiles, pick_best_checkpoint,
    generate_a_held_out_splits,
)

# ── Configuration ───────────────────────────────────────────────────
DATA_PATH = ROOT / 'data' / 'ea_ip.csv'
PREDICTIONS_DIR = ROOT / 'predictions' / 'HPG2Stage_Ablation'
OUTPUT_DIR = ROOT / 'output' / 'stage2d_ablation'
DATASET_NAME = 'ea_ip'

MODELS = {
    '2D1':              '2d1_arch',
    '2D1_FiLM':         '2d1_film',
    '2D1_NonlinearMix': '2d1_nlmix',
    '2D1_FiLM_NonlinearMix': '2d1_film_nlmix',
}
TARGETS = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
TARGET_SHORT = {'EA vs SHE (eV)': 'EA', 'IP vs SHE (eV)': 'IP'}
SEED = 42

# Training hyperparameters (match original HPG2Stage runs)
EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_single_run(data_A, data_B, fA, fB, orig_Xd,
                     train_idx, val_idx, test_idx,
                     variant_key, variant_code, target, fold_idx,
                     featurizer, args_template):
    """Train a single model run and return (y_true, y_pred, elapsed_time)."""
    from chemprop.data import CopolymerDataset, build_dataloader

    copolymer_mode = f'stage2d_{variant_code}'

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

    # Normalize targets
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)

    # Checkpoint path
    checkpoint_path = (ROOT / 'checkpoints' / 'HPG2Stage_Ablation' /
                       f'{DATASET_NAME}__{target}__{copolymer_mode}__lomao__fold{fold_idx}')
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
    val_loader   = build_dataloader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
    test_loader  = build_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)

    # Skip if already done
    done_flag = checkpoint_path / "TRAINING_COMPLETE"
    inprog_flag = checkpoint_path / "TRAINING_IN_PROGRESS"

    t0 = time.time()

    if done_flag.exists():
        best_ckpt_path, _ = pick_best_checkpoint(checkpoint_path)
        if best_ckpt_path is not None:
            logger.info(f"  Skipping (COMPLETE): {checkpoint_path.name}")
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
            logger.warning(f"  TRAINING_COMPLETE but no checkpoint. Retraining.")
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

    elapsed = time.time() - t0

    # Get predictions
    y_pred = trainer.predict(model=mpnn, dataloaders=test_loader)
    y_true = np.array([test_ds[j].y for j in range(len(test_ds))], dtype=float)

    if isinstance(y_pred, list):
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    elif hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()

    # Count trainable parameters
    n_params = sum(p.numel() for p in mpnn.parameters() if p.requires_grad)

    return y_true.flatten(), y_pred.flatten(), elapsed, n_params


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE-DEVIATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_arch_deviation_metrics(y_true, y_pred, test_idx, df):
    """Compute R²(Δ) and MAE(Δ) for architecture recovery.

    Δ = deviation from the composition-group mean, where a group is
    defined by (smiles_A, smiles_B, fracA).
    """
    sub = df.iloc[test_idx].copy()
    sub['y_true'] = y_true
    sub['y_pred'] = y_pred

    # Group key: same (A, B, fracA) but different poly_type
    sub['group_key'] = (sub['smiles_A'].astype(str) + '||' +
                        sub['smiles_B'].astype(str) + '||' +
                        sub['fracA'].astype(str))

    # Remove singleton groups (no architecture variation to recover)
    grp_counts = sub['group_key'].value_counts()
    multi_arch = grp_counts[grp_counts > 1].index
    sub = sub[sub['group_key'].isin(multi_arch)]

    if len(sub) < 4:
        return np.nan, np.nan

    group_mean_true = sub.groupby('group_key')['y_true'].transform('mean')
    group_mean_pred = sub.groupby('group_key')['y_pred'].transform('mean')

    delta_true = sub['y_true'].values - group_mean_true.values
    delta_pred = sub['y_pred'].values - group_mean_pred.values

    r2_d = r2_score(delta_true, delta_pred)
    mae_d = mean_absolute_error(delta_true, delta_pred)
    return r2_d, mae_d


# ═══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_ablation_summary(results_df, param_counts, timing):
    """Generate ablation_summary.md comparing all variants."""
    lines = []
    lines.append("# Stage 2D Ablation Study: Fusion Mechanism Comparison")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This study tests whether alternative chemistry–architecture fusion")
    lines.append("mechanisms improve upon the baseline 2D1 (additive residual) model.")
    lines.append("")
    lines.append("- **Evaluation split**: Leave-One-Monomer-A-Out (LOMAO), 9 folds")
    lines.append("- **Dataset**: ea_ip (42,966 copolymers, 9 unique monomer A)")
    lines.append("- **Targets**: EA vs SHE (eV), IP vs SHE (eV)")
    lines.append("")

    # Baseline row
    baseline_name = '2D1'

    # Per-model sections
    lines.append("## Model Descriptions")
    lines.append("")

    descriptions = {
        '2D1': (
            "**Baseline**: Additive architecture residual.\n"
            "h_poly = h_mix + α_arch · r_arch, where r_arch = MLP(z)."
        ),
        '2D1_FiLM': (
            "**FiLM Architecture Modulation**: Multiplicative scaling of residual.\n"
            "γ, β = MLP(e_arch); h_poly = h_mix + γ ⊙ (α_arch · r_arch) + β.\n"
            "Tests: does architecture modulate rather than simply offset?"
        ),
        '2D1_NonlinearMix': (
            "**Nonlinear Composition Pooling**: Nonlinear chemistry interaction.\n"
            "h_mix_NL = f_A·h_A + f_B·h_B + f_A·f_B·g([h_A, h_B, |h_A−h_B|, h_A⊙h_B]).\n"
            "Tests: does nonlinear monomer interaction improve prediction?"
        ),
        '2D1_FiLM_NonlinearMix': (
            "**Combined**: Both FiLM modulation and nonlinear composition pooling.\n"
            "Tests: do the two modifications complement each other?"
        ),
    }

    for model_name in MODELS.keys():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(descriptions.get(model_name, ""))
        lines.append("")
        if model_name in param_counts:
            lines.append(f"- **Trainable parameters**: {param_counts[model_name]:,}")
        if model_name in timing:
            total_s = timing[model_name]
            lines.append(f"- **Total training time**: {total_s:.0f}s ({total_s/60:.1f}min)")
        lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")

    # Aggregate results: mean ± std across folds
    agg_rows = []
    for model_name in MODELS.keys():
        sub = results_df[results_df['Model'] == model_name]
        if sub.empty:
            continue
        row = {'Model': model_name}
        for col in ['EA_R2', 'IP_R2', 'DeltaEA_R2', 'DeltaIP_R2',
                     'EA_MAE', 'IP_MAE', 'DeltaEA_MAE', 'DeltaIP_MAE']:
            vals = sub[col].dropna()
            if len(vals) > 0:
                row[col] = f"{vals.mean():.4f} ± {vals.std():.4f}"
            else:
                row[col] = "N/A"
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    # Markdown table
    header = "| " + " | ".join(agg_df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(agg_df.columns)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in agg_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    lines.append("")

    # Comparison with baseline
    lines.append("## Comparison with Baseline (2D1)")
    lines.append("")

    baseline_sub = results_df[results_df['Model'] == baseline_name]
    for model_name in MODELS.keys():
        if model_name == baseline_name:
            continue
        variant_sub = results_df[results_df['Model'] == model_name]
        if variant_sub.empty or baseline_sub.empty:
            continue

        lines.append(f"### {model_name} vs {baseline_name}")
        lines.append("")

        for col in ['EA_R2', 'IP_R2', 'DeltaEA_R2', 'DeltaIP_R2',
                     'EA_MAE', 'IP_MAE', 'DeltaEA_MAE', 'DeltaIP_MAE']:
            bl_vals = baseline_sub[col].dropna().values
            var_vals = variant_sub[col].dropna().values
            if len(bl_vals) == 0 or len(var_vals) == 0:
                continue
            bl_mean = bl_vals.mean()
            var_mean = var_vals.mean()
            diff = var_mean - bl_mean
            if abs(bl_mean) > 1e-10:
                pct = 100 * diff / abs(bl_mean)
            else:
                pct = 0.0

            # For R² higher is better; for MAE lower is better
            if 'R2' in col:
                better = "↑" if diff > 0 else "↓"
            else:
                better = "↓" if diff < 0 else "↑"

            lines.append(
                f"- **{col}**: {var_mean:.4f} vs {bl_mean:.4f} "
                f"(Δ = {diff:+.4f}, {pct:+.2f}%) {better}"
            )
        lines.append("")

    # Conclusion
    lines.append("## Conclusions")
    lines.append("")
    lines.append("### 1. Does multiplicative architecture modulation outperform additive fusion?")
    lines.append("")
    # Compute
    bl_ea = results_df[results_df['Model'] == '2D1']['EA_R2'].dropna().mean()
    bl_ip = results_df[results_df['Model'] == '2D1']['IP_R2'].dropna().mean()
    film_sub = results_df[results_df['Model'] == '2D1_FiLM']
    if not film_sub.empty:
        film_ea = film_sub['EA_R2'].dropna().mean()
        film_ip = film_sub['IP_R2'].dropna().mean()
        if film_ea > bl_ea and film_ip > bl_ip:
            lines.append("FiLM modulation improves both EA and IP R² over additive fusion.")
        elif film_ea > bl_ea or film_ip > bl_ip:
            lines.append("FiLM modulation improves one target but not both.")
        else:
            lines.append("FiLM modulation does not improve over additive fusion.")
    else:
        lines.append("FiLM results not available.")
    lines.append("")

    lines.append("### 2. Does nonlinear composition pooling improve prediction?")
    lines.append("")
    nlmix_sub = results_df[results_df['Model'] == '2D1_NonlinearMix']
    if not nlmix_sub.empty:
        nlmix_ea = nlmix_sub['EA_R2'].dropna().mean()
        nlmix_ip = nlmix_sub['IP_R2'].dropna().mean()
        if nlmix_ea > bl_ea and nlmix_ip > bl_ip:
            lines.append("Nonlinear composition pooling improves both EA and IP R².")
        elif nlmix_ea > bl_ea or nlmix_ip > bl_ip:
            lines.append("Nonlinear composition pooling improves one target but not both.")
        else:
            lines.append("Nonlinear composition pooling does not improve over linear pooling.")
    else:
        lines.append("NonlinearMix results not available.")
    lines.append("")

    lines.append("### 3. Do the two modifications complement each other?")
    lines.append("")
    combined_sub = results_df[results_df['Model'] == '2D1_FiLM_NonlinearMix']
    if not combined_sub.empty:
        comb_ea = combined_sub['EA_R2'].dropna().mean()
        comb_ip = combined_sub['IP_R2'].dropna().mean()
        best_single = max(
            film_sub['EA_R2'].dropna().mean() if not film_sub.empty else -1,
            nlmix_sub['EA_R2'].dropna().mean() if not nlmix_sub.empty else -1,
        )
        if comb_ea > best_single:
            lines.append("The combined model outperforms both individual modifications on EA R².")
        else:
            lines.append("The combined model does not outperform the best individual modification on EA R².")
    else:
        lines.append("Combined results not available.")
    lines.append("")

    lines.append("### 4. Which model should be carried forward?")
    lines.append("")
    # Find best model by mean of EA_R2 + IP_R2
    best_model = None
    best_score = -np.inf
    for model_name in MODELS.keys():
        sub = results_df[results_df['Model'] == model_name]
        if sub.empty:
            continue
        score = sub['EA_R2'].dropna().mean() + sub['IP_R2'].dropna().mean()
        if score > best_score:
            best_score = score
            best_model = model_name
    if best_model:
        lines.append(f"**Recommendation: {best_model}** (highest mean EA+IP R² = {best_score/2:.4f})")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Stage 2D Ablation Study')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only generate splits, do not train')
    parser.add_argument('--folds', type=str, default=None,
                        help='Comma-separated fold indices (default: all 9)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model keys (default: all 4)')
    parser.add_argument('--target', type=str, default=None,
                        help="Run a single target: 'EA' or 'IP' (default: both)")
    cli_args = parser.parse_args()

    # Map short target names to full column names
    target_map = {'EA': 'EA vs SHE (eV)', 'IP': 'IP vs SHE (eV)'}
    if cli_args.target is not None:
        if cli_args.target not in target_map:
            raise ValueError(f"--target must be one of {list(target_map.keys())}, got {cli_args.target}")
        cli_args.target = target_map[cli_args.target]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows from {DATA_PATH}")

    # ── LOMAO splits ──────────────────────────────────────────────────
    smis_A_raw = df['smiles_A'].astype(str).values
    train_indices, val_indices, test_indices, held_out_monomers = \
        generate_a_held_out_splits(
            smis_A_raw, len(df), SEED,
            protocol='leave_one_A_out', logger=logger,
        )
    n_folds = len(train_indices)
    logger.info(f"LOMAO: {n_folds} folds")

    folds_to_run = list(range(n_folds))
    if cli_args.folds is not None:
        folds_to_run = [int(x) for x in cli_args.folds.split(',')]

    models_to_run = list(MODELS.keys())
    if cli_args.models is not None:
        models_to_run = [x.strip() for x in cli_args.models.split(',')]

    print("=" * 70)
    print("STAGE 2D ABLATION STUDY")
    print("=" * 70)
    print(f"  Models: {models_to_run}")
    print(f"  Folds:  {folds_to_run} (of {n_folds})")
    print(f"  Dry run: {cli_args.dry_run}")
    print("=" * 70)

    if cli_args.dry_run:
        print("[DRY RUN] Split generation complete. Exiting.")
        return

    # ── Prepare data ──────────────────────────────────────────────────
    smis_A = df['smiles_A'].astype(str).values
    smis_B = df['smiles_B'].astype(str).values
    fracA_arr = df['fracA'].astype(float).values
    fracB_arr = df['fracB'].astype(float).values

    # Stage2D ordinal architecture encoding
    arch_ordinal = df['poly_type'].astype(str).str.lower().str.strip().map(ARCH_LABEL_MAP)
    orig_Xd = arch_ordinal.values.astype(np.float32).reshape(-1, 1)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_args = argparse.Namespace(
        model_name='DMPNN',
        dataset_name=DATASET_NAME,
        polymer_type='copolymer',
        copolymer_mode='stage2d_2d1_arch',
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
        results_subdir='HPG2Stage_Ablation',
        aux_task='off',
        _aux_cols=[],
        _n_aux_targets=0,
    )

    # ── Training loop ─────────────────────────────────────────────────
    all_results = []
    param_counts = {}
    timing = defaultdict(float)

    targets_to_run = TARGETS if cli_args.target is None else [cli_args.target]
    for target in targets_to_run:
        target_short = TARGET_SHORT[target]
        logger.info(f"\n{'='*60}\nTarget: {target}\n{'='*60}")

        ys = df[target].astype(float).values.reshape(-1, 1)

        data_A, data_B, fA, fB = create_copolymer_data(
            smis_A, smis_B, fracA_arr, fracB_arr, ys,
            orig_Xd, 'DMPNN',
        )
        logger.info(f"  Created {len(data_A)} copolymer datapoints for {target}")

        for fold_idx in folds_to_run:
            tr = train_indices[fold_idx]
            va = val_indices[fold_idx]
            te = test_indices[fold_idx]
            held_out = held_out_monomers[fold_idx] if held_out_monomers else f"fold{fold_idx}"

            for model_name in models_to_run:
                variant_code = MODELS[model_name]
                logger.info(
                    f"\n  {model_name} ({variant_code}) | fold={fold_idx} | "
                    f"held_out={held_out} | "
                    f"n_train={len(tr)} | n_val={len(va)} | n_test={len(te)}"
                )

                y_true, y_pred, elapsed, n_params = train_single_run(
                    data_A, data_B, fA, fB, orig_Xd,
                    tr, va, te,
                    model_name, variant_code, target, fold_idx,
                    featurizer, train_args,
                )

                param_counts[model_name] = n_params
                timing[model_name] += elapsed

                # Overall metrics
                r2_val = r2_score(y_true, y_pred)
                mae_val = mean_absolute_error(y_true, y_pred)

                # Architecture-deviation metrics
                r2_d, mae_d = compute_arch_deviation_metrics(
                    y_true, y_pred, te, df,
                )

                row = {
                    'Model': model_name,
                    'Fold': fold_idx,
                    'HeldOut': held_out,
                    'Target': target_short,
                }
                if target_short == 'EA':
                    row.update({
                        'EA_R2': r2_val,
                        'EA_MAE': mae_val,
                        'DeltaEA_R2': r2_d,
                        'DeltaEA_MAE': mae_d,
                    })
                else:
                    row.update({
                        'IP_R2': r2_val,
                        'IP_MAE': mae_val,
                        'DeltaIP_R2': r2_d,
                        'DeltaIP_MAE': mae_d,
                    })
                all_results.append(row)

                logger.info(
                    f"    {target_short} R²={r2_val:.4f} MAE={mae_val:.4f} | "
                    f"Δ{target_short} R²={r2_d:.4f} MAE(Δ)={mae_d:.4f}"
                )

                # Save predictions
                pred_file = (PREDICTIONS_DIR /
                             f'{DATASET_NAME}__{target}__stage2d_{variant_code}__lomao__fold{fold_idx}.npz')
                np.savez_compressed(
                    pred_file,
                    y_true=y_true,
                    y_pred=y_pred,
                    test_indices=te,
                    fold=fold_idx,
                    held_out_monomer=held_out,
                    n_train=len(tr),
                    n_val=len(va),
                    n_test=len(te),
                )
                logger.info(f"    Saved: {pred_file.name}")

    # ── Aggregate results ─────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    # Pivot: combine EA and IP rows for same (Model, Fold)
    ea_df = results_df[results_df['Target'] == 'EA'].drop(columns=['Target']).reset_index(drop=True)
    ip_df = results_df[results_df['Target'] == 'IP'].drop(columns=['Target']).reset_index(drop=True)
    merged = ea_df.merge(ip_df, on=['Model', 'Fold', 'HeldOut'], how='outer')

    # Ensure column order
    col_order = ['Model', 'Fold', 'HeldOut',
                 'EA_R2', 'IP_R2', 'DeltaEA_R2', 'DeltaIP_R2',
                 'EA_MAE', 'IP_MAE', 'DeltaEA_MAE', 'DeltaIP_MAE']
    for c in col_order:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[col_order]

    # Save per-fold results
    merged.to_csv(OUTPUT_DIR / 'results_ablation.csv', index=False)
    logger.info(f"\nSaved: {OUTPUT_DIR / 'results_ablation.csv'}")

    # Also save the summary CSV in the format requested
    summary_cols = ['Model', 'EA_R2', 'IP_R2', 'DeltaEA_R2', 'DeltaIP_R2',
                    'EA_MAE', 'IP_MAE', 'DeltaEA_MAE', 'DeltaIP_MAE']
    summary = merged.groupby('Model')[summary_cols[1:]].mean().reset_index()
    summary = summary[summary_cols]
    summary.to_csv(OUTPUT_DIR / 'results_ablation_summary.csv', index=False)
    logger.info(f"Saved: {OUTPUT_DIR / 'results_ablation_summary.csv'}")

    # ── Generate ablation_summary.md ──────────────────────────────────
    md = generate_ablation_summary(merged, param_counts, timing)
    with open(OUTPUT_DIR / 'ablation_summary.md', 'w') as f:
        f.write(md)
    logger.info(f"Saved: {OUTPUT_DIR / 'ablation_summary.md'}")

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print(f"Results:     {OUTPUT_DIR / 'results_ablation.csv'}")
    print(f"Summary:     {OUTPUT_DIR / 'results_ablation_summary.csv'}")
    print(f"Report:      {OUTPUT_DIR / 'ablation_summary.md'}")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    set_seed(SEED)
    main()
