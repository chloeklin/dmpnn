#!/usr/bin/env python
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from utils import (
    set_seed,
    setup_training_environment,
    load_and_preprocess_data,
    process_data,
    determine_split_strategy,
    generate_data_splits,
    build_experiment_paths,
    embedding_files,
    have_all_embeddings,
)



from evaluate_utils import fit_and_score_baselines, extract_embeddings_for_rep

def save_results_table(results_df: pd.DataFrame, results_dir: Path, args, descriptor_columns: List[str] | None) -> Path:
    out_dir = results_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [args.dataset_name]
    if args.target:
        parts.append(args.target)
    parts.append("attentivefp_baselines")
    out_csv = out_dir / ("__".join(parts) + ".csv")

    # Column ordering
    base_cols = ["target", "split", "baseline"]
    metric_cols = [c for c in results_df.columns if c.startswith("test/")] \
                  or (["test/mae", "test/r2", "test/rmse"] if args.task_type == "reg" else ["test/accuracy", "test/f1", "test/roc_auc"])
    extra_cols = [c for c in results_df.columns if c not in set(base_cols + metric_cols)]

    results_df = results_df[base_cols + metric_cols + extra_cols]
    results_df.to_csv(out_csv, index=False)
    return out_csv





def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    from utils import (
        create_base_argument_parser,
        add_model_specific_args,
        validate_train_size_argument,
    )

    parser = create_base_argument_parser("AttentiveFP baselines on learned embeddings")
    parser = add_model_specific_args(parser, "attentivefp")
    parser.add_argument("--force_recompute", action="store_true", help="Recompute embeddings even if cached files exist")
    args = parser.parse_args()

    validate_train_size_argument(args, parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_info = setup_training_environment(args, model_type="graph")
    chemprop_dir = setup_info['chemprop_dir']
    checkpoint_dir = setup_info['checkpoint_dir']
    results_dir = setup_info['results_dir']

    smiles_column = setup_info['smiles_column']
    descriptor_columns = setup_info['descriptor_columns']
    SEED = setup_info['SEED']
    REPLICATES = setup_info['REPLICATES']

    set_seed(SEED)

    df_input, target_columns = load_and_preprocess_data(args, setup_info)
    if args.target:
        if args.target not in target_columns:
            logger.error(f"Target '{args.target}' not found. Available: {target_columns}")
            raise SystemExit(1)
        target_columns = [args.target]

    _, df_input, combined_descriptor_data, n_classes_per_target = process_data(
        df_input, smiles_column, descriptor_columns, target_columns, args
    )

    emb_dir = results_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for target in target_columns:
        y_all = df_input.loc[:, target].astype(float if args.task_type == 'reg' else int).to_numpy()

        n_splits, local_reps = determine_split_strategy(len(y_all), REPLICATES)
        train_indices, val_indices, test_indices = generate_data_splits(args, y_all.reshape(-1, 1), n_splits, local_reps, SEED)

        for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
            # Build experiment naming for this split
            ckpt_path, preprocessing_path, desc_suf, rdkit_suf, bn_suf, size_suf, fusion_suf, aux_suf = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            embedding_prefix = f"{args.dataset_name}__{args.model_name}__{target}{desc_suf}{rdkit_suf}{bn_suf}{fusion_suf}{aux_suf}{size_suf}"
            mask_f, Xtr_f, Xva_f, Xte_f = embedding_files(emb_dir, embedding_prefix, i)

            # Ensure embeddings exist (compute if missing and we can load a checkpoint)
            if not have_all_embeddings(mask_f, Xtr_f, Xva_f, Xte_f) or args.force_recompute:
                smis = df_input[smiles_column].tolist()
                extract_embeddings_for_rep(
                    args, setup_info, target, i, tr, va, te,
                    smis, df_input, combined_descriptor_data,
                    ckpt_path, emb_dir, embedding_prefix, n_classes_per_target, device
                )

            # Load embeddings
            if not have_all_embeddings(mask_f, Xtr_f, Xva_f, Xte_f):
                logger.warning(f"[{target}] split {i}: embeddings still missing after attempt; skipping")
                continue
            
            X_train = np.load(Xtr_f)
            X_val   = np.load(Xva_f)
            X_test  = np.load(Xte_f)
            
            # Load target scaler from checkpoint for proper scaling
            target_scaler = None
            if args.task_type == 'reg':
                checkpoint_file = ckpt_path / "best.pt"
                if checkpoint_file.exists():
                    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                    target_scaler = checkpoint.get("target_scaler", None)
                    if target_scaler is None:
                        logger.warning(f"[{target}] split {i}: No target_scaler found in checkpoint, using unscaled targets")
                else:
                    logger.warning(f"[{target}] split {i}: Checkpoint file not found, using unscaled targets")
            
            # Align y sizes with potential dropped samples and apply scaling
            y_train = y_all[tr][:len(X_train)]
            y_val   = y_all[va][:len(X_val)]
            y_test  = y_all[te][:len(X_test)]
            
            # Apply target scaling if available (for regression)
            if target_scaler is not None and args.task_type == 'reg':
                y_train = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
                y_val = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
                # Note: y_test stays unscaled for evaluation (same as AttentiveFP training)

            # Fit baselines and collect scores
            n_classes = n_classes_per_target.get(target)
            variant_bits = [s for s in [desc_suf, rdkit_suf, bn_suf, size_suf] if s]
            variant_label = "+".join(b.lstrip("_") for b in variant_bits) or "default"
            split_scores = fit_and_score_baselines(X_train, y_train, X_val, y_val, X_test, y_test, args.task_type, n_classes,
                                                    dataset_name=args.dataset_name,
                                                    model_name=args.model_name,
                                                    target=target,
                                                    replicate=i,
                                                    variant_label=variant_label)
            for name, metrics in split_scores.items():
                rows.append({
                    "target": target,
                    "split": i,
                    "baseline": name,
                    **metrics,
                })

    if rows:
        df = pd.DataFrame(rows)
        out_csv = save_results_table(df, results_dir, args, descriptor_columns)
        logging.info(f"✅ Saved baseline results to {out_csv}")
    else:
        logging.warning("No rows to save — did we skip all splits due to missing checkpoints/embeddings?")


if __name__ == "__main__":
    main()
