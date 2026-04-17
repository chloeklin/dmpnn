import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import os, json
import torch
from dataclasses import replace
from pathlib import Path

from chemprop import data, featurizers
from utils import (set_seed, process_data, 
                  create_all_data, build_model_and_trainer, get_metric_list,
                  build_experiment_paths, validate_checkpoint_compatibility, manage_preprocessing_cache,
                  setup_training_environment, load_and_preprocess_data, determine_split_strategy, 
                  generate_data_splits, save_aggregate_results, get_encodings_from_loader, save_predictions,
                  create_base_argument_parser, add_model_specific_args, validate_train_size_argument,
                  setup_model_environment, save_model_results, pick_best_checkpoint,
                  create_copolymer_data, create_multi_monomer_copolymer_data,
                  build_copolymer_model_and_trainer,
                  generate_a_held_out_splits, save_fold_assignments, canonicalize_smiles)

# polymer_input integration — canonical polymer spec utilities
# These are used for optional PolymerSpec-based validation and scalar-feature
# extraction when --polymer_input flag is set.  The existing SMILES-based flow
# remains the default; polymer_input provides structured parsing & validation.
try:
    from polymer_input import (
        PolymerParser, SchemaMapping, validate_polymer_spec,
        extract_scalar_features, collect_scalar_keys,
    )
    from polymer_input.featurizers.dmpnn import DMPNNFeaturizer
    from polymer_input.featurizers.ppg import PPGFeaturizer
    from polymer_input.featurizers.wdmpnn import WDMPNNFeaturizer
    _HAS_POLYMER_INPUT = True
except ImportError:
    _HAS_POLYMER_INPUT = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments using modular parser
parser = create_base_argument_parser('Train a Chemprop model for regression or classification')
parser = add_model_specific_args(parser, "dmpnn")

args = parser.parse_args()

# Validate arguments
validate_train_size_argument(args, parser)

# Validate split_type compatibility
if getattr(args, 'split_type', 'random') == 'a_held_out' and getattr(args, 'polymer_type', 'homo') != 'copolymer':
    parser.error("--split_type a_held_out requires --polymer_type copolymer (needs smiles_A column)")

# Validate auxiliary task arguments
aux_task = getattr(args, 'aux_task', 'off')
if aux_task == 'predict_descriptors':
    if not getattr(args, 'aux_descriptor_cols', None):
        parser.error("--aux_descriptor_cols is required when --aux_task=predict_descriptors")
    if getattr(args, 'incl_desc', False) or getattr(args, 'incl_rdkit', False):
        parser.error("--aux_task=predict_descriptors is incompatible with --incl_desc and --incl_rdkit. "
                      "Descriptors must NOT be model inputs in aux_task mode.")
    if getattr(args, 'fusion_mode', 'late_concat') == 'film':
        parser.error("--aux_task=predict_descriptors is incompatible with --fusion_mode=film")
    # Parse comma-separated column names
    args._aux_cols = [c.strip() for c in args.aux_descriptor_cols.split(',')]
    args._n_aux_targets = len(args._aux_cols)
    logger.info(f"Auxiliary task: predict_descriptors with columns {args._aux_cols}, lambda_aux={args.lambda_aux}")
else:
    args._aux_cols = []
    args._n_aux_targets = 0

# Setup training environment with common configuration
setup_info = setup_model_environment(args, "dmpnn")

# Extract commonly used variables for backward compatibility
config = setup_info['config']
chemprop_dir = setup_info['chemprop_dir']
checkpoint_dir = setup_info['checkpoint_dir']
results_dir = setup_info['results_dir']

# Create predictions directory if saving predictions
if args.save_predictions:
    predictions_dir = chemprop_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Predictions will be saved to: {predictions_dir}")
else:
    predictions_dir = None
smiles_column = setup_info['smiles_column']
ignore_columns = setup_info['ignore_columns']
descriptor_columns = setup_info['descriptor_columns']
SEED = setup_info['SEED']
REPLICATES = setup_info['REPLICATES']
EPOCHS = setup_info['EPOCHS']
PATIENCE = setup_info['PATIENCE']
num_workers = setup_info['num_workers']

# Check model configuration
model_config = config['MODELS'].get(args.model_name, {})
if not model_config:
    logger.warning(f"No configuration found for model '{args.model_name}'. Using defaults.")

# === Set Random Seed ===
set_seed(SEED)
import lightning.pytorch as pl
pl.seed_everything(SEED, workers=True)

# === Load and Preprocess Data ===
df_input, target_columns = load_and_preprocess_data(args, setup_info)

# Debug: Show what columns we're working with
logger.info(f"DataFrame columns after preprocessing: {list(df_input.columns)}")
logger.info(f"Detected target columns: {target_columns}")
logger.info(f"DataFrame shape: {df_input.shape}")

# Check for any remaining string columns
for col in df_input.columns:
    if df_input[col].dtype == 'object':
        logger.warning(f"Column '{col}' still has object dtype after preprocessing")


if args.pretrain_monomer:
    assert args.model_name == "DMPNN", "Monomer pretraining uses the small-molecule D-MPNN."
    # Parse multiclass specs
    mc_map = {}
    if args.multiclass_targets:
        for tok in args.multiclass_targets.split(","):
            name, k = tok.split(":")
            mc_map[name.strip()] = int(k)

    # Build task_specs aligned with target_columns
    task_specs = []
    for t in target_columns:
        if t in mc_map:
            task_specs.append(("multi", mc_map[t]))
        else:
            task_specs.append(("reg", None))
    args.task_specs = task_specs

    # Select the mixed head
    args.task_type = 'mixed-reg-multi'

    # Use all numeric target columns at once (masked multitask)
    ys_df = df_input[target_columns].copy()  
    
    # Debug: Check for non-numeric columns
    for col in target_columns:
        if ys_df[col].dtype == 'object':
            logger.warning(f"Target column '{col}' has object dtype. Sample values: {ys_df[col].dropna().head().tolist()}")
            # Try to convert to numeric
            try:
                ys_df[col] = pd.to_numeric(ys_df[col], errors='coerce')
                logger.info(f"Successfully converted '{col}' to numeric")
            except Exception as e:
                logger.error(f"Failed to convert '{col}' to numeric: {e}")
    
    # Factorize multiclass columns (0..K-1), keep NaNs
    for tname in target_columns:
        if tname in mc_map:
            col = ys_df[tname]
            not_nan = col.notna()
            codes, classes = pd.factorize(col[not_nan].astype(str), sort=True)
            tmp = pd.Series(np.nan, index=col.index, dtype=float)
            tmp.loc[not_nan] = codes.astype(float)
            ys_df[tname] = tmp

    # Build datapoints with ALL targets
    smis, df_input, combined_descriptor_data, _ = process_data(
        df_input, smiles_column, descriptor_columns, target_columns, args
    )
    

    # Generate splits FIRST (we'll need train indices for stats)
    ys_full = ys_df.values  # shape [N, T] with NaNs
    first_mc = next((i for i,(k,_) in enumerate(task_specs) if k == "multi"), None)

    # Choose a stratification column if we have a multiclass task
    y_strat = ys_df.iloc[:, first_mc].values if first_mc is not None else None

    # --- PATCH: if stratification labels contain NaNs, fallback to unstratified ---
    use_strat = (first_mc is not None)
    if use_strat:
        y_strat_np = y_strat.astype(float)
        if np.isnan(y_strat_np).any():
            logger.warning("NaNs found in stratification labels for pretrain_monomer; "
                        "falling back to unstratified splits.")
            use_strat = False

    n_splits, local_reps = determine_split_strategy(len(ys_full), REPLICATES)

    from copy import deepcopy
    split_args = deepcopy(args)
    # Only stratify if we have a valid multiclass column WITHOUT NaNs
    split_args.task_type = 'multi' if use_strat else 'reg'
    ys_for_split = (y_strat if use_strat else ys_full)

    train_indices, val_indices, test_indices = generate_data_splits(
        split_args, ys_for_split, n_splits, local_reps, SEED
    )
    # Identify regression target indices
    reg_idx = [i for i,(k,_) in enumerate(args.task_specs) if k == 'reg']
    
    # For pretrain_monomer (single split path in your code), pick split 0.
    tr = train_indices[0]

    # Train-only stats (μ, σ) per regression column
    if reg_idx:
        reg_data = ys_full[tr][:, reg_idx]
        mu = np.nanmean(reg_data, axis=0)
        sd = np.nanstd(reg_data, axis=0)
    else:
        mu = np.array([])
        sd = np.array([])
    # handle all-NaN columns
    nan_cols = np.isnan(mu) | np.isnan(sd)
    mu[nan_cols] = 0.0
    sd[nan_cols] = 1.0
    sd[sd < 1e-8] = 1.0

    # Map μ,σ to per-task lists aligned with task_specs
    reg_mu_per_task = [None] * len(task_specs)
    reg_sd_per_task = [None] * len(task_specs)
    rj = 0
    for t, (kind, _) in enumerate(task_specs):
        if kind == "reg":
            reg_mu_per_task[t] = float(mu[rj])
            reg_sd_per_task[t] = float(sd[rj])
            rj += 1

    args.reg_mu_per_task = reg_mu_per_task
    args.reg_sd_per_task = reg_sd_per_task

    # Create normalized targets for datapoints
    ys_norm = ys_full.copy()
    if reg_idx:  # Only normalize if there are regression columns
        ys_norm[:, reg_idx] = (ys_norm[:, reg_idx] - mu) / sd

    all_data = create_all_data(smis, ys_norm, combined_descriptor_data, args.model_name)

    train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

    # Build datasets (small-molecule featurizer)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train = data.MoleculeDataset(train_data[0], featurizer)
    val   = data.MoleculeDataset(val_data[0],   featurizer)
    test  = data.MoleculeDataset(test_data[0],  featurizer)


    # Metrics list: pick a default (RMSE for reg, etc.)
    n_classes_arg = None
    metric_list = []

    # Paths and model
    checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix, copoly_suffix = \
        build_experiment_paths(args, chemprop_dir, checkpoint_dir, "__multitask__", descriptor_columns, 0)

    processed_descriptor_data = None
    mpnn, trainer = build_model_and_trainer(
        args=args,
        combined_descriptor_data=processed_descriptor_data,
        n_classes=n_classes_arg,
        scaler=None,
        checkpoint_path=checkpoint_path,
        batch_norm=args.batch_norm,
        metric_list=metric_list,
        early_stopping_patience=PATIENCE,
        max_epochs=EPOCHS,
        save_checkpoint=args.save_checkpoint,
    )

    # Train one model (no per-target loop)
    trainer.fit(mpnn, data.build_dataloader(train, batch_size=args.batch_size, num_workers=num_workers,pin_memory=True),
                      data.build_dataloader(val, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True))
    _ = trainer.test(dataloaders=data.build_dataloader(test, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True))

    # Optionally export embeddings for all monomers now
    if args.export_embeddings:
        mpnn.eval()
        # Re-embed ALL datapoints (train+val+test order) for convenience
        full_loader = data.build_dataloader(data.MoleculeDataset(all_data, featurizer), batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        X_full = get_encodings_from_loader(mpnn, full_loader)
        # Save as .npy; map to smiles with df_input[smiles_column]
        emb_dir = checkpoint_dir / "embeddings"; emb_dir.mkdir(parents=True, exist_ok=True)
        emb_base = f"{args.dataset_name}__{args.model_name}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{fusion_suffix}{aux_suffix}{size_suffix}"
        np.save(emb_dir / f"{emb_base}__monomer_encoder.npy", X_full)
        pd.DataFrame({
            "smiles": [dp.smiles for dp in all_data],
        }).assign(idx=np.arange(len(all_data))).to_csv(emb_dir / f"{emb_base}__monomer_index.csv", index=False)

    # Exit after pretraining (we don’t do per-target loops in this mode)
    save_aggregate_results([], results_dir, args.model_name, args.dataset_name, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, logger)
    raise SystemExit(0)


# ========================= HPG BRANCH =========================
# HPG encodes polymers as hierarchical graphs (fragment + atom nodes).
# For copolymers, each monomer is a fragment node with directed connections.
# For homopolymers, a single fragment with a self-loop.
# This branch handles BOTH cases and exits via SystemExit.
if args.model_name == "HPG":
    from chemprop.featurizers.molgraph.hpg import HPGMolGraphFeaturizer, HPGMolGraphFeaturizerEdgeTyped
    from chemprop.data.hpg import HPGDatapoint, HPGDataset, hpg_collate_fn, BatchHPGMolGraph
    from chemprop.models.hpg import HPGMPNN
    from torch.utils.data import DataLoader
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    is_copolymer = (args.polymer_type == "copolymer")

    hpg_variant = getattr(args, 'hpg_variant', 'baseline')
    use_frac_pooling = hpg_variant in ('frac', 'frac_polytype', 'frac_edgeTyped', 'frac_archAware', 'relMsg')
    # Select featurizer: edgeTyped uses 4-dim typed edges; all others use standard d_e=1
    # frac_archAware intentionally reuses the standard featurizer (no edge typing)
    hpg_d_e = 4 if hpg_variant == 'frac_edgeTyped' else 1
    hpg_featurizer = HPGMolGraphFeaturizerEdgeTyped() if hpg_variant == 'frac_edgeTyped' else HPGMolGraphFeaturizer()

    logger.info(f"\n=== HPG Training ===")
    logger.info(f"Dataset          : {args.dataset_name}")
    logger.info(f"Polymer type     : {args.polymer_type}")
    logger.info(f"HPG variant      : {hpg_variant}")
    logger.info(f"Frac pooling     : {use_frac_pooling}")
    logger.info(f"Target columns   : {target_columns}")
    logger.info(f"incl_desc        : {args.incl_desc}")
    logger.info(f"incl_poly_type   : {getattr(args, 'incl_poly_type', False)}")
    logger.info(f"Split type       : {args.split_type}")
    logger.info("================================\n")

    # Filter to specific target if specified
    if args.target:
        if args.target not in target_columns:
            logger.error(f"Specified target '{args.target}' not found. Available: {target_columns}")
            exit(1)
        target_columns = [args.target]

    # ── Parse SMILES into HPG format ──
    if is_copolymer:
        sA_col = "smilesA" if "smilesA" in df_input.columns else "smiles_A"
        sB_col = "smilesB" if "smilesB" in df_input.columns else "smiles_B"
        smis_A = df_input[sA_col].astype(str).tolist()
        smis_B = df_input[sB_col].astype(str).tolist()
        fracA_arr = df_input["fracA"].values.astype(float)
        fracB_arr = df_input["fracB"].values.astype(float)
    else:
        smis_homo = df_input[smiles_column].astype(str).tolist()

    # ── Descriptors (X_d) — variant-dependent wiring ──
    # HPG_baseline   : uses incl_desc / incl_poly_type flags as before
    # HPG_frac       : fractions go through pooling, NO X_d
    # HPG_frac_polytype : fractions through pooling, X_d = poly_type one-hot only
    combined_descriptor_data_hpg = None
    _poly_oh = None

    if hpg_variant == 'baseline':
        # ---- Original behaviour: honour incl_desc / incl_poly_type flags ----
        hpg_desc_parts = []
        if args.incl_desc:
            if is_copolymer and "fracA" in df_input.columns and "fracB" in df_input.columns:
                frac_data = df_input[["fracA", "fracB"]].values.astype(np.float32)
                hpg_desc_parts.append(frac_data)
                logger.info("HPG baseline + incl_desc: including fracA, fracB as scalar descriptors")
            if descriptor_columns:
                hpg_desc_parts.append(df_input[descriptor_columns].values.astype(np.float32))
                logger.info(f"HPG baseline + incl_desc: including {len(descriptor_columns)} config descriptors: {descriptor_columns}")
        if hpg_desc_parts:
            combined_descriptor_data_hpg = np.hstack(hpg_desc_parts)

        if getattr(args, 'incl_poly_type', False) and 'poly_type' in df_input.columns:
            _POLY_CATS = sorted(df_input['poly_type'].dropna().astype(str).unique().tolist())
            _poly_oh = (
                pd.get_dummies(df_input['poly_type'].astype(str))
                .reindex(columns=_POLY_CATS, fill_value=0)
                .values.astype(np.float32)
            )
            if combined_descriptor_data_hpg is not None:
                combined_descriptor_data_hpg = np.hstack([combined_descriptor_data_hpg, _poly_oh])
            else:
                combined_descriptor_data_hpg = _poly_oh

    elif hpg_variant in ('frac', 'frac_edgeTyped', 'frac_archAware', 'relMsg'):
        # ---- Fractions enter through pooling, NO X_d at all ----
        logger.info(f"HPG_{hpg_variant}: fractions used for pooling; no X_d")

    elif hpg_variant == 'frac_polytype':
        # ---- Fractions through pooling; polytype one-hot as X_d ----
        if 'poly_type' not in df_input.columns:
            raise ValueError("hpg_variant='frac_polytype' requires a 'poly_type' column in the dataset")
        _POLY_CATS = sorted(df_input['poly_type'].dropna().astype(str).unique().tolist())
        _poly_oh = (
            pd.get_dummies(df_input['poly_type'].astype(str))
            .reindex(columns=_POLY_CATS, fill_value=0)
            .values.astype(np.float32)
        )
        combined_descriptor_data_hpg = _poly_oh
        logger.info(f"HPG_frac_polytype: fractions for pooling; polytype one-hot ({_poly_oh.shape[1]}d) as X_d")

    # ── Featurize all samples into HPGMolGraph objects ──
    hpg_graphs = []
    skip_indices = []
    N = len(df_input)
    for idx in range(N):
        try:
            if is_copolymer:
                frag_smiles = [smis_A[idx], smis_B[idx]]
                connections = [(0, 1, 1.0)]
                # Attach per-fragment fractions for frac pooling variants
                ff = (np.array([fracA_arr[idx], fracB_arr[idx]], dtype=np.float32)
                      if use_frac_pooling else None)
                mg = hpg_featurizer(frag_smiles, connections, frag_fracs=ff)
            else:
                mg = hpg_featurizer([smis_homo[idx]])
            hpg_graphs.append(mg)
        except Exception as e:
            logger.warning(f"HPG featurizer failed for sample {idx}: {e}")
            hpg_graphs.append(None)
            skip_indices.append(idx)

    if skip_indices:
        logger.warning(f"Skipped {len(skip_indices)} samples due to featurization errors")

    all_results = []
    for target in target_columns:
        ys = df_input[target].astype(float).values
        if args.task_type != 'reg':
            ys = ys.astype(int)

        # Build HPGDatapoints (skip failed featurizations)
        all_hpg_dps = []
        valid_indices = []
        for idx in range(N):
            if hpg_graphs[idx] is None:
                continue
            if not np.isfinite(ys[idx]):
                continue
            y_val = np.array([ys[idx]], dtype=np.float32)
            x_d = combined_descriptor_data_hpg[idx] if combined_descriptor_data_hpg is not None else None
            dp = HPGDatapoint(mg=hpg_graphs[idx], y=y_val, x_d=x_d)
            all_hpg_dps.append(dp)
            valid_indices.append(idx)
        valid_indices = np.array(valid_indices)
        logger.info(f"[{target}] {len(all_hpg_dps)} valid HPG datapoints")

        # ── Splits ──
        n_samples = len(all_hpg_dps)
        n_splits, local_reps = determine_split_strategy(n_samples, REPLICATES)

        if is_copolymer and args.split_type == "a_held_out":
            valid_smiles_A = [smis_A[i] for i in valid_indices]
            n_splits = 5
            train_indices_hpg, val_indices_hpg, test_indices_hpg = generate_a_held_out_splits(
                valid_smiles_A, n_samples, SEED, n_splits=n_splits, logger=logger
            )
        else:
            # Flatten y values for stratified splitting (each dp.y is a 1D array)
            dummy_ys = np.array([dp.y[0] if isinstance(dp.y, np.ndarray) else dp.y for dp in all_hpg_dps])
            train_indices_hpg, val_indices_hpg, test_indices_hpg = generate_data_splits(
                args, dummy_ys, n_splits, local_reps, SEED
            )

        # ── Train size subsampling ──
        if args.train_size is not None and args.train_size.lower() != "full":
            target_train_size = int(args.train_size)
            for si in range(len(train_indices_hpg)):
                orig = len(train_indices_hpg[si])
                if target_train_size < orig:
                    rng = np.random.default_rng(SEED + si)
                    train_indices_hpg[si] = rng.choice(train_indices_hpg[si], size=target_train_size, replace=False)
                    logger.info(f"Split {si}: Training set reduced {orig} → {target_train_size}")

        results_all = []
        for i in range(len(train_indices_hpg)):
            tr = train_indices_hpg[i]
            va = val_indices_hpg[i]
            te = test_indices_hpg[i]

            train_dps = [all_hpg_dps[j] for j in tr]
            val_dps = [all_hpg_dps[j] for j in va]
            test_dps = [all_hpg_dps[j] for j in te]

            train_ds = HPGDataset(train_dps)
            val_ds = HPGDataset(val_dps)
            test_ds = HPGDataset(test_dps)

            # ── Normalize targets (regression only) ──
            scaler = None
            if args.task_type == 'reg':
                scaler = train_ds.normalize_targets()
                val_ds.normalize_targets(scaler)
                # test targets intentionally left unscaled

            # ── Normalize descriptors ──
            if combined_descriptor_data_hpg is not None:
                desc_scaler = train_ds.normalize_inputs("X_d")
                val_ds.normalize_inputs("X_d", desc_scaler)
                test_ds.normalize_inputs("X_d", desc_scaler)

            # ── DataLoaders ──
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=hpg_collate_fn, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=hpg_collate_fn, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=hpg_collate_fn, num_workers=num_workers, pin_memory=True)

            # ── Build experiment paths ──
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix, copoly_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )

            # ── Metrics ──
            n_classes_arg = None
            if args.task_type == 'multi':
                unique_classes = df_input[target].dropna().unique()
                n_classes_arg = len(unique_classes)
                logger.info(f"[{target}] Detected {n_classes_arg} classes for multi-class classification")
            metric_list = get_metric_list(args.task_type, target=target, n_classes=n_classes_arg, df_input=df_input)

            # ── Build HPG model + trainer ──
            d_xd = combined_descriptor_data_hpg.shape[1] if combined_descriptor_data_hpg is not None else 0
            # For multi-class, n_tasks = n_classes (output logits for each class)
            # For regression and binary, n_tasks = 1
            n_tasks_hpg = n_classes_arg if args.task_type == 'multi' else 1
            pooling_type = (
                "frac_arch_aware" if hpg_variant == 'frac_archAware'
                else "frac_weighted" if use_frac_pooling
                else "sum"
            )
            # mp_type selects message-passing mechanism; all Phase 1 variants
            # use the default 'gat'; Phase 2A relMsg uses 'rel_msg'
            mp_type = "rel_msg" if hpg_variant == 'relMsg' else "gat"
            mpnn = HPGMPNN(
                d_v=hpg_featurizer.d_v,
                d_e=hpg_d_e,
                d_h=getattr(args, "hpg_hidden_dim", 128),
                d_ffn=getattr(args, "hpg_ffn_dim", 64),
                depth=getattr(args, "hpg_depth", 6),
                num_heads=getattr(args, "hpg_num_heads", 8),
                dropout_mp=getattr(args, "hpg_dropout_mp", 0.0),
                dropout_ffn=getattr(args, "hpg_dropout_ffn", 0.2),
                n_tasks=n_tasks_hpg,
                d_xd=d_xd,
                mp_type=mp_type,
                pooling_type=pooling_type,
                task_type="regression" if args.task_type == "reg" else "classification",
                metrics=metric_list or [],
                criterion=None,
            )

            # ── Output transform for regression (unscale predictions) ──
            if scaler is not None:
                from chemprop.nn.transforms import UnscaleTransform
                mpnn._output_transform = UnscaleTransform.from_standard_scaler(scaler)

            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=True),
            ]
            if args.save_checkpoint:
                callbacks.append(ModelCheckpoint(
                    dirpath=str(checkpoint_path),
                    filename="best-{epoch:03d}-{val_loss:.4f}",
                    monitor="val_loss", mode="min", save_top_k=1, save_last=True,
                    save_weights_only=False, auto_insert_metric_name=False,
                ))

            trainer = pl.Trainer(
                max_epochs=EPOCHS, callbacks=callbacks,
                gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                enable_progress_bar=True, enable_model_summary=(i == 0),
                log_every_n_steps=10, check_val_every_n_epoch=1,
                num_sanity_val_steps=0, enable_checkpointing=args.save_checkpoint,
                default_root_dir=str(checkpoint_path),
            )

            # ── Train ──
            logger.info(f"[{target}] split {i}: Training HPG ({len(train_dps)} train, {len(val_dps)} val, {len(test_dps)} test)")
            trainer.fit(mpnn, train_loader, val_loader)

            # ── Test ──
            results = trainer.test(model=mpnn, dataloaders=test_loader)
            test_metrics = results[0]
            test_metrics["split"] = i
            results_all.append(test_metrics)
            logger.info(f"[{target}] split {i}: {test_metrics}")

        # Aggregate results for this target
        results_df = pd.DataFrame(results_all)
        numeric_cols = [col for col in results_df.columns if col != "split"]
        logger.info(f"\n[{target}] Mean across {len(results_all)} splits:\n{results_df[numeric_cols].mean()}")
        logger.info(f"\n[{target}] Std across {len(results_all)} splits:\n{results_df[numeric_cols].std()}")
        results_df["target"] = target
        all_results.append(results_df)

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        save_model_results(combined_results, args, args.model_name, results_dir, logger)

    raise SystemExit(0)
# ========================= END HPG BRANCH =========================


# ========================= COPOLYMER BRANCH =========================
# wDMPNN bypasses the copolymer branch: it reads from WDMPNN_Input and
# runs through the standard homopolymer path below.
if args.polymer_type == "copolymer" and args.model_name != "wDMPNN":
    from chemprop.data.copolymer import CopolymerDataset, MultiMonomerCopolymerDataset
    from chemprop.models.copolymer import CopolymerMPNN

    copolymer_mode = args.copolymer_mode
    fusion_type = getattr(args, 'fusion_type', 'sum_fusion')
    is_multi_monomer = df_input.attrs.get("multi_monomer", False)
    logger.info(f"\n=== Copolymer Training (mode={copolymer_mode}, fusion={fusion_type}) ===")
    logger.info(f"Dataset          : {args.dataset_name}")
    logger.info(f"Model            : {args.model_name}")
    logger.info(f"Target columns   : {target_columns}")
    logger.info(f"Copolymer mode   : {copolymer_mode}")
    logger.info(f"Fusion type      : {fusion_type}")
    logger.info(f"Multi-monomer    : {is_multi_monomer}")
    logger.info(f"Split type       : {args.split_type}")
    logger.info("================================\n")

    # Filter to specific target if specified
    if args.target:
        if args.target not in target_columns:
            logger.error(f"Specified target '{args.target}' not found. Available: {target_columns}")
            exit(1)
        target_columns = [args.target]
        logger.info(f"Training on single target: {args.target}")

    if is_multi_monomer:
        # Multi-monomer: extract per-row lists of SMILES and fractions
        smiles_A_lists = df_input["smilesA_list"].tolist()
        frac_A_lists = df_input["fracA_list"].tolist()
        smiles_B_lists = df_input["smilesB_list"].tolist()
        frac_B_lists = df_input["fracB_list"].tolist()
    else:
        # Single-monomer: extract scalar SMILES columns
        sA_col = "smilesA" if "smilesA" in df_input.columns else "smiles_A"
        sB_col = "smilesB" if "smilesB" in df_input.columns else "smiles_B"
        smis_A = df_input[sA_col].astype(str).tolist()
        smis_B = df_input[sB_col].astype(str).tolist()
        fracA_arr = df_input["fracA"].values.astype(float)
        fracB_arr = df_input["fracB"].values.astype(float)

    # Process descriptors with global cleaning (matches homopolymer branch)
    combined_descriptor_data = None
    orig_Xd_copoly = None
    copoly_constant_features = []
    if descriptor_columns:
        combined_descriptor_data = df_input[descriptor_columns].values.astype(np.float32)
        logger.info(f"Using {len(descriptor_columns)} descriptor columns: {descriptor_columns}")

        # Global descriptor cleaning (inf→NaN, constant removal) — same as homopolymer
        orig_Xd_copoly = np.asarray(combined_descriptor_data, dtype=np.float64)
        inf_mask = np.isinf(orig_Xd_copoly)
        if np.any(inf_mask):
            logger.warning(f"Found {np.sum(inf_mask)} infinite values in descriptors, replacing with NaN")
            orig_Xd_copoly[inf_mask] = np.nan

        Xd_temp_df = pd.DataFrame(orig_Xd_copoly)
        non_na_uniques = Xd_temp_df.nunique(dropna=True)
        copoly_constant_features = non_na_uniques[non_na_uniques < 2].index.tolist()
        if copoly_constant_features:
            logger.info(f"Removing {len(copoly_constant_features)} constant descriptor features globally")
            orig_Xd_copoly = np.delete(orig_Xd_copoly, copoly_constant_features, axis=1)

        nan_mask = np.isnan(orig_Xd_copoly)
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} NaN values in descriptors, will use per-split median imputation")
        logger.info(f"Copolymer descriptor shape after global cleaning: {orig_Xd_copoly.shape}")

    # ── poly_type one-hot (incl_poly_type) ─────────────────────────────────
    # Appended to orig_Xd_copoly so it flows through the standard
    # imputation / correlation-removal / scaling pipeline.
    # copolymer_mode is auto-upgraded to a *_meta variant so that
    # CopolymerMPNN._apply_mode forwards X_d to the predictor head.
    if getattr(args, 'incl_poly_type', False):
        if 'poly_type' not in df_input.columns:
            raise ValueError('--incl_poly_type requires a poly_type column in the dataset')
        _POLY_CATS = sorted(df_input['poly_type'].dropna().astype(str).unique().tolist())
        _poly_oh = (
            pd.get_dummies(df_input['poly_type'].astype(str))
            .reindex(columns=_POLY_CATS, fill_value=0)
            .values.astype(np.float64)
        )
        logger.info(f'incl_poly_type: categories={_POLY_CATS}, one-hot shape={_poly_oh.shape}')
        if orig_Xd_copoly is not None:
            orig_Xd_copoly = np.hstack([orig_Xd_copoly, _poly_oh])
        else:
            orig_Xd_copoly = _poly_oh.copy()
        if combined_descriptor_data is not None:
            combined_descriptor_data = np.hstack([combined_descriptor_data, _poly_oh.astype(np.float32)])
        else:
            combined_descriptor_data = _poly_oh.astype(np.float32)
        # Auto-upgrade base mode → meta variant so X_d is forwarded
        _UPGRADES = {'mean': 'mean_meta', 'mix': 'mix_meta', 'mix_pair': 'mix_pair_meta', 'mix_pair_attn': 'mix_pair_attn_meta', 'attention': 'attention_meta', 'frac_attn': 'frac_attn_meta', 'frac_attn_pair': 'frac_attn_pair_meta', 'frac_attn_pair_attn': 'frac_attn_pair_attn_meta', 'self_attn': 'self_attn_meta', 'interact': 'interact_meta', 'mix_frac': 'mix_frac_meta'}
        if copolymer_mode in _UPGRADES:
            args.copolymer_mode = _UPGRADES[copolymer_mode]
            copolymer_mode = args.copolymer_mode
            logger.info(f'incl_poly_type: copolymer_mode upgraded to {copolymer_mode}')

    featurizer_copoly = featurizers.SimpleMoleculeMolGraphFeaturizer()

    all_results = []

    for target in target_columns:
        ys = df_input[target].astype(float).values.reshape(-1, 1)

        # Create copolymer datapoints
        if is_multi_monomer:
            data_A, data_B, fA, fB = create_multi_monomer_copolymer_data(
                smiles_A_lists, frac_A_lists, smiles_B_lists, frac_B_lists,
                ys, combined_descriptor_data, args.model_name,
            )
            n_datapoints = len(data_A)
        else:
            data_A, data_B, fA, fB = create_copolymer_data(
                smis_A, smis_B, fracA_arr, fracB_arr, ys,
                combined_descriptor_data, args.model_name,
            )
            n_datapoints = len(data_A)
        logger.info(f"[{target}] Created {n_datapoints} copolymer datapoints")

        # Determine splits
        if args.split_type == "a_held_out":
            # --- A-held-out: group by smiles_A identity, always 5-fold CV ---
            # Build smiles_A array aligned with valid datapoints (after filtering)
            valid_smiles_A = []
            for idx in range(len(df_input)):
                y_val = ys[idx]
                if is_multi_monomer:
                    has_A = bool(smiles_A_lists[idx])
                    has_B = bool(smiles_B_lists[idx])
                    sA_val = smiles_A_lists[idx][0] if has_A else ""
                else:
                    has_A = bool(smis_A[idx])
                    has_B = bool(smis_B[idx])
                    sA_val = smis_A[idx] if has_A else ""
                if has_A and has_B and pd.notna(y_val).any():
                    valid_smiles_A.append(sA_val)
            valid_smiles_A = np.array(valid_smiles_A, dtype=str)
            assert len(valid_smiles_A) == n_datapoints, \
                f"smiles_A length ({len(valid_smiles_A)}) != n_datapoints ({n_datapoints})"

            n_splits = 5
            train_indices, val_indices, test_indices = generate_a_held_out_splits(
                valid_smiles_A, n_datapoints, SEED, n_splits=n_splits, logger=logger
            )

            # Save fold assignments for reproducibility
            save_fold_assignments(
                train_indices, val_indices, test_indices,
                valid_smiles_A, args.dataset_name, SEED, results_dir, logger=logger
            )
        else:
            # --- Default: existing split logic ---
            n_splits, local_reps = determine_split_strategy(n_datapoints, REPLICATES)

            # Generate splits using group keys if available
            if "group_key" in df_input.columns:
                # Build group array aligned with data (after filtering)
                valid_indices = []
                for idx in range(len(df_input)):
                    y_val = ys[idx]
                    if is_multi_monomer:
                        has_A = bool(smiles_A_lists[idx])
                        has_B = bool(smiles_B_lists[idx])
                    else:
                        has_A = bool(smis_A[idx])
                        has_B = bool(smis_B[idx])
                    if has_A and has_B and pd.notna(y_val).any():
                        valid_indices.append(idx)
                groups = df_input["group_key"].values[valid_indices]

                from sklearn.model_selection import GroupKFold, GroupShuffleSplit
                train_indices_list, val_indices_list, test_indices_list = [], [], []
                idx_all = np.arange(n_datapoints)

                if n_splits > 1:
                    for rep in range(local_reps):
                        gkf = GroupKFold(n_splits=n_splits)
                        for train_val_idx, test_idx in gkf.split(idx_all, groups=groups):
                            tv_groups = groups[train_val_idx]
                            unique_groups = np.unique(tv_groups)
                            rng = np.random.default_rng(SEED + rep)
                            n_val_groups = max(1, int(0.1 * len(unique_groups)))
                            val_group_set = set(rng.choice(unique_groups, size=n_val_groups, replace=False))
                            val_mask = np.array([g in val_group_set for g in tv_groups])
                            val_idx = train_val_idx[val_mask]
                            tr_idx = train_val_idx[~val_mask]
                            train_indices_list.append(tr_idx)
                            val_indices_list.append(val_idx)
                            test_indices_list.append(test_idx)
                else:
                    for rep in range(local_reps):
                        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED + rep)
                        train_val_idx, test_idx = next(gss.split(idx_all, groups=groups))
                        tv_groups = groups[train_val_idx]
                        gss_inner = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=SEED + rep)
                        tr_local, val_local = next(gss_inner.split(np.arange(len(train_val_idx)), groups=tv_groups))
                        train_indices_list.append(train_val_idx[tr_local])
                        val_indices_list.append(train_val_idx[val_local])
                        test_indices_list.append(test_idx)

                train_indices = train_indices_list
                val_indices = val_indices_list
                test_indices = test_indices_list
            else:
                train_indices, val_indices, test_indices = generate_data_splits(args, ys, n_splits, local_reps, SEED)

        # Apply train_size subsampling
        if args.train_size is not None and args.train_size.lower() != "full":
            target_train_size = int(args.train_size)
            for si in range(len(train_indices)):
                orig_size = len(train_indices[si])
                new_size = min(target_train_size, orig_size)
                if new_size < orig_size:
                    rng = np.random.default_rng(SEED + si)
                    train_indices[si] = rng.choice(train_indices[si], size=new_size, replace=False)
                    logger.info(f"Split {si}: Training set reduced from {orig_size} to {new_size}")

        num_splits = len(train_indices)
        results_all = []

        for i in range(num_splits):
            tr, va, te = train_indices[i], val_indices[i], test_indices[i]

            # Build experiment paths FIRST (needed for preprocessing cache)
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix, copoly_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )

            # Per-split descriptor preprocessing (matches homopolymer branch)
            descriptor_scaler = None
            preprocessing_reused = False
            processed_descriptor_data = None
            if orig_Xd_copoly is not None:
                from sklearn.impute import SimpleImputer
                float32_max = np.finfo(np.float32).max
                float32_min = np.finfo(np.float32).min

                # Try to load cached preprocessing
                preprocessing_reused, cached_scaler, cached_mask, cache_meta = manage_preprocessing_cache(
                    preprocessing_path, i, orig_Xd_copoly, None, None, logger
                )

                if preprocessing_reused and cached_mask is not None and len(cached_mask) > 0:
                    mask = cached_mask
                    imputer_stats = None
                    if cache_meta is not None:
                        imputer_stats = (cache_meta.get("cleaning") or {}).get("imputer_statistics")
                    if imputer_stats is not None:
                        imputer = SimpleImputer(strategy='median')
                        imputer.statistics_ = np.array(imputer_stats)
                        imputer.n_features_in_ = orig_Xd_copoly.shape[1]
                        imputer._fit_dtype = orig_Xd_copoly.dtype
                        all_data_clean = imputer.transform(orig_Xd_copoly)
                    else:
                        all_data_clean = orig_Xd_copoly.copy()
                    all_data_clean = np.clip(all_data_clean, float32_min, float32_max).astype(np.float32)
                    descriptor_scaler = cached_scaler
                    logger.info(f"Split {i}: reused cached copolymer preprocessing.")
                else:
                    # Compute fresh preprocessing (leakage-free)
                    imputer = SimpleImputer(strategy='median')
                    imputer.fit(orig_Xd_copoly[tr])
                    all_data_clean = imputer.transform(orig_Xd_copoly)
                    all_data_clean = np.clip(all_data_clean, float32_min, float32_max).astype(np.float32)

                    # Remove correlated features using training data only
                    correlated_features = []
                    if all_data_clean.shape[1] > 1:
                        train_df = pd.DataFrame(all_data_clean[tr])
                        corr_matrix = train_df.corr(method="pearson" if args.task_type == "reg" else "spearman").abs()
                        upper_tri = corr_matrix.where(
                            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                        )
                        correlated_features = [col for col in upper_tri.columns if any(upper_tri[col] >= 0.90)]
                        if correlated_features:
                            logger.info(f"Split {i}: Removing {len(correlated_features)} correlated features")
                            keep_features = [col for col in range(all_data_clean.shape[1]) if col not in correlated_features]
                        else:
                            keep_features = list(range(all_data_clean.shape[1]))
                    else:
                        keep_features = list(range(all_data_clean.shape[1]))

                    mask = np.zeros(all_data_clean.shape[1], dtype=bool)
                    mask[keep_features] = True

                    # Build and save preprocessing metadata (includes split_type + seed)
                    preprocessing_metadata = {
                        "cleaning": {
                            "imputation_strategy": "median",
                            "float32_max": float(float32_max),
                            "float32_min": float(float32_min),
                            "imputer_statistics": imputer.statistics_.tolist()
                        },
                        "data_info": {
                            "original_data_shape": list(orig_Xd_copoly.shape),
                            "descriptor_columns": descriptor_columns,
                            "rdkit_included": args.incl_rdkit,
                            "constant_features_removed": copoly_constant_features,
                            "post_constant_shape": list(orig_Xd_copoly.shape)
                        },
                        "splits": {
                            "train_indices": tr.tolist(),
                            "val_indices": va.tolist(),
                            "test_indices": te.tolist(),
                            "random_seed": SEED,
                            "split_type": args.split_type,
                            "correlation_threshold": 0.90,
                            "correlation_method": "pearson" if args.task_type == "reg" else "spearman"
                        },
                        "split_specific": {
                            "split_id": i,
                            "correlated_features": correlated_features,
                            "keep_features": keep_features,
                            "correlation_mask": mask.tolist()
                        },
                        "target": target,
                        "task_type": args.task_type,
                        "split_type": args.split_type,
                    }
                    manage_preprocessing_cache(
                        preprocessing_path, i, orig_Xd_copoly,
                        {i: preprocessing_metadata}, None, logger
                    )
                    logger.info(f"Split {i}: computed fresh copolymer preprocessing, {int(mask.sum())} features kept")

                # Update x_d on A-side datapoints from cleaned data
                for j in range(n_datapoints):
                    data_A[j].x_d = all_data_clean[j][mask].astype(np.float32)

                processed_descriptor_data = orig_Xd_copoly[:, mask]
            else:
                # No descriptors: save split configuration metadata only
                preprocessing_path.mkdir(parents=True, exist_ok=True)
                split_config = {
                    "split_type": args.split_type,
                    "random_seed": SEED,
                    "split_id": i,
                    "target": target,
                    "task_type": args.task_type,
                    "train_size": len(tr),
                    "val_size": len(va),
                    "test_size": len(te),
                }
                split_config_file = preprocessing_path / f"split_config_{i}.json"
                with open(split_config_file, "w") as f:
                    json.dump(split_config, f, indent=2)

            # Build dataset for each split
            if is_multi_monomer:
                train_dA = [data_A[j] for j in tr]
                train_dB = [data_B[j] for j in tr]
                val_dA = [data_A[j] for j in va]
                val_dB = [data_B[j] for j in va]
                test_dA = [data_A[j] for j in te]
                test_dB = [data_B[j] for j in te]
                train_fA = [fA[j] for j in tr]
                train_fB = [fB[j] for j in tr]
                val_fA = [fA[j] for j in va]
                val_fB = [fB[j] for j in va]
                test_fA = [fA[j] for j in te]
                test_fB = [fB[j] for j in te]

                train_ds = MultiMonomerCopolymerDataset(train_dA, train_dB, train_fA, train_fB, featurizer_copoly)
                val_ds = MultiMonomerCopolymerDataset(val_dA, val_dB, val_fA, val_fB, featurizer_copoly)
                test_ds = MultiMonomerCopolymerDataset(test_dA, test_dB, test_fA, test_fB, featurizer_copoly)
            else:
                train_dA = [data_A[j] for j in tr]
                train_dB = [data_B[j] for j in tr]
                val_dA = [data_A[j] for j in va]
                val_dB = [data_B[j] for j in va]
                test_dA = [data_A[j] for j in te]
                test_dB = [data_B[j] for j in te]

                train_ds = CopolymerDataset(train_dA, train_dB, fA[tr], fB[tr], featurizer_copoly)
                val_ds = CopolymerDataset(val_dA, val_dB, fA[va], fB[va], featurizer_copoly)
                test_ds = CopolymerDataset(test_dA, test_dB, fA[te], fB[te], featurizer_copoly)

            # Normalize targets (regression only)
            scaler = None
            if args.task_type == "reg":
                scaler = train_ds.normalize_targets()
                val_ds.normalize_targets(scaler)

            # Normalize descriptors with caching (matches homopolymer branch)
            if orig_Xd_copoly is not None:
                if descriptor_scaler is not None:
                    # Use cached scaler
                    train_ds.normalize_inputs("X_d", descriptor_scaler)
                    val_ds.normalize_inputs("X_d", descriptor_scaler)
                    test_ds.normalize_inputs("X_d", descriptor_scaler)
                else:
                    # Fit new scaler on training data
                    desc_scaler = train_ds.normalize_inputs("X_d")
                    val_ds.normalize_inputs("X_d", desc_scaler)
                    test_ds.normalize_inputs("X_d", desc_scaler)
                    # Persist the fitted scaler
                    manage_preprocessing_cache(
                        preprocessing_path, i, orig_Xd_copoly, None, desc_scaler, logger
                    )

            # Metrics - determine n_classes for multi-class classification
            n_classes_arg = None
            if args.task_type == 'multi':
                # Get unique classes from the target column
                unique_classes = df_input[target].dropna().unique()
                n_classes_arg = len(unique_classes)
                logger.info(f"[{target}] Detected {n_classes_arg} classes for multi-class classification")
            metric_list = get_metric_list(args.task_type, target=target, n_classes=n_classes_arg, df_input=df_input)

            # Set n_classes on args for multi-class classification
            if args.task_type == 'multi':
                args._n_classes = n_classes_arg

            # Build model and trainer
            mpnn, trainer = build_copolymer_model_and_trainer(
                args=args,
                combined_descriptor_data=processed_descriptor_data,
                scaler=scaler,
                checkpoint_path=checkpoint_path,
                copolymer_mode=copolymer_mode,
                batch_norm=args.batch_norm,
                metric_list=metric_list,
                early_stopping_patience=PATIENCE,
                max_epochs=EPOCHS,
                save_checkpoint=args.save_checkpoint,
            )

            # Dataloaders
            train_loader = data.build_dataloader(train_ds, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
            val_loader = data.build_dataloader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
            test_loader = data.build_dataloader(test_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

            # Skip-training logic
            inprog_flag = checkpoint_path / "TRAINING_IN_PROGRESS"
            done_flag = checkpoint_path / "TRAINING_COMPLETE"
            best_ckpt_path, best_val_loss = None, None
            skip_training = False

            if done_flag.exists():
                best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
                if best_ckpt_path is not None:
                    skip_training = True
                    logger.info(f"[{target}] split {i}: Found TRAINING_COMPLETE; skipping.")

            if skip_training and best_ckpt_path:
                logger.info(f"Loading copolymer checkpoint: {best_ckpt_path}")
                use_cuda = torch.cuda.is_available()
                map_location = None if use_cuda else torch.device("cpu")
                
                # Rebuild model with same architecture, then load weights
                mpnn_fresh, _ = build_copolymer_model_and_trainer(
                    args=args,
                    combined_descriptor_data=processed_descriptor_data,
                    scaler=scaler,
                    checkpoint_path=checkpoint_path,
                    copolymer_mode=copolymer_mode,
                    batch_norm=args.batch_norm,
                    metric_list=metric_list,
                    early_stopping_patience=PATIENCE,
                    max_epochs=EPOCHS,
                    save_checkpoint=args.save_checkpoint,
                )
                
                # Load checkpoint weights into the fresh model
                checkpoint = torch.load(best_ckpt_path, map_location=map_location, weights_only=False)
                mpnn_fresh.load_state_dict(checkpoint['state_dict'])
                mpnn = mpnn_fresh
                
                if use_cuda:
                    mpnn = mpnn.to(torch.device("cuda"))
                mpnn.eval()
            else:
                inprog_flag.touch(exist_ok=True)
                try:
                    trainer.fit(mpnn, train_loader, val_loader)
                    best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
                    if best_ckpt_path:
                        with open(checkpoint_path / "best.json", "w") as f:
                            json.dump({"best_ckpt": best_ckpt_path, "best_val_loss": best_val_loss}, f, indent=2)
                        done_flag.touch()
                finally:
                    if inprog_flag.exists():
                        inprog_flag.unlink(missing_ok=True)

            # Test
            results = trainer.test(model=mpnn, dataloaders=test_loader)
            test_metrics = results[0]
            test_metrics["split"] = i
            results_all.append(test_metrics)

            # Save predictions if requested
            if args.save_predictions:
                logger.info(f"Extracting predictions for split {i}, target {target}")
                
                # Use trainer.predict for unscaled outputs (applies same transform as trainer.test)
                y_pred = trainer.predict(model=mpnn, dataloaders=test_loader)
                
                # Extract y_true directly from test dataset to match loader order
                if is_multi_monomer:
                    y_true = np.array([test_ds[j].y for j in range(len(test_ds))], dtype=float)
                else:
                    y_true = np.array([test_ds[j].y for j in range(len(test_ds))], dtype=float)
                
                # Extract IDs/indices for order verification
                test_ids = []
                for j in range(len(test_ds)):
                    dp = test_ds[j]
                    if hasattr(dp, 'id') and dp.id is not None:
                        test_ids.append(dp.id)
                    elif hasattr(dp, 'smiles'):
                        test_ids.append(dp.smiles)
                    else:
                        test_ids.append(f"idx_{j}")
                
                # Convert predictions to numpy - handle list of tensors properly
                if isinstance(y_pred, list):
                    import torch
                    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                elif hasattr(y_pred, 'cpu'):
                    y_pred = y_pred.cpu().numpy()
                
                # Save predictions with IDs and training configuration metadata
                split_type_sfx = f"__{args.split_type}"
                save_predictions(
                    y_true, y_pred, predictions_dir, args.dataset_name, target, args.model_name,
                    desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, copoly_suffix, i, logger,
                    test_ids=test_ids,
                    copolymer_mode=copolymer_mode,
                    polymer_type=args.polymer_type,
                    task_type=args.task_type,
                    fusion_mode=getattr(args, 'fusion_mode', None),
                    aux_task=getattr(args, 'aux_task', None),
                    split_type_suffix=split_type_sfx
                )

            # Export embeddings if requested
            if args.export_embeddings:
                logger.info(f"Exporting copolymer embeddings for split {i}, target {target}")
                mpnn.eval()

                def _get_copolymer_embeddings(model, loader):
                    all_z_A, all_z_B, all_z_final = [], [], []
                    device = next(model.parameters()).device
                    with torch.no_grad():
                        for batch in loader:
                            is_mm = len(batch) == 11
                            if is_mm:
                                bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d, *_ = batch
                                bmg_A.to(device); bmg_B.to(device)
                                fracs_A = fracs_A.to(device); fracs_B = fracs_B.to(device)
                                counts_A = counts_A.to(device); counts_B = counts_B.to(device)
                                if X_d is not None:
                                    X_d = X_d.to(device)
                                comps = model.fingerprint_components_multi_monomer(
                                    bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d
                                )
                            else:
                                bmg_A, bmg_B, fracA_t, fracB_t, X_d, *_ = batch
                                bmg_A.to(device); bmg_B.to(device)
                                fracA_t = fracA_t.to(device); fracB_t = fracB_t.to(device)
                                if X_d is not None:
                                    X_d = X_d.to(device)
                                comps = model.fingerprint_components(bmg_A, bmg_B, fracA_t, fracB_t, X_d)
                            all_z_A.append(comps["z_A"].cpu().numpy())
                            all_z_B.append(comps["z_B"].cpu().numpy())
                            all_z_final.append(comps["z_final"].cpu().numpy())
                    return {
                        "z_A": np.concatenate(all_z_A),
                        "z_B": np.concatenate(all_z_B),
                        "z_final": np.concatenate(all_z_final),
                    }

                emb_train = _get_copolymer_embeddings(mpnn, train_loader)
                emb_test = _get_copolymer_embeddings(mpnn, test_loader)

                embeddings_dir = results_dir / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                split_type_suffix = f"__{args.split_type}" if args.split_type != "random" else ""
                _ft = getattr(args, 'fusion_type', 'sum_fusion')
                _is_pw = (copolymer_mode.startswith('frac_attn_pair') or copolymer_mode.startswith('mix_pair'))
                _fusion_sfx = f"__fusion_{_ft}" if _is_pw else ""
                emb_prefix = f"{args.dataset_name}__{args.model_name}__{target}__copoly_{copolymer_mode}{_fusion_sfx}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{split_type_suffix}{size_suffix}"

                for key in ["z_A", "z_B", "z_final"]:
                    np.save(embeddings_dir / f"{emb_prefix}__{key}_train_split_{i}.npy", emb_train[key])
                    np.save(embeddings_dir / f"{emb_prefix}__{key}_test_split_{i}.npy", emb_test[key])
                logger.info(f"Split {i}: Saved copolymer embeddings to {embeddings_dir}")
                logger.info(f"  - z_A train: {emb_train['z_A'].shape}, z_B train: {emb_train['z_B'].shape}, z_final train: {emb_train['z_final'].shape}")

        # Aggregate results for this target
        results_df = pd.DataFrame(results_all)
        numeric_cols = [col for col in results_df.columns if col != "split"]
        mean_metrics = results_df[numeric_cols].mean()
        std_metrics = results_df[numeric_cols].std()
        logger.info(f"\n[{target}] Mean across {len(results_all)} splits:\n{mean_metrics}")
        logger.info(f"\n[{target}] Std across {len(results_all)} splits:\n{std_metrics}")
        results_df["target"] = target
        all_results.append(results_df)

    # Save final results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        save_model_results(combined_results, args, args.model_name, results_dir, logger)

    raise SystemExit(0)
# ========================= END COPOLYMER BRANCH =========================


# Filter to specific target if specified
if args.target:
    if args.target not in target_columns:
        logger.error(f"Specified target '{args.target}' not found in dataset. Available targets: {target_columns}")
        exit(1)
    target_columns = [args.target]
    logger.info(f"Training on single target: {args.target}")



logger.info("\n=== Training Configuration ===")
logger.info(f"Dataset          : {args.dataset_name}")
logger.info(f"Task type        : {args.task_type}")
logger.info(f"Model            : {args.model_name}")
logger.info(f"SMILES column    : {smiles_column}")
logger.info(f"Descriptor cols  : {descriptor_columns}")
logger.info(f"Ignore cols      : {ignore_columns}")
logger.info(f"Target columns   : {target_columns}")
logger.info(f"Descriptors      : {'Enabled' if args.incl_desc else 'Disabled'}")
logger.info(f"RDKit desc.      : {'Enabled' if args.incl_rdkit else 'Disabled'}")

if args.target:
    logger.info(f"Single target    : {args.target}")
if args.train_size is not None:
    if args.train_size.lower() == "full":
        logger.info(f"Training size    : full (no subsampling)")
    else:
        logger.info(f"Training size    : {args.train_size} samples")
logger.info("================================\n")


smis, df_input, combined_descriptor_data, n_classes_per_target = process_data(df_input, smiles_column, descriptor_columns, target_columns, args)

# Choose featurizer based on model type
# PPG uses PPGMolGraphFeaturizer for periodic polymer graph construction
# Small molecule models use SimpleMoleculeMolGraphFeaturizer
# Polymer models (wDMPNN) use PolymerMolGraphFeaturizer
#
# NOTE: These are the same Chemprop featurizers used internally by
# polymer_input.featurizers.{dmpnn,ppg,wdmpnn}.  The polymer_input
# featurizers add a PolymerSpec → Chem.Mol conversion step on top;
# here we rely on Chemprop's from_smi datapoint path which handles
# that conversion.  See polymer_input/README.md for the full pipeline.
small_molecule_models = ["DMPNN", "DMPNN_DiffPool", "GIN", "GIN0", "GINE", "GAT", "GATv2", "AttentiveFP"]
if args.model_name == "PPG":
    featurizer = featurizers.PPGMolGraphFeaturizer()
elif args.model_name in small_molecule_models:
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
else:
    featurizer = featurizers.PolymerMolGraphFeaturizer()

if _HAS_POLYMER_INPUT:
    logger.info("polymer_input package available — PolymerSpec validation and scalar extraction enabled")
else:
    logger.debug("polymer_input package not installed — using standard SMILES-based flow only")
      

# Store all results for aggregate saving
all_results = []

# === Auxiliary task: load raw descriptor targets ===
aux_raw = None  # shape [N, T_aux] or None
if args._n_aux_targets > 0:
    missing_aux = [c for c in args._aux_cols if c not in df_input.columns]
    if missing_aux:
        logger.error(f"Auxiliary descriptor columns not found in DataFrame: {missing_aux}")
        logger.error(f"Available columns: {list(df_input.columns)}")
        exit(1)
    aux_raw = df_input[args._aux_cols].values.astype(np.float64)
    logger.info(f"Loaded auxiliary targets: {args._aux_cols}, shape={aux_raw.shape}")

for target in target_columns:
    # Extract target values
    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg':
        ys = ys.astype(int)
    ys = ys.reshape(-1, 1) # reshaping target to be 2D

    # Append raw auxiliary targets as extra columns in ys
    # They will be standardized per-split below
    if aux_raw is not None:
        ys = np.concatenate([ys, aux_raw], axis=1)  # [N, 1 + T_aux]

    all_data = create_all_data(smis, ys, combined_descriptor_data, args.model_name)

    # Determine split strategy and generate splits
    n_splits, local_reps = determine_split_strategy(len(ys), REPLICATES)

    if getattr(args, 'split_type', 'random') == 'a_held_out':
        sA_col = "smilesA" if "smilesA" in df_input.columns else "smiles_A"
        valid_smiles_A = df_input[sA_col].astype(str).tolist()
        n_splits = 5
        logger.info(f"Using A-held-out 5-fold CV (group by {sA_col})")
        train_indices, val_indices, test_indices = generate_a_held_out_splits(
            valid_smiles_A, len(all_data), SEED, n_splits=n_splits, logger=logger
        )
        save_fold_assignments(
            train_indices, val_indices, test_indices,
            valid_smiles_A, args.dataset_name, SEED, results_dir, logger=logger
        )
    else:
        if n_splits > 1:
            logger.info(f"Using {n_splits}-fold cross-validation with {local_reps} replicate(s)")
        else:
            logger.info(f"Using holdout validation with {local_reps} replicate(s)")
        train_indices, val_indices, test_indices = generate_data_splits(args, ys, n_splits, local_reps, SEED)
    
    # Apply train_size subsampling if specified
    if args.train_size is not None and args.train_size.lower() != "full":
        target_train_size = int(args.train_size)
        logger.info(f"Subsampling training data to {target_train_size} samples")
        
        for i in range(len(train_indices)):
            original_train_size = len(train_indices[i])
            new_train_size = min(target_train_size, original_train_size)
            
            if new_train_size < original_train_size:
                # Use per-split RNG for reproducible but distinct subsampling
                rng = np.random.default_rng(SEED + i)  # stable, split-specific
                subsampled_indices = rng.choice(
                    train_indices[i], 
                    size=new_train_size, 
                    replace=False
                )
                train_indices[i] = subsampled_indices
                logger.info(f"Split {i}: Training set reduced from {original_train_size} to {new_train_size} samples")
            else:
                logger.info(f"Split {i}: Training set size ({original_train_size}) is already <= target size ({target_train_size}), keeping all samples")
    elif args.train_size is not None and args.train_size.lower() == "full":
        logger.info("Using full training set (no subsampling)")


    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # Descriptor cleaning (if incl_desc or incl_rdkit is enabled)
    if combined_descriptor_data is not None:
        # Initial data preparation (no imputation yet to avoid leakage)
        orig_Xd = np.asarray(combined_descriptor_data, dtype=np.float64)  # Use float64 first
        
        # Replace inf with NaN
        inf_mask = np.isinf(orig_Xd)
        if np.any(inf_mask):
            logger.warning(f"Found {np.sum(inf_mask)} infinite values, replacing with NaN")
            orig_Xd[inf_mask] = np.nan
        
        
        # Store float32 limits for later use
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        
        logger.debug(f"Original descriptor data shape: {orig_Xd.shape}")
        logger.debug(f"Original descriptor data - Inf count: {np.sum(inf_mask)}")
        
        # Create temporary DataFrame for constant feature detection (with NaNs)
        Xd_temp_df = pd.DataFrame(orig_Xd)
        
        # 1) Remove constants on FULL dataset (consistent across splits)
        non_na_uniques = Xd_temp_df.nunique(dropna=True)
        constant_features = non_na_uniques[non_na_uniques < 2].index.tolist()
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features from full dataset")
            # Remove constant features from original data
            orig_Xd = np.delete(orig_Xd, constant_features, axis=1)
        
        # Check for NaN values but don't impute yet
        nan_mask = np.isnan(orig_Xd)
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} NaN values, will use per-split median imputation")
        

        # Store base cleaning metadata (imputer stats will be added per split)
        base_cleaning_metadata = {
            "imputation_strategy": "median",
            "float32_max": float(float32_max),
            "float32_min": float(float32_min),
            "inf_values_found": bool(np.any(inf_mask)),
            "nan_values_found": bool(np.any(nan_mask))
        }
        
        data_metadata = {
            "original_data_shape": list(orig_Xd.shape),
            "descriptor_columns": descriptor_columns if descriptor_columns else [],
            "rdkit_included": args.incl_rdkit,
            "constant_features_removed": constant_features,
            "post_constant_shape": list(orig_Xd.shape)  # After constant removal
        }
        
        split_metadata = {
            "train_indices": [idx.tolist() for idx in train_indices],
            "val_indices": [idx.tolist() for idx in val_indices], 
            "test_indices": [idx.tolist() for idx in test_indices],
            "random_seed": SEED,
            "split_type": getattr(args, 'split_type', 'random'),
            "correlation_threshold": 0.90,
            "correlation_method": "pearson" if args.task_type == "reg" else "spearman"
        }
    
    # Initialize preprocessing metadata and object storage (outside descriptor block)
    split_preprocessing_metadata = {}
    split_imputers = {}
    
    if combined_descriptor_data is not None:
            
        for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
            checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix, copoly_suffix = build_experiment_paths(
                args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
            )
            # Try to load cached preprocessing BEFORE doing any heavy work
            preprocessing_reused, cached_scaler, cached_mask, cache_meta = manage_preprocessing_cache(
                preprocessing_path, i, orig_Xd, None, None, logger
            )

            if preprocessing_reused and cached_mask is not None:
                imputer_stats = None
                if cache_meta is not None:
                    imputer_stats = ((cache_meta.get("cleaning") or {}).get("imputer_statistics"))
                imputer = None
                if imputer_stats is not None:
                    stats = np.asarray(imputer_stats, dtype=float)
                    imputer = SimpleImputer(strategy="median")
                    imputer.statistics_ = stats
                    imputer.n_features_in_ = stats.shape[0]
                    imputer._fit_dtype = np.asarray(orig_Xd, dtype=np.float64).dtype

                base = orig_Xd.copy()
                if imputer is not None:
                    base = imputer.transform(base)
                elif np.isnan(base).any():
                    # cache didn’t have stats => fit on TRAIN ONLY for this split
                    tmp_imputer = SimpleImputer(strategy="median")
                    tmp_imputer.fit(orig_Xd[tr])
                    base = tmp_imputer.transform(base)
                    imputer = tmp_imputer
                base = np.clip(base, float32_min, float32_max).astype(np.float32)
                mask = np.array(cached_mask, dtype=bool)

                def _apply(datapoints, row_indices):
                    for dp, ridx in zip(datapoints, row_indices):
                        dp.x_d = base[ridx][mask]
                _apply(train_data[i], tr); _apply(val_data[i], va); _apply(test_data[i], te)

                split_preprocessing_metadata[i] = cache_meta or {}
                # Ensure correlation_mask reflects what we actually used
                split_preprocessing_metadata[i].setdefault("split_specific", {})
                split_preprocessing_metadata[i]["split_specific"].update({
                    "split_id": i,
                    "correlation_mask": mask.tolist(),
                })
                split_imputers[i] = imputer
                logger.info(f"Split {i}: reused cached preprocessing (imputer+mask).")
            else:    
                # Initialize default values
                correlated_features = []
                # Per-split data cleaning: fit imputer on training data only
                
                # Get training data for this split (after constant removal)
                train_data_split = orig_Xd[tr]
                
                # Fit imputer on training data only
                imputer = None
                if np.any(nan_mask):
                    imputer = SimpleImputer(strategy='median')
                    train_data_clean = imputer.fit_transform(train_data_split)
                else:
                    train_data_clean = train_data_split.copy()
                
                # Apply imputation to all splits using training-fitted imputer
                if imputer is not None:
                    all_data_clean = imputer.transform(orig_Xd)
                else:
                    all_data_clean = orig_Xd.copy()
                
                # Clip and convert to float32
                all_data_clean = np.clip(all_data_clean, float32_min, float32_max)
                all_data_clean = all_data_clean.astype(np.float32)
                
                # Create DataFrame for correlation analysis
                train_df = pd.DataFrame(all_data_clean[tr])
                
                # Find highly correlated features in training set
                if train_df.shape[1] > 1:
                    corr_matrix = train_df.corr(method="pearson" if args.task_type == "reg" else "spearman").abs()
                    upper_tri = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    correlated_features = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.90)]
                    
                    if correlated_features:
                        logger.info(f"Split {i}: Removing {len(correlated_features)} correlated features based on training set")
                        keep_features = [col for col in range(train_df.shape[1]) if col not in correlated_features]
                    else:
                        keep_features = list(range(train_df.shape[1]))
                else:
                    keep_features = list(range(train_df.shape[1]))
                
                # Create mask for features to keep after correlation removal
                mask = np.zeros(all_data_clean.shape[1], dtype=bool)
                mask[keep_features] = True
                
                # Update cleaning metadata with per-split imputer statistics
                cleaning_metadata = base_cleaning_metadata.copy()
                cleaning_metadata["imputer_statistics"] = imputer.statistics_.tolist() if imputer is not None else None

                # Store split-specific preprocessing metadata
                preprocessing_metadata = {
                    "cleaning": cleaning_metadata,
                    "data_info": data_metadata,
                    "splits": split_metadata,
                    "split_specific": {
                        "split_id": i,
                        "correlated_features": correlated_features,
                        "keep_features": keep_features,
                        "correlation_mask": mask.tolist()
                    },
                    "target": target,
                    "task_type": args.task_type
                }
                
                # Store metadata and imputer for later saving (after checkpoint_path is created)
                split_preprocessing_metadata[i] = preprocessing_metadata
                split_imputers[i] = imputer

                # Apply preprocessing and masking to datapoints
                def _apply_preprocessing_and_mask(datapoints, row_indices):
                    for dp, ridx in zip(datapoints, row_indices):
                        # Get cleaned data for this row
                        row_clean = all_data_clean[ridx]  # Already imputed, clipped, and converted
                        # Apply correlation mask (keep only non-correlated features)
                        dp.x_d = row_clean[mask].astype(np.float32)
                
                _apply_preprocessing_and_mask(train_data[i], tr)
                _apply_preprocessing_and_mask(val_data[i], va)
                _apply_preprocessing_and_mask(test_data[i], te)

                # Enhanced sanity check with more detailed debugging
                logger.debug(f"Split {i}: Applied preprocessing - shape: {all_data_clean.shape}")
                logger.debug(f"Split {i}: Features kept: {np.sum(mask)} out of {len(mask)}")
                logger.debug(f"Split {i}: Imputer fitted on {len(tr)} training samples")
                def _check(dps):
                    arrs = []
                    for dp in dps:
                        x = np.asarray(dp.x_d, dtype=np.float32)
                        if not np.isfinite(x).all():
                            nan_mask = ~np.isfinite(x)
                            logger.debug(f"Found {nan_mask.sum()} non-finite values in a datapoint")
                            logger.debug(f"Non-finite indices: {np.where(nan_mask)[0]}")
                            logger.debug(f"Non-finite values: {x[nan_mask]}")
                        arrs.append(x)
                    
                    X = np.stack(arrs, axis=0)   # will fail if lengths differ
                    logger.debug(f"Final tabular data - shape: {X.shape}, dtype: {X.dtype}")
                    logger.debug(f"Final tabular data - finite values: {np.isfinite(X).all()}")
                    logger.debug(f"Final tabular data - NaN count: {np.isnan(X).sum()}")
                    logger.debug(f"Final tabular data - Inf count: {np.isinf(X).sum()}")
                    
                    if not np.isfinite(X).all():
                        # Print some statistics about the non-finite values
                        nan_mask = ~np.isfinite(X)
                        logger.debug("\nNon-finite value statistics:")
                        logger.debug(f"Total non-finite values: {nan_mask.sum()}")
                        logger.debug(f"Non-finite values per feature: {np.sum(nan_mask, axis=0)}")
                        logger.debug(f"Samples with non-finite values: {np.any(nan_mask, axis=1).sum()}")
                    
                    return X
                    
                logger.debug("\n=== Training Data Check ===")
                _check(train_data[i])
                logger.debug("\n=== Validation Data Check ===")
                _check(val_data[i])
                logger.debug("\n=== Test Data Check ===")
                _check(test_data[i])




    # === Train ===
    results_all = []
    num_splits = len(train_data)  # robust for both CV and holdout
    for i in range(num_splits):
        # Build experiment paths
        checkpoint_path, preprocessing_path, desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, fusion_suffix, aux_suffix, copoly_suffix = build_experiment_paths(
            args, chemprop_dir, checkpoint_dir, target, descriptor_columns, i
        )

        if combined_descriptor_data is not None:
            imputer = split_imputers[i]
            if imputer is not None:
                all_data_clean_i = imputer.transform(orig_Xd)
            else:
                all_data_clean_i = orig_Xd.copy()
            all_data_clean_i = np.clip(all_data_clean_i, float32_min, float32_max).astype(np.float32)

            mask_i = np.array(
                split_preprocessing_metadata[i]['split_specific']['correlation_mask'],
                dtype=bool
            )

            # overwrite x_d for ONLY this split right before dataset construction
            for dp, ridx in zip(train_data[i], train_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
            for dp, ridx in zip(val_data[i],   val_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]
            for dp, ridx in zip(test_data[i],  test_indices[i]):
                dp.x_d = all_data_clean_i[ridx][mask_i]

            # ---- cache/scaler handling (descriptor-only) ----
            preprocessing_reused, cached_scaler, correlation_mask, constant_features = manage_preprocessing_cache(
                preprocessing_path, i, orig_Xd, split_preprocessing_metadata, None, logger
            )
            m_local = np.array(split_preprocessing_metadata[i]['split_specific']['correlation_mask'], dtype=bool)
            cm = np.array(correlation_mask, dtype=bool)
            assert m_local.shape == cm.shape and np.all(m_local == cm), \
                "Local correlation mask != cached correlation mask (split consistency issue)"

            processed_descriptor_data = orig_Xd[:, mask_i]
            descriptor_scaler = cached_scaler  # Will be used after dataset creation
        else:
            preprocessing_reused = False
            descriptor_scaler = None
            processed_descriptor_data = None

        # Create datasets after cleaning
        # Small molecule models (DMPNN, PPG, GAT, GIN, etc.) use MoleculeDataset
        # Only wDMPNN uses PolymerDataset (for polymer-specific datapoints with edges)
        small_molecule_models = ["DMPNN", "DMPNN_DiffPool", "PPG", "GIN", "GIN0", "GINE", "GAT", "GATv2", "AttentiveFP"]
        DS = data.MoleculeDataset if args.model_name in small_molecule_models else data.PolymerDataset
        train = DS(train_data[i], featurizer)
        val = DS(val_data[i], featurizer)
        test = DS(test_data[i], featurizer)
        
        # Now normalize inputs if we have descriptors
        if combined_descriptor_data is not None:
            if descriptor_scaler is not None:
                # Use cached scaler
                train.normalize_inputs("X_d", descriptor_scaler)
                val.normalize_inputs("X_d", descriptor_scaler)
                test.normalize_inputs("X_d", descriptor_scaler)
            else:
                # Fit new scaler on training data
                descriptor_scaler = train.normalize_inputs("X_d")
                val.normalize_inputs("X_d", descriptor_scaler)
                test.normalize_inputs("X_d", descriptor_scaler)
                # persist the fitted scaler
                _ = manage_preprocessing_cache(
                    preprocessing_path, i, orig_Xd, split_preprocessing_metadata, descriptor_scaler, logger
                )

        # Chemprop convention:
        # - Fit scaler on train, apply to train/val targets (for training stability)
        # - DO NOT scale test targets; predictions are unscaled by output_transform
        #
        # When aux targets are appended, we must standardize them SEPARATELY
        # because normalize_targets() returns a scaler used for output_transform
        # which must only cover the main target columns.
        if args._n_aux_targets > 0:
            n_aux = args._n_aux_targets
            # 1) Strip aux columns from datapoints before main normalization
            # dp.y may be 1D (flat list) or 2D ([[...]]); flatten to 1D for safe indexing
            aux_train_raw = np.array([np.asarray(dp.y).flatten()[-n_aux:] for dp in train_data[i]], dtype=np.float64)
            aux_val_raw = np.array([np.asarray(dp.y).flatten()[-n_aux:] for dp in val_data[i]], dtype=np.float64)
            aux_test_raw = np.array([np.asarray(dp.y).flatten()[-n_aux:] for dp in test_data[i]], dtype=np.float64)

            # Temporarily remove aux columns for main normalization
            # Use replace() for slotted dataclasses (can't directly assign to dp.y)
            train_data[i] = [replace(dp, y=np.asarray(dp.y).flatten()[:-n_aux]) for dp in train_data[i]]
            val_data[i] = [replace(dp, y=np.asarray(dp.y).flatten()[:-n_aux]) for dp in val_data[i]]
            test_data[i] = [replace(dp, y=np.asarray(dp.y).flatten()[:-n_aux]) for dp in test_data[i]]

            # Rebuild datasets without aux columns for normalization
            train = DS(train_data[i], featurizer)
            val = DS(val_data[i], featurizer)
            test = DS(test_data[i], featurizer)

        if args.task_type == 'reg':
            scaler = train.normalize_targets()
            val.normalize_targets(scaler)
            # test targets intentionally left unscaled

        if args._n_aux_targets > 0:
            n_aux = args._n_aux_targets
            # 2) Standardize aux targets using training-set stats
            aux_mu = np.nanmean(aux_train_raw, axis=0)
            aux_sd = np.nanstd(aux_train_raw, axis=0)
            aux_sd[aux_sd < 1e-8] = 1.0  # prevent division by zero

            def _standardize_aux(raw):
                return ((raw - aux_mu) / aux_sd).astype(np.float32)

            aux_train_std = _standardize_aux(aux_train_raw)
            aux_val_std = _standardize_aux(aux_val_raw)
            aux_test_std = _standardize_aux(aux_test_raw)

            # 3) Re-append standardized aux columns to datapoints' y
            # Use replace() for slotted dataclasses and update the dataset's data list
            train.data = [
                replace(dp, y=np.concatenate([dp.y, aux_train_std[j:j+1]], axis=-1) if dp.y.ndim == 2 
                        else np.concatenate([dp.y, aux_train_std[j]]))
                for j, dp in enumerate(train.data)
            ]
            val.data = [
                replace(dp, y=np.concatenate([dp.y, aux_val_std[j:j+1]], axis=-1) if dp.y.ndim == 2 
                        else np.concatenate([dp.y, aux_val_std[j]]))
                for j, dp in enumerate(val.data)
            ]
            test.data = [
                replace(dp, y=np.concatenate([dp.y, aux_test_std[j:j+1]], axis=-1) if dp.y.ndim == 2 
                        else np.concatenate([dp.y, aux_test_std[j]]))
                for j, dp in enumerate(test.data)
            ]

            logger.info(f"Split {i}: Aux targets standardized (mu={aux_mu}, sd={aux_sd})")
        

        # Modular metric selection
        n_classes_arg = n_classes_per_target[target] if args.task_type == 'multi' else None
        metric_list = get_metric_list(
            args.task_type,
            target=target,
            n_classes=n_classes_arg,
            df_input=df_input
        )
        batch_norm = args.batch_norm
        
        
        # Create dataloaders
        train_loader = data.build_dataloader(train, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
        val_loader = data.build_dataloader(val, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        test_loader = data.build_dataloader(test, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        
        # Clean up incompatible checkpoints if preprocessing changed
        # Only delete checkpoints if descriptors are used and preprocessing actually changed
        if not preprocessing_reused and checkpoint_path.exists() and combined_descriptor_data is not None:
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed incompatible checkpoint directory: {checkpoint_path}")
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        
        # Assertion to prevent descriptor dimension regression
        desc_len_seen = len(train[0].x_d) if getattr(train[0], "x_d", None) is not None else 0
        if processed_descriptor_data is not None:
            logger.info(f"[split {i}] datapoint descriptor dim: {desc_len_seen}, processed dim: {processed_descriptor_data.shape[1]}")
            assert processed_descriptor_data.shape[1] == desc_len_seen, \
                "Descriptor dim mismatch: datapoints vs processed_descriptor_data."
        
        scaler_arg = scaler if args.task_type == 'reg' else None
        mpnn, trainer = build_model_and_trainer(
            args=args,
            combined_descriptor_data=processed_descriptor_data,
            n_classes=n_classes_arg,
            scaler=scaler_arg,
            checkpoint_path=checkpoint_path,
            batch_norm=batch_norm,
            metric_list=metric_list,
            early_stopping_patience=PATIENCE,
            max_epochs=EPOCHS,
            save_checkpoint=args.save_checkpoint,
            featurizer=featurizer,
        )
        # Validate checkpoint compatibility and get resume path
        descriptor_dim = processed_descriptor_data.shape[1] if processed_descriptor_data is not None else 0
        last_ckpt = validate_checkpoint_compatibility(
            checkpoint_path, preprocessing_path, i, descriptor_dim, logger
        )
        # ---- Skip training logic (align with AttentiveFP semantics) ----
        inprog_flag = checkpoint_path / "TRAINING_IN_PROGRESS"
        done_flag   = checkpoint_path / "TRAINING_COMPLETE"

        best_ckpt_path, best_val_loss = None, None
        skip_training = False

        if done_flag.exists():
            best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
            if best_ckpt_path is not None:
                skip_training = True
                logger.info(f"[{target}] split {i}: Found TRAINING_COMPLETE; skipping training.\n"
                            f"  -> best_ckpt: {best_ckpt_path}"
                            + (f" (val_loss={best_val_loss:.6f})" if best_val_loss is not None else ""))
        else:
            logger.info(f"[{target}] split {i}: No TRAINING_COMPLETE flag; will (re)train.")


        # Train or skip
        if skip_training and best_ckpt_path:
            logger.info(f"Loading checkpoint for evaluation: {best_ckpt_path}")
            from chemprop import models
            use_cuda = torch.cuda.is_available()
            map_location = None if use_cuda else torch.device("cpu")
            mpnn = models.MPNN.load_from_checkpoint(best_ckpt_path, map_location=map_location)
            if use_cuda:
                mpnn = mpnn.to(torch.device("cuda"))
            mpnn.eval()
        else:
            inprog_flag.touch(exist_ok=True)
            try:
                trainer.fit(mpnn, train_loader, val_loader, ckpt_path=last_ckpt)
                best_ckpt_path, best_val_loss = pick_best_checkpoint(checkpoint_path)
                if best_ckpt_path is None:
                    logger.warning(f"[{target}] split {i}: training finished but no checkpoint found.")
                else:
                    with open(checkpoint_path / "best.json", "w") as f:
                        json.dump({"best_ckpt": best_ckpt_path, "best_val_loss": best_val_loss}, f, indent=2)
                    done_flag.touch()
            finally:
                if inprog_flag.exists():
                    inprog_flag.unlink(missing_ok=True)


        results = trainer.test(model=mpnn, dataloaders=test_loader)
        # results = trainer.test(dataloaders=test_loader)
        test_metrics = results[0]
        test_metrics['split'] = i  # Add split index to metrics
        results_all.append(test_metrics)
        
        # Save predictions if requested
        if args.save_predictions:
            logger.info(f"Extracting predictions for split {i}, target {target}")
            
            # Use trainer.predict for unscaled outputs (applies same transform as trainer.test)
            # y_pred = trainer.predict(dataloaders=test_loader)
            y_pred = trainer.predict(model=mpnn, dataloaders=test_loader)
            
            # Extract y_true and IDs directly from test dataset to match loader order
            y_true = np.array([dp.y[0] if isinstance(dp.y, (list, np.ndarray)) else dp.y for dp in test], dtype=float)
            
            # Extract IDs/indices for order verification
            test_ids = []
            for dp in test:
                if hasattr(dp, 'id') and dp.id is not None:
                    test_ids.append(dp.id)
                elif hasattr(dp, 'smiles'):
                    test_ids.append(dp.smiles)  # Use SMILES as fallback ID
                else:
                    test_ids.append(f"idx_{len(test_ids)}")  # Fallback to index
            
            # Convert predictions to numpy - handle list of tensors properly
            if isinstance(y_pred, list):
                # Concatenate tensors from different batches
                import torch
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
            elif hasattr(y_pred, 'cpu'):
                y_pred = y_pred.cpu().numpy()
            
            # Save predictions with IDs and training configuration metadata
            save_predictions(
                y_true, y_pred, predictions_dir, args.dataset_name, target, args.model_name,
                desc_suffix, rdkit_suffix, batch_norm_suffix, size_suffix, copoly_suffix, i, logger,
                test_ids=test_ids,
                copolymer_mode=None,
                polymer_type=getattr(args, 'polymer_type', None),
                task_type=args.task_type,
                fusion_mode=getattr(args, 'fusion_mode', None),
                aux_task=getattr(args, 'aux_task', None)
            )
        
        # Export embeddings if requested
        if args.export_embeddings:
            logger.info(f"Exporting embeddings for split {i}, target {target}")
            
            # Set model to evaluation mode and extract embeddings
            mpnn.eval()
            X_train = get_encodings_from_loader(mpnn, train_loader)
            X_val = get_encodings_from_loader(mpnn, val_loader)
            X_test = get_encodings_from_loader(mpnn, test_loader)
            
            # Apply same filtering as in evaluate_model.py (remove low-variance features)
            eps = 1e-8
            std_train = X_train.std(axis=0)
            keep = std_train > eps
            
            X_train = X_train[:, keep]
            X_val = X_val[:, keep]
            X_test = X_test[:, keep]
            
            logger.info(f"Split {i}: Kept {int(keep.sum())} / {len(keep)} embedding dimensions")
            
            # Create embeddings directory with target/model/size specificity
            embeddings_dir = results_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Build embedding filename prefix with all identifiers including model name
            embedding_prefix = f"{args.dataset_name}__{args.model_name}__{target}{desc_suffix}{rdkit_suffix}{batch_norm_suffix}{fusion_suffix}{aux_suffix}{size_suffix}"
            
            # Save embeddings as numpy arrays with full identifiers
            np.save(embeddings_dir / f"{embedding_prefix}__X_train_split_{i}.npy", X_train)
            np.save(embeddings_dir / f"{embedding_prefix}__X_val_split_{i}.npy", X_val)
            np.save(embeddings_dir / f"{embedding_prefix}__X_test_split_{i}.npy", X_test)
            
            # Save feature mask for reproducibility
            np.save(embeddings_dir / f"{embedding_prefix}__feature_mask_split_{i}.npy", keep)
            
            logger.info(f"Split {i}: Saved embeddings to {embeddings_dir}")
            logger.info(f"  - X_train: {X_train.shape}")
            logger.info(f"  - X_val: {X_val.shape}")
            logger.info(f"  - X_test: {X_test.shape}")
    

    # Convert to DataFrame
    results_df = pd.DataFrame(results_all)
    # Calculate mean/std only for numeric metric columns (exclude 'split')
    numeric_cols = [col for col in results_df.columns if col != 'split']
    mean_metrics = results_df[numeric_cols].mean()
    std_metrics = results_df[numeric_cols].std()

    n_evals = len(results_all)
    logger.info(f"\n[{target}] Mean across {n_evals} splits:\n{mean_metrics}")
    logger.info(f"\n[{target}] Std across {n_evals} splits:\n{std_metrics}")


    # Always add target column for proper result organization
    results_df['target'] = target
    all_results.append(results_df)
    
# Save final results using modular function
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    save_model_results(combined_results, args, args.model_name, results_dir, logger)