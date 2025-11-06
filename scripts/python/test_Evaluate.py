import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re
import torch

import logging

from utils import (
    set_seed, process_data, 
    determine_split_strategy, 
    generate_data_splits, 
    build_sklearn_models,
    load_and_preprocess_data,
    pick_best_checkpoint,
    get_encodings_from_loader,
)


from chemprop import data, featurizers, models

from sklearn.metrics import (
    r2_score, accuracy_score, f1_score, roc_auc_score
)

from attentivefp_utils import build_attentivefp_loaders, extract_attentivefp_embeddings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_config_from_checkpoint_path(checkpoint_path: str) -> dict:
    """Extract training configuration from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Configuration with keys 'descriptors', 'rdkit', 'batch_norm', 'train_size'
    """
    # Extract the experiment name from the path
    # Format: /path/to/checkpoints/MODEL/dataset__target__[desc]__[rdkit]__[batch_norm]__[sizeN]__repN/...
    path_parts = Path(checkpoint_path).parts
    
    # Find the experiment directory (contains dataset name and suffixes)
    experiment_dir = None
    for part in path_parts:
        if '__rep' in part:
            experiment_dir = part
            break
    
    if not experiment_dir:
        logger.warning(f"Could not extract experiment configuration from checkpoint path: {checkpoint_path}")
        return {'descriptors': False, 'rdkit': False, 'batch_norm': False, 'train_size': 'full'}
    
    # Parse the experiment directory name
    config = {
        'descriptors': '__desc' in experiment_dir,
        'rdkit': '__rdkit' in experiment_dir,
        'batch_norm': '__batch_norm' in experiment_dir,
        'train_size': 'full'
    }
    
    # Extract train_size if present
    if '__size' in experiment_dir:
        import re
        size_match = re.search(r'__size(\d+)', experiment_dir)
        if size_match:
            config['train_size'] = size_match.group(1)
    
    return config


# === Modular Functions to Eliminate Code Duplication ===

def get_metric_columns(task_type: str, results_df: pd.DataFrame) -> list[str]:
    cols = {
        "reg": ["test/mae", "test/r2", "test/rmse"],
        "binary": ["test/accuracy", "test/f1", "test/roc_auc"],
        "multi": ["test/accuracy", "test/f1"],  # roc_auc often absent for multi
    }[task_type]
    return [c for c in cols if c in results_df.columns]



def build_results_filename(args, results_dir: Path, descriptor_columns: list = None) -> Path:
    """Build results filename with appropriate suffixes."""
    # Create model results directory
    model_results_dir = results_dir / args.model_name
    model_results_dir.mkdir(exist_ok=True)
    
    # Build filename with suffixes
    filename_parts = [args.dataset_name]
    
    # Add descriptor suffixes (for DMPNN pipeline)
    if descriptor_columns:
        filename_parts.append("desc")
    
    if hasattr(args, 'incl_rdkit') and args.incl_rdkit:
        filename_parts.append("rdkit")
    
    if hasattr(args, 'batch_norm') and args.batch_norm:
        filename_parts.append("batch_norm")
    
    # Add target suffix if specific target
    if args.target:
        filename_parts.append(args.target)
    
    # Join with double underscores and add suffix
    if len(filename_parts) == 1:
        filename = f"{filename_parts[0]}_baseline.csv"
    else:
        filename = "__".join(filename_parts) + "_baseline.csv"
    
    return model_results_dir / filename


def save_evaluation_results(results_df: pd.DataFrame, args, results_dir: Path, 
                          descriptor_columns: list = None, model_name: str = None) -> Path:
    """Save evaluation results with proper formatting and return the output path."""
    if results_df.empty:
        logger.warning("No results to save - empty DataFrame")
        return None
    
    # Build output filename
    out_csv = build_results_filename(args, results_dir, descriptor_columns)
    
    # Organize columns: target, split, then metrics, then model
    base_cols = ["target", "split"]
    metric_cols = get_metric_columns(args.task_type, results_df)
    extra_cols = [c for c in results_df.columns if c not in base_cols + metric_cols]
    results_df = results_df[base_cols + metric_cols + extra_cols]
    
    # Save to CSV
    results_df.to_csv(out_csv, index=False)
    
    return out_csv


def print_evaluation_summary(results_df: pd.DataFrame, model_name: str = None):
    """Print summary statistics for evaluation results."""
    model_label = f"{model_name} " if model_name else ""
    logger.info(f"\n=== {model_label}Evaluation Summary ===")
    
    for col in results_df.columns:
        if col.startswith('test/'):
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            logger.info(f"{col}: {mean_val:.4f} ± {std_val:.4f}")


def save_and_summarize_results(results_df: pd.DataFrame, args, results_dir: Path,
                             descriptor_columns: list = None, model_name: str = None) -> Path:
    """Complete results saving and summary printing."""
    if results_df.empty:
        logger.warning("No results to save")
        return None
    
    # Save results
    out_csv = save_evaluation_results(results_df, args, results_dir, descriptor_columns, model_name)
    
    if out_csv:
        logger.info(f"✅ {model_name or args.model_name} results saved to: {out_csv}")
        
        # Print summary statistics
        print_evaluation_summary(results_df, model_name)
    
    return out_csv

def make_experiment_tokens(args, target, has_desc: bool) -> dict:
    return dict(
        desc="__desc" if has_desc else "",
        rdkit="__rdkit" if getattr(args, "incl_rdkit", False) else "",
        bn="__batch_norm" if getattr(args, "batch_norm", False) else "",
        size=("" if not getattr(args, "train_size", None)
              or str(args.train_size).lower() == "full"
              else f"__size{args.train_size}")
    )

def experiment_base(args, target, descriptor_columns):
    t = make_experiment_tokens(args, target, bool(descriptor_columns))
    return f"{args.dataset_name}__{target}{t['desc']}{t['rdkit']}{t['bn']}{t['size']}__rep{{rep}}"


def paths_from_base(args, setup_info, base_name_with_placeholder, rep: int):
    chemprop_dir    = Path(setup_info["chemprop_dir"])
    checkpoint_root = Path(setup_info["checkpoint_dir"])   # already model-specific from setup
    results_dir     = Path(setup_info["results_dir"])

    exp_name = base_name_with_placeholder.format(rep=rep)

    # per-rep dirs
    ckpt_dir   = checkpoint_root / exp_name                           # DMPNN: directory of lightning ckpts; AttentiveFP: contains best.pt
    preproc_dir= chemprop_dir / "preprocessing" / exp_name

    # train_graph.py primary embeddings location (+ file prefix)
    primary_embeddings_dir = results_dir / "embeddings"

    # <dataset>__<model>__<target><desc><rdkit><bn><size>  (no __rep in file prefix)
    prefix_no_rep = re.sub(r"__rep\{.*\}$", "", base_name_with_placeholder)
    # drop the leading "<dataset>__"
    right = prefix_no_rep.split("__", 1)[1] if "__" in prefix_no_rep else prefix_no_rep
    embedding_prefix = f"{args.dataset_name}__{args.model_name}__{right}"

    return ckpt_dir, preproc_dir, primary_embeddings_dir, embedding_prefix

def resolve_checkpoint_for_rep(args, setup_info, target: str, rep: int, descriptor_columns) -> Path | None:
    """
    Returns a file path for the checkpoint to load for this rep, or None if missing.
    AttentiveFP -> <dir>/best.pt
    DMPNN-family -> pick_best_checkpoint(<dir>)
    """
    # (A) If the user passed a path, rewrite only the __repN suffix to this rep:
    if args.checkpoint_path:
        p = Path(args.checkpoint_path)
        # If it's a file, operate on parent; else on dir name
        base = p.parent if p.suffix else p
        name = base.name
        import re
        name = re.sub(r'__rep\d+', f'__rep{rep}', name) if '__rep' in name else f"{name}__rep{rep}"
        candidate_dir = base.parent / name
        if args.model_name == "AttentiveFP":
            ckpt = candidate_dir / "best.pt"
            return ckpt if ckpt.exists() else None
        else:
            ckpt, _ = pick_best_checkpoint(candidate_dir)
            return Path(ckpt) if ckpt else None

    # (B) Normal discovery via your experiment naming
    base = experiment_base(args, target, setup_info['descriptor_columns'])
    ckpt_dir, *_ = paths_from_base(args, setup_info, base, rep)
    if args.model_name == "AttentiveFP":
        ckpt = ckpt_dir / "best.pt"
        return ckpt if ckpt.exists() else None
    else:
        ckpt, _ = pick_best_checkpoint(ckpt_dir)
        return Path(ckpt) if ckpt else None


# ---------- tiny loaders for each model family ----------

def build_dmpnn_loaders(args, setup_info, target, train_idx, val_idx, test_idx, smis, df_input,
                        combined_descriptor_data):
    from utils import create_all_data
    small_molecule_models = ["DMPNN", "DMPNN_DiffPool", "PPG"]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() \
        if args.model_name in small_molecule_models else featurizers.PolymerMolGraphFeaturizer()

    ys = df_input.loc[:, target].astype(float).values
    if args.task_type != 'reg': ys = ys.astype(int)
    ys = ys.reshape(-1, 1)

    all_data = create_all_data(smis, ys, combined_descriptor_data, args.model_name)
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, [train_idx], [val_idx], [test_idx]
    )
    # index 0 because split_data_by_indices returns lists per split
    train = data.MoleculeDataset(train_data[0], featurizer)
    val   = data.MoleculeDataset(val_data[0],   featurizer)
    test  = data.MoleculeDataset(test_data[0],  featurizer)

    # Regression target scaling like train_graph.py
    if args.task_type == 'reg':
        scaler = train.normalize_targets()
        val.normalize_targets(scaler)
        # test targets intentionally unscaled
    else:
        scaler = None

    num_workers = setup_info['num_workers']
    use_workers = num_workers if torch.cuda.is_available() else 0

    train_loader = data.build_dataloader(train, batch_size=args.batch_size, num_workers=use_workers, shuffle=False)
    val_loader   = data.build_dataloader(val,   batch_size=args.batch_size, num_workers=use_workers, shuffle=False)
    test_loader  = data.build_dataloader(test,  batch_size=args.batch_size, num_workers=use_workers, shuffle=False)
    return train_loader, val_loader, test_loader



# ---------- main overall process skeleton ----------
def extract_embeddings_for_rep(args, setup_info, target, rep, tr, va, te,
                               smis, df_input, combined_descriptor_data,
                               smiles_col, base_ckpt_dir: Path,
                               keep_eps: float = 1e-8):
    """Returns (X_train, X_val, X_test, keep_mask) and writes cache to both primary & temp."""
    import torch
    # Build loaders
    if args.model_name == "AttentiveFP":
        from attentivefp_utils import create_attentivefp_model
        df_tr, df_va, df_te = df_input.iloc[tr].reset_index(drop=True), df_input.iloc[va].reset_index(drop=True), df_input.iloc[te].reset_index(drop=True)
        train_loader, val_loader, test_loader, _ = build_attentivefp_loaders(
            args, df_tr, df_va, df_te, smiles_col, target, eval=True
        )
        ckpt = resolve_checkpoint_for_rep(args, setup_info, target, rep, setup_info['descriptor_columns'])
        if not ckpt or not ckpt.exists():
            logger.warning(f"[rep {rep}] missing AttentiveFP ckpt: {ckpt}")
            return None
        logger.info(f"[rep {rep}] checkpoint: {ckpt}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden = getattr(args, "hidden", 200)
        n_classes = None if args.task_type != 'multi' else int(df_input[target].dropna().nunique())
        model = create_attentivefp_model(args.task_type, n_classes, hidden_channels=hidden,
                                         num_layers=2, num_timesteps=2, dropout=0.0).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict({k: v.to(device) for k, v in state["state_dict"].items()})
        X_train = extract_attentivefp_embeddings(model, train_loader, device)
        X_val   = extract_attentivefp_embeddings(model, val_loader,   device)
        X_test  = extract_attentivefp_embeddings(model, test_loader,  device)
    else:
        # DMPNN-family
        train_loader, val_loader, test_loader = build_dmpnn_loaders(
            args, setup_info, target, tr, va, te, smis, df_input, combined_descriptor_data
        )
        ckpt = resolve_checkpoint_for_rep(args, setup_info, target, rep, setup_info['descriptor_columns'])
        if not ckpt:
            logger.warning(f"[rep {rep}] missing checkpoint dir")
            return None
        logger.info(f"[rep {rep}] checkpoint: {ckpt}")
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        if args.model_name == "DMPNN_DiffPool":
            # If you’ve got a helper in utils to instantiate, call that; otherwise leave as-is.
            from chemprop import nn, models
            base_mp_cls = nn.BondMessagePassing
            depth = getattr(args, "diffpool_depth", 1)
            ratio = getattr(args, "diffpool_ratio", 0.5)
            mp = nn.BondMessagePassingWithDiffPool(base_mp_cls=base_mp_cls, depth=depth, ratio=ratio)
            desc_dim = (combined_descriptor_data.shape[1] if combined_descriptor_data is not None else 0)
            input_dim = mp.output_dim + desc_dim
            if args.task_type == 'reg':
                predictor = nn.RegressionFFN(output_transform=None, n_tasks=1, input_dim=input_dim)
            elif args.task_type == 'binary':
                predictor = nn.BinaryClassificationFFN(input_dim=input_dim)
            else:
                n_classes = max(2, int(df_input[target].dropna().nunique()))
                predictor = nn.MulticlassClassificationFFN(n_classes=n_classes, input_dim=input_dim)
            mpnn = models.MPNN(message_passing=mp, agg=nn.IdentityAggregation(),
                               predictor=predictor, batch_norm=args.batch_norm, metrics=[])
            import torch as _t
            state = _t.load(ckpt, map_location=map_location)
            mpnn.load_state_dict(state["state_dict"], strict=False)
            mpnn.eval()
        else:
            from chemprop import models
            mpnn = models.MPNN.load_from_checkpoint(str(ckpt), map_location=map_location)
            mpnn.eval()
        X_train = get_encodings_from_loader(mpnn, train_loader)
        X_val   = get_encodings_from_loader(mpnn, val_loader)
        X_test  = get_encodings_from_loader(mpnn, test_loader)

    # Low-variance mask
    std_train = X_train.std(axis=0)
    keep = std_train > keep_eps
    return X_train[:, keep], X_val[:, keep], X_test[:, keep], keep


def _cache_files(rep_idx: int, prefix: str | None = None):
    # with prefix -> "<prefix>__X_train_split_i.npy"; else "X_train_split_i.npy"
    stem = (lambda name: f"{prefix}__{name}" if prefix else name)
    return (
        stem(f"X_train_split_{rep_idx}.npy"),
        stem(f"X_val_split_{rep_idx}.npy"),
        stem(f"X_test_split_{rep_idx}.npy"),
        stem(f"feature_mask_split_{rep_idx}.npy"),
    )

def resolve_embeddings_for_rep(rep_idx: int, primary_dir: Path, base_ckpt_dir: Path, prefix: str | None) -> tuple[Path, bool, tuple[str,str,str,str]]:
    f_tr, f_va, f_te, f_mask = _cache_files(rep_idx, prefix)
    # 1) primary
    primary_have = all((primary_dir / f).exists() for f in (f_tr, f_va, f_te, f_mask))
    if primary_have:
        return primary_dir, True, (f_tr, f_va, f_te, f_mask)
    # 2) temp next to checkpoint base (no prefix there to keep it short)
    f_tr2, f_va2, f_te2, f_mask2 = _cache_files(rep_idx, None)
    temp_dir = base_ckpt_dir / "temp_embeddings"
    temp_have = all((temp_dir / f).exists() for f in (f_tr2, f_va2, f_te2, f_mask2))
    return temp_dir, temp_have, (f_tr2, f_va2, f_te2, f_mask2)

def load_cached_embeddings(d: Path, names: tuple[str,str,str,str]):
    f_tr, f_va, f_te, f_mask = names
    X_tr = np.load(d / f_tr); X_va = np.load(d / f_va); X_te = np.load(d / f_te); keep = np.load(d / f_mask)
    return X_tr, X_va, X_te, keep

def save_cached_embeddings(d: Path, names: tuple[str,str,str,str], X_tr, X_va, X_te, keep):
    d.mkdir(parents=True, exist_ok=True)
    f_tr, f_va, f_te, f_mask = names
    np.save(d / f_tr, X_tr); np.save(d / f_va, X_va); np.save(d / f_te, X_te); np.save(d / f_mask, keep)

def make_base_ckpt_dir(ckpt_dir: Path) -> Path:
    """
    Convert /.../checkpoints/<MODEL>/<dataset>__<target>...__repN
    ->     /.../checkpoints/<MODEL>/<dataset>__repN
    """
    name = ckpt_dir.name
    m = re.match(r'^([^_]+)__.*__(rep\d+)$', name)  # group1=dataset, group2=repN
    if m:
        dataset, rep = m.groups()
        return ckpt_dir.parent / f"{dataset}__{rep}"
    # fallback: keep original dir
    return ckpt_dir




# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Chemprop model for regression or classification')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset file (without .csv extension)')
parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                    help='Type of task: "reg" for regression or "binary" or "multi" for classification')
parser.add_argument('--incl_desc', action='store_true',
                    help='Use dataset-specific descriptors')
parser.add_argument('--incl_rdkit', action='store_true',
                    help='Include RDKit 2D descriptors')
parser.add_argument('--model_name', type=str, default="DMPNN",
                    help='Name of the model to use')
parser.add_argument("--polymer_type", type=str, choices=["homo", "copolymer"], default="homo",
                    help='Type of polymer: "homo" for homopolymer or "copolymer" for copolymer')
parser.add_argument('--target', type=str, default=None,
                    help='Specific target to evaluate (if not provided, evaluates all targets)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to specific checkpoint file to evaluate (overrides automatic checkpoint discovery)')
parser.add_argument('--preprocessing_path', type=str, default=None,
                    help='Path to preprocessing directory (overrides automatic preprocessing path discovery)')
parser.add_argument('--batch_norm', action='store_true',
                    help='Use batch normalization models for evaluation')
parser.add_argument('--train_size', type=str, default=None,
                    help='Training size used during model training (for path matching)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for dataloaders during embedding extraction')


args = parser.parse_args()

# Auto-detect task type for specific datasets
if args.dataset_name == 'polyinfo' and args.task_type == 'reg':
    args.task_type = 'multi'
    logger.info(f"Auto-detected task type for {args.dataset_name}: {args.task_type}")

logger.info("\n=== Evaluation Configuration ===")
logger.info(f"Dataset       : {args.dataset_name}")
logger.info(f"Task type     : {args.task_type}")
logger.info(f"Model         : {args.model_name}")
logger.info(f"Target        : {args.target if args.target else 'All targets'}")
logger.info(f"Descriptors   : {'Enabled' if args.incl_desc else 'Disabled'}")
logger.info(f"RDKit desc.   : {'Enabled' if args.incl_rdkit else 'Disabled'}")
logger.info(f"Batch norm    : {'Enabled' if args.batch_norm else 'Disabled'}")
if args.train_size is not None:
    logger.info(f"Train size    : {args.train_size}")
logger.info("===============================\n")

# Setup evaluation environment with model-specific configuration
from utils import setup_model_environment

# Determine model type once
if args.model_name in ["DMPNN", "wDMPNN", "DMPNN_DiffPool", "PPG"]:
    model_type = "dmpnn"
elif args.model_name == "AttentiveFP":
    model_type = "attentivefp"
elif args.model_name == "Graphormer":
    model_type = "graphormer"
else:
    model_type = "dmpnn"
    logger.warning(f"Unknown model type {args.model_name}, using DMPNN environment setup")

setup_info = setup_model_environment(args, model_type)
set_seed(setup_info['SEED'])


# data
df_input, target_columns = load_and_preprocess_data(args, setup_info)
if args.target:
    if args.target not in target_columns:
        logger.error(f"Target '{args.target}' not in {target_columns}")
        sys.exit(1)
    target_columns = [args.target]

smiles_col = setup_info['smiles_column']
descriptor_columns = setup_info['descriptor_columns']
smis, df_input, combined_descriptor_data, n_classes = process_data(
    df_input, smiles_col, descriptor_columns, target_columns, args
)

all_results = []
for target in target_columns:
    ys = df_input[target].astype(float if args.task_type=='reg' else int).values.reshape(-1,1)
    n_splits, local_reps = determine_split_strategy(len(ys), setup_info['REPLICATES'])
    train_indices, val_indices, test_indices = generate_data_splits(
        args, ys, n_splits, local_reps, setup_info['SEED'], df_input=df_input
    )

    base = experiment_base(args, target, descriptor_columns)
    for i, (tr, va, te) in enumerate(zip(train_indices, val_indices, test_indices)):
        ckpt_dir, preproc_dir, emb_dir, emb_prefix = paths_from_base(args, setup_info, base, i)

        base_ckpt_dir = make_base_ckpt_dir(ckpt_dir)

        use_dir, have, fnames = resolve_embeddings_for_rep(i, emb_dir, base_ckpt_dir, prefix=emb_prefix)

        if have:
            X_train, X_val, X_test, keep = load_cached_embeddings(i, use_dir)
        else:
            extracted = extract_embeddings_for_rep(
                args, setup_info, target, i, tr, va, te, smis, df_input,
                combined_descriptor_data, smiles_col, base_ckpt_dir
            )
            if extracted is None:
                logger.warning(f"[rep {i}] skipped (no checkpoint)")
                continue
            X_train, X_val, X_test, keep = extracted
            save_cached_embeddings(i, use_dir, X_train, X_val, X_test, keep)

        # Targets aligned with possibly shorter loader outputs
        y_train = df_input.loc[tr, target].to_numpy()[:len(X_train)]
        y_val   = df_input.loc[va, target].to_numpy()[:len(X_val)]
        y_test  = df_input.loc[te, target].to_numpy()[:len(X_test)]

        # Baselines
        from sklearn.preprocessing import StandardScaler
        target_scaler = None
        if args.task_type == "reg":
            target_scaler = StandardScaler().fit(y_train.reshape(-1,1))
            y_train_scaled = target_scaler.transform(y_train.reshape(-1,1)).ravel()
            y_val_scaled   = target_scaler.transform(y_val.reshape(-1,1)).ravel()
        else:
            y_train_scaled, y_val_scaled = y_train, y_val

        num_classes = None if args.task_type == "reg" else int(np.unique(y_train).size)
        specs = build_sklearn_models(args.task_type, num_classes, scaler_flag=True)

        fold_rows = []
        for name, (mdl, needs_scaler) in specs.items():
            # scale X if requested
            Xtr = X_train; Xva = X_val; Xte = X_test
            if needs_scaler:
                xs = StandardScaler().fit(Xtr)
                Xtr = xs.transform(Xtr); Xva = xs.transform(Xva); Xte = xs.transform(Xte)

            if args.task_type == "reg":
                if name == "XGB":
                    mdl.set_params(early_stopping_rounds=30, eval_metric="rmse")
                    mdl.fit(Xtr, y_train_scaled, eval_set=[(Xva, y_val_scaled)], verbose=False)
                else:
                    mdl.fit(Xtr, y_train_scaled)
                yp = mdl.predict(Xte)
                yp = target_scaler.inverse_transform(yp.reshape(-1,1)).ravel()
                fold_rows.append(dict(target=target, split=i,
                                      **{"test/mae": np.mean(np.abs(yp-y_test)),
                                         "test/rmse": float(np.sqrt(np.mean((yp-y_test)**2))),
                                         "test/r2": float(r2_score(y_test, yp))},
                                      model=name))
            else:
                if name == "XGB":
                    mdl.set_params(early_stopping_rounds=30, eval_metric=("mlogloss" if args.task_type=="multi" else "logloss"))
                    mdl.fit(Xtr, y_train_scaled, eval_set=[(Xva, y_val_scaled)], verbose=False)
                else:
                    mdl.fit(Xtr, y_train_scaled)
                yp = mdl.predict(Xte)
                row = dict(target=target, split=i,
                           **{"test/accuracy": float(accuracy_score(y_test, yp)),
                              "test/f1": float(f1_score(y_test, yp, average=("macro" if args.task_type=="multi" else "binary")))},
                           model=name)
                if hasattr(mdl, "predict_proba"):
                    try:
                        proba = mdl.predict_proba(Xte)
                        if args.task_type == "binary":
                            # ensure both classes present
                            if len(np.unique(y_test)) == 2 and proba.shape[1] == 2:
                                row["test/roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                        else:
                            classes_sorted = sorted(np.unique(np.concatenate([y_train, y_test])))
                            if len(classes_sorted) > 1:
                                from sklearn.preprocessing import label_binarize
                                yb = label_binarize(y_test, classes=classes_sorted)
                                if yb.shape[1] > 1:
                                    row["test/roc_auc"] = float(
                                        roc_auc_score(yb, proba[:, :yb.shape[1]], average="macro", multi_class="ovr")
                                    )
                    except Exception:
                        pass

                fold_rows.append(row)
        all_results.extend(fold_rows)

# Save once
if all_results:
    results_df = pd.DataFrame(all_results)
    out = save_and_summarize_results(results_df, args, setup_info['results_dir'],
                                     descriptor_columns, model_name="baselines")
    logger.info(f"Saved: {out}")
