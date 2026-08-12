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
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
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
from frozen_splits import load_frozen_b_heldout_splits
from regeneration import checkpoint_record, runtime_environment, split_indices_sha256
from utils import generate_a_held_out_splits, set_seed

DATA_PATH = ROOT_DIR / "data" / "ea_ip.csv"
META_PATH = ROOT_DIR / "metadata" / "splits" / "monomer_heldout.json"
B_META_PATH = ROOT_DIR / "metadata" / "splits" / "monomer_b_heldout.json"
B_CLUSTERED_META_PATH = ROOT_DIR / "metadata" / "splits" / "monomer_b_heldout_clustered.json"
PREDICTIONS_DIR = ROOT_DIR / "predictions"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "HPG_Gen"
TARGETS = ["EA vs SHE (eV)", "IP vs SHE (eV)"]
MODEL_TO_POOLING = {
    "hpg_sum": "sum",
    "hpg_frac": "frac_weighted",
    "hpg_hier": "hpg_hier",
    "hpg_hier_wedge": "hpg_hier",
    "hpg_hier_octamer": "hpg_hier",
    "hpg_hier_octamer_stoich": "hpg_hier",
    "hpg_hier_octamer_edges": "hpg_hier",
    "hpg_hier_attention": "hpg_hier",
    "hpg_hier_junction": "hpg_hier",
    "hpg_hier_junction1": "hpg_hier",
}

_VARIANT_FLAGS = {
    "hpg_hier":          {"stage2_edge_weight": "feature",    "stage2_mode": "transition_graph", "stage2_readout": "stoich_weighted", "junction_coupling": "off", "n_coupling_steps": 0},
    "hpg_hier_wedge":    {"stage2_edge_weight": "multiplier", "stage2_mode": "transition_graph", "stage2_readout": "stoich_weighted", "junction_coupling": "off", "n_coupling_steps": 0},
    "hpg_hier_octamer":  {"stage2_edge_weight": "feature",    "stage2_mode": "octamer_sequence", "stage2_readout": "attention", "junction_coupling": "off", "n_coupling_steps": 0},
    "hpg_hier_octamer_stoich": {"stage2_edge_weight": "feature", "stage2_mode": "octamer_sequence", "stage2_readout": "stoich_weighted", "junction_coupling": "off", "n_coupling_steps": 0},
    # M2 — arm D's 8-slot topology + mean readout (stoich_weighted-on-octamer == mean pooling,
    # HANDOFF §7), with the 17-d junction edge features restored into the path layers.
    # M2 vs HPG-hier isolates topology; arm D vs M2 isolates the edge features.
    "hpg_hier_octamer_edges": {"stage2_edge_weight": "feature", "stage2_mode": "octamer_sequence", "stage2_readout": "stoich_weighted", "junction_coupling": "off", "n_coupling_steps": 0, "octamer_edge_features": True},
    "hpg_hier_attention": {"stage2_edge_weight": "feature", "stage2_mode": "transition_graph", "stage2_readout": "attention", "junction_coupling": "off", "n_coupling_steps": 0},
    "hpg_hier_junction": {"stage2_edge_weight": "feature",    "stage2_mode": "transition_graph", "stage2_readout": "stoich_weighted", "junction_coupling": "on",  "n_coupling_steps": 2},
    "hpg_hier_junction1": {"stage2_edge_weight": "feature",   "stage2_mode": "transition_graph", "stage2_readout": "stoich_weighted", "junction_coupling": "on",  "n_coupling_steps": 1},
}


def _resolve_variant(model_token: str, args) -> dict:
    """Resolve the variant flag table with any CLI overrides.

    Returns a fresh dict describing what the model will actually use, including
    the resolved ``stage2_readout`` (overridable via ``--stage2_readout``) and a
    default for ``octamer_edge_features`` so the sidecar is self-contained.
    """
    variant = _VARIANT_FLAGS.get(model_token, _VARIANT_FLAGS["hpg_hier"])
    resolved = dict(variant)
    resolved.setdefault("octamer_edge_features", False)
    resolved["stage2_readout"] = args.stage2_readout or resolved["stage2_readout"]
    return resolved


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
    parser.add_argument("--stage2_readout", choices=("stoich_weighted", "attention"), default=None)
    parser.add_argument("--octamer_len", type=int, default=8)
    parser.add_argument("--n_random_samples", type=int, default=16)
    parser.add_argument("--n_coupling_steps", type=int, default=2)
    parser.add_argument("--octamer_position_embeddings", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--b_split_metadata", type=Path, default=None)
    parser.add_argument("--prediction_dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--frozen_protocol", action="store_true")
    parser.add_argument("--allow_non_cuda", action="store_true",
                        help="Explicitly allow frozen-protocol runs on a non-CUDA accelerator. "
                             "Output filenames will be stamped with __localsmoke so they cannot "
                             "be mistaken for protocol runs (incident: 2026-08-11 local MPS/CUDA mix).")
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--stability_fix", choices=("none", "best_checkpoint", "row_val_best", "fixed_epochs", "arm_c"), default="none")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_lomao_splits(df: pd.DataFrame, split_seed: int):
    if split_seed != 42:
        raise AssertionError(f"monomer_heldout requires fixed split_seed=42, got {split_seed}")
    train, val, test, _ = generate_a_held_out_splits(
        df["smiles_A"].astype(str).values, len(df), seed=split_seed,
        protocol="leave_one_A_out", logger=logger,
    )
    metadata = json.loads(META_PATH.read_text())["folds"]
    if len(test) != len(metadata):
        raise AssertionError(f"Expected {len(metadata)} monomer-heldout folds, got {len(test)}")
    for fold, indices in enumerate(test):
        expected = np.asarray(metadata[fold]["global_test_indices"], dtype=int)
        if not np.array_equal(indices, expected):
            raise AssertionError(
                f"monomer_heldout fold {fold} test indices differ from metadata for split_seed={split_seed}"
            )
    return train, val, test


def _build_splits(df: pd.DataFrame, split_type: str, split_seed: int, b_split_metadata: Path):
    if split_type == "group_disjoint":
        keys = build_group_keys(df)
        splits = generate_group_disjoint_splits(df, n_splits=5, seed=42)
    elif split_type == "pair_disjoint":
        keys = build_pair_keys(df)
        splits = generate_pair_disjoint_splits(df, n_splits=5, seed=42)
    elif split_type == "monomer_heldout":
        return _load_lomao_splits(df, split_seed)
    elif split_type in {"monomer_b_heldout", "monomer_b_heldout_clustered"}:
        metadata_path = b_split_metadata or (B_CLUSTERED_META_PATH if split_type.endswith("_clustered") else B_META_PATH)
        return load_frozen_b_heldout_splits(df, split_seed, metadata_path)
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


def _build_hier_graphs(df: pd.DataFrame, stage2_edge: str, stage2_mode: str = "transition_graph",
                       octamer_len: int = 8, n_random_samples: int = 16, octamer_rng_seed: int = 42,
                       junction_coupling: str = "off"):
    if "WDMPNN_Input" not in df:
        raise ValueError("hpg_hier requires the WDMPNN_Input column")
    featurizer = TwoStageHPGFeaturizer()
    return [
        featurizer(
            value, stage2_edge=stage2_edge, stage2_mode=stage2_mode,
            octamer_len=octamer_len, n_random_samples=n_random_samples, octamer_rng_seed=octamer_rng_seed,
            junction_coupling=junction_coupling,
        )
        for value in df["WDMPNN_Input"].astype(str)
    ]


def _repeat_suffix(args: argparse.Namespace) -> str:
    repeat = "" if args.repeat is None else f"__repeat{args.repeat}"
    fix = "" if args.stability_fix == "none" else f"__{args.stability_fix}"
    return repeat + fix


def _smoke_suffix(args: argparse.Namespace) -> str:
    """Return a token that marks a non-CUDA local smoke test."""
    return "__localsmoke" if args.allow_non_cuda else ""


def _check_frozen_protocol_accelerator(args: argparse.Namespace, env: dict) -> None:
    """Raise if a frozen-protocol run would train on a non-CUDA accelerator.

    The 2026-08-11 local MPS / CUDA mix showed that protocol baselines and
    experimental arms must not be produced on different hardware-software stacks.
    --allow_non_cuda is the only escape hatch; it stamps __localsmoke on outputs.
    """
    if not args.frozen_protocol:
        return
    if env["accelerator"] == "cuda" or args.allow_non_cuda:
        return
    raise RuntimeError(
        "Frozen protocol training must use CUDA (Gadi V100 baseline). "
        f"Detected accelerator is '{env['accelerator']}'. "
        "Use --allow_non_cuda only for deliberate local smoke tests; "
        "those runs are stamped with __localsmoke and cannot be used as protocol data. "
        "See incident: 2026-08-11 local MPS / CUDA mix."
    )


class ValidationLossHistory(Callback):
    def __init__(self):
        self.values = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val_loss")
        if value is not None:
            self.values.append(float(value.detach().cpu()))


def _row_validation_split(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, fold: int, split_seed: int):
    pool = np.sort(np.concatenate([train_idx, val_idx]))
    groups = df.iloc[pool]["smiles_A"].astype(str).to_numpy()
    rng = np.random.default_rng(split_seed + fold)
    selected = []
    for group in np.unique(groups):
        candidates = pool[groups == group]
        count = max(1, int(round(0.1 * len(candidates))))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    new_val = np.sort(np.asarray(selected, dtype=int))
    new_train = np.setdiff1d(pool, new_val, assume_unique=True)
    return new_train, new_val


def _prediction_path(prediction_dir: Path, target: str, model_token: str, split_type: str, fold: int, args: argparse.Namespace) -> Path:
    filename = make_prediction_filename(target, model_token, split_type, fold, seed=args.seed)
    return prediction_dir / (filename[:-4] + _repeat_suffix(args) + _smoke_suffix(args) + ".npz")


def _runtime_environment() -> dict:
    cuda_available = torch.cuda.is_available()
    driver_version = None
    if cuda_available:
        try:
            driver_version = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip().splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver_version = None
    return {
        "accelerator": "cuda" if cuda_available else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "driver_version": driver_version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "trainer_deterministic": None,
        # No runner calls torch.use_deterministic_algorithms, so full determinism is not
        # in force and fixed-seed runs are not bit-reproducible.
        "deterministic_algorithms_requested": False,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _resolve_lr_config(model) -> dict:
    """Read the optimizer/LR-schedule configuration actually used by ``model``.

    HPGHierMPNN trains with a flat Adam optimizer (no scheduler), so
    max_lr/final_lr both equal init_lr. HPGMPNN (hpg_sum/hpg_frac) uses a
    Noam-like warmup/decay schedule, so warmup_epochs is also reported.
    Values are read from the constructed model, never hard-coded.
    """
    if hasattr(model, "warmup_epochs"):
        return {
            "optimizer": "Adam",
            "lr_schedule": "NoamLR",
            "init_lr": float(model.init_lr),
            "max_lr": float(model.max_lr),
            "final_lr": float(model.final_lr),
            "warmup_epochs": int(model.warmup_epochs),
        }
    init_lr = float(model.hparams.init_lr)
    return {
        "optimizer": "Adam",
        "lr_schedule": "none",
        "init_lr": init_lr,
        "max_lr": init_lr,
        "final_lr": init_lr,
    }


def _train_hier_fold(graphs, values, train_idx, val_idx, test_idx, target, split_type, fold, args, model_token="hpg_hier"):
    set_seed(args.seed if args.frozen_protocol else args.seed + fold)
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
    resolved_variant = _resolve_variant(model_token, args)
    resolved_stage2_readout = resolved_variant["stage2_readout"]
    if resolved_variant["stage2_mode"] == "octamer_sequence" and resolved_stage2_readout not in {"attention", "stoich_weighted"}:
        raise ValueError(
            f"octamer_sequence with readout '{resolved_stage2_readout}' is not implemented — "
            "only 'attention' and 'stoich_weighted' are supported. "
            "See HANDOFF §7 (arm D is now implemented)."
        )
    checkpoint_path = args.checkpoint_dir / f"ea_ip__{standard_target_token(target)}__{model_token}__{split_type}__fold{fold}__s{args.seed}{_repeat_suffix(args)}{_smoke_suffix(args)}"
    model = HPGHierMPNN(
        atom_fdim=75, bond_fdim=graphs[0].monomer_graphs[0].E.shape[1], d_h=128,
        stage1_pool=args.stage1_pool, stage2_depth=args.stage2_depth,
        stage2_edge_weight=resolved_variant["stage2_edge_weight"],
        stage2_mode=resolved_variant["stage2_mode"],
        stage2_readout=resolved_variant["stage2_readout"],
        octamer_len=args.octamer_len,
        n_random_samples=args.n_random_samples,
        junction_coupling=resolved_variant["junction_coupling"],
        n_coupling_steps=resolved_variant["n_coupling_steps"],
        use_position_embeddings=(args.octamer_position_embeddings == "on"),
        octamer_edge_features=resolved_variant["octamer_edge_features"],
    )
    model._output_transform = UnscaleTransform.from_standard_scaler(scaler)
    checkpoint = ModelCheckpoint(dirpath=str(checkpoint_path), monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    history = ValidationLossHistory()
    callbacks = [checkpoint, history]
    if args.stability_fix != "fixed_epochs":
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"))
    trainer = pl.Trainer(max_epochs=args.epochs, min_epochs=args.min_epochs, accelerator="auto", devices=1, logger=False,
                         default_root_dir=str(checkpoint_path), enable_model_summary=False, callbacks=callbacks)
    trainer.fit(model, loaders[0], loaders[1])
    compare_checkpoints = args.frozen_protocol or args.stability_fix in {"best_checkpoint", "row_val_best", "arm_c"}
    final_batches = trainer.predict(model=model, dataloaders=loaders[2])
    final_predictions = torch.cat([batch.detach().cpu() for batch in final_batches]).numpy().reshape(-1)
    if compare_checkpoints:
        if not checkpoint.best_model_path:
            raise RuntimeError(f"No best checkpoint was saved for {checkpoint_path}")
        best_batches = trainer.predict(model=model, dataloaders=loaders[2], ckpt_path=checkpoint.best_model_path)
        predictions = torch.cat([batch.detach().cpu() for batch in best_batches]).numpy().reshape(-1)
    else:
        predictions = final_predictions
    best_epoch = int(np.argmin(history.values)) + 1 if history.values else None
    return predictions, {
        "_final_y_pred": final_predictions if compare_checkpoints else None,
        "_optimizer_lr_config": _resolve_lr_config(model),
        "epochs_actually_run": len(history.values),
        "best_epoch": best_epoch,
        "best_val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        "prediction_checkpoint": checkpoint_record(checkpoint.best_model_path) if compare_checkpoints else checkpoint_record(checkpoint.last_model_path),
        "final_prediction_checkpoint": checkpoint_record(checkpoint.last_model_path),
        "validation_loss_curve": history.values,
        "n_octamer_params": (
            sum(p.numel() for p in model.octamer_encoder.parameters())
            if model.octamer_encoder is not None else 0
        ),
    }


def _train_fold(graphs, values, train_idx, val_idx, test_idx, pooling_type, target, split_type, fold, args):
    set_seed(args.seed if args.frozen_protocol else args.seed + fold)
    train_ds = _dataset(graphs, values, train_idx)
    val_ds = _dataset(graphs, values, val_idx)
    test_ds = _dataset(graphs, values, test_idx)
    scaler = train_ds.normalize_targets()
    val_ds.normalize_targets(scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=hpg_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=hpg_collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=hpg_collate_fn, num_workers=args.num_workers)
    checkpoint_path = args.checkpoint_dir / f"ea_ip__{standard_target_token(target)}__{pooling_type}__{split_type}__fold{fold}__s{args.seed}{_repeat_suffix(args)}"
    model = HPGMPNN(d_v=HPG_ATOM_FDIM, d_e=1, d_h=128, d_ffn=64, depth=6, num_heads=8,
                    dropout_mp=0.0, dropout_ffn=0.2, n_tasks=1, pooling_type=pooling_type,
                    task_type="regression")
    model._output_transform = UnscaleTransform.from_standard_scaler(scaler)
    history = ValidationLossHistory()
    checkpoint = ModelCheckpoint(dirpath=str(checkpoint_path), monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1, logger=False,
        default_root_dir=str(checkpoint_path), enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"), checkpoint, history],
    )
    trainer.fit(model, train_loader, val_loader)
    final_batches = trainer.predict(model=model, dataloaders=test_loader)
    final_predictions = torch.cat([batch.detach().cpu() for batch in final_batches]).numpy().reshape(-1)
    if args.frozen_protocol:
        best_batches = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=checkpoint.best_model_path)
        predictions = torch.cat([batch.detach().cpu() for batch in best_batches]).numpy().reshape(-1)
    else:
        predictions = final_predictions
    return predictions, {
        "_final_y_pred": final_predictions if args.frozen_protocol else None,
        "_optimizer_lr_config": _resolve_lr_config(model),
        "epochs_actually_run": len(history.values),
        "best_epoch": int(np.argmin(history.values)) + 1 if history.values else None,
        "best_val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,
        "prediction_checkpoint": checkpoint_record(checkpoint.best_model_path if args.frozen_protocol else checkpoint.last_model_path),
        "final_prediction_checkpoint": checkpoint_record(checkpoint.last_model_path),
        "validation_loss_curve": history.values,
    }


def main() -> None:
    args = parse_args()
    split_types = [item.strip() for item in args.split_types.split(",")]
    if args.frozen_protocol:
        if args.stability_fix != "none" or args.repeat is not None or args.min_epochs != 1 or args.patience != 15:
            raise ValueError("Frozen protocol requires stability_fix=none, no repeat, min_epochs=1, and patience=15")
        if args.prediction_dir.resolve() == PREDICTIONS_DIR.resolve() or args.checkpoint_dir.resolve() == CHECKPOINT_DIR.resolve():
            raise ValueError("Frozen protocol requires fresh prediction and checkpoint directories")
        _check_frozen_protocol_accelerator(args, runtime_environment())
    models = [item.strip() for item in args.models.split(",")]
    targets = TARGETS if args.targets is None else [item.strip() for item in args.targets.split(",")]
    invalid = set(models) - set(MODEL_TO_POOLING)
    if invalid:
        raise ValueError(f"Unknown models: {sorted(invalid)}")
    requested_paths = []
    for split_type in split_types:
        n_folds = 9 if split_type in {"monomer_heldout", "monomer_b_heldout", "monomer_b_heldout_clustered"} else 5
        folds = list(range(n_folds)) if args.folds is None else [int(item) for item in args.folds.split(",")]
        if any(fold < 0 or fold >= n_folds for fold in folds):
            raise ValueError(f"Invalid fold requested for {split_type}: {folds}")
        for target in targets:
            for model_token in models:
                requested_paths.extend(
                    _prediction_path(args.prediction_dir / split_subdir(split_type), target, model_token, split_type, fold, args)
                    for fold in folds
                )
    if not args.force and not args.dry_run and all(path.exists() for path in requested_paths):
        logger.info("All requested predictions already exist; exiting without loading data.")
        return
    df = pd.read_csv(DATA_PATH)
    hier_tokens = {m for m in models if MODEL_TO_POOLING.get(m) == "hpg_hier"}
    standard_graphs = _build_graphs(df) if any(MODEL_TO_POOLING.get(m) != "hpg_hier" for m in models) else None
    hier_graphs_by_token: dict = {}
    for tok in hier_tokens:
        vf = _VARIANT_FLAGS[tok]
        hier_graphs_by_token[tok] = _build_hier_graphs(
            df, args.stage2_edge,
            stage2_mode=vf["stage2_mode"],
            octamer_len=args.octamer_len,
            n_random_samples=args.n_random_samples,
            octamer_rng_seed=args.seed,
            junction_coupling=vf["junction_coupling"],
        )
    split_sets = {split_type: _build_splits(df, split_type, args.split_seed, args.b_split_metadata) for split_type in split_types}
    if args.stability_fix == "row_val_best":
        if set(split_types) != {"monomer_heldout"}:
            raise ValueError("row_val_best is restricted to monomer_heldout stability tests")
        trains, vals, tests = split_sets["monomer_heldout"]
        adjusted = [_row_validation_split(df, trains[fold], vals[fold], fold, args.split_seed) for fold in range(len(trains))]
        split_sets["monomer_heldout"] = ([item[0] for item in adjusted], [item[1] for item in adjusted], tests)
    for split_type, (trains, vals, tests) in split_sets.items():
        folds = list(range(len(trains))) if args.folds is None else [int(item) for item in args.folds.split(",")]
        for target in targets:
            values = df[target].to_numpy(dtype=np.float32)
            for model_token in models:
                for fold in folds:
                    prediction_dir = args.prediction_dir / split_subdir(split_type)
                    prediction_path = _prediction_path(prediction_dir, target, model_token, split_type, fold, args)
                    if prediction_path.exists() and not args.force:
                        logger.info("Skipping existing prediction: %s", prediction_path)
                        continue
                    if args.dry_run:
                        logger.info("Dry run: %s %s %s fold=%d", model_token, target, split_type, fold)
                        continue
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    is_hier = MODEL_TO_POOLING.get(model_token) == "hpg_hier"
                    started_at = time.monotonic()
                    if is_hier:
                        y_pred, training_summary = _train_hier_fold(
                            hier_graphs_by_token[model_token], values,
                            trains[fold], vals[fold], tests[fold],
                            target, split_type, fold, args, model_token=model_token,
                        )
                    else:
                        y_pred, training_summary = _train_fold(
                            standard_graphs, values, trains[fold], vals[fold], tests[fold],
                            MODEL_TO_POOLING[model_token], target, split_type, fold, args,
                        )
                    y_true = values[tests[fold]].astype(np.float64)
                    final_y_pred = training_summary.pop("_final_y_pred", None)
                    optimizer_lr_config = training_summary.pop("_optimizer_lr_config", {})
                    if y_pred.shape != y_true.shape:
                        raise AssertionError(f"Prediction shape {y_pred.shape} != target shape {y_true.shape}")
                    if final_y_pred is not None and final_y_pred.shape != y_true.shape:
                        raise AssertionError(f"Final prediction shape {final_y_pred.shape} != target shape {y_true.shape}")
                    # Canonical semantics:
                    #   y_pred        = predictions from the best validation-loss checkpoint
                    #   y_pred_final  = predictions from the final (patience-expired) model
                    # Analysis scripts must read y_pred as the primary result; y_pred_final
                    # is retained only for a checkpoint-gap diagnostic.
                    np.savez_compressed(
                        prediction_path, y_true=y_true, y_pred=y_pred.astype(np.float64),
                        y_pred_final=(np.asarray([], dtype=np.float64) if final_y_pred is None else final_y_pred.astype(np.float64)),
                        test_indices=tests[fold],
                        split_type=standard_split_name(split_type), model=standard_model_name(model_token),
                        target=standard_target_token(target), fold=fold, seed=args.seed,
                        repeat=(-1 if args.repeat is None else args.repeat),
                        n_train=len(trains[fold]), n_val=len(vals[fold]), n_test=len(tests[fold]),
                        prediction_scale="physical_units",
                        split_seed=args.split_seed,
                        split_indices_sha256=split_indices_sha256(trains[fold], vals[fold], tests[fold]),
                        stage2_readout=(args.stage2_readout or _VARIANT_FLAGS.get(model_token, {}).get("stage2_readout")),
                        smiles_A=df.iloc[tests[fold]]["smiles_A"].to_numpy(),
                        smiles_B=df.iloc[tests[fold]]["smiles_B"].to_numpy(), fracA=df.iloc[tests[fold]]["fracA"].to_numpy(),
                        fracB=df.iloc[tests[fold]]["fracB"].to_numpy(), poly_type=df.iloc[tests[fold]]["poly_type"].to_numpy(),
                    )
                    try:
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
                        ).strip()
                    except (OSError, subprocess.CalledProcessError):
                        git_commit = None
                    assert resolved_variant["stage2_readout"] == resolved_stage2_readout, (
                        f"resolved_variant['stage2_readout'] ({resolved_variant['stage2_readout']!r}) "
                        f"does not match resolved_stage2_readout ({resolved_stage2_readout!r})"
                    )
                    n_octamer_params = int(training_summary.pop("n_octamer_params", 0))
                    provenance = {
                        "cli_args": vars(args),
                        "resolved_config": {
                            "model": model_token, "target": target, "split_type": split_type, "fold": fold,
                            "seed": args.seed, "split_seed": args.split_seed, "epochs": args.epochs,
                            "epoch_cap": args.epochs,
                            "patience": args.patience, "min_epochs": args.min_epochs, "batch_size": args.batch_size,
                            "frozen_protocol": args.frozen_protocol,
                            "allow_non_cuda": args.allow_non_cuda,
                            "octamer_position_embeddings": args.octamer_position_embeddings,
                            "octamer_len": args.octamer_len,
                            "n_octamer_params": int(n_octamer_params),
                            "split_indices_sha256": split_indices_sha256(trains[fold], vals[fold], tests[fold]),
                            **optimizer_lr_config,
                        },
                        "resolved_variant": resolved_variant,
                        "resolved_stage2_readout": resolved_stage2_readout,
                        "git_commit": git_commit,
                        "pbs_job_id": os.environ.get("PBS_JOBID"),
                        "runtime_environment": runtime_environment(),
                        "wall_time_seconds": time.monotonic() - started_at,
                        **training_summary,
                    }
                    prediction_path.with_suffix(".config.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n")
                    logger.info("Saved: %s", prediction_path)


if __name__ == "__main__":
    main()
