"""Infrastructure for the chemistry-disjoint EA/IP representation benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from chemprop.data.collate import BatchMolGraph, BatchPolymerMolGraph
from chemprop.data.datapoints import PolymerDatapoint
from chemprop.featurizers.molgraph.molecule import (
    SimpleMoleculeMolGraphFeaturizer,
    wDMPNNPublishedPolymerMolGraphFeaturizer,
)
from chemprop.schedulers import build_NoamLike_LRSched
from evaluation.htpmd_benchmark import (
    NativeRepresentationEncoder,
    SharedPredictor,
    TargetStandardizer,
    reload_best_checkpoint,
    save_best_checkpoint,
    set_initialization_seed,
)

EA_IP_TARGETS = ("EA vs SHE (eV)", "IP vs SHE (eV)")
EA_IP_MODELS = ("dmpnn", "gin", "gat", "wd-published", "wd-published-xn8")
GENERIC_MODELS = {"dmpnn", "gin", "gat"}
PUBLISHED_INPUT_COLUMN = "poly_chemprop_input"
LOCAL_INPUT_COLUMN = "WDMPNN_Input"
PROVENANCE_FIELDS = {
    "dataset", "target", "model", "split_id", "split_seed", "init_seed",
    "split_reference", "row_ids", "embedding_dimension", "trainable_parameter_count",
    "predictor_specification", "input_semantics", "checkpoint_selected",
    "epoch_selected", "test_metrics",
}


def validate_ea_ip_frame(frame: pd.DataFrame) -> None:
    required = {
        "smiles_A", "smiles_B", "fracA", "fracB", "poly_type",
        LOCAL_INPUT_COLUMN, *EA_IP_TARGETS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EA/IP data is missing required columns: {sorted(missing)}")
    if len(frame) != 42966:
        raise ValueError(f"Expected 42966 EA/IP rows, found {len(frame)}")
    fractions = frame[["fracA", "fracB"]].to_numpy(dtype=np.float64)
    if not np.isfinite(fractions).all() or not np.allclose(fractions.sum(axis=1), 1.0):
        raise ValueError("EA/IP constituent fractions must be finite and sum to one")


def certify_published_parity(local: pd.DataFrame, published: pd.DataFrame, atol: float = 1e-8) -> dict:
    validate_ea_ip_frame(local)
    required = {PUBLISHED_INPUT_COLUMN, *EA_IP_TARGETS}
    missing = required - set(published.columns)
    if missing:
        raise ValueError(f"Published EA/IP data is missing columns: {sorted(missing)}")
    if len(local) != len(published):
        raise ValueError("Local and published EA/IP row counts differ")
    input_equal = local[LOCAL_INPUT_COLUMN].astype(str).eq(published[PUBLISHED_INPUT_COLUMN].astype(str))
    target_differences = {
        target: np.abs(local[target].to_numpy(dtype=np.float64) - published[target].to_numpy(dtype=np.float64))
        for target in EA_IP_TARGETS
    }
    result = {
        "rows_compared": len(local),
        "input_matches": int(input_equal.sum()),
        "input_mismatches": int((~input_equal).sum()),
        "target_max_abs_difference": {
            target: float(values.max()) for target, values in target_differences.items()
        },
        "targets_within_tolerance": {
            target: bool(np.all(values <= atol)) for target, values in target_differences.items()
        },
        "row_order_aligned": bool(input_equal.all()),
    }
    if not input_equal.all() or not all(result["targets_within_tolerance"].values()):
        raise ValueError(f"Local EA/IP data does not match the published input: {result}")
    return result


def canonical_constituent(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Cannot parse EA/IP constituent SMILES {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def chemistry_pair_id(smiles_a: str, smiles_b: str) -> str:
    return "|||".join(sorted((canonical_constituent(smiles_a), canonical_constituent(smiles_b))))


def chemistry_pair_ids(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([
        chemistry_pair_id(smiles_a, smiles_b)
        for smiles_a, smiles_b in zip(frame["smiles_A"], frame["smiles_B"])
    ])


def _split_digest(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def generate_chemistry_disjoint_splits(
    frame: pd.DataFrame,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> dict:
    validate_ea_ip_frame(frame)
    groups = chemistry_pair_ids(frame)
    row_ids = np.arange(len(frame))
    splits = []
    for split_id, seed in enumerate(seeds):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=int(seed))
        train_val_pos, test_pos = next(outer.split(row_ids, groups=groups))
        train_val_ids = row_ids[train_val_pos]
        train_val_groups = groups[train_val_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=int(seed))
        train_pos, validation_pos = next(inner.split(train_val_ids, groups=train_val_groups))
        split = {
            "split_id": split_id,
            "split_seed": int(seed),
            "train": sorted(map(int, train_val_ids[train_pos])),
            "validation": sorted(map(int, train_val_ids[validation_pos])),
            "test": sorted(map(int, row_ids[test_pos])),
        }
        split["digests"] = {
            partition: _split_digest(split[partition])
            for partition in ("train", "validation", "test")
        }
        splits.append(split)
    artifact = {
        "dataset": "ea_ip",
        "n_rows": len(frame),
        "n_chemistry_groups": int(len(set(groups))),
        "chemistry_identity": "lexically sorted pair of RDKit canonical isomeric constituent SMILES",
        "proportions": [0.8, 0.1, 0.1],
        "splits": splits,
    }
    validate_split_artifact(frame, artifact)
    return artifact


def validate_split_artifact(frame: pd.DataFrame, artifact: dict) -> None:
    if artifact["n_rows"] != len(frame):
        raise ValueError("Split artifact row count does not match EA/IP data")
    groups = chemistry_pair_ids(frame)
    if artifact["n_chemistry_groups"] != len(set(groups)):
        raise ValueError("Split artifact chemistry-group count does not match EA/IP data")
    expected = set(range(len(frame)))
    for split in artifact["splits"]:
        partitions = {name: set(split[name]) for name in ("train", "validation", "test")}
        if set.union(*partitions.values()) != expected:
            raise ValueError(f"Split {split['split_id']} does not cover every row")
        if any(partitions[left] & partitions[right] for left, right in (
            ("train", "validation"), ("train", "test"), ("validation", "test")
        )):
            raise ValueError(f"Split {split['split_id']} has overlapping row membership")
        group_sets = {name: set(groups[list(indices)]) for name, indices in partitions.items()}
        if any(group_sets[left] & group_sets[right] for left, right in (
            ("train", "validation"), ("train", "test"), ("validation", "test")
        )):
            raise ValueError(f"Split {split['split_id']} has chemistry-pair leakage")
        for name in partitions:
            if split["digests"][name] != _split_digest(split[name]):
                raise ValueError(f"Split {split['split_id']} {name} digest is invalid")


def save_split_artifact(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def load_split_artifact(frame: pd.DataFrame, path: Path) -> dict:
    artifact = json.loads(path.read_text())
    validate_split_artifact(frame, artifact)
    return artifact


def add_xn8(published_input: str) -> str:
    value = str(published_input)
    if "~" in value:
        raise ValueError("Published EA/IP input must not already contain Xn")
    if "<" not in value:
        raise ValueError("Published EA/IP input must contain connection rules")
    return value + "~8"


def remove_xn8(xn8_input: str) -> str:
    value = str(xn8_input)
    if not value.endswith("~8") or value.count("~") != 1:
        raise ValueError("EA/IP Xn8 input must contain exactly one final ~8 suffix")
    return value[:-2]


def mutation_safe_published_graph(polymer_input: str):
    datapoint = PolymerDatapoint.from_smi(str(polymer_input))
    featurizer = wDMPNNPublishedPolymerMolGraphFeaturizer()
    return featurizer(datapoint.mol, list(datapoint.edges))


def featurize_ea_ip(frame: pd.DataFrame, model: str):
    if model in GENERIC_MODELS:
        featurizer = SimpleMoleculeMolGraphFeaturizer()
        graphs_a = [featurizer(Chem.MolFromSmiles(smiles)) for smiles in frame["smiles_A"]]
        graphs_b = [featurizer(Chem.MolFromSmiles(smiles)) for smiles in frame["smiles_B"]]
        fractions = frame[["fracA", "fracB"]].to_numpy(dtype=np.float32)
        return graphs_a, graphs_b, fractions
    if model == "wd-published":
        return [mutation_safe_published_graph(value) for value in frame[LOCAL_INPUT_COLUMN]]
    if model == "wd-published-xn8":
        return [mutation_safe_published_graph(add_xn8(value)) for value in frame[LOCAL_INPUT_COLUMN]]
    raise ValueError(f"Unknown EA/IP model {model!r}")


class EAIPDataset(Dataset):
    def __init__(self, inputs, targets: np.ndarray, row_ids: Sequence[int]):
        self.inputs = inputs
        self.targets = np.asarray(targets, dtype=np.float32)
        self.row_ids = np.asarray(row_ids, dtype=np.int64)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, index):
        row_id = int(self.row_ids[index])
        if isinstance(self.inputs, tuple):
            graphs_a, graphs_b, fractions = self.inputs
            value = (graphs_a[row_id], graphs_b[row_id], fractions[row_id])
        else:
            value = self.inputs[row_id]
        return value, self.targets[index], row_id


def make_collate(model: str):
    def collate(items):
        inputs, targets, row_ids = zip(*items)
        if model in GENERIC_MODELS:
            graphs_a, graphs_b, fractions = zip(*inputs)
            batch = (
                BatchMolGraph(graphs_a),
                BatchMolGraph(graphs_b),
                torch.as_tensor(np.stack(fractions), dtype=torch.float32),
            )
        else:
            batch = BatchPolymerMolGraph(inputs)
        return (
            batch,
            torch.as_tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1),
            torch.as_tensor(row_ids, dtype=torch.long),
        )
    return collate


class EAIPRepresentationModel(nn.Module):
    def __init__(self, model: str, example_graph):
        super().__init__()
        self.model = model
        encoder_name = model if model in GENERIC_MODELS else "wdmpnn-published"
        self.encoder = NativeRepresentationEncoder(encoder_name, example_graph)
        self.embedding_dimension = self.encoder.output_dim
        self.predictor = SharedPredictor(self.embedding_dimension)

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        if self.model in GENERIC_MODELS:
            batch_a, batch_b, fractions = batch
            embedding_a = self.encoder(batch_a)
            embedding_b = self.encoder(batch_b)
            embedding = fractions[:, :1] * embedding_a + fractions[:, 1:] * embedding_b
        else:
            embedding = self.encoder(batch)
        return self.predictor(embedding), embedding


def _move_batch(batch, device: torch.device):
    if isinstance(batch, tuple):
        graph_a, graph_b, fractions = batch
        graph_a.to(device)
        graph_b.to(device)
        return graph_a, graph_b, fractions.to(device)
    batch.to(device)
    return batch


def _predict(model, loader, device, max_batches: int | None = None):
    model.eval()
    predictions, targets, row_ids, embeddings = [], [], [], []
    with torch.no_grad():
        for batch_index, (inputs, target, rows) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            prediction, embedding = model(_move_batch(inputs, device))
            predictions.append(prediction.cpu())
            targets.append(target)
            row_ids.append(rows)
            embeddings.append(embedding.cpu())
    return (
        torch.cat(predictions).numpy(), torch.cat(targets).numpy(),
        torch.cat(row_ids).numpy(), torch.cat(embeddings).numpy(),
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    residual = y_true - y_pred
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.sum(residual ** 2) / ss_total) if ss_total > 0 else float("nan"),
    }


def fit_one_run(
    frame: pd.DataFrame,
    inputs,
    model_name: str,
    target: str,
    split: dict,
    split_reference: Path,
    init_seed: int,
    output_dir: Path,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 30,
    warmup_epochs: int = 2,
    max_batches: int | None = None,
    device: str = "cpu",
) -> dict:
    if target not in EA_IP_TARGETS:
        raise ValueError(f"Target {target!r} is not an EA/IP benchmark target")
    set_initialization_seed(init_seed)
    torch_device = torch.device(device)
    target_scaler = TargetStandardizer.fit(frame.iloc[split["train"]][target].to_numpy(dtype=np.float64))
    datasets = {}
    loaders = {}
    generator = torch.Generator().manual_seed(init_seed)
    for partition in ("train", "validation", "test"):
        row_ids = split[partition]
        targets = target_scaler.transform(frame.iloc[row_ids][target].to_numpy(dtype=np.float64))
        datasets[partition] = EAIPDataset(inputs, targets, row_ids)
        loaders[partition] = DataLoader(
            datasets[partition], batch_size=batch_size, shuffle=partition == "train",
            collate_fn=make_collate(model_name), generator=generator if partition == "train" else None,
            num_workers=0,
        )
    example_graph = inputs[0][0] if isinstance(inputs, tuple) else inputs[0]
    model = EAIPRepresentationModel(model_name, example_graph).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / batch_size))
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    scheduler = build_NoamLike_LRSched(
        optimizer,
        max(1, warmup_epochs * steps_per_epoch),
        max(1, (max_epochs - warmup_epochs) * steps_per_epoch),
        1e-4, 1e-3, 1e-4,
    )
    run_name = f"{model_name}__{target.replace(' ', '_')}__split{split['split_id']}__init{init_seed}"
    run_dir = output_dir / run_name
    checkpoint_path = run_dir / "best.pt"
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_index, (batch, targets, _) in enumerate(loaders["train"]):
            if max_batches is not None and batch_index >= max_batches:
                break
            optimizer.zero_grad()
            predictions, _ = model(_move_batch(batch, torch_device))
            loss = torch.mean((predictions - targets.to(torch_device)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
        val_pred, val_true, _, _ = _predict(model, loaders["validation"], torch_device, max_batches)
        validation_loss = float(np.mean((val_pred - val_true) ** 2))
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            save_best_checkpoint(checkpoint_path, model, epoch, validation_loss)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    selected = reload_best_checkpoint(model, checkpoint_path, torch_device)
    test_pred_scaled, test_true_scaled, test_rows, test_embeddings = _predict(
        model, loaders["test"], torch_device, max_batches,
    )
    test_predictions = target_scaler.inverse(test_pred_scaled)
    test_targets = target_scaler.inverse(test_true_scaled)
    test_metrics = _metrics(test_targets, test_predictions)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "predictions.npz", row_ids=test_rows, y_true=test_targets,
        y_pred=test_predictions, embeddings=test_embeddings,
    )
    provenance = {
        "dataset": "ea_ip",
        "target": target,
        "model": model_name,
        "split_id": split["split_id"],
        "split_seed": split["split_seed"],
        "init_seed": int(init_seed),
        "split_reference": str(split_reference),
        "row_ids": {partition: split[partition] for partition in ("train", "validation", "test")},
        "split_digests": split["digests"],
        "embedding_dimension": model.embedding_dimension,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "predictor_specification": SharedPredictor.specification,
        "input_semantics": (
            "shared_encoder_fraction_weighted_mix"
            if model_name in GENERIC_MODELS
            else "published_wd_input_xn8" if model_name == "wd-published-xn8"
            else "published_wd_input_default_xn1"
        ),
        "target_standardizer": {
            "mean": target_scaler.mean,
            "scale": target_scaler.scale,
        },
        "checkpoint_selected": str(checkpoint_path),
        "epoch_selected": int(selected["epoch"]),
        "validation_loss_selected": float(selected["validation_loss"]),
        "test_metrics": test_metrics,
        "protocol": {
            "optimizer": "Adam", "initial_lr": 1e-4, "maximum_lr": 1e-3,
            "final_lr": 1e-4, "warmup_epochs": warmup_epochs,
            "batch_size": batch_size, "maximum_epochs": max_epochs,
            "early_stopping_patience": patience, "selection_metric": "validation_mse",
            "max_batches_per_partition": max_batches,
        },
    }
    missing = PROVENANCE_FIELDS - provenance.keys()
    if missing:
        raise AssertionError(f"Missing provenance fields: {sorted(missing)}")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
