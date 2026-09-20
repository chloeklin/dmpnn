"""Infrastructure for the chemistry-disjoint HTPMD representation benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from chemprop import nn as cpnn
from chemprop.data.collate import BatchMolGraph, BatchPolymerMolGraph
from chemprop.data.datapoints import PolymerDatapoint
from chemprop.data.hpg import BatchHPGMolGraph
from chemprop.featurizers.molgraph.hpg_published import (
    HPGPublishedConnection,
    HPGPublishedMolGraphFeaturizer,
    HPGPublishedPolymerGraph,
)
from chemprop.featurizers.molgraph.molecule import (
    SimpleMoleculeMolGraphFeaturizer,
    wDMPNNPublishedPolymerMolGraphFeaturizer,
)
from chemprop.models.hpg_published import HPGPublishedPolymerEncoder
from chemprop.schedulers import build_NoamLike_LRSched


HTPMD_TARGETS = (
    "Conductivity",
    "TFSI Diffusivity",
    "Li Diffusivity",
    "Poly Diffusivity",
    "Transference Number",
)
REPRESENTATIONS = ("dmpnn", "gin", "gat", "wdmpnn-published", "hpg-published")
INFORMATION_CONDITIONS = ("native", "dop", "state")
POLYMER_REPRESENTATIONS = {"wdmpnn-published", "hpg-published"}
PROVENANCE_FIELDS = {
    "dataset", "target", "representation", "information_condition", "split_id",
    "split_seed", "init_seed", "split_reference", "row_ids", "embedding_dimension",
    "trainable_parameter_count", "predictor_specification", "external_variables",
    "wd_connection_rule", "checkpoint_selected", "epoch_selected", "test_metrics",
}
_STAR_RE = re.compile(r"\[\*:\d+\]|\[\*\]|\*")


def validate_htpmd_frame(frame: pd.DataFrame) -> None:
    required = {"smiles", "psmiles", "DoP", "Molality", *HTPMD_TARGETS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"HTPMD data is missing required columns: {sorted(missing)}")
    if len(frame) != 5950:
        raise ValueError(f"Expected 5950 HTPMD rows, found {len(frame)}")


def canonical_chemistry(psmiles: str) -> str:
    mol = Chem.MolFromSmiles(str(psmiles))
    if mol is None:
        raise ValueError(f"Cannot parse HTPMD pSMILES {psmiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def htpmd_generic_molecule(psmiles: str) -> tuple[Chem.Mol, np.ndarray]:
    mol = Chem.MolFromSmiles(str(psmiles))
    if mol is None:
        raise ValueError(f"Cannot parse HTPMD pSMILES {psmiles!r}")
    editable = Chem.RWMol(mol)
    wildcard_indices = []
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError("HTPMD attachment markers must have exactly one real-atom neighbor")
        neighbors[0].SetBoolProp("htpmd_attachment_site", True)
        wildcard_indices.append(atom.GetIdx())
    if len(wildcard_indices) != 2:
        raise ValueError("HTPMD repeat units must have exactly two attachment markers")
    for index in sorted(wildcard_indices, reverse=True):
        editable.RemoveAtom(index)
    clean = editable.GetMol()
    Chem.SanitizeMol(clean)
    attachment = np.asarray([
        [1.0 if atom.HasProp("htpmd_attachment_site") else 0.0]
        for atom in clean.GetAtoms()
    ], dtype=np.float32)
    return clean, attachment


def _split_digest(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def generate_chemistry_disjoint_splits(
    frame: pd.DataFrame,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> dict:
    groups = frame["psmiles"].map(canonical_chemistry).to_numpy()
    row_ids = np.arange(len(frame))
    splits = []
    for split_id, seed in enumerate(seeds):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=int(seed))
        train_val_pos, test_pos = next(outer.split(row_ids, groups=groups))
        train_val_ids = row_ids[train_val_pos]
        train_val_groups = groups[train_val_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=int(seed))
        train_pos, val_pos = next(inner.split(train_val_ids, groups=train_val_groups))
        train_ids = train_val_ids[train_pos]
        val_ids = train_val_ids[val_pos]
        split = {
            "split_id": split_id,
            "split_seed": int(seed),
            "train": sorted(map(int, train_ids)),
            "validation": sorted(map(int, val_ids)),
            "test": sorted(map(int, test_pos)),
        }
        split["digests"] = {
            partition: _split_digest(split[partition])
            for partition in ("train", "validation", "test")
        }
        splits.append(split)
    artifact = {
        "dataset": "htpmd",
        "n_rows": len(frame),
        "chemistry_identity": "RDKit canonical isomeric psmiles",
        "proportions": [0.8, 0.1, 0.1],
        "splits": splits,
    }
    validate_split_artifact(frame, artifact)
    return artifact


def validate_split_artifact(frame: pd.DataFrame, artifact: dict) -> None:
    if artifact["n_rows"] != len(frame):
        raise ValueError("Split artifact row count does not match HTPMD data")
    chemistry = frame["psmiles"].map(canonical_chemistry).to_numpy()
    expected = set(range(len(frame)))
    for split in artifact["splits"]:
        partitions = {name: set(split[name]) for name in ("train", "validation", "test")}
        if set.union(*partitions.values()) != expected:
            raise ValueError(f"Split {split['split_id']} does not cover every row")
        if any(partitions[left] & partitions[right] for left, right in (
            ("train", "validation"), ("train", "test"), ("validation", "test")
        )):
            raise ValueError(f"Split {split['split_id']} has overlapping row membership")
        chemistry_sets = {
            name: set(chemistry[list(indices)]) for name, indices in partitions.items()
        }
        if any(chemistry_sets[left] & chemistry_sets[right] for left, right in (
            ("train", "validation"), ("train", "test"), ("validation", "test")
        )):
            raise ValueError(f"Split {split['split_id']} has chemistry leakage")
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


def set_initialization_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def htpmd_wd_input(psmiles: str, dop: float, convention: str = "corrected") -> str:
    next_label = 1

    def replace(_: re.Match) -> str:
        nonlocal next_label
        replacement = f"[*:{next_label}]"
        next_label += 1
        return replacement

    labeled = _STAR_RE.sub(replace, str(psmiles))
    if next_label != 3:
        raise ValueError("HTPMD homopolymers must have exactly two attachment points")
    if convention == "corrected":
        weight = 1.0
    elif convention == "historical":
        weight = 0.5
    else:
        raise ValueError("wD connection convention must be 'corrected' or 'historical'")
    return f"{labeled}|1.0|<1-2:{weight:.1f}:{weight:.1f}~{float(dop):g}"


def htpmd_hpg_spec(psmiles: str, dop: float) -> HPGPublishedPolymerGraph:
    return HPGPublishedPolymerGraph.from_psmiles(
        [str(psmiles)],
        [HPGPublishedConnection(0, 0, float(dop), "R", "Q")],
    )


def external_columns(representation: str, condition: str) -> tuple[str, ...]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}")
    if condition not in INFORMATION_CONDITIONS:
        raise ValueError(f"Unknown information condition {condition!r}")
    if condition == "native":
        return ()
    if condition == "dop":
        return () if representation in POLYMER_REPRESENTATIONS else ("DoP",)
    return ("Molality",) if representation in POLYMER_REPRESENTATIONS else ("DoP", "Molality")


@dataclass(frozen=True)
class TrainStandardizer:
    columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, row_ids: Sequence[int], columns: Sequence[str]):
        columns = tuple(columns)
        if not columns:
            return cls((), (), ())
        values = frame.iloc[list(row_ids)][list(columns)].to_numpy(dtype=np.float64)
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales[scales < 1e-12] = 1.0
        return cls(columns, tuple(map(float, means)), tuple(map(float, scales)))

    def transform(self, frame: pd.DataFrame, row_ids: Sequence[int]) -> np.ndarray:
        if not self.columns:
            return np.empty((len(row_ids), 0), dtype=np.float32)
        values = frame.iloc[list(row_ids)][list(self.columns)].to_numpy(dtype=np.float64)
        return ((values - np.asarray(self.means)) / np.asarray(self.scales)).astype(np.float32)


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray):
        mean = float(np.mean(values))
        scale = float(np.std(values))
        return cls(mean, scale if scale >= 1e-12 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


class HTPMDGraphDataset(Dataset):
    def __init__(self, graphs: Sequence, external: np.ndarray, targets: np.ndarray, row_ids: Sequence[int]):
        self.graphs = list(graphs)
        self.external = np.asarray(external, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.row_ids = np.asarray(row_ids, dtype=np.int64)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index], self.external[index], self.targets[index], self.row_ids[index]


def make_collate(representation: str):
    def collate(items):
        graphs, external, targets, row_ids = zip(*items)
        if representation == "hpg-published":
            batch = BatchHPGMolGraph(graphs)
        elif representation == "wdmpnn-published":
            batch = BatchPolymerMolGraph(graphs)
        else:
            batch = BatchMolGraph(graphs)
        return (
            batch,
            torch.as_tensor(np.stack(external), dtype=torch.float32),
            torch.as_tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1),
            torch.as_tensor(row_ids, dtype=torch.long),
        )
    return collate


def featurize_htpmd(frame: pd.DataFrame, representation: str, wd_convention: str = "corrected") -> list:
    if representation in {"dmpnn", "gin", "gat"}:
        featurizer = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=1)
        graphs = []
        for psmiles in frame["psmiles"]:
            molecule, attachment = htpmd_generic_molecule(psmiles)
            graphs.append(featurizer(molecule, atom_features_extra=attachment))
        return graphs
    if representation == "wdmpnn-published":
        featurizer = wDMPNNPublishedPolymerMolGraphFeaturizer()
        graphs = []
        for row in frame.itertuples(index=False):
            datapoint = PolymerDatapoint.from_smi(htpmd_wd_input(row.psmiles, row.DoP, wd_convention))
            graphs.append(featurizer(datapoint.mol, datapoint.edges))
        return graphs
    if representation == "hpg-published":
        featurizer = HPGPublishedMolGraphFeaturizer()
        return [featurizer(htpmd_hpg_spec(row.psmiles, row.DoP)) for row in frame.itertuples(index=False)]
    raise ValueError(f"Unknown representation {representation!r}")


class NativeRepresentationEncoder(nn.Module):
    def __init__(self, representation: str, graph):
        super().__init__()
        self.representation = representation
        if representation == "dmpnn":
            self.message_passing = cpnn.BondMessagePassing(d_v=graph.V.shape[1], d_e=graph.E.shape[1])
            self.aggregation = cpnn.MeanAggregation()
            self.output_dim = self.message_passing.output_dim
        elif representation == "gin":
            self.message_passing = cpnn.GINMessagePassing(
                d_v=graph.V.shape[1], d_e=graph.E.shape[1], eps_learnable=True,
                mlp_layers=2, use_edge_features=True,
            )
            self.aggregation = cpnn.MeanAggregation()
            self.output_dim = self.message_passing.output_dim
        elif representation == "gat":
            self.message_passing = cpnn.GATMessagePassing(
                d_v=graph.V.shape[1], d_e=graph.E.shape[1], num_heads=4,
                concat_heads=True, attention_dropout=0.0, use_edge_features=True,
            )
            self.aggregation = cpnn.MeanAggregation()
            self.output_dim = self.message_passing.output_dim
        elif representation == "wdmpnn-published":
            self.message_passing = cpnn.WeightedBondMessagePassing(
                d_v=graph.V.shape[1], d_e=graph.E.shape[1],
            )
            self.aggregation = cpnn.WeightedMeanAggregation()
            self.output_dim = self.message_passing.output_dim
        elif representation == "hpg-published":
            self.message_passing = HPGPublishedPolymerEncoder()
            self.aggregation = None
            self.output_dim = self.message_passing.output_dim
        else:
            raise ValueError(f"Unknown representation {representation!r}")

    def forward(self, batch) -> Tensor:
        if self.representation == "hpg-published":
            return self.message_passing(batch)
        node_embeddings = self.message_passing(batch)
        if self.representation == "wdmpnn-published":
            graph_embeddings = self.aggregation(node_embeddings, batch)
            return graph_embeddings * batch.degree_of_polym.unsqueeze(1)
        return self.aggregation(node_embeddings, batch.batch)


class SharedPredictor(nn.Module):
    specification = "Linear(native+external,300)->ReLU->Dropout(0)->Linear(300,1)"

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(300, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class HTPMDRepresentationModel(nn.Module):
    def __init__(self, representation: str, example_graph, external_dim: int):
        super().__init__()
        self.encoder = NativeRepresentationEncoder(representation, example_graph)
        self.embedding_dimension = self.encoder.output_dim
        self.external_dim = external_dim
        self.predictor = SharedPredictor(self.embedding_dimension + external_dim)

    def forward(self, graph_batch, external: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.encoder(graph_batch)
        if external.shape[1] != self.external_dim:
            raise ValueError("External scalar width does not match benchmark condition")
        predictor_input = torch.cat((embedding, external), dim=1) if self.external_dim else embedding
        return self.predictor(predictor_input), embedding


def _move_graph(batch, device: torch.device):
    batch.to(device)
    return batch


def save_best_checkpoint(path: Path, model: nn.Module, epoch: int, validation_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
    }, path)


def reload_best_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _predict(model, loader, device, max_batches: int | None = None):
    model.eval()
    predictions, targets, row_ids, embeddings = [], [], [], []
    with torch.no_grad():
        for batch_index, (graphs, external, target, rows) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            graphs = _move_graph(graphs, device)
            prediction, embedding = model(graphs, external.to(device))
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
    graphs: Sequence,
    representation: str,
    condition: str,
    target: str,
    split: dict,
    split_reference: Path,
    init_seed: int,
    output_dir: Path,
    wd_convention: str = "corrected",
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 30,
    warmup_epochs: int = 2,
    max_batches: int | None = None,
    device: str = "cpu",
) -> dict:
    if target not in HTPMD_TARGETS:
        raise ValueError(f"Target {target!r} is not an HTPMD benchmark target")
    set_initialization_seed(init_seed)
    torch_device = torch.device(device)
    columns = external_columns(representation, condition)
    external_scaler = TrainStandardizer.fit(frame, split["train"], columns)
    target_scaler = TargetStandardizer.fit(frame.iloc[split["train"]][target].to_numpy(dtype=np.float64))

    datasets = {}
    loaders = {}
    generator = torch.Generator().manual_seed(init_seed)
    for partition in ("train", "validation", "test"):
        row_ids = split[partition]
        external = external_scaler.transform(frame, row_ids)
        targets = target_scaler.transform(frame.iloc[row_ids][target].to_numpy(dtype=np.float64))
        datasets[partition] = HTPMDGraphDataset(
            [graphs[index] for index in row_ids], external, targets, row_ids,
        )
        loaders[partition] = DataLoader(
            datasets[partition], batch_size=batch_size, shuffle=partition == "train",
            collate_fn=make_collate(representation), generator=generator if partition == "train" else None,
            num_workers=0,
        )

    model = HTPMDRepresentationModel(representation, graphs[0], len(columns)).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / batch_size))
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    cooldown_steps = max(1, (max_epochs - warmup_epochs) * steps_per_epoch)
    scheduler = build_NoamLike_LRSched(
        optimizer, warmup_steps, cooldown_steps, 1e-4, 1e-3, 1e-4,
    )

    run_name = f"{representation}__{condition}__{target.replace(' ', '_')}__split{split['split_id']}__init{init_seed}"
    run_dir = output_dir / run_name
    checkpoint_path = run_dir / "best.pt"
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_index, (graph_batch, external, targets, _) in enumerate(loaders["train"]):
            if max_batches is not None and batch_index >= max_batches:
                break
            graph_batch = _move_graph(graph_batch, torch_device)
            optimizer.zero_grad()
            predictions, _ = model(graph_batch, external.to(torch_device))
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
        "dataset": "htpmd",
        "target": target,
        "representation": representation,
        "information_condition": condition,
        "split_id": split["split_id"],
        "split_seed": split["split_seed"],
        "init_seed": int(init_seed),
        "split_reference": str(split_reference),
        "row_ids": {
            partition: split[partition] for partition in ("train", "validation", "test")
        },
        "split_digests": split["digests"],
        "embedding_dimension": model.embedding_dimension,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "predictor_specification": SharedPredictor.specification,
        "external_variables": list(columns),
        "external_standardizer": asdict(external_scaler),
        "target_standardizer": asdict(target_scaler),
        "wd_connection_rule": (
            f"<1-2:{1.0 if wd_convention == 'corrected' else 0.5:.1f}:"
            f"{1.0 if wd_convention == 'corrected' else 0.5:.1f}~DoP"
            if representation == "wdmpnn-published" else None
        ),
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
