from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph.hpg_hier import TwoStageHPGGraph


@dataclass(repr=False, eq=False, slots=True)
class BatchTwoStageHPG:
    graphs: Sequence[TwoStageHPGGraph]
    atom_graph: BatchMolGraph = field(init=False)
    monomer_batch: Tensor = field(init=False)
    polymer_batch: Tensor = field(init=False)
    monomer_fracs: Tensor = field(init=False)
    stage2_edge_index: Tensor = field(init=False)
    stage2_edge_features: Tensor = field(init=False)

    def __post_init__(self):
        atom_graphs = [monomer_graph for graph in self.graphs for monomer_graph in graph.monomer_graphs]
        self.atom_graph = BatchMolGraph(atom_graphs)
        self.monomer_batch = torch.arange(len(self.graphs), dtype=torch.long).repeat_interleave(2)
        self.polymer_batch = self.monomer_batch
        self.monomer_fracs = torch.from_numpy(np.concatenate([graph.monomer_fracs for graph in self.graphs])).float()
        edge_indices, edge_features = [], []
        for polymer_idx, graph in enumerate(self.graphs):
            edge_indices.append(graph.stage2_edge_index + 2 * polymer_idx)
            edge_features.append(graph.stage2_edge_features)
        self.stage2_edge_index = torch.from_numpy(np.hstack(edge_indices)).long()
        self.stage2_edge_features = torch.from_numpy(np.concatenate(edge_features)).float()

    def __len__(self) -> int:
        return len(self.graphs)

    def to(self, device: str | torch.device):
        self.atom_graph.to(device)
        self.monomer_batch = self.monomer_batch.to(device)
        self.polymer_batch = self.polymer_batch.to(device)
        self.monomer_fracs = self.monomer_fracs.to(device)
        self.stage2_edge_index = self.stage2_edge_index.to(device)
        self.stage2_edge_features = self.stage2_edge_features.to(device)
        return self


class TwoStageHPGDatapoint:
    __slots__ = ("graph", "y", "weight")

    def __init__(self, graph: TwoStageHPGGraph, y: np.ndarray | None = None, weight: float = 1.0):
        self.graph = graph
        self.y = y
        self.weight = weight


class TwoStageHPGDataset(torch.utils.data.Dataset):
    def __init__(self, datapoints: Sequence[TwoStageHPGDatapoint]):
        self.data = list(datapoints)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> TwoStageHPGDatapoint:
        return self.data[index]

    def normalize_targets(self, scaler=None):
        from sklearn.preprocessing import StandardScaler

        values = np.asarray([datapoint.y for datapoint in self.data if datapoint.y is not None], dtype=np.float64)
        if scaler is None:
            scaler = StandardScaler().fit(values)
        for datapoint in self.data:
            if datapoint.y is not None:
                datapoint.y = scaler.transform(datapoint.y.reshape(1, -1)).reshape(-1).astype(np.float32)
        return scaler


def two_stage_hpg_collate_fn(batch: Sequence[TwoStageHPGDatapoint]):
    graph = BatchTwoStageHPG([datapoint.graph for datapoint in batch])
    targets = torch.tensor(np.stack([datapoint.y for datapoint in batch]), dtype=torch.float32)
    weights = torch.tensor([datapoint.weight for datapoint in batch], dtype=torch.float32).unsqueeze(1)
    masks = torch.zeros_like(targets, dtype=torch.bool)
    return graph, None, targets, weights, masks, masks
