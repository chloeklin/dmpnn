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
    octamer_sequences: Tensor | None = field(init=False)       # (n_polymers * K, octamer_len) long, or None
    octamer_polymer_batch: Tensor | None = field(init=False)   # (n_polymers * K,) mapping replicate→polymer
    junction_edge_index: Tensor | None = field(init=False)     # (2, total_junc_edges) global atom indices
    junction_edge_weights: Tensor | None = field(init=False)   # (total_junc_edges,)

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

        if self.graphs[0].octamer_sequences is not None:
            all_seqs = np.concatenate([g.octamer_sequences for g in self.graphs], axis=0)
            self.octamer_sequences = torch.from_numpy(all_seqs).long()
            self.octamer_polymer_batch = torch.cat([
                torch.full((g.octamer_sequences.shape[0],), polymer_idx, dtype=torch.long)
                for polymer_idx, g in enumerate(self.graphs)
            ])
        else:
            self.octamer_sequences = None
            self.octamer_polymer_batch = None

        has_junction = all(g.junction_edge_index is not None for g in self.graphs)
        if has_junction:
            junc_indices, junc_weights = [], []
            atom_offset = 0
            for graph in self.graphs:
                n_A = graph.monomer_graphs[0].V.shape[0]
                n_B = graph.monomer_graphs[1].V.shape[0]
                junc_indices.append(graph.junction_edge_index + atom_offset)
                junc_weights.append(graph.junction_edge_weights)
                atom_offset += n_A + n_B
            self.junction_edge_index = torch.from_numpy(np.hstack(junc_indices)).long()
            self.junction_edge_weights = torch.from_numpy(np.concatenate(junc_weights)).float()
        else:
            self.junction_edge_index = None
            self.junction_edge_weights = None

    def __len__(self) -> int:
        return len(self.graphs)

    def to(self, device: str | torch.device):
        self.atom_graph.to(device)
        self.monomer_batch = self.monomer_batch.to(device)
        self.polymer_batch = self.polymer_batch.to(device)
        self.monomer_fracs = self.monomer_fracs.to(device)
        self.stage2_edge_index = self.stage2_edge_index.to(device)
        self.stage2_edge_features = self.stage2_edge_features.to(device)
        if self.octamer_sequences is not None:
            self.octamer_sequences = self.octamer_sequences.to(device)
            self.octamer_polymer_batch = self.octamer_polymer_batch.to(device)
        if self.junction_edge_index is not None:
            self.junction_edge_index = self.junction_edge_index.to(device)
            self.junction_edge_weights = self.junction_edge_weights.to(device)
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
