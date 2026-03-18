"""HPG (Hierarchical Polymer Graph) data structures for chemprop.

An HPGMolGraph flattens the three-level hierarchy (fragment nodes, atom nodes,
and three edge types) into a single graph — matching the original HPG-GAT paper.

Node ordering: fragment nodes first (indices 0..n_fragments-1), then atom nodes.
Edge types (all stored as 1-D scalar features):
  - fragment↔fragment: degree of polymerisation (default 1.0)
  - atom↔atom: bond order (1, 1.5, 2, 3)
  - atom→fragment: 1.0  (directed, atoms to their owning fragment)
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor


class HPGMolGraph(NamedTuple):
    """A single hierarchical polymer graph."""

    V: np.ndarray
    """Node feature matrix [n_nodes, d_v].  Fragment nodes first, then atoms."""

    E: np.ndarray
    """Scalar edge features [n_edges, 1]."""

    edge_index: np.ndarray
    """COO edge index [2, n_edges]."""

    n_fragments: int
    """Number of fragment-level nodes (first n_fragments rows of V)."""

    n_atoms: int
    """Number of atom-level nodes."""


@dataclass(repr=False, eq=False, slots=True)
class BatchHPGMolGraph:
    """Batched HPG graphs for use with PyTorch data loaders."""

    mgs: InitVar[Sequence[HPGMolGraph]]

    V: Tensor = field(init=False)
    E: Tensor = field(init=False)
    edge_index: Tensor = field(init=False)
    batch: Tensor = field(init=False)
    """Graph index for each node."""
    frag_mask: Tensor = field(init=False)
    """Boolean mask: True for fragment nodes, False for atom nodes."""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[HPGMolGraph]):
        self.__size = len(mgs)

        if self.__size == 0:
            self.V = torch.empty((0, 0), dtype=torch.float32)
            self.E = torch.empty((0, 1), dtype=torch.float32)
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.batch = torch.empty((0,), dtype=torch.long)
            self.frag_mask = torch.empty((0,), dtype=torch.bool)
            return

        Vs, Es, edge_indexes, batch_indexes, frag_masks = [], [], [], [], []
        num_nodes = 0

        for i, mg in enumerate(mgs):
            n = mg.V.shape[0]
            Vs.append(np.asarray(mg.V, dtype=np.float32))
            Es.append(np.asarray(mg.E, dtype=np.float32))
            ei = np.asarray(mg.edge_index, dtype=np.int64) + num_nodes
            edge_indexes.append(ei)
            batch_indexes.append(np.full(n, i, dtype=np.int64))

            fm = np.zeros(n, dtype=bool)
            fm[:mg.n_fragments] = True
            frag_masks.append(fm)

            num_nodes += n

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.batch = torch.from_numpy(np.concatenate(batch_indexes)).long()
        self.frag_mask = torch.from_numpy(np.concatenate(frag_masks)).bool()

    def __len__(self) -> int:
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.frag_mask = self.frag_mask.to(device)


# ---------------------------------------------------------------------------
#  Dataset + collation for training
# ---------------------------------------------------------------------------

class HPGDatapoint:
    """A single training sample for HPG."""

    __slots__ = ("mg", "y", "x_d", "weight", "lt_mask", "gt_mask")

    def __init__(
        self,
        mg: HPGMolGraph,
        y: np.ndarray | None = None,
        x_d: np.ndarray | None = None,
        weight: float = 1.0,
        lt_mask: np.ndarray | None = None,
        gt_mask: np.ndarray | None = None,
    ):
        self.mg = mg
        self.y = y
        self.x_d = x_d
        self.weight = weight
        self.lt_mask = lt_mask
        self.gt_mask = gt_mask


class HPGDataset(torch.utils.data.Dataset):
    """Minimal Dataset wrapper around a list of :class:`HPGDatapoint`."""

    def __init__(self, datapoints: Sequence[HPGDatapoint]):
        self.data = list(datapoints)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> HPGDatapoint:
        return self.data[idx]

    # ------ helpers used by train_graph.py (match MoleculeDataset API) ------

    def normalize_targets(self, scaler=None):
        """Fit-transform or transform targets, returning the fitted scaler."""
        from sklearn.preprocessing import StandardScaler

        ys = np.array([dp.y for dp in self.data if dp.y is not None], dtype=np.float64)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(ys)
        for dp in self.data:
            if dp.y is not None:
                dp.y = scaler.transform(dp.y.reshape(1, -1)).flatten().astype(np.float32)
        return scaler

    def normalize_inputs(self, key: str = "X_d", scaler=None):
        """Fit-transform or transform X_d descriptors, returning the fitted scaler."""
        from sklearn.preprocessing import StandardScaler

        if key != "X_d":
            raise ValueError(f"HPGDataset only supports key='X_d', got {key!r}")
        xds = np.array([dp.x_d for dp in self.data if dp.x_d is not None], dtype=np.float64)
        if len(xds) == 0:
            return scaler
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(xds)
        for dp in self.data:
            if dp.x_d is not None:
                dp.x_d = scaler.transform(dp.x_d.reshape(1, -1)).flatten().astype(np.float32)
        return scaler


def hpg_collate_fn(batch: Sequence[HPGDatapoint]):
    """Collate a list of :class:`HPGDatapoint` into a training batch tuple.

    Returns the 6-tuple ``(bmg, X_d, targets, weights, lt_mask, gt_mask)``
    expected by :meth:`HPGMPNN.training_step`.
    """
    mgs = [dp.mg for dp in batch]
    bmg = BatchHPGMolGraph(mgs)

    n = len(batch)
    n_tasks = batch[0].y.shape[0] if batch[0].y is not None else 1

    # Targets
    if batch[0].y is not None:
        targets = torch.tensor(np.stack([dp.y for dp in batch]), dtype=torch.float32)
    else:
        targets = torch.full((n, n_tasks), float("nan"), dtype=torch.float32)

    # Weights
    weights = torch.tensor([dp.weight for dp in batch], dtype=torch.float32).unsqueeze(1)

    # X_d
    if batch[0].x_d is not None:
        X_d = torch.tensor(np.stack([dp.x_d for dp in batch]), dtype=torch.float32)
    else:
        X_d = None

    # Masks
    if batch[0].lt_mask is not None:
        lt_mask = torch.tensor(np.stack([dp.lt_mask for dp in batch]), dtype=torch.bool)
    else:
        lt_mask = torch.zeros(n, n_tasks, dtype=torch.bool)

    if batch[0].gt_mask is not None:
        gt_mask = torch.tensor(np.stack([dp.gt_mask for dp in batch]), dtype=torch.bool)
    else:
        gt_mask = torch.zeros(n, n_tasks, dtype=torch.bool)

    return bmg, X_d, targets, weights, lt_mask, gt_mask
