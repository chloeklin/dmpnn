from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop.data.datasets import Datum, MolAtomBondDatum, PolymerDatum
from chemprop.data.molgraph import MolGraph, PolymerMolGraph

@dataclass(repr=False, eq=False, slots=True)
class BatchPolymerMolGraph:
    """A :class:`BatchPolymerMolGraph` represents a batch of individual :class:`PolymerMolGraph`\s.

    It has all the attributes of a ``PolymerMolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[PolymerMolGraph]]
    V: Tensor = field(init=False)
    E: Tensor = field(init=False)
    atom_weights: Tensor = field(init=False)
    edge_weights: Tensor = field(init=False)
    edge_index: Tensor = field(init=False)
    rev_edge_index: Tensor = field(init=False)
    batch: Tensor = field(init=False)
    degree_of_polym: Tensor = field(init=False)  

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[PolymerMolGraph]):
        self.__size = len(mgs)

        Vs = []
        Es = []
        W_atoms_list = []
        W_bonds_list = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        degree_of_polym = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            W_atoms_list.append(mg.atom_weights)
            W_bonds_list.append(mg.edge_weights)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            degree_of_polym.append(mg.degree_of_polym)

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.atom_weights = torch.from_numpy(np.concatenate(W_atoms_list)).float()
        self.edge_weights = torch.from_numpy(np.concatenate(W_bonds_list)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()
        self.degree_of_polym = torch.from_numpy(np.array(degree_of_polym, dtype=np.float32)).float()

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.atom_weights = self.atom_weights.to(device)
        self.edge_weights = self.edge_weights.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)
        self.degree_of_polym = self.degree_of_polym.to(device)


@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    
    def __post_init__(self, mgs: Sequence["MolGraph"]):
        self.__size = len(mgs)

        if self.__size == 0:
            # Create empty tensors with correct ranks
            self.V = torch.empty((0, 0), dtype=torch.float32)
            self.E = torch.empty((0, 0), dtype=torch.float32)
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.rev_edge_index = torch.empty((0,), dtype=torch.long)
            self.batch = torch.empty((0,), dtype=torch.long)
            return

        Vs, Es, edge_indexes, rev_edge_indexes, batch_indexes = [], [], [], [], []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            # Defensive casts to expected numpy dtypes
            V_np = np.asarray(mg.V, dtype=np.float32)
            E_np = np.asarray(mg.E, dtype=np.float32)
            ei_np = np.asarray(mg.edge_index, dtype=np.int64)   # shape (2, E_i)
            rei_np = np.asarray(mg.rev_edge_index, dtype=np.int64)

            # Offsets for concatenation
            ei_np = ei_np + np.array([[num_nodes], [num_nodes]], dtype=np.int64)
            rei_np = rei_np + num_edges

            Vs.append(V_np)
            Es.append(E_np)
            edge_indexes.append(ei_np)
            rev_edge_indexes.append(rei_np)
            batch_indexes.append(np.full((V_np.shape[0],), i, dtype=np.int64))

            num_nodes += V_np.shape[0]
            num_edges += ei_np.shape[1]

        # Concatenate and convert to tensors (float32/long)
        self.V = torch.from_numpy(np.concatenate(Vs, axis=0))          # (sum V_i, d_v)
        self.E = torch.from_numpy(np.concatenate(Es, axis=0))          # (sum E_i, d_e)
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes))    # (2, sum E_i)
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes))
        self.batch = torch.from_numpy(np.concatenate(batch_indexes))

        # Final dtypes
        self.V = self.V.to(torch.float32)
        self.E = self.E.to(torch.float32)
        self.edge_index = self.edge_index.to(torch.long)
        self.rev_edge_index = self.rev_edge_index.to(torch.long)
        self.batch = self.batch.to(torch.long)

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


@dataclass(repr=False, eq=False, slots=True)
class BatchCuikMolGraph:
    V: Tensor
    """the atom feature matrix"""
    E: Tensor
    """the bond feature matrix"""
    edge_index: Tensor
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self):
        self.__size = self.V.shape[0]

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph | BatchCuikMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None

class TrainingPolymerBatch(NamedTuple):
    bmg: BatchPolymerMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_batch(batch: Iterable[Datum]) -> TrainingBatch:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch(
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )

def collate_polymer_batch(batch: Iterable[PolymerDatum]) -> TrainingPolymerBatch:
    
    mgs, edges, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingPolymerBatch(
        BatchPolymerMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )

def collate_cuik_batch(batch: Iterable[Datum]) -> TrainingBatch:
    (
        atom_feats,
        bond_feats,
        edge_index,
        rev_edge_index,
        _batch,
        V_ds,
        x_ds,
        ys,
        weights,
        lt_masks,
        gt_masks,
    ) = batch
    return TrainingBatch(
        BatchCuikMolGraph(
            V=atom_feats,
            E=bond_feats,
            edge_index=edge_index,
            rev_edge_index=rev_edge_index,
            batch=_batch,
        ),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


@dataclass(repr=False, eq=False, slots=True)
class BatchMolAtomBondGraph(BatchMolGraph):
    bond_batch: Tensor = field(init=False)
    """A tensor of indices that show which :class:`MolGraph` each bond belongs to in the batch"""

    def __post_init__(self, mgs: Sequence[MolGraph]):
        # inheriting a dataclass with slots=True requires explicit arguments to super
        super(BatchMolAtomBondGraph, self).__post_init__(mgs)

        bond_batch_indexes = []
        for i, mg in enumerate(mgs):
            bond_batch_indexes.append([i] * len(mg.E))

        self.bond_batch = torch.tensor(np.concatenate(bond_batch_indexes)).long()

    def to(self, device):
        super(BatchMolAtomBondGraph, self).to(device)
        self.bond_batch = self.bond_batch.to(device)


class MolAtomBondTrainingBatch(NamedTuple):
    bmg: BatchMolAtomBondGraph
    V_d: Tensor | None
    E_d: Tensor | None
    X_d: Tensor | None
    Ys: tuple[Tensor | None, Tensor | None, Tensor | None]
    w: tuple[Tensor | None, Tensor | None, Tensor | None]
    lt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    gt_masks: tuple[Tensor | None, Tensor | None, Tensor | None]
    constraints: tuple[Tensor | None, Tensor | None]


def collate_mol_atom_bond_batch(batch: Iterable[MolAtomBondDatum]) -> MolAtomBondTrainingBatch:
    mgs, V_ds, E_ds, x_ds, yss, weights, lt_maskss, gt_maskss, constraintss = zip(*batch)

    weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    weights_tensors = []
    for ys in zip(*yss):
        if ys[0] is None:
            weights_tensors.append(None)
        elif ys[0].ndim == 1:
            weights_tensors.append(weights)
        else:
            repeats = torch.tensor([y.shape[0] for y in ys])
            weights_tensors.append(torch.repeat_interleave(weights, repeats, dim=0))

    if constraintss[0][0] is None and constraintss[0][1] is None:
        constraintss = None
    else:
        constraintss = [
            None if constraints[0] is None else torch.from_numpy(np.array(constraints)).float()
            for constraints in zip(*constraintss)
        ]

    return MolAtomBondTrainingBatch(
        BatchMolAtomBondGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if E_ds[0] is None else torch.from_numpy(np.concatenate(E_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        [None if ys[0] is None else torch.from_numpy(np.vstack(ys)).float() for ys in zip(*yss)],
        weights_tensors,
        [
            None if lt_masks[0] is None else torch.from_numpy(np.vstack(lt_masks))
            for lt_masks in zip(*lt_maskss)
        ],
        [
            None if gt_masks[0] is None else torch.from_numpy(np.vstack(gt_masks))
            for gt_masks in zip(*gt_maskss)
        ],
        constraintss,
    )


class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
