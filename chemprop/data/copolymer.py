"""Copolymer dataset and collate: two separate MoleculeDatapoints per sample."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import Mol
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from torch import Tensor

from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.base import Featurizer
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.molgraph.cache import MolGraphCache, MolGraphCacheOnTheFly


# --------------------------------------------------------------------------- #
#  Datum                                                                       #
# --------------------------------------------------------------------------- #
class CopolymerDatum(NamedTuple):
    """One training sample for a copolymer: two mol-graphs + fractions + meta."""
    mg_A: MolGraph
    mg_B: MolGraph
    fracA: float
    fracB: float
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


# --------------------------------------------------------------------------- #
#  Training batch                                                              #
# --------------------------------------------------------------------------- #
class CopolymerTrainingBatch(NamedTuple):
    bmg_A: BatchMolGraph
    bmg_B: BatchMolGraph
    fracA: Tensor
    fracB: Tensor
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


# --------------------------------------------------------------------------- #
#  Collate function                                                            #
# --------------------------------------------------------------------------- #
def collate_copolymer_batch(batch: Sequence[CopolymerDatum]) -> CopolymerTrainingBatch:
    mg_As, mg_Bs, fracAs, fracBs, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return CopolymerTrainingBatch(
        bmg_A=BatchMolGraph(mg_As),
        bmg_B=BatchMolGraph(mg_Bs),
        fracA=torch.tensor(fracAs, dtype=torch.float),
        fracB=torch.tensor(fracBs, dtype=torch.float),
        X_d=None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        Y=None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        w=torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        lt_mask=None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        gt_mask=None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class CopolymerDataset(Dataset):
    """Dataset for copolymer samples: each sample has two MoleculeDatapoints + fractions.

    Parameters
    ----------
    data_A : list[MoleculeDatapoint]
        Monomer A datapoints (targets & x_d stored here).
    data_B : list[MoleculeDatapoint]
        Monomer B datapoints (targets ignored, only mol is used).
    fracA : np.ndarray
        Composition fraction for monomer A, shape ``(N,)``.
    fracB : np.ndarray
        Composition fraction for monomer B, shape ``(N,)``.
    featurizer : Featurizer
        Mol→MolGraph featurizer (shared for both monomers).
    """

    data_A: list[MoleculeDatapoint]
    data_B: list[MoleculeDatapoint]
    fracA: np.ndarray
    fracB: np.ndarray
    featurizer: Featurizer[Mol, MolGraph] = field(
        default_factory=SimpleMoleculeMolGraphFeaturizer
    )

    def __post_init__(self):
        assert len(self.data_A) == len(self.data_B) == len(self.fracA)
        self.reset()
        self.cache = False

    def __len__(self) -> int:
        return len(self.data_A)

    def __getitem__(self, idx: int) -> CopolymerDatum:
        dA = self.data_A[idx]
        dB = self.data_B[idx]
        mg_A = self._cache_A[idx]
        mg_B = self._cache_B[idx]
        return CopolymerDatum(
            mg_A=mg_A,
            mg_B=mg_B,
            fracA=float(self.fracA[idx]),
            fracB=float(self.fracB[idx]),
            x_d=self.X_d[idx],
            y=self.Y[idx],
            weight=dA.weight,
            lt_mask=dA.lt_mask,
            gt_mask=dA.gt_mask,
        )

    # ---- target / descriptor arrays (mirror MoleculeDataset API) ----

    @cached_property
    def _Y(self) -> np.ndarray:
        return np.array([d.y for d in self.data_A], float)

    @property
    def Y(self) -> np.ndarray:
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        self.__Y = np.array(Y, float)

    @cached_property
    def _X_d(self) -> np.ndarray:
        return np.array([d.x_d for d in self.data_A])

    @property
    def X_d(self) -> np.ndarray:
        return self.__X_d

    @X_d.setter
    def X_d(self, X_d: ArrayLike):
        self.__X_d = np.array(X_d)

    @property
    def d_xd(self) -> int:
        return 0 if self.X_d[0] is None else self.X_d.shape[1]

    @property
    def data(self):
        """Alias so that external code accessing .data works (returns A side)."""
        return self.data_A

    @data.setter
    def data(self, value):
        self.data_A = value

    @property
    def names(self) -> list[str]:
        return [f"{dA.name}|||{dB.name}" for dA, dB in zip(self.data_A, self.data_B)]

    # ---- normalization (same interface as MoleculeDataset) ----

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        if scaler is None:
            scaler = StandardScaler().fit(self._Y)
        self.Y = scaler.transform(self._Y)
        return scaler

    def normalize_inputs(
        self, key: str = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        if key != "X_d":
            raise ValueError(f"CopolymerDataset only supports key='X_d', got '{key}'")
        X = self.X_d if self.X_d[0] is not None else None
        if X is None:
            return scaler
        if scaler is None:
            scaler = StandardScaler().fit(X)
        self.X_d = scaler.transform(X)
        return scaler

    def reset(self):
        self.__Y = self._Y
        self.__X_d = self._X_d

    # ---- mol-graph caching ----

    @property
    def cache(self) -> bool:
        return self.__cache

    @cache.setter
    def cache(self, cache: bool = False):
        self.__cache = cache
        self._init_cache()

    def _init_cache(self):
        CacheClass = MolGraphCache if self.__cache else MolGraphCacheOnTheFly
        mols_A = [d.mol for d in self.data_A]
        mols_B = [d.mol for d in self.data_B]
        V_fs_A = [d.V_f for d in self.data_A]
        E_fs_A = [d.E_f for d in self.data_A]
        V_fs_B = [d.V_f for d in self.data_B]
        E_fs_B = [d.E_f for d in self.data_B]
        self._cache_A = CacheClass(mols_A, V_fs_A, E_fs_A, self.featurizer)
        self._cache_B = CacheClass(mols_B, V_fs_B, E_fs_B, self.featurizer)
