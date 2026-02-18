from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, Iterable

import numpy as np

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import Featurizer, S


class MolGraphCacheFacade(Sequence[MolGraph], Generic[S]):
    """
    A :class:`MolGraphCacheFacade` provided an interface for caching
    :class:`~chemprop.data.molgraph.MolGraph`\s.

    .. note::
        This class only provides a facade for a cached dataset, but it *does not guarantee*
        whether the underlying data is truly cached.


    Parameters
    ----------
    inputs : Iterable[S]
        The inputs to be featurized.
    V_fs : Iterable[np.ndarray]
        The node features for each input.
    E_fs : Iterable[np.ndarray]
        The edge features for each input.
    featurizer : Featurizer[S, MolGraph]
        The featurizer with which to generate the
        :class:`~chemprop.data.molgraph.MolGraph`\s.
    """

    @abstractmethod
    def __init__(
        self,
        inputs: Iterable[S],
        V_fs: Iterable[np.ndarray],
        E_fs: Iterable[np.ndarray],
        featurizer: Featurizer[S, MolGraph],
    ):
        pass


class MolGraphCache(MolGraphCacheFacade):
    """
    A :class:`MolGraphCache` precomputes the corresponding
    :class:`~chemprop.data.molgraph.MolGraph`\s and caches them in memory.
    """

    def __init__(
        self,
        inputs: Iterable[S],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        featurizer: Featurizer[S, MolGraph],
    ):
        self._mgs = []
        for input_data, V_f, E_f in zip(inputs, V_fs, E_fs):
            # If input is a PolymerDatapoint, extract mol and edges
            if hasattr(input_data, 'mol') and hasattr(input_data, 'edges'):
                self._mgs.append(featurizer(input_data.mol, input_data.edges, V_f, E_f))
            # If input is a MoleculeDatapoint, extract mol only
            elif hasattr(input_data, 'mol'):
                self._mgs.append(featurizer(input_data.mol, V_f, E_f))
            else:
                self._mgs.append(featurizer(input_data, V_f, E_f))

    def __len__(self) -> int:
        return len(self._mgs)

    def __getitem__(self, index: int) -> MolGraph:
        return self._mgs[index]


class MolGraphCacheOnTheFly(MolGraphCacheFacade):
    """
    A :class:`MolGraphCacheOnTheFly` computes the corresponding
    :class:`~chemprop.data.molgraph.MolGraph`\s as they are requested.
    """

    def __init__(
        self,
        inputs: Iterable[S],
        V_fs: Iterable[np.ndarray | None],
        E_fs: Iterable[np.ndarray | None],
        featurizer: Featurizer[S, MolGraph],
    ):
        self._inputs = list(inputs)
        self._V_fs = list(V_fs)
        self._E_fs = list(E_fs)
        self._featurizer = featurizer

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> MolGraph:
        input_data = self._inputs[index]
        # If input is a PolymerDatapoint, extract mol and edges
        if hasattr(input_data, 'mol') and hasattr(input_data, 'edges'):
            return self._featurizer(input_data.mol, input_data.edges, self._V_fs[index], self._E_fs[index])
        # If input is a MoleculeDatapoint, extract mol only
        elif hasattr(input_data, 'mol'):
            return self._featurizer(input_data.mol, self._V_fs[index], self._E_fs[index])
        # Otherwise, assume it's a regular molecule
        return self._featurizer(input_data, self._V_fs[index], self._E_fs[index])
