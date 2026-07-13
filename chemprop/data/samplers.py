from itertools import chain
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


class SeededSampler(Sampler):
    """A :class`SeededSampler` is a class for iterating through a dataset in a randomly seeded
    fashion"""

    def __init__(self, N: int, seed: int):
        if seed is None:
            raise ValueError("arg 'seed' was `None`! A SeededSampler must be seeded!")

        self.idxs = np.arange(N)
        self.rg = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        self.rg.shuffle(self.idxs)

        return iter(self.idxs)

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return len(self.idxs)


class ClassBalanceSampler(Sampler):
    """A :class:`ClassBalanceSampler` samples data from a :class:`MolGraphDataset` such that
    positive and negative classes are equally sampled

    Parameters
    ----------
    dataset : MolGraphDataset
        the dataset from which to sample
    seed : int
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """

    def __init__(self, Y: np.ndarray, seed: Optional[int] = None, shuffle: bool = False):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)

        idxs = np.arange(len(Y))
        actives = Y.any(1)

        self.pos_idxs = idxs[actives]
        self.neg_idxs = idxs[~actives]

        self.length = 2 * min(len(self.pos_idxs), len(self.neg_idxs))

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        if self.shuffle:
            self.rg.shuffle(self.pos_idxs)
            self.rg.shuffle(self.neg_idxs)

        return chain(*zip(self.pos_idxs, self.neg_idxs))

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return self.length


class GroupAwareSampler(Sampler):
    """Sampler that keeps chemistry groups intact within every batch.

    Each element of a chemistry group (same monomer pair + composition,
    differing only in poly_type / architecture) is guaranteed to appear in
    the same batch.  Multiple complete groups are packed into each batch.

    Batching algorithm
    ------------------
    1. Groups are shuffled at the start of every epoch.
    2. Groups are greedily packed into a batch until the next complete group
       would overflow ``batch_size``; at that point the current batch is
       emitted and a new one is started.
    3. The last (potentially smaller) batch is always emitted unless it would
       be a singleton (size 1) with ``drop_last=True``.

    This guarantees that no group is ever split across two batches.

    Parameters
    ----------
    group_ids : array-like of int, shape (N,)
        Integer group label for each dataset sample.  Samples with the same
        label belong to the same chemistry group.
    batch_size : int
        Target batch size.  Actual batch sizes may vary because groups are
        kept intact; each batch contains ``floor(batch_size / group_size)``
        complete groups.
    seed : int | None
        Random seed for group-order shuffling.  ``None`` uses a fresh random
        state each epoch.
    drop_last : bool
        If True, drop the last batch if it would be a singleton (size 1).
        Mirrors the ``drop_last`` semantics in :func:`build_dataloader`.
    """

    def __init__(
        self,
        group_ids: "np.ndarray",
        batch_size: int,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        group_ids = np.asarray(group_ids, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.rg = np.random.default_rng(seed)

        # Build ordered list of (group_label, [sample_indices]) pairs.
        # Use stable sort so group order is deterministic before shuffling.
        unique_groups, inverse = np.unique(group_ids, return_inverse=True)
        self._groups: List[List[int]] = [[] for _ in range(len(unique_groups))]
        for sample_idx, group_label in enumerate(inverse):
            self._groups[group_label].append(sample_idx)

        # Validate: every group must fit inside a single batch.
        for g in self._groups:
            if len(g) > batch_size:
                raise ValueError(
                    f"GroupAwareSampler: a chemistry group of size {len(g)} exceeds "
                    f"batch_size={batch_size}.  Increase batch_size or merge groups."
                )

        self._n_samples = len(group_ids)

    def _make_batches(self) -> List[List[int]]:
        """Build a list of batches for one epoch (groups shuffled fresh)."""
        group_order = np.arange(len(self._groups))
        self.rg.shuffle(group_order)

        batches: List[List[int]] = []
        current: List[int] = []

        for g_idx in group_order:
            members = self._groups[g_idx]
            if current and len(current) + len(members) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(members)

        if current:
            if self.drop_last and len(current) == 1:
                pass  # drop singleton tail
            else:
                batches.append(current)

        return batches

    def __iter__(self) -> Iterator[int]:
        for batch in self._make_batches():
            yield from batch

    def __len__(self) -> int:
        return self._n_samples
