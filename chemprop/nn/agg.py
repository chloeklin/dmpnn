from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from chemprop.nn.hparams import HasHParams
from chemprop.utils import ClassRegistry

__all__ = [
    "Aggregation",
    "AggregationRegistry",
    "MeanAggregation",
    "SumAggregation",
    "NormAggregation",
    "AttentiveAggregation",
    "IdentityAggregation",
    "WeightedMeanAggregation",
]


class Aggregation(nn.Module, HasHParams):
    """An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
    a batch of graph-level representations

    .. note::
        this class is abstract and cannot be instantiated.

    See also
    --------
    :class:`~chemprop.v2.models.modules.agg.MeanAggregation`
    :class:`~chemprop.v2.models.modules.agg.SumAggregation`
    :class:`~chemprop.v2.models.modules.agg.NormAggregation`
    """

    def __init__(self, dim: int = 0, *args, **kwargs):
        super().__init__()

        self.dim = dim
        self.hparams = {"dim": dim, "cls": self.__class__}

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        """Aggregate the graph-level representations of a batch of graphs into their respective
        global representations

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

        Parameters
        ----------
        H : Tensor
            a tensor of shape ``V x d`` containing the batched node-level representations of ``b``
            graphs
        batch : Tensor
            a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to

        Returns
        -------
        Tensor
            a tensor of shape ``b x d`` containing the graph-level representations
        """


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    r"""Average the graph-level representation:

    .. math::
        \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v
    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="mean", include_self=False
        )


@AggregationRegistry.register("identity")
class IdentityAggregation(Aggregation):
    @property
    def hparams(self): return {"cls": self.__class__}
    def forward(self, H, batch_or_bmg):
        return H

@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    r"""Sum the graph-level representation:

    .. math::
        \mathbf h = \sum_{v \in V} \mathbf h_v

    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="sum", include_self=False
        )


@AggregationRegistry.register("norm")
class NormAggregation(SumAggregation):
    r"""Sum the graph-level representation and divide by a normalization constant:

    .. math::
        \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v
    """

    def __init__(self, dim: int = 0, *args, norm: float = 100.0, **kwargs):
        super().__init__(dim, **kwargs)

        self.norm = norm
        self.hparams["norm"] = norm

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return super().forward(H, batch) / self.norm


class AttentiveAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.hparams["output_size"] = output_size
        self.W = nn.Linear(output_size, 1)

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        dim_size = batch.max().int() + 1
        attention_logits = self.W(H).exp()
        Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, batch.unsqueeze(1), attention_logits, reduce="sum", include_self=False
        )
        alphas = attention_logits / Z[batch]
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, alphas * H, reduce="sum", include_self=False
        )

@AggregationRegistry.register("identity")
class IdentityAggregation(Aggregation):
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return H  # already graph-level

@AggregationRegistry.register("weighted_mean")
class WeightedMeanAggregation(Aggregation):
    r"""
    Graph readout that computes a weighted mean over node embeddings.

    • Polymer graphs: pass the BatchPolymerMolGraph (must have `.batch` and `.atom_weights`).
      g_i = (sum_{v in graph i} w_v * h_v) / (sum_{v in graph i} w_v)

    • Non-polymer graphs: if given a batch index tensor, falls back to an unweighted mean.

    Notes
    -----
    This class accepts either:
      - (H: [num_nodes, d], bmg: object with .batch [num_nodes] and .atom_weights [num_nodes])
      - (H: [num_nodes, d], batch: LongTensor [num_nodes])
    """

    def forward(self, H: Tensor, batch_or_bmg: Any) -> Tensor:
        # --- Polymer path: BatchPolymerMolGraph with weights ---
        if hasattr(batch_or_bmg, "batch") and hasattr(batch_or_bmg, "atom_weights"):
            bmg = batch_or_bmg
            batch = bmg.batch
            w = bmg.atom_weights.to(H.dtype).to(H.device).unsqueeze(-1)  # [num_nodes, 1]
            hw = H * w                                                   # [num_nodes, d]

            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
            d = H.size(1)

            sum_hw = torch.zeros(num_graphs, d, dtype=H.dtype, device=H.device)
            sum_w  = torch.zeros(num_graphs, 1, dtype=H.dtype, device=H.device)

            sum_hw.index_add_(0, batch, hw)
            sum_w.index_add_(0, batch, w)

            return sum_hw / sum_w.clamp_min(1e-12)

        # --- Fallback: unweighted mean using a batch index tensor ---
        batch = batch_or_bmg  # expect LongTensor [num_nodes]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        d = H.size(1)

        index = batch.unsqueeze(1).expand(-1, d)

        # numerator: per-graph sum of H
        num = torch.zeros(num_graphs, d, dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index, H, reduce="sum", include_self=False
        )

        # denominator: per-graph counts
        ones = torch.ones(batch.size(0), 1, dtype=H.dtype, device=H.device)
        den = torch.zeros(num_graphs, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
            0, batch.unsqueeze(1), ones, reduce="sum", include_self=False
        ).clamp_min_(1e-12)

        return num / den