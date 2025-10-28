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


@AggregationRegistry.register("attn")
class AttentiveAggregation(Aggregation):
    """
    Global attention pooling (Li et al., 2016): 
        e_v = a^T tanh(W h_v)
        α_v = softmax_v(e_v)  (softmax taken per-graph)
        h_G = Σ_v α_v h_v
    """
    def __init__(self, dim: int = 0, *, output_size: int, hidden_size: int | None = None, bias: bool = True, **kwargs):
        super().__init__(dim, **kwargs)
        hs = hidden_size or output_size
        self.proj = nn.Linear(output_size, hs, bias=bias)   # W
        self.score = nn.Linear(hs, 1, bias=False)           # a^T
        self.hparams.update({"output_size": output_size, "hidden_size": hs, "cls": self.__class__})

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        # H: [num_nodes, d], batch: [num_nodes] with graph indices
        V, d = H.size(0), H.size(1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1

        # e = a^T tanh(W H)
        e = self.score(torch.tanh(self.proj(H))).squeeze(-1)    # [V]

        # per-graph softmax with numerical stability: subtract max per graph
        # max over nodes in each graph
        max_per_g = torch.full((num_graphs, 1), float("-inf"), dtype=H.dtype, device=H.device)
        max_per_g.scatter_reduce_(0, batch.unsqueeze(1), e.unsqueeze(1), reduce="amax", include_self=True)
        e_shift = e - max_per_g[batch, 0]

        a = torch.exp(e_shift)                                   # unnormalized attention weights [V]

        # denominator: sum over nodes in each graph
        denom = torch.zeros(num_graphs, 1, dtype=H.dtype, device=H.device)
        denom.index_add_(0, batch, a.unsqueeze(1))
        alpha = a / denom[batch, 0].clamp_min(1e-12)             # [V]

        # weighted sum per graph
        out = torch.zeros(num_graphs, d, dtype=H.dtype, device=H.device)
        out.index_add_(0, batch, alpha.unsqueeze(1) * H)         # [G, d]
        return out




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