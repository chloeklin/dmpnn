"""Edge-aware GAT layers for HPG (Hierarchical Polymer Graph).

Implements the GAT architecture from the original HPG-GAT paper in pure
PyTorch (no DGL dependency), using scatter operations for message passing.

Architecture per layer:
  1. Project node features:  h_trans = W_node(h)  → [N, num_heads * out_feats]
  2. Project edge features:  e_trans = W_edge(e)  → [E, num_heads]
  3. Attention scores:  a_ij = LeakyReLU(src·attn_src + dst·attn_dst + e_trans)
  4. Softmax over incoming edges per node
  5. Weighted message:  m_ij = alpha_ij * h_src_trans
  6. Aggregate:  h_new = mean_over_heads( sum_j m_ij )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from chemprop.data.hpg import BatchHPGMolGraph


class HPGGATLayer(nn.Module):
    """A single edge-aware Graph Attention layer matching the original HPG-GAT.

    Parameters
    ----------
    in_feats : int
        Input node feature dimension.
    out_feats : int
        Output feature dimension per head.
    edge_feats : int
        Edge feature dimension (1 for scalar features).
    num_heads : int
        Number of attention heads.
    activation : callable | None
        Activation applied after aggregation.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        edge_feats: int = 1,
        num_heads: int = 8,
        activation: callable | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats

        self.W_node = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.W_edge = nn.Linear(edge_feats, num_heads, bias=False)
        self.attention_src = nn.Parameter(torch.empty(num_heads, out_feats))
        self.attention_dst = nn.Parameter(torch.empty(num_heads, out_feats))
        self.activation = activation

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attention_src)
        nn.init.xavier_uniform_(self.attention_dst)

    def forward(self, h: Tensor, edge_index: Tensor, edge_feat: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : Tensor [N, in_feats]
        edge_index : Tensor [2, E]   (src, dst)
        edge_feat : Tensor [E, edge_feats]

        Returns
        -------
        Tensor [N, out_feats]
        """
        src, dst = edge_index  # [E]
        N = h.size(0)

        # 1) Project nodes
        h_trans = self.W_node(h)  # [N, heads * out]
        h_trans = h_trans.view(N, self.num_heads, self.out_feats)  # [N, H, F]

        # 2) Project edges
        e_trans = self.W_edge(edge_feat)  # [E, H]

        # 3) Attention scores
        src_feat = h_trans[src]  # [E, H, F]
        dst_feat = h_trans[dst]  # [E, H, F]

        attn = (
            (src_feat * self.attention_src).sum(dim=-1)   # [E, H]
            + (dst_feat * self.attention_dst).sum(dim=-1) # [E, H]
            + e_trans                                      # [E, H]
        )
        attn = F.leaky_relu(attn, negative_slope=0.2)  # [E, H]

        # 4) Per-destination softmax (edge_softmax)
        alpha = self._edge_softmax(attn, dst, N)  # [E, H]

        # 5) Weighted messages
        msg = src_feat * alpha.unsqueeze(-1)  # [E, H, F]

        # 6) Aggregate: scatter-sum over destination nodes
        out = torch.zeros(N, self.num_heads, self.out_feats,
                          dtype=msg.dtype, device=msg.device)
        dst_expand = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg)
        out.scatter_add_(0, dst_expand, msg)

        # Mean over heads (matching original: h_new.mean(dim=1))
        out = out.mean(dim=1)  # [N, F]

        if self.activation is not None:
            out = self.activation(out)

        return out

    @staticmethod
    def _edge_softmax(attn: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
        """Numerically stable softmax over edges grouped by destination node.

        Parameters
        ----------
        attn : [E, H]
        dst : [E]
        num_nodes : int

        Returns
        -------
        Tensor [E, H]
        """
        H = attn.size(1)

        # Max per destination for numerical stability
        max_val = torch.full((num_nodes, H), float("-inf"),
                             dtype=attn.dtype, device=attn.device)
        dst_exp = dst.unsqueeze(-1).expand(-1, H)
        max_val.scatter_reduce_(0, dst_exp, attn, reduce="amax", include_self=True)
        attn_shifted = attn - max_val[dst]

        exp_attn = torch.exp(attn_shifted)

        # Sum per destination
        sum_exp = torch.zeros(num_nodes, H, dtype=attn.dtype, device=attn.device)
        sum_exp.scatter_add_(0, dst_exp, exp_attn)

        return exp_attn / sum_exp[dst].clamp_min(1e-12)


class HPGMessagePassing(nn.Module):
    """Stack of HPG-GAT layers with shared weights, following the original architecture.

    Parameters
    ----------
    d_v : int
        Input node feature dimension.
    d_h : int
        Hidden dimension (= out_feats of each GAT layer).
    d_e : int
        Edge feature dimension (1 for HPG scalar features).
    depth : int
        Number of GAT layers.
    num_heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout probability applied after each GAT layer.
    """

    def __init__(
        self,
        d_v: int = 49,
        d_h: int = 128,
        d_e: int = 1,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_h = d_h

        dims = [d_v] + [d_h] * depth
        self.layers = nn.ModuleList([
            HPGGATLayer(
                in_feats=dims[i],
                out_feats=dims[i + 1],
                edge_feats=d_e,
                num_heads=num_heads,
                activation=F.leaky_relu,
            )
            for i in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.d_h

    def forward(self, bmg: BatchHPGMolGraph) -> Tensor:
        """Run GAT message passing on a batched HPG graph.

        Parameters
        ----------
        bmg : BatchHPGMolGraph

        Returns
        -------
        Tensor [N, d_h]
            Node-level hidden representations after message passing.
        """
        h = bmg.V
        for layer in self.layers:
            h = self.dropout(layer(h, bmg.edge_index, bmg.E))
        return h
