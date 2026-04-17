"""Edge-aware GAT layers for HPG (Hierarchical Polymer Graph).

Implements the GAT architecture from the original HPG-GAT paper in pure
PyTorch (no DGL dependency), using scatter operations for message passing.

Architecture per layer (HPGGATLayer):
  1. Project node features:  h_trans = W_node(h)  → [N, num_heads * out_feats]
  2. Project edge features:  e_trans = W_edge(e)  → [E, num_heads]
  3. Attention scores:  a_ij = LeakyReLU(src·attn_src + dst·attn_dst + e_trans)
  4. Softmax over incoming edges per node
  5. Weighted message:  m_ij = alpha_ij * h_src_trans   ← edge enters ONLY attention
  6. Aggregate:  h_new = mean_over_heads( sum_j m_ij )

HPGRelMsgGATLayer (Phase 2A variant):
  Same as above, but step 5 becomes:
  5. Weighted message:  m_ij = alpha_ij * W_msg([h_src, e_ij])
                                           ^^^^^^^^^^^^^^^^^^^^^^
                                           edge NOW enters message CONTENT
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


# ---------------------------------------------------------------------------
#  Phase 2A: HPG_relMsg — relation-message-aware GAT layer
# ---------------------------------------------------------------------------

class HPGRelMsgGATLayer(nn.Module):
    """GAT layer where edge features enter the propagated **message content**.

    ``HPG_relMsg`` tests whether HPG improves when edge / relation information
    is incorporated into propagated message content rather than being used only
    to modulate attention weights.

    Compared with :class:`HPGGATLayer`:
    - Attention mechanism: **unchanged** (fair comparison).
    - Message step changes from::

          m_ij = alpha_ij * W_node(h_src)               # original

      to::

          m_ij = alpha_ij * W_msg( cat(h_src, e_ij) )   # HPG_relMsg

    where ``W_msg : Linear(in_feats + edge_feats, out_feats * num_heads)``.

    Design assumption (Phase 2A)
    ----------------------------
    A **single shared** ``W_msg`` is used for all edge types.  The 1-D scalar
    edge feature already implicitly encodes relation type (bond-order ∈
    {1.0, 1.5, 2.0, 3.0} for atom–atom bonds; 1.0 for atom→fragment; degree
    scalar for fragment–fragment), so the MLP can learn to differentiate.
    Per-relation routing (one MLP per relation type) is left to Phase 2B and
    would require storing an explicit edge-type tensor in BatchHPGMolGraph.

    Parameters
    ----------
    in_feats : int
        Input node feature dimension.
    out_feats : int
        Output feature dimension per head.
    edge_feats : int
        Edge feature dimension (1 for standard HPG scalar features).
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
        self.activation = activation

        # ── Attention weights (identical to HPGGATLayer) ──
        self.W_node = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.W_edge = nn.Linear(edge_feats, num_heads, bias=False)
        self.attention_src = nn.Parameter(torch.empty(num_heads, out_feats))
        self.attention_dst = nn.Parameter(torch.empty(num_heads, out_feats))

        # ── Message content MLP (new in HPG_relMsg) ──
        # W_msg : (in_feats + edge_feats) → out_feats * num_heads
        # Raw source embedding (before W_node) is concatenated with the edge
        # feature so the message MLP controls its own joint projection.
        self.W_msg = nn.Linear(in_feats + edge_feats, out_feats * num_heads, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attention_src)
        nn.init.xavier_uniform_(self.attention_dst)
        nn.init.xavier_uniform_(self.W_msg.weight)

    def forward(self, h: Tensor, edge_index: Tensor, edge_feat: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : Tensor [N, in_feats]
        edge_index : Tensor [2, E]
        edge_feat : Tensor [E, edge_feats]

        Returns
        -------
        Tensor [N, out_feats]
        """
        src, dst = edge_index
        N = h.size(0)
        E = src.size(0)

        # ── 1) Attention: same as HPGGATLayer ──
        h_trans = self.W_node(h).view(N, self.num_heads, self.out_feats)  # [N, H, F]
        e_trans = self.W_edge(edge_feat)                                   # [E, H]

        src_feat = h_trans[src]   # [E, H, F]
        dst_feat = h_trans[dst]   # [E, H, F]

        attn = (
            (src_feat * self.attention_src).sum(dim=-1)   # [E, H]
            + (dst_feat * self.attention_dst).sum(dim=-1) # [E, H]
            + e_trans                                      # [E, H]
        )
        attn  = F.leaky_relu(attn, negative_slope=0.2)
        alpha = HPGGATLayer._edge_softmax(attn, dst, N)   # [E, H]

        # ── 2) Message content: cat(h_src, e_ij) → W_msg → [E, H, F] ──
        h_src_raw = h[src]                               # [E, in_feats]
        msg_input = torch.cat([h_src_raw, edge_feat], dim=-1)  # [E, in_feats + edge_feats]
        msg = self.W_msg(msg_input).view(E, self.num_heads, self.out_feats)  # [E, H, F]

        # ── 3) Attention-weighted message ──
        msg = msg * alpha.unsqueeze(-1)                  # [E, H, F]

        # ── 4) Aggregate: scatter-sum → mean over heads ──
        out = torch.zeros(N, self.num_heads, self.out_feats,
                          dtype=msg.dtype, device=msg.device)
        dst_expand = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg)
        out.scatter_add_(0, dst_expand, msg)
        out = out.mean(dim=1)                            # [N, F]

        if self.activation is not None:
            out = self.activation(out)
        return out


class HPGRelMsgMessagePassing(nn.Module):
    """Stack of :class:`HPGRelMsgGATLayer` layers — drop-in replacement for
    :class:`HPGMessagePassing` for the ``HPG_relMsg`` variant.

    Interface identical to :class:`HPGMessagePassing`; swap by setting
    ``mp_type="rel_msg"`` in :class:`~chemprop.models.hpg.HPGMPNN`.
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
            HPGRelMsgGATLayer(
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
        h = bmg.V
        for layer in self.layers:
            h = self.dropout(layer(h, bmg.edge_index, bmg.E))
        return h
