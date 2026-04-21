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


# ---------------------------------------------------------------------------
#  Phase 2B: HPG_fragGraph — lightweight fragment-level graph layer
# ---------------------------------------------------------------------------

class HPGFragGraphLayer(nn.Module):
    """Lightweight fragment-level message-passing layer for HPG_fragGraph.

    HPG_fragGraph tests whether explicit fragment adjacency improves polymer
    property prediction by introducing one lightweight polymer-structure layer
    on top of the existing HPG fragment embeddings, while keeping
    fraction-weighted pooling unchanged.

    Fragment-fragment edges are already stored in BatchHPGMolGraph.edge_index
    (from the featurizer's ``ff_src / ff_dst`` block).  They are identified
    at runtime as edges where both endpoints are fragment nodes:
        is_ff = bmg.frag_mask[src] & bmg.frag_mask[dst]
    No changes to the featurizer or data classes are required.

    Message passing (minimal residual, adjacency-only):

        m_i = sum_{j in N_frag(i)} W(u_j)
        z_i = u_i + m_i

    where:
    - u_i / z_i are the input / output embeddings for fragment node i
    - W  = nn.Linear(d_h, d_h, bias=False), **zero-initialized**
    - N_frag(i) = fragment-level neighbours of fragment i

    Zero-initialization ensures the model starts identically to HPG_frac
    (W=0 → messages are zero → z_i = u_i) and only learns fragment
    adjacency signal if it is informative.

    Batching: edge_index uses globally-shifted node indices, so
    scatter_add automatically operates within each polymer.
    Atom nodes are left unchanged (their positions receive zero aggregation
    because ff_dst only contains fragment indices).

    Parameters
    ----------
    d_h : int
        Hidden dimension (must match the HPG encoder output dimension).
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.W = nn.Linear(d_h, d_h, bias=False)
        nn.init.zeros_(self.W.weight)  # start identical to HPG_frac

    def forward(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Apply one fragment-level message-passing step.

        Parameters
        ----------
        H : Tensor [N, d_h]
            Node embeddings for all nodes (fragments + atoms).
        bmg : BatchHPGMolGraph
            Must have ``frag_mask`` set.

        Returns
        -------
        Tensor [N, d_h]
            Updated embeddings: fragment positions updated via W + residual;
            atom positions unchanged (zero aggregation).
        """
        src, dst = bmg.edge_index  # each [E]

        # --- Identify fragment-fragment edges ---
        is_ff = bmg.frag_mask[src] & bmg.frag_mask[dst]  # [E] bool

        if not is_ff.any():
            return H  # no fragment edges — pass through unchanged

        ff_src = src[is_ff]  # [E_ff]
        ff_dst = dst[is_ff]  # [E_ff]

        # --- Message: m_ij = W(H[src_j]) ---
        msgs = self.W(H[ff_src])  # [E_ff, d_h]

        # --- Aggregate: scatter-sum into destination nodes ---
        aggregated = torch.zeros_like(H)  # [N, d_h]
        aggregated.scatter_add_(
            0,
            ff_dst.unsqueeze(-1).expand_as(msgs),
            msgs,
        )

        # --- Residual update: z = H + aggregated ---
        return H + aggregated


# ---------------------------------------------------------------------------
#  Phase 2B rerun: HPG_frac_archGraph — architecture-aware fragment update
# ---------------------------------------------------------------------------

class HPGArchGraphLayer(nn.Module):
    """Architecture-aware binary-fragment update for HPG_frac_archGraph.

    For a binary copolymer with fragment embeddings h_A, h_B and per-polymer
    architecture weights [w_AA, w_AB, w_BA, w_BB]:

        m_A = w_AA · W_AA(h_A)  +  w_BA · W_BA(h_B)
        m_B = w_AB · W_AB(h_A)  +  w_BB · W_BB(h_B)
        z_A = h_A + m_A
        z_B = h_B + m_B

    Architecture weights are determined at featurization time from the
    ``poly_type`` column of the dataset:

        alternating : [w_AA=0,   w_AB=1,   w_BA=1,   w_BB=0  ]
        block       : [w_AA=1,   w_AB=γ,   w_BA=γ,   w_BB=1  ]  (γ=0.1)
        random      : [w_AA=f_A, w_AB=f_B, w_BA=f_A, w_BB=f_B]

    All four weight matrices are zero-initialized so the model starts
    identically to HPG_frac (m_A = m_B = 0 → z = h) and only learns
    architecture-aware interactions if they are informative.

    Parameters
    ----------
    d_h : int
        Hidden dimension (must match HPG encoder output).
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.W_AA = nn.Linear(d_h, d_h, bias=False)
        self.W_AB = nn.Linear(d_h, d_h, bias=False)
        self.W_BA = nn.Linear(d_h, d_h, bias=False)
        self.W_BB = nn.Linear(d_h, d_h, bias=False)
        for lin in (self.W_AA, self.W_AB, self.W_BA, self.W_BB):
            nn.init.zeros_(lin.weight)  # start identical to HPG_frac

    def forward(
        self,
        h_A: Tensor,           # [B, d_h]
        h_B: Tensor,           # [B, d_h]
        arch_weights: Tensor,  # [B, 4]  = [w_AA, w_AB, w_BA, w_BB]
    ) -> tuple[Tensor, Tensor]:
        """Return updated (z_A, z_B), each [B, d_h]."""
        w_AA = arch_weights[:, 0:1]   # [B, 1]
        w_AB = arch_weights[:, 1:2]
        w_BA = arch_weights[:, 2:3]
        w_BB = arch_weights[:, 3:4]

        m_A = w_AA * self.W_AA(h_A) + w_BA * self.W_BA(h_B)
        m_B = w_AB * self.W_AB(h_A) + w_BB * self.W_BB(h_B)
        return h_A + m_A, h_B + m_B


# ---------------------------------------------------------------------------
#  Phase 3B: HPG_pairInteract — fixed pairwise interaction layer
# ---------------------------------------------------------------------------

class HPGPairInteractLayer(nn.Module):
    """Pairwise monomer interaction MLP for HPG_pairInteract (Phase 3B).

    HPG_pairInteract tests whether polymer properties depend on explicit
    non-additive pairwise monomer interactions beyond additive fraction-
    weighted pooling.

    Pair feature (symmetric, no bias in construction):
        pair_feat(i,j) = [h_i + h_j,  h_i ⊙ h_j,  |h_i − h_j|]   # 3·d_h

    MLP mapping:
        phi(h_i, h_j) = MLP_pair(pair_feat(i,j))   # d_h output

    The output layer of MLP_pair is **zero-initialized** so the interaction
    term h_int = Σ_{i<j} f_i f_j phi_ij starts at zero, keeping the model
    identical to HPG_frac at initialization.

    Parameters
    ----------
    d_h : int
        Hidden dimension (must match HPG encoder output).
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        # Zero-init output layer → phi = 0 at start → h_int = 0 → h_poly = h_mix
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pair_feat: Tensor) -> Tensor:
        """Compute pair representation from pre-built pair features.

        Parameters
        ----------
        pair_feat : Tensor [P, 3·d_h]

        Returns
        -------
        Tensor [P, d_h]
        """
        return self.mlp(pair_feat)


# ---------------------------------------------------------------------------
#  Phase 3C: HPG_pairInteractAttn — attention-weighted pairwise interaction
# ---------------------------------------------------------------------------

class HPGPairInteractAttnLayer(nn.Module):
    """Attention-weighted pairwise interaction layer for HPG_pairInteractAttn (Phase 3C).

    HPG_pairInteractAttn tests whether the relevance of pairwise monomer
    interactions must itself be learned, rather than assuming all pairs
    contribute proportional to their fraction product (Phase 3B).

    This module holds two heads over the same pair feature:
        phi     : pair representation    pair_feat → d_h  (MLP, 2-layer)
        score   : pair importance scalar pair_feat → 1    (Linear)

    Both output layers are **zero-initialized**:
        score  = 0 → beta_ij uniform over fraction product → beta starts near f_i f_j
        phi    = 0 → v_ij = 0 → h_int = 0 → h_poly = h_mix  (identical to HPG_frac)

    Parameters
    ----------
    d_h : int
        Hidden dimension (must match HPG encoder output).
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        self.score = nn.Linear(3 * d_h, 1)
        # Zero-init both output layers for conservative start (h_int = 0 at init)
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, pair_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Compute pair representation and scalar score.

        Parameters
        ----------
        pair_feat : Tensor [P, 3·d_h]

        Returns
        -------
        v_ij : Tensor [P, d_h]   pair representation
        t_ij : Tensor [P]        scalar importance score (un-normalized)
        """
        return self.phi(pair_feat), self.score(pair_feat).squeeze(-1)


# ---------------------------------------------------------------------------
#  Phase 4: HPG_pairInteractGate — gated pairwise interaction
# ---------------------------------------------------------------------------

class HPGPairInteractGateLayer(nn.Module):
    """Gated pairwise interaction layer for HPG_pairInteractGate (Phase 4).

    HPG_pairInteractGate decomposes the polymer representation into:
      - an additive composition term  (h_mix, identical to HPG_frac)
      - a pairwise interaction correction  (h_int, same MLP as Phase 3B)
    and learns a scalar gate λ ∈ [0, 1] to control how much interaction
    signal is used for each polymer:

        h_poly = h_mix + λ · h_int

    The pair MLP output layer is zero-initialized (h_int = 0 at start),
    and the gate bias is set to a negative value so λ ≈ 0 at start.
    Combined, the model starts identical to HPG_frac.

    Parameters
    ----------
    d_h : int
        Hidden dimension (must match HPG encoder output).
    gate_init_bias : float
        Initial bias for the gate linear layer. A negative value ensures
        λ = sigmoid(bias) ≈ 0 at initialization.  Default: -3.0
        (sigmoid(-3) ≈ 0.047).
    """

    def __init__(self, d_h: int, gate_init_bias: float = -3.0):
        super().__init__()
        # Pair interaction MLP (same architecture as Phase 3B)
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        # Zero-init output layer → phi = 0 at start → h_int = 0
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

        # Scalar gate:  λ = sigmoid(Linear(h_mix))
        # Input: h_mix [B, d_h],  output: λ [B, 1]
        self.gate = nn.Linear(d_h, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)

    def forward_pair(self, pair_feat: Tensor) -> Tensor:
        """Compute pair representation from pre-built pair features.

        Parameters
        ----------
        pair_feat : Tensor [P, 3·d_h]

        Returns
        -------
        Tensor [P, d_h]
        """
        return self.pair_mlp(pair_feat)

    def forward_gate(self, h_mix: Tensor) -> Tensor:
        """Compute scalar gate λ ∈ [0, 1] per polymer.

        Parameters
        ----------
        h_mix : Tensor [B, d_h]

        Returns
        -------
        Tensor [B, 1]   — sigmoid-gated scalar
        """
        return torch.sigmoid(self.gate(h_mix))
