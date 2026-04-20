"""HPG-GAT model implemented as a PyTorch Lightning module.

Follows chemprop's MPNN conventions (training_step, validation_step, etc.)
so it can be used with the same Trainer infrastructure.

Architecture (matching the original HPG-GAT paper):
  1. HPGMessagePassing: stack of edge-aware GAT layers on the hierarchical graph
  2. Polymer readout (sum-pooling or fraction-weighted pooling) → [B, d_h]
  3. Linear projection d_h → d_ffn
  4. Optional concatenation of scalar features (X_d, e.g. polytype)
  5. FFN → prediction

Variants (selected via ``pooling_type`` and ``mp_type``):
  pooling_type:
  - ``"sum"``             : sum over ALL nodes             (HPG_baseline)
  - ``"frac_weighted"``   : Σ f_i · h_i over fragments    (HPG_frac / HPG_frac_polytype / HPG_frac_edgeTyped / HPG_relMsg)
  - ``"frac_arch_aware"`` : context-conditioned monomer    (HPG_frac_archAware)
  - ``"frac_graph"``       : fragment-level adjacency MP then fraction-weighted pool (HPG_fragGraph)
       update before pooling:
         m        = Σ_j f_j h_j
         h̃_i    = h_i + W(m - f_i h_i)
         h_poly   = Σ_i f_i h̃_i
       W is initialised to zero so the variant starts identically to
       HPG_frac and only diverges once gradients learn the interaction.

  mp_type:
  - ``"gat"``     : standard HPGGATLayer (default — all Phase 1 variants)
  - ``"rel_msg"`` : HPGRelMsgGATLayer — edge features enter message CONTENT
                    m_ij = alpha_ij * W_msg([h_src, e_ij])  (HPG_relMsg)
"""

from __future__ import annotations

import logging
from typing import Iterable

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from chemprop.data.hpg import BatchHPGMolGraph
from chemprop.nn.hpg import (
    HPGFragGraphLayer,
    HPGMessagePassing,
    HPGPairInteractAttnLayer,
    HPGPairInteractGateLayer,
    HPGPairInteractLayer,
    HPGRelMsgMessagePassing,
)
from chemprop.nn.metrics import ChempropMetric
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)


# Valid pooling types for HPG polymer readout.
VALID_POOLING_TYPES = (
    "sum",
    "frac_weighted",
    "frac_arch_aware",
    "frac_graph",
    "attn_pool",             # Phase 3A: fraction-aware learned attention pooling
    "pair_interact",         # Phase 3B: fixed pairwise monomer interaction
    "pair_interact_attn",    # Phase 3C: attention-weighted pairwise interaction
    "pair_interact_gate",    # Phase 4:  gated pairwise interaction
)


class HPGMPNN(pl.LightningModule):
    """Hierarchical Polymer Graph model with GAT message passing.

    Parameters
    ----------
    d_v : int
        Input node feature dimension (must match featurizer output).
    d_e : int
        Edge feature dimension. 1 for all scalar-edge variants (default);
        4 for ``HPG_frac_edgeTyped`` which uses typed 4-dim edge features.
    d_h : int
        Hidden dimension for GAT layers.
    d_ffn : int
        Intermediate FFN dimension after graph pooling.
    depth : int
        Number of GAT layers.
    num_heads : int
        Attention heads per GAT layer.
    dropout_mp : float
        Dropout in message passing layers.
    dropout_ffn : float
        Dropout before final prediction.
    n_tasks : int
        Number of output targets.
    d_xd : int
        Dimension of extra scalar features (X_d). 0 if none.
    mp_type : str
        Message-passing mechanism:
        - ``"gat"``     : standard HPGGATLayer (default, all Phase 1 variants).
        - ``"rel_msg"`` : HPGRelMsgGATLayer; edge features enter message content
          as ``m_ij = alpha_ij * W_msg([h_src, e_ij])``.
    pooling_type : str
        Polymer readout strategy:
        - ``"sum"``: sum over all nodes (original HPG_baseline).
        - ``"frac_weighted"``: Σ f_i · h_i over fragment nodes only,
          where f_i are monomer fractions from ``bmg.frag_fracs``.
        - ``"frac_arch_aware"``: one lightweight interaction step before
          pooling (HPG_frac_archAware).  Uses standard scalar edges
          (d_e=1) — explicit edge typing is intentionally excluded to
          keep this a clean ablation of polymer-level interaction only.
          Math:  m = Σ_j f_j h_j
                 h̃_i = h_i + W(m − f_i h_i)
                 h_poly = Σ_i f_i h̃_i
        - ``"frac_graph"``: one lightweight fragment-level graph MP step
          then fraction-weighted pooling (HPG_fragGraph, Phase 2B).
          Uses fragment-fragment edges already in bmg.edge_index.
          Math:  m_i = Σ_{j in N_frag(i)} W(u_j)  (W zero-init)
                 z_i = u_i + m_i
                 h_poly = Σ_i f_i z_i
    task_type : str
        ``"regression"`` or ``"classification"``.
    metrics : Iterable[ChempropMetric] | None
        Metrics for validation/test logging.
    criterion : ChempropMetric | None
        Loss function. If None, uses MSE for regression.
    warmup_epochs : int
        Learning rate warmup epochs.
    init_lr, max_lr, final_lr : float
        Learning rate schedule parameters.
    """

    def __init__(
        self,
        d_v: int = 49,
        d_e: int = 1,
        d_h: int = 128,
        d_ffn: int = 64,
        depth: int = 6,
        num_heads: int = 8,
        dropout_mp: float = 0.0,
        dropout_ffn: float = 0.2,
        n_tasks: int = 1,
        d_xd: int = 0,
        mp_type: str = "gat",
        pooling_type: str = "sum",
        task_type: str = "regression",
        metrics: Iterable[ChempropMetric] | None = None,
        criterion: ChempropMetric | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        if pooling_type not in VALID_POOLING_TYPES:
            raise ValueError(
                f"Invalid pooling_type={pooling_type!r}. "
                f"Choose from {VALID_POOLING_TYPES}"
            )
        _VALID_MP_TYPES = ("gat", "rel_msg")
        if mp_type not in _VALID_MP_TYPES:
            raise ValueError(
                f"Invalid mp_type={mp_type!r}. Choose from {_VALID_MP_TYPES}"
            )
        super().__init__()
        self.save_hyperparameters()

        # Message passing — swap layer stack based on mp_type
        _mp_cls = HPGRelMsgMessagePassing if mp_type == "rel_msg" else HPGMessagePassing
        self.message_passing = _mp_cls(
            d_v=d_v, d_h=d_h, d_e=d_e, depth=depth,
            num_heads=num_heads, dropout=dropout_mp,
        )

        # Arch-aware interaction layer: W in h̃_i = h_i + W(m - f_i h_i)
        # Only allocated for frac_arch_aware; zero-init so training starts
        # identical to HPG_frac and gradually learns the interaction signal.
        # bias=False keeps parameter count minimal and avoids constant shifts.
        if pooling_type == "frac_arch_aware":
            self.arch_interact = nn.Linear(d_h, d_h, bias=False)
            nn.init.zeros_(self.arch_interact.weight)
        else:
            self.arch_interact = None

        # Fragment-level graph layer (Phase 2B: HPG_fragGraph).
        # Only allocated for frac_graph; W is zero-initialized so the model
        # starts identically to HPG_frac and learns adjacency signal only
        # if it is informative for prediction.
        if pooling_type == "frac_graph":
            self.frag_graph_layer = HPGFragGraphLayer(d_h=d_h)
        else:
            self.frag_graph_layer = None

        # Phase 3A — HPG_attnPool: fraction-aware attention scorer.
        # scorer(h_i) = s_i;  alpha_i ∝ f_i * exp(s_i)  (per-polymer softmax).
        # Zero-init: at start s_i=0 → alpha_i = f_i → identical to HPG_frac.
        if pooling_type == "attn_pool":
            self.attn_scorer = nn.Linear(d_h, 1)
            nn.init.zeros_(self.attn_scorer.weight)
            nn.init.zeros_(self.attn_scorer.bias)
        else:
            self.attn_scorer = None

        # Phase 3B — HPG_pairInteract: fixed pairwise interaction MLP.
        # h_int = Σ_{i<j} f_i f_j phi(pair_feat); output layer zero-init.
        if pooling_type == "pair_interact":
            self.pair_interact_layer = HPGPairInteractLayer(d_h=d_h)
        else:
            self.pair_interact_layer = None

        # Phase 3C — HPG_pairInteractAttn: attention-weighted pair interactions.
        # beta_ij = softmax(score(pair_feat) + log(f_i f_j)); h_int = Σ beta_ij v_ij.
        # Both phi and score output layers zero-init.
        if pooling_type == "pair_interact_attn":
            self.pair_interact_attn_layer = HPGPairInteractAttnLayer(d_h=d_h)
        else:
            self.pair_interact_attn_layer = None

        # Phase 4 — HPG_pairInteractGate: gated pairwise interaction.
        # h_poly = h_mix + λ · h_int, where λ = sigmoid(Linear(h_mix)).
        # pair MLP output layer zero-init (h_int=0); gate bias=-3 (λ≈0.05).
        # Combined → model starts ≈ HPG_frac.
        if pooling_type == "pair_interact_gate":
            self.pair_interact_gate_layer = HPGPairInteractGateLayer(d_h=d_h)
        else:
            self.pair_interact_gate_layer = None

        # Graph readout: project pooled embedding down
        self.linear_pool = nn.Linear(d_h, d_ffn)

        # FFN: pool_dim [+ xd_dim] → hidden → prediction
        ffn_in = d_ffn + d_xd
        self.ffn = nn.Sequential(
            nn.Linear(ffn_in, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(512, n_tasks),
        )

        # Scalar feature projection (optional)
        if d_xd > 0:
            self.xd_transform = nn.Identity()
        else:
            self.xd_transform = None

        # Loss and metrics
        self.task_type = task_type
        if criterion is not None:
            self._criterion = criterion
        else:
            from chemprop.nn.metrics import MSE
            self._criterion = MSE()

        if metrics is not None:
            self.metrics = nn.ModuleList([*metrics, self._criterion.clone()])
        else:
            self.metrics = nn.ModuleList([self._criterion.clone()])

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self._output_transform = None  # Set externally for regression unscaling
        self._init_weights()

    def _init_weights(self):
        for m in [self.linear_pool, *self.ffn]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    @property
    def output_dim(self) -> int:
        return self.hparams["n_tasks"]

    @property
    def criterion(self) -> ChempropMetric:
        return self._criterion

    # ------------------------------------------------------------------
    #  Polymer readout helpers
    # ------------------------------------------------------------------

    def _pool_sum(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Sum-pooling over ALL nodes per graph → [B, d_h]."""
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        idx = bmg.batch.unsqueeze(-1).expand(-1, H.size(1))
        H_graph = torch.zeros(B, H.size(1), dtype=H.dtype, device=H.device)
        H_graph.scatter_add_(0, idx, H)
        return H_graph

    def _frag_fracs_checked(self, bmg: BatchHPGMolGraph, variant: str):
        """Return (H_frag, frag_batch, fracs) or raise a clear error."""
        if bmg.frag_fracs is None:
            raise ValueError(
                f"{variant} pooling requires bmg.frag_fracs to be set. "
                "Ensure fractions are provided in the HPGMolGraph."
            )
        frag_mask  = bmg.frag_mask               # [N] bool
        frag_batch = bmg.batch[frag_mask]         # [F_total]
        fracs      = bmg.frag_fracs               # [F_total]
        return frag_mask, frag_batch, fracs

    def _pool_frac_weighted(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Fraction-weighted pooling over **fragment** nodes only.

        Computes  h_poly = Σ_i  f_i · h_i  per polymer, where
        h_i are fragment node embeddings and f_i are monomer fractions.

        Parameters
        ----------
        H : Tensor [N, d_h]
            Node embeddings after message passing (all nodes).
        bmg : BatchHPGMolGraph
            Must have ``frag_fracs`` set (non-None).

        Returns
        -------
        Tensor [B, d_h]
        """
        frag_mask, frag_batch, fracs = self._frag_fracs_checked(bmg, "frac_weighted")
        H_frag = H[frag_mask]                      # [F_total, d_h]

        # Weighted embeddings
        weighted = H_frag * fracs.unsqueeze(-1)     # [F_total, d_h]

        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        idx = frag_batch.unsqueeze(-1).expand(-1, weighted.size(1))
        H_graph = torch.zeros(B, weighted.size(1), dtype=weighted.dtype, device=weighted.device)
        H_graph.scatter_add_(0, idx, weighted)
        return H_graph

    @staticmethod
    def _build_pairs(
        frag_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Build all within-polymer unordered fragment pairs (i < j).

        Used by Phase 3B (pair_interact) and 3C (pair_interact_attn).
        Polymers with only one fragment contribute no pairs and are
        handled correctly (their h_int stays zero).

        Parameters
        ----------
        frag_batch : Tensor [F_total]
            Polymer index for each fragment node.

        Returns
        -------
        (idx_i, idx_j, poly_idx) or None if no pairs exist.
            idx_i, idx_j : Tensor [P]   fragment-local pair indices (i < j)
            poly_idx     : Tensor [P]   polymer index for each pair
        """
        F_total = frag_batch.shape[0]
        if F_total < 2:
            return None

        # [F, F] mask: same polymer AND upper triangular (i < j)
        same_poly = frag_batch.unsqueeze(0) == frag_batch.unsqueeze(1)  # [F, F]
        upper_tri = torch.ones(
            F_total, F_total, dtype=torch.bool, device=frag_batch.device
        ).triu(diagonal=1)
        pair_mask = same_poly & upper_tri  # [F, F]

        if not pair_mask.any():
            return None  # all polymers are singletons

        idx_i, idx_j = pair_mask.nonzero(as_tuple=True)  # each [P]
        poly_idx = frag_batch[idx_i]                      # [P]
        return idx_i, idx_j, poly_idx

    def _pool_frag_graph(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Fragment-graph update then fraction-weighted pooling (HPG_fragGraph).

        The HPG encoder (lower-level) is unchanged.  One lightweight
        fragment-level message-passing step runs AFTER the encoder and
        BEFORE fraction-weighted pooling.

        Math (per polymer, index i over its fragment nodes):

            m_i     = sum_{j in N_frag(i)} W(u_j)   # neighbour messages
            z_i     = u_i + m_i                       # residual update
            h_poly  = sum_i  f_i * z_i                # fraction-weighted pool

        W (``self.frag_graph_layer.W``) is zero-initialized, so the variant
        starts identically to HPG_frac and diverges only as W learns.

        Parameters
        ----------
        H   : Tensor [N, d_h]       all-node embeddings from HPG encoder
        bmg : BatchHPGMolGraph      must have frag_fracs set

        Returns
        -------
        Tensor [B, d_h]
        """
        Z = self.frag_graph_layer(H, bmg)  # [N, d_h] — fragment nodes updated
        return self._pool_frac_weighted(Z, bmg)  # identical pooling as HPG_frac

    def _pool_frac_arch_aware(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Context-conditioned fraction-weighted pooling (HPG_frac_archAware).

        Built on top of HPG_frac.  Explicit edge typing is intentionally
        excluded — this variant tests polymer-level interaction only.

        Math (per polymer, index i over its fragment nodes):

            m        = Σ_j  f_j · h_j          # global mixture
            h̃_i    = h_i + W(m − f_i · h_i)  # context-conditioned update
            h_poly   = Σ_i  f_i · h̃_i         # final pooling

        W (``self.arch_interact``) is linear d_h → d_h, no bias, zero-init,
        so the variant starts identical to HPG_frac and learns the
        interaction signal incrementally.

        Batching: ``scatter_add`` is used throughout so variable numbers
        of monomers per polymer are handled correctly.

        Parameters
        ----------
        H   : Tensor [N, d_h]       node embeddings (all nodes)
        bmg : BatchHPGMolGraph      must have frag_fracs set

        Returns
        -------
        Tensor [B, d_h]
        """
        frag_mask, frag_batch, fracs = self._frag_fracs_checked(bmg, "frac_arch_aware")
        H_frag = H[frag_mask]                       # [F_total, d_h]
        f      = fracs.unsqueeze(-1)                # [F_total, 1]  broadcast-ready

        # ── Step 1: compute per-polymer global mixture m = Σ_j f_j h_j ──
        weighted = H_frag * f                       # [F_total, d_h]
        B   = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        idx = frag_batch.unsqueeze(-1).expand_as(weighted)
        m_poly = torch.zeros(B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device)
        m_poly.scatter_add_(0, idx, weighted)       # [B, d_h]

        # ── Step 2: broadcast m back to each fragment's polymer ──
        m_each = m_poly[frag_batch]                 # [F_total, d_h]

        # ── Step 3: context-conditioned update h̃_i = h_i + W(m - f_i h_i) ──
        H_tilde = H_frag + self.arch_interact(m_each - f * H_frag)  # [F_total, d_h]

        # ── Step 4: final fraction-weighted pooling h_poly = Σ f_i h̃_i ──
        weighted_tilde = H_tilde * f                # [F_total, d_h]
        H_graph = torch.zeros(B, H_tilde.size(1), dtype=H_tilde.dtype, device=H_tilde.device)
        H_graph.scatter_add_(0, idx, weighted_tilde)
        return H_graph

    def _pool_attn_pool(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Fraction-aware attention pooling (HPG_attnPool, Phase 3A).

        Tests whether adaptive importance weighting on top of fraction priors
        improves polymer property prediction beyond HPG_frac.

        Math (per polymer, index i over its fragment nodes):

            s_i      = scorer(h_i)                       # learned scalar
            alpha_i  = softmax( s_i + log(f_i + eps) )  # per-polymer
                     ∝ f_i · exp(s_i)
            h_poly   = Σ_i  alpha_i · h_i

        At init (scorer = 0):  s_i = 0  →  alpha_i = f_i  (identical to HPG_frac)

        Parameters
        ----------
        H   : Tensor [N, d_h]  node embeddings (all nodes)
        bmg : BatchHPGMolGraph  must have frag_fracs set

        Returns
        -------
        Tensor [B, d_h]
        """
        frag_mask, frag_batch, fracs = self._frag_fracs_checked(bmg, "attn_pool")
        H_frag = H[frag_mask]                             # [F_total, d_h]
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1

        # Step 1: log-space unnormalized weights:  s_i + log(f_i + eps)
        s     = self.attn_scorer(H_frag).squeeze(-1)     # [F_total]
        log_w = s + torch.log(fracs.clamp(min=1e-8))     # [F_total]

        # Step 2: per-polymer max subtraction for numerical stability
        per_poly_max = torch.full(
            (B,), float("-inf"), dtype=log_w.dtype, device=log_w.device
        )
        per_poly_max.scatter_reduce_(
            0, frag_batch, log_w, reduce="amax", include_self=True
        )
        log_w_stable = log_w - per_poly_max[frag_batch]  # [F_total]

        # Step 3: softmax attention weights (sum to 1 per polymer)
        exp_w = torch.exp(log_w_stable)                  # [F_total]
        Z = torch.zeros(B, dtype=exp_w.dtype, device=exp_w.device)
        Z.scatter_add_(0, frag_batch, exp_w)
        alpha = exp_w / Z[frag_batch].clamp(min=1e-8)    # [F_total]

        # Step 4: weighted sum over fragment embeddings
        idx = frag_batch.unsqueeze(-1).expand_as(H_frag)
        H_graph = torch.zeros(
            B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
        )
        H_graph.scatter_add_(0, idx, H_frag * alpha.unsqueeze(-1))
        return H_graph

    def _pool_pair_interact(self, H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Fraction-weighted pairwise interaction pooling (HPG_pairInteract, Phase 3B).

        Tests whether polymer properties depend on explicit non-additive
        pairwise monomer interactions beyond additive fraction-weighted pooling.

        Math (per polymer, index i,j over pairs with i < j in same polymer):

            h_mix      = Σ_i  f_i · h_i                          # HPG_frac baseline
            pair_feat  = [h_i+h_j,  h_i⊙h_j,  |h_i−h_j|]      # 3·d_h, symmetric
            phi_ij     = MLP_pair(pair_feat)                      # d_h
            h_int      = Σ_{i<j}  f_i · f_j · phi_ij
            h_poly     = h_mix + h_int

        At init (MLP output layer = 0):  phi_ij = 0  →  h_int = 0  →  h_poly = h_mix

        Parameters
        ----------
        H   : Tensor [N, d_h]  node embeddings (all nodes)
        bmg : BatchHPGMolGraph  must have frag_fracs set

        Returns
        -------
        Tensor [B, d_h]
        """
        frag_mask, frag_batch, fracs = self._frag_fracs_checked(bmg, "pair_interact")
        H_frag = H[frag_mask]                             # [F_total, d_h]
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1

        # ── Additive mixture (identical to HPG_frac) ──
        weighted = H_frag * fracs.unsqueeze(-1)           # [F_total, d_h]
        idx      = frag_batch.unsqueeze(-1).expand_as(H_frag)
        h_mix    = torch.zeros(
            B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
        )
        h_mix.scatter_add_(0, idx, weighted)

        # ── Pairwise interaction term ──
        pairs = self._build_pairs(frag_batch)
        if pairs is None:
            return h_mix  # all singletons — fall back to HPG_frac
        idx_i, idx_j, poly_idx = pairs

        h_i = H_frag[idx_i]                               # [P, d_h]
        h_j = H_frag[idx_j]                               # [P, d_h]
        f_i = fracs[idx_i]                                # [P]
        f_j = fracs[idx_j]                                # [P]

        pair_feat = torch.cat(
            [h_i + h_j, h_i * h_j, (h_i - h_j).abs()], dim=1
        )                                                  # [P, 3·d_h]
        phi_ij = self.pair_interact_layer(pair_feat)      # [P, d_h]
        w_ij   = (f_i * f_j).unsqueeze(-1)                # [P, 1]

        h_int = torch.zeros(
            B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
        )
        h_int.scatter_add_(
            0,
            poly_idx.unsqueeze(-1).expand_as(phi_ij),
            w_ij * phi_ij,
        )
        return h_mix + h_int

    def _pool_pair_interact_attn(
        self, H: Tensor, bmg: BatchHPGMolGraph
    ) -> Tensor:
        """Attention-weighted pairwise interaction pooling (HPG_pairInteractAttn, Phase 3C).

        Tests whether the relevance of pairwise monomer interactions must
        itself be learned, rather than assuming all pairs contribute
        proportional to their fraction product (Phase 3B).

        Math (per polymer, index i,j over pairs with i < j in same polymer):

            h_mix      = Σ_i  f_i · h_i
            pair_feat  = [h_i+h_j,  h_i⊙h_j,  |h_i−h_j|]      # 3·d_h
            v_ij       = phi(pair_feat)                           # d_h
            t_ij       = score_pair(pair_feat)                    # scalar
            beta_ij    = softmax( t_ij + log(f_i·f_j + eps) )   # per-polymer over pairs
            h_int      = Σ_{i<j}  beta_ij · v_ij
            h_poly     = h_mix + h_int

        At init (phi and score_pair output layers = 0):  v_ij = 0  →  h_poly = h_mix

        Parameters
        ----------
        H   : Tensor [N, d_h]  node embeddings (all nodes)
        bmg : BatchHPGMolGraph  must have frag_fracs set

        Returns
        -------
        Tensor [B, d_h]
        """
        frag_mask, frag_batch, fracs = self._frag_fracs_checked(
            bmg, "pair_interact_attn"
        )
        H_frag = H[frag_mask]                             # [F_total, d_h]
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1

        # ── Additive mixture ──
        weighted = H_frag * fracs.unsqueeze(-1)
        idx      = frag_batch.unsqueeze(-1).expand_as(H_frag)
        h_mix    = torch.zeros(
            B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
        )
        h_mix.scatter_add_(0, idx, weighted)

        # ── Pair-attention interaction ──
        pairs = self._build_pairs(frag_batch)
        if pairs is None:
            return h_mix
        idx_i, idx_j, poly_idx = pairs

        h_i = H_frag[idx_i]
        h_j = H_frag[idx_j]
        f_i = fracs[idx_i]
        f_j = fracs[idx_j]

        pair_feat = torch.cat(
            [h_i + h_j, h_i * h_j, (h_i - h_j).abs()], dim=1
        )                                                  # [P, 3·d_h]
        v_ij, t_ij = self.pair_interact_attn_layer(pair_feat)  # [P, d_h], [P]

        # Per-polymer pair attention with fraction prior
        log_pair_w = t_ij + torch.log((f_i * f_j).clamp(min=1e-8))  # [P]

        # Numerically stable per-polymer softmax over pairs
        per_poly_max = torch.full(
            (B,), float("-inf"), dtype=log_pair_w.dtype, device=log_pair_w.device
        )
        per_poly_max.scatter_reduce_(
            0, poly_idx, log_pair_w, reduce="amax", include_self=True
        )
        log_w_stable = log_pair_w - per_poly_max[poly_idx]

        exp_w = torch.exp(log_w_stable)                  # [P]
        Z = torch.zeros(B, dtype=exp_w.dtype, device=exp_w.device)
        Z.scatter_add_(0, poly_idx, exp_w)
        beta = exp_w / Z[poly_idx].clamp(min=1e-8)       # [P]

        h_int = torch.zeros(
            B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
        )
        h_int.scatter_add_(
            0,
            poly_idx.unsqueeze(-1).expand_as(v_ij),
            beta.unsqueeze(-1) * v_ij,
        )
        return h_mix + h_int

    def _pool_pair_interact_gate(
        self, H: Tensor, bmg: BatchHPGMolGraph
    ) -> Tensor:
        """Gated pairwise interaction pooling (HPG_pairInteractGate, Phase 4).

        HPG_pairInteractGate decomposes the polymer representation into:
          - an additive composition term  (h_mix, identical to HPG_frac)
          - a pairwise interaction correction  (h_int, same MLP as Phase 3B)
        and learns a scalar gate λ ∈ [0, 1] to control how much interaction
        signal is used for each polymer.

        Math (per polymer, index i,j over pairs with i < j in same polymer):

            h_mix      = Σ_i  f_i · h_i                          # HPG_frac baseline
            pair_feat  = [h_i+h_j,  h_i⊙h_j,  |h_i−h_j|]      # 3·d_h, symmetric
            phi_ij     = MLP_pair(pair_feat)                      # d_h
            h_int      = Σ_{i<j}  f_i · f_j · phi_ij
            λ          = sigmoid(Linear(h_mix))                   # [B, 1]
            h_poly     = h_mix + λ · h_int

        At init:  MLP output layer = 0 → h_int = 0,
                  gate bias = -3    → λ ≈ 0.047.
        Combined → h_poly ≈ h_mix  (identical to HPG_frac).

        Diagnostics (stored on ``self`` in eval mode for later analysis):
            self._diag_lambda   : Tensor [B, 1]   gate values
            self._diag_norm_mix : Tensor [B]       ‖h_mix‖₂ per polymer
            self._diag_norm_int : Tensor [B]       ‖h_int‖₂ per polymer
            self._diag_ratio    : Tensor [B]       ‖h_int‖/(‖h_mix‖+ε)

        Parameters
        ----------
        H   : Tensor [N, d_h]  node embeddings (all nodes)
        bmg : BatchHPGMolGraph  must have frag_fracs set

        Returns
        -------
        Tensor [B, d_h]
        """
        frag_mask, frag_batch, fracs = self._frag_fracs_checked(
            bmg, "pair_interact_gate"
        )
        H_frag = H[frag_mask]                             # [F_total, d_h]
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1

        # ── Additive mixture (identical to HPG_frac) ──
        weighted = H_frag * fracs.unsqueeze(-1)           # [F_total, d_h]
        idx      = frag_batch.unsqueeze(-1).expand_as(H_frag)
        h_mix    = torch.zeros(
            B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
        )
        h_mix.scatter_add_(0, idx, weighted)

        # ── Pairwise interaction term ──
        pairs = self._build_pairs(frag_batch)
        if pairs is None:
            # All singletons — no pairs, h_int = 0, fall back to h_mix
            h_int = torch.zeros_like(h_mix)
        else:
            idx_i, idx_j, poly_idx = pairs

            h_i = H_frag[idx_i]                           # [P, d_h]
            h_j = H_frag[idx_j]                           # [P, d_h]
            f_i = fracs[idx_i]                             # [P]
            f_j = fracs[idx_j]                             # [P]

            pair_feat = torch.cat(
                [h_i + h_j, h_i * h_j, (h_i - h_j).abs()], dim=1
            )                                              # [P, 3·d_h]
            phi_ij = self.pair_interact_gate_layer.forward_pair(pair_feat)  # [P, d_h]
            w_ij   = (f_i * f_j).unsqueeze(-1)             # [P, 1]

            h_int = torch.zeros(
                B, H_frag.size(1), dtype=H_frag.dtype, device=H_frag.device
            )
            h_int.scatter_add_(
                0,
                poly_idx.unsqueeze(-1).expand_as(phi_ij),
                w_ij * phi_ij,
            )

        # ── Gate ──
        lam = self.pair_interact_gate_layer.forward_gate(h_mix)  # [B, 1]

        # ── Diagnostics (eval mode only, no grad overhead during training) ──
        if not self.training:
            with torch.no_grad():
                self._diag_lambda   = lam.detach()
                self._diag_norm_mix = h_mix.detach().norm(dim=1)
                self._diag_norm_int = h_int.detach().norm(dim=1)
                eps = 1e-8
                self._diag_ratio    = self._diag_norm_int / (self._diag_norm_mix + eps)

        return h_mix + lam * h_int

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def fingerprint(self, bmg: BatchHPGMolGraph, X_d: Tensor | None = None) -> Tensor:
        """Compute graph-level fingerprint.

        Returns
        -------
        Tensor [B, d_ffn] or [B, d_ffn + d_xd]
        """
        H = self.message_passing(bmg)  # [N, d_h]

        # --- Polymer readout ---
        pt = self.hparams["pooling_type"]
        if pt == "frac_weighted":
            H_graph = self._pool_frac_weighted(H, bmg)
        elif pt == "frac_arch_aware":
            H_graph = self._pool_frac_arch_aware(H, bmg)
        elif pt == "frac_graph":
            H_graph = self._pool_frag_graph(H, bmg)
        elif pt == "attn_pool":
            H_graph = self._pool_attn_pool(H, bmg)
        elif pt == "pair_interact":
            H_graph = self._pool_pair_interact(H, bmg)
        elif pt == "pair_interact_attn":
            H_graph = self._pool_pair_interact_attn(H, bmg)
        elif pt == "pair_interact_gate":
            H_graph = self._pool_pair_interact_gate(H, bmg)
        else:
            H_graph = self._pool_sum(H, bmg)

        H_graph = F.leaky_relu(self.linear_pool(H_graph))  # [B, d_ffn]

        # --- Optional metadata concatenation (e.g. polytype) ---
        if X_d is not None and self.xd_transform is not None:
            X_d_t = self.xd_transform(X_d)
            H_graph = torch.cat([H_graph, X_d_t], dim=1)

        return H_graph

    def forward(self, bmg: BatchHPGMolGraph, X_d: Tensor | None = None) -> Tensor:
        return self.ffn(self.fingerprint(bmg, X_d))

    # ------------------------------------------------------------------
    #  Lightning training loop
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        bmg, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self.forward(bmg, X_d)
        loss = self._criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        batch_size = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        self.log("train_loss", self._criterion, batch_size=batch_size,
                 prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        bmg, X_d, targets, weights, lt_mask, gt_mask = batch
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self.forward(bmg, X_d)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)

        batch_size = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        self.log("val_loss", self.metrics[-1], batch_size=batch_size,
                 prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch, label: str):
        bmg, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self.forward(bmg, X_d)
        if self._output_transform is not None:
            preds = self._output_transform(preds)
        batch_size = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            metric_alias = getattr(m, "alias", None) or getattr(type(m), "alias", None)
            if metric_alias is not None:
                self.log(f"{label}/{metric_alias}", m, batch_size=batch_size,
                         logger=True, prog_bar=False)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, X_d, *_ = batch
        preds = self.forward(bmg, X_d)
        if self._output_transform is not None:
            preds = self._output_transform(preds)
        return preds

    # ------------------------------------------------------------------
    #  Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps,
            self.init_lr, self.max_lr, self.final_lr,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

    def get_batch_size(self, batch) -> int:
        bmg = batch[0]
        return int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
