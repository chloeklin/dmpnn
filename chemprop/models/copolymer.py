"""Copolymer MPNN model: shared encoder for two monomers with composition-aware integration."""
from __future__ import annotations

import logging
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched

logger = logging.getLogger(__name__)


class MonomerScorer(nn.Module):
    """Lightweight MLP that maps a monomer embedding to a scalar attention score.

    Architecture: Linear(d_in, hidden) → ReLU → Linear(hidden, 1).
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        """Return scalar scores for each monomer embedding.

        Parameters
        ----------
        h : Tensor [N, d_in]

        Returns
        -------
        Tensor [N, 1]
        """
        return self.net(h)


class PairInteractionMLP(nn.Module):
    """MLP that maps a symmetric pair feature vector to a pair embedding.

    Input: ``[h_i + h_j, h_i * h_j, |h_i - h_j|]``  →  dim = 3 * d_in.
    Output: pair embedding of dimension d_out (defaults to d_in).
    """

    def __init__(self, d_in: int, d_out: int | None = None, hidden: int = 128):
        super().__init__()
        d_out = d_out or d_in
        self.net = nn.Sequential(
            nn.Linear(3 * d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, pair_feat: Tensor) -> Tensor:
        return self.net(pair_feat)


class PairScorer(nn.Module):
    """Lightweight MLP that maps a pair feature vector to a scalar score.

    Input: ``[h_i + h_j, h_i * h_j, |h_i - h_j|]``  →  dim = 3 * d_in.
    Output: scalar score per pair.
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair_feat: Tensor) -> Tensor:
        return self.net(pair_feat)


class SelfAttentionPooling(nn.Module):
    """Single-head, fraction-aware self-attention over monomer embeddings.

    Produces a single polymer embedding directly (no h_mix / h_int split).

    Steps
    -----
    1. Linear projections: q = W_Q h, k = W_K h, v = W_V h
    2. Scaled dot-product attention with fraction bias:
       scores = (q k^T) / sqrt(d_k) + log(f_j + eps)
    3. alpha = softmax(scores, dim=-1)
    4. h_attn = alpha @ v
    5. Residual: h_out = h + h_attn
    6. Pooling: h_poly = sum_i f_i * h_out_i

    Parameters
    ----------
    d : int
        Embedding dimension (= d_mp).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.scale = d ** 0.5

    def forward(self, H: Tensor, F: Tensor) -> Tensor:
        """Compute fraction-aware self-attention pooled polymer embedding.

        Parameters
        ----------
        H : Tensor [B, N, d]
            Monomer embeddings.
        F : Tensor [B, N]
            Monomer fractions (sum to 1 per sample).

        Returns
        -------
        Tensor [B, d]
            Single polymer embedding.
        """
        Q = self.W_Q(H)                                    # [B, N, d]
        K = self.W_K(H)                                    # [B, N, d]
        V = self.W_V(H)                                    # [B, N, d]

        # Scaled dot-product scores + fraction bias
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, N, N]
        frac_bias = torch.log(F + 1e-8).unsqueeze(1)       # [B, 1, N]
        scores = scores + frac_bias                         # broadcast → [B, N, N]

        alpha = torch.softmax(scores, dim=-1)               # [B, N, N]

        # Context aggregation + residual
        h_attn = torch.bmm(alpha, V)                        # [B, N, d]
        h_out = H + h_attn                                  # [B, N, d]

        # Fraction-weighted pooling
        h_poly = (F.unsqueeze(-1) * h_out).sum(dim=1)       # [B, d]
        return h_poly


VALID_FUSION_TYPES = ("sum_fusion", "concat_fusion", "gated_fusion", "scalar_residual_fusion")


class SumFusion(nn.Module):
    """``h = h_mix + h_int``  (additive residual — the original default)."""

    def forward(self, h_mix: Tensor, h_int: Tensor) -> Tensor:
        return h_mix + h_int


class ConcatFusion(nn.Module):
    """``h = MLP([h_mix || h_int])``  →  output dim = d."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, h_mix: Tensor, h_int: Tensor) -> Tensor:
        return self.net(torch.cat([h_mix, h_int], dim=1))


class GatedFusion(nn.Module):
    """Element-wise gating: ``g = σ(MLP([h_mix || h_int]))``, ``h = (1-g)·h_mix + g·h_int``."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, h_mix: Tensor, h_int: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_mix, h_int], dim=1)))
        return (1 - g) * h_mix + g * h_int


class ScalarResidualFusion(nn.Module):
    """``h = h_mix + λ · h_int``  where λ is a learned scalar (init 0.1)."""

    def __init__(self, init_lambda: float = 0.1):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, h_mix: Tensor, h_int: Tensor) -> Tensor:
        return h_mix + self.lam * h_int


def build_fusion_module(fusion_type: str, d: int) -> nn.Module:
    """Factory: create a fusion module by name.

    Parameters
    ----------
    fusion_type : str
        One of :data:`VALID_FUSION_TYPES`.
    d : int
        Embedding dimension (= d_mp).
    """
    if fusion_type == "sum_fusion":
        return SumFusion()
    if fusion_type == "concat_fusion":
        return ConcatFusion(d)
    if fusion_type == "gated_fusion":
        return GatedFusion(d)
    if fusion_type == "scalar_residual_fusion":
        return ScalarResidualFusion()
    raise ValueError(f"Unknown fusion_type '{fusion_type}'. Valid: {VALID_FUSION_TYPES}")


class CopolymerMPNN(pl.LightningModule):
    """Two-monomer copolymer model with shared GNN encoder.

    Encodes monomer A and monomer B independently through a **shared** message-passing
    encoder + aggregation, then combines their graph-level embeddings using one of
    several integration modes:

    **Mean family** (unweighted average):

    * **mean**: ``z = (z_A + z_B) / 2`` → head input: ``z``
    * **mean_meta**: same ``z`` → head input: ``[z || meta]``

    **Mix family** (fraction-weighted mean-field embedding):

    * **mix**: ``z = fracA * z_A + fracB * z_B`` → head input: ``z``
    * **mix_meta**: same ``z`` → head input: ``[z || meta]``
    * **mix_frac**: same ``z`` → head input: ``[z || fracA || fracB]``
    * **mix_frac_meta**: same ``z`` → head input: ``[z || fracA || fracB || meta]``

    **Mix + pairwise fixed** (mixture h_mix + fraction-product–weighted pairwise):

    * **mix_pair**: ``h = fuse(h_mix, Σ_{i<j} (f_i·f_j)·φ_ij)`` → head input: ``h``
    * **mix_pair_meta**: same ``h`` → head input: ``[h || meta]``

    **Mix + pairwise attention** (mixture h_mix + attention-weighted pairwise):

    * **mix_pair_attn**: ``h = fuse(h_mix, Σ_{i<j} β_ij·φ_ij)`` → head input: ``h``
    * **mix_pair_attn_meta**: same ``h`` → head input: ``[h || meta]``

    **Attention family** (learned attention without fraction prior):

    * **attention**: ``α_i = softmax(s_i)``, ``z = Σ α_i z_i`` → head input: ``z``
    * **attention_meta**: same ``z`` → head input: ``[z || meta]``

    **Fraction-aware attention family** (learned attention with fraction prior):

    * **frac_attn**: ``α_i = softmax(s_i + log(f_i + ε))``, ``z = Σ α_i z_i`` → head input: ``z``
    * **frac_attn_meta**: same ``z`` → head input: ``[z || meta]``

    **Frac-attn + pairwise fixed** (attention h_mix + fraction-product–weighted pairwise):

    * **frac_attn_pair**: ``h = fuse(h_mix, Σ_{i<j} (f_i·f_j)·φ_ij)`` → head input: ``h``
    * **frac_attn_pair_meta**: same ``h`` → head input: ``[h || meta]``

    **Frac-attn + pairwise attention** (attention h_mix + attention-weighted pairwise):

    * **frac_attn_pair_attn**: ``h = fuse(h_mix, Σ_{i<j} β_ij·φ_ij)`` → head input: ``h``
    * **frac_attn_pair_attn_meta**: same ``h`` → head input: ``[h || meta]``

    **Interact family** (interaction-aware concatenation):

    * **interact**: head input: ``[z_A || z_B || |z_A-z_B| || z_A⊙z_B || fracA || fracB]``
    * **interact_meta**: head input: ``[z_A || z_B || |z_A-z_B| || z_A⊙z_B || fracA || fracB || meta]``

    where ``meta`` = additional scalar descriptors (e.g. RDKit, dataset-specific).

    **Self-attention family** (transformer-style self-attention over monomers):

    * **self_attn**: fraction-aware self-attention → fraction-weighted pool → ``h_poly``
    * **self_attn_meta**: same ``h_poly`` → head input: ``[h_poly || meta]``

    **Fusion strategies** (for pairwise modes only):

    * **sum_fusion**: ``h = h_mix + h_int``  (default, backward-compatible)
    * **concat_fusion**: ``h = MLP([h_mix || h_int])``
    * **gated_fusion**: ``g = σ(MLP([h_mix || h_int]))``, ``h = (1-g)·h_mix + g·h_int``
    * **scalar_residual_fusion**: ``h = h_mix + λ·h_int``  (λ learned, init 0.1)
    """

    VALID_MODES = (
        "mean", "mean_meta",
        "mix", "mix_meta", "mix_frac", "mix_frac_meta",
        "mix_pair", "mix_pair_meta",
        "mix_pair_attn", "mix_pair_attn_meta",
        "attention", "attention_meta",
        "frac_attn", "frac_attn_meta",
        "frac_attn_pair", "frac_attn_pair_meta",
        "frac_attn_pair_attn", "frac_attn_pair_attn_meta",
        "self_attn", "self_attn_meta",
        "interact", "interact_meta",
    )

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        copolymer_mode: str = "mix",
        fusion_type: str = "sum_fusion",
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["X_d_transform", "message_passing", "agg", "predictor"]
        )
        self.hparams["X_d_transform"] = X_d_transform
        
        # Handle both fresh init and checkpoint loading
        # When loading from checkpoint, these might be dicts instead of modules
        self.hparams.update(
            {
                "message_passing": getattr(message_passing, "hparams", message_passing),
                "agg": getattr(agg, "hparams", agg),
                "predictor": getattr(predictor, "hparams", predictor),
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        if copolymer_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown copolymer_mode '{copolymer_mode}'. "
                f"Valid modes: {self.VALID_MODES}"
            )
        self.copolymer_mode = copolymer_mode

        # Build scorer MLP for attention-based modes
        if copolymer_mode.startswith("attention") or copolymer_mode.startswith("frac_attn"):
            self.monomer_scorer = MonomerScorer(self.message_passing.output_dim)
        else:
            self.monomer_scorer = None

        # Build self-attention pooling module
        if copolymer_mode.startswith("self_attn"):
            self.self_attn_pool = SelfAttentionPooling(self.message_passing.output_dim)
        else:
            self.self_attn_pool = None

        # Build pair interaction MLPs and fusion module for pairwise modes
        _is_pairwise = (copolymer_mode.startswith("frac_attn_pair")
                        or copolymer_mode.startswith("mix_pair"))
        if _is_pairwise:
            if fusion_type not in VALID_FUSION_TYPES:
                raise ValueError(
                    f"Unknown fusion_type '{fusion_type}'. "
                    f"Valid: {VALID_FUSION_TYPES}"
                )
            d_mp = self.message_passing.output_dim
            self.pair_mlp = PairInteractionMLP(d_mp)
            _is_pair_attn = (copolymer_mode.startswith("frac_attn_pair_attn")
                             or copolymer_mode.startswith("mix_pair_attn"))
            if _is_pair_attn:
                self.pair_scorer = PairScorer(d_mp)
            else:
                self.pair_scorer = None
            self.fuse = build_fusion_module(fusion_type, d_mp)
        else:
            self.pair_mlp = None
            self.pair_scorer = None
            self.fuse = None
        self.fusion_type = fusion_type

        # Initialize metrics - handle checkpoint loading where criterion might not exist yet
        if metrics:
            metric_list = [*metrics]
            if self.criterion is not None:
                metric_list.append(self.criterion.clone())
            self.metrics = nn.ModuleList(metric_list)
        else:
            metric_list = []
            if hasattr(self.predictor, '_T_default_metric'):
                metric_list.append(self.predictor._T_default_metric())
            if self.criterion is not None:
                metric_list.append(self.criterion.clone())
            self.metrics = nn.ModuleList(metric_list) if metric_list else nn.ModuleList()

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    # ------------------------------------------------------------------ properties
    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    # ------------------------------------------------------------------ encoder
    def _encode_single(self, bmg: BatchMolGraph) -> Tensor:
        """Run shared message passing + aggregation + BN on one set of graphs."""
        H_v = self.message_passing(bmg)
        H = self.agg(H_v, bmg.batch)
        if H.size(0) == bmg.V.size(0):
            H = self.agg(H, bmg.batch)
        H = self.bn(H)
        return H

    @staticmethod
    def _weighted_block_embedding(
        flat_embeddings: Tensor,
        counts: Tensor,
        fracs: Tensor,
    ) -> Tensor:
        """Compute per-sample block embeddings from variable-length monomer lists.

        Parameters
        ----------
        flat_embeddings : Tensor [total_monomers, d]
            Embeddings for all monomers (across all samples) concatenated.
        counts : Tensor [batch_size] (long)
            Number of monomers per sample.
        fracs : Tensor [total_monomers]
            Intra-block fractions for each monomer (aligned with flat_embeddings).

        Returns
        -------
        Tensor [batch_size, d]
            Fraction-weighted sum of monomer embeddings per sample.
        """
        d = flat_embeddings.size(1)
        batch_size = counts.size(0)
        device = flat_embeddings.device
        result = flat_embeddings.new_zeros(batch_size, d)
        offset = 0
        for i in range(batch_size):
            n = counts[i].item()
            w = fracs[offset:offset + n].unsqueeze(1)   # [n, 1]
            emb = flat_embeddings[offset:offset + n]     # [n, d]
            result[i] = (w * emb).sum(dim=0)
            offset += n
        return result

    def fingerprint(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracA: Tensor,
        fracB: Tensor,
        X_d: Tensor | None = None,
    ) -> Tensor:
        """Compute copolymer fingerprint from two monomer graphs + fractions.

        Returns
        -------
        Tensor
            Shape ``[batch, d_final]`` where ``d_final`` depends on integration mode.
        """
        z_A = self._encode_single(bmg_A)  # [B, d]
        z_B = self._encode_single(bmg_B)  # [B, d]
        return self._apply_mode(z_A, z_B, fracA, fracB, X_d)

    def fingerprint_multi_monomer(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracs_A: Tensor,
        fracs_B: Tensor,
        counts_A: Tensor,
        counts_B: Tensor,
        X_d: Tensor | None = None,
    ) -> Tensor:
        """Compute copolymer fingerprint from multi-monomer blocks.

        Each block has variable-length monomer lists. Block embeddings are
        computed as fraction-weighted sums of monomer embeddings.

        Parameters
        ----------
        bmg_A : BatchMolGraph
            Batched mol-graphs for ALL block-A monomers (flattened across samples).
        bmg_B : BatchMolGraph
            Same for block B.
        fracs_A : Tensor [total_A_monomers]
            Intra-block fractions for block-A monomers.
        fracs_B : Tensor [total_B_monomers]
        counts_A : Tensor [batch_size]
            Number of A-monomers per sample.
        counts_B : Tensor [batch_size]
        X_d : Tensor | None
            Optional descriptor features.
        """
        # Encode all monomers at once
        flat_z_A = self._encode_single(bmg_A)  # [total_A, d]
        flat_z_B = self._encode_single(bmg_B)  # [total_B, d]

        # Weighted average per sample
        z_A = self._weighted_block_embedding(flat_z_A, counts_A, fracs_A)  # [B, d]
        z_B = self._weighted_block_embedding(flat_z_B, counts_B, fracs_B)  # [B, d]

        # For mix modes, fracA/fracB (block-level) are always 0.5/0.5
        batch_size = counts_A.size(0)
        half = flat_z_A.new_full((batch_size,), 0.5)
        return self._apply_mode(z_A, z_B, half, half, X_d)

    def _attention_fuse(
        self,
        embeddings: list[Tensor],
        fracs: list[Tensor] | None = None,
    ) -> Tensor:
        """Fuse monomer embeddings via learned attention, optionally with fraction prior.

        **attention** (``fracs=None``):
            ``α_i = softmax(s_i)``
        **fraction_aware_attention** (``fracs`` provided):
            ``α_i = softmax(s_i + log(f_i + ε))``

        The operation is permutation-invariant: re-ordering (embedding, fraction)
        pairs yields the same output.

        Parameters
        ----------
        embeddings : list[Tensor]
            Each element has shape ``[B, d]``.
        fracs : list[Tensor] | None
            Each element has shape ``[B, 1]``.  If ``None``, pure attention is used.

        Returns
        -------
        Tensor [B, d]
        """
        H = torch.stack(embeddings, dim=1)           # [B, N, d]
        scores = self.monomer_scorer(H).squeeze(-1)   # [B, N]

        if fracs is not None:
            F = torch.cat(fracs, dim=1)               # [B, N]
            scores = scores + torch.log(F + 1e-8)

        alpha = torch.softmax(scores, dim=1)          # [B, N]
        z = (alpha.unsqueeze(-1) * H).sum(dim=1)      # [B, d]
        return z

    def _pairwise_fixed(
        self,
        embeddings: list[Tensor],
        fracs: list[Tensor],
    ) -> Tensor:
        """Frac-attn h_mix + fraction-product–weighted pairwise interaction.

        ``h_int = Σ_{i<j} (f_i · f_j) · φ_ij``  where
        ``φ_ij = MLP_pair([h_i+h_j, h_i*h_j, |h_i−h_j|])``.
        Returns ``fuse(h_mix, h_int)``.
        """
        h_mix = self._attention_fuse(embeddings, fracs)

        H = torch.stack(embeddings, dim=1)           # [B, N, d]
        F = torch.cat(fracs, dim=1)                   # [B, N]
        B, N, d = H.shape

        h_int = H.new_zeros(B, d)
        for i in range(N):
            for j in range(i + 1, N):
                pair_feat = torch.cat([
                    H[:, i] + H[:, j],
                    H[:, i] * H[:, j],
                    (H[:, i] - H[:, j]).abs(),
                ], dim=1)                             # [B, 3d]
                phi_ij = self.pair_mlp(pair_feat)      # [B, d]
                w_ij = (F[:, i] * F[:, j]).unsqueeze(1)  # [B, 1]
                h_int = h_int + w_ij * phi_ij

        return self.fuse(h_mix, h_int)

    def _pairwise_attention(
        self,
        embeddings: list[Tensor],
        fracs: list[Tensor],
    ) -> Tensor:
        """Frac-attn h_mix + attention-weighted pairwise interaction.

        ``β_ij = softmax(t_ij + log(f_i·f_j + ε))`` over all pairs ``i<j``.
        ``h_int = Σ_{i<j} β_ij · φ_ij``.
        Returns ``fuse(h_mix, h_int)``.
        """
        h_mix = self._attention_fuse(embeddings, fracs)

        H = torch.stack(embeddings, dim=1)           # [B, N, d]
        F = torch.cat(fracs, dim=1)                   # [B, N]
        B, N, d = H.shape

        pair_phis = []
        pair_logits = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_feat = torch.cat([
                    H[:, i] + H[:, j],
                    H[:, i] * H[:, j],
                    (H[:, i] - H[:, j]).abs(),
                ], dim=1)                              # [B, 3d]
                phi_ij = self.pair_mlp(pair_feat)       # [B, d]
                t_ij = self.pair_scorer(pair_feat).squeeze(-1)  # [B]
                f_ij = F[:, i] * F[:, j]
                pair_phis.append(phi_ij)
                pair_logits.append(t_ij + torch.log(f_ij + 1e-8))

        if not pair_phis:
            return h_mix

        Phi = torch.stack(pair_phis, dim=1)            # [B, P, d]
        logits = torch.stack(pair_logits, dim=1)       # [B, P]
        beta = torch.softmax(logits, dim=1)            # [B, P]
        h_int = (beta.unsqueeze(-1) * Phi).sum(dim=1)  # [B, d]

        return self.fuse(h_mix, h_int)

    def _mixture_pairwise_fixed(
        self,
        embeddings: list[Tensor],
        fracs: list[Tensor],
    ) -> Tensor:
        """Mixture h_mix + fraction-product–weighted pairwise interaction.

        ``h_mix = Σ f_i · h_i``  (simple mixture),
        ``h_int = Σ_{i<j} (f_i · f_j) · φ_ij``.
        Returns ``fuse(h_mix, h_int)``.
        """
        H = torch.stack(embeddings, dim=1)           # [B, N, d]
        F = torch.cat(fracs, dim=1)                   # [B, N]
        B, N, d = H.shape

        # Simple mixture: h_mix = Σ f_i · h_i
        h_mix = (F.unsqueeze(-1) * H).sum(dim=1)     # [B, d]

        h_int = H.new_zeros(B, d)
        for i in range(N):
            for j in range(i + 1, N):
                pair_feat = torch.cat([
                    H[:, i] + H[:, j],
                    H[:, i] * H[:, j],
                    (H[:, i] - H[:, j]).abs(),
                ], dim=1)                             # [B, 3d]
                phi_ij = self.pair_mlp(pair_feat)      # [B, d]
                w_ij = (F[:, i] * F[:, j]).unsqueeze(1)  # [B, 1]
                h_int = h_int + w_ij * phi_ij

        return self.fuse(h_mix, h_int)

    def _mixture_pairwise_attention(
        self,
        embeddings: list[Tensor],
        fracs: list[Tensor],
    ) -> Tensor:
        """Mixture h_mix + attention-weighted pairwise interaction.

        ``h_mix = Σ f_i · h_i``  (simple mixture),
        ``β_ij = softmax(t_ij + log(f_i·f_j + ε))`` over all pairs ``i<j``,
        ``h_int = Σ_{i<j} β_ij · φ_ij``.
        Returns ``fuse(h_mix, h_int)``.
        """
        H = torch.stack(embeddings, dim=1)           # [B, N, d]
        F = torch.cat(fracs, dim=1)                   # [B, N]
        B, N, d = H.shape

        # Simple mixture: h_mix = Σ f_i · h_i
        h_mix = (F.unsqueeze(-1) * H).sum(dim=1)     # [B, d]

        pair_phis = []
        pair_logits = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_feat = torch.cat([
                    H[:, i] + H[:, j],
                    H[:, i] * H[:, j],
                    (H[:, i] - H[:, j]).abs(),
                ], dim=1)                              # [B, 3d]
                phi_ij = self.pair_mlp(pair_feat)       # [B, d]
                t_ij = self.pair_scorer(pair_feat).squeeze(-1)  # [B]
                f_ij = F[:, i] * F[:, j]
                pair_phis.append(phi_ij)
                pair_logits.append(t_ij + torch.log(f_ij + 1e-8))

        if not pair_phis:
            return h_mix

        Phi = torch.stack(pair_phis, dim=1)            # [B, P, d]
        logits = torch.stack(pair_logits, dim=1)       # [B, P]
        beta = torch.softmax(logits, dim=1)            # [B, P]
        h_int = (beta.unsqueeze(-1) * Phi).sum(dim=1)  # [B, d]

        return self.fuse(h_mix, h_int)

    def _apply_mode(
        self,
        z_A: Tensor,
        z_B: Tensor,
        fracA: Tensor,
        fracB: Tensor,
        X_d: Tensor | None = None,
    ) -> Tensor:
        """Apply the copolymer integration mode to block-level embeddings."""

        # Keep fracA/fracB as column vectors for broadcasting
        fA = fracA.unsqueeze(1) if fracA.ndim == 1 else fracA  # [B, 1]
        fB = fracB.unsqueeze(1) if fracB.ndim == 1 else fracB  # [B, 1]

        mode = self.copolymer_mode

        # --- Mean family: z = (z_A + z_B) / 2 (no fraction weighting) ---
        if mode.startswith("mean"):
            z = (z_A + z_B) / 2  # [B, d]
            if mode == "mean":
                parts = [z]
            elif mode == "mean_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown mean sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Mix + pairwise attention: h = h_mix_mixture + Σ β_ij·φ_ij ---
        # (checked before mix_pair because of startswith ordering)
        if mode.startswith("mix_pair_attn"):
            z = self._mixture_pairwise_attention([z_A, z_B], fracs=[fA, fB])
            if mode == "mix_pair_attn":
                parts = [z]
            elif mode == "mix_pair_attn_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown mix_pair_attn sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Mix + pairwise fixed: h = h_mix_mixture + Σ (f_i·f_j)·φ_ij ---
        if mode.startswith("mix_pair"):
            z = self._mixture_pairwise_fixed([z_A, z_B], fracs=[fA, fB])
            if mode == "mix_pair":
                parts = [z]
            elif mode == "mix_pair_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown mix_pair sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Mix family: z = fracA * z_A + fracB * z_B ---
        if mode.startswith("mix"):
            z = fA * z_A + fB * z_B  # [B, d]
            if mode == "mix":
                parts = [z]
            elif mode == "mix_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            elif mode == "mix_frac":
                parts = [z, fA, fB]
            elif mode == "mix_frac_meta":
                parts = [z, fA, fB]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown mix sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Attention family: α_i = softmax(s_i), z = Σ α_i z_i ---
        # Pure learned attention — no fraction prior. Useful as an ablation
        # baseline for fraction_aware_attention.
        if mode.startswith("attention"):
            z = self._attention_fuse([z_A, z_B], fracs=None)
            if mode == "attention":
                parts = [z]
            elif mode == "attention_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown attention sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Frac-attn + pairwise attention: h = h_mix + Σ β_ij·φ_ij ---
        # (checked before frac_attn_pair because of startswith ordering)
        if mode.startswith("frac_attn_pair_attn"):
            z = self._pairwise_attention([z_A, z_B], fracs=[fA, fB])
            if mode == "frac_attn_pair_attn":
                parts = [z]
            elif mode == "frac_attn_pair_attn_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown frac_attn_pair_attn sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Frac-attn + pairwise fixed: h = h_mix + Σ (f_i·f_j)·φ_ij ---
        if mode.startswith("frac_attn_pair"):
            z = self._pairwise_fixed([z_A, z_B], fracs=[fA, fB])
            if mode == "frac_attn_pair":
                parts = [z]
            elif mode == "frac_attn_pair_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown frac_attn_pair sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Fraction-aware attention family: α_i = softmax(s_i + log(f_i + ε)) ---
        # Fractions act as a soft prior on the attention weights. When the
        # learned scores are zero-initialised, the initial weights equal the
        # monomer fractions, and the network can subsequently refine them.
        if mode.startswith("frac_attn"):
            z = self._attention_fuse([z_A, z_B], fracs=[fA, fB])
            if mode == "frac_attn":
                parts = [z]
            elif mode == "frac_attn_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown frac_attn sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Self-attention family: transformer-style self-attention over monomers ---
        if mode.startswith("self_attn"):
            H = torch.stack([z_A, z_B], dim=1)             # [B, 2, d]
            F_cat = torch.cat([fA, fB], dim=1)              # [B, 2]
            z = self.self_attn_pool(H, F_cat)               # [B, d]
            if mode == "self_attn":
                parts = [z]
            elif mode == "self_attn_meta":
                parts = [z]
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown self_attn sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        # --- Interact family ---
        if mode.startswith("interact"):
            parts = [
                z_A,
                z_B,
                (z_A - z_B).abs(),
                z_A * z_B,
                fA,
                fB,
            ]
            if mode == "interact":
                pass  # base interact, no meta
            elif mode == "interact_meta":
                if X_d is not None:
                    parts.append(self.X_d_transform(X_d))
            else:
                raise ValueError(f"Unknown interact sub-mode: {mode}")
            return torch.cat(parts, dim=1)

        raise ValueError(f"Unknown copolymer_mode: {mode}")

    def fingerprint_components(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracA: Tensor,
        fracB: Tensor,
        X_d: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return individual embedding components for export."""
        z_A = self._encode_single(bmg_A)
        z_B = self._encode_single(bmg_B)
        z_final = self.fingerprint(bmg_A, bmg_B, fracA, fracB, X_d)
        return {"z_A": z_A, "z_B": z_B, "z_final": z_final}

    def fingerprint_components_multi_monomer(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracs_A: Tensor,
        fracs_B: Tensor,
        counts_A: Tensor,
        counts_B: Tensor,
        X_d: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return individual embedding components for multi-monomer export."""
        flat_z_A = self._encode_single(bmg_A)
        flat_z_B = self._encode_single(bmg_B)
        z_A = self._weighted_block_embedding(flat_z_A, counts_A, fracs_A)
        z_B = self._weighted_block_embedding(flat_z_B, counts_B, fracs_B)
        batch_size = counts_A.size(0)
        half = flat_z_A.new_full((batch_size,), 0.5)
        z_final = self._apply_mode(z_A, z_B, half, half, X_d)
        return {"z_A": z_A, "z_B": z_B, "z_final": z_final}

    def forward(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracA: Tensor,
        fracB: Tensor,
        X_d: Tensor | None = None,
    ) -> Tensor:
        return self.predictor(self.fingerprint(bmg_A, bmg_B, fracA, fracB, X_d))

    def forward_multi_monomer(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracs_A: Tensor,
        fracs_B: Tensor,
        counts_A: Tensor,
        counts_B: Tensor,
        X_d: Tensor | None = None,
    ) -> Tensor:
        return self.predictor(
            self.fingerprint_multi_monomer(bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d)
        )

    # ------------------------------------------------------------------ batch dispatch helpers
    def _is_multi_monomer_batch(self, batch) -> bool:
        """Detect whether batch is a MultiMonomerCopolymerBatch (11 fields) vs CopolymerTrainingBatch (9 fields)."""
        return len(batch) == 11

    def _unpack_batch(self, batch):
        """Unpack batch and return (Z, targets, weights, lt_mask, gt_mask, batch_size)."""
        if self._is_multi_monomer_batch(batch):
            bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d, targets, weights, lt_mask, gt_mask = batch
            Z = self.fingerprint_multi_monomer(bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d)
            batch_size = counts_A.size(0)
        else:
            bmg_A, bmg_B, fracA, fracB, X_d, targets, weights, lt_mask, gt_mask = batch
            Z = self.fingerprint(bmg_A, bmg_B, fracA, fracB, X_d)
            batch_size = len(fracA)
        return Z, targets, weights, lt_mask, gt_mask, batch_size

    def _unpack_batch_for_pred(self, batch):
        """Unpack batch and return predictions (for val/test/predict)."""
        if self._is_multi_monomer_batch(batch):
            bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d, targets, weights, lt_mask, gt_mask = batch
            preds = self.forward_multi_monomer(bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d)
            batch_size = counts_A.size(0)
        else:
            bmg_A, bmg_B, fracA, fracB, X_d, targets, weights, lt_mask, gt_mask = batch
            preds = self(bmg_A, bmg_B, fracA, fracB, X_d)
            batch_size = len(fracA)
        return preds, targets, weights, lt_mask, gt_mask, batch_size

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_idx):
        Z, targets, weights, lt_mask, gt_mask, batch_size = self._unpack_batch(batch)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True,
                 on_epoch=True, on_step=False, sync_dist=True)
        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        if hasattr(self.message_passing, 'V_d_transform'):
            self.message_passing.V_d_transform.train()
        if hasattr(self.message_passing, 'graph_transform'):
            self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        if hasattr(self.predictor, 'output_transform'):
            self.predictor.output_transform.train()

    def validation_step(self, batch, batch_idx: int = 0):
        preds, targets, weights, lt_mask, gt_mask, batch_size = self._unpack_batch_for_pred(batch)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        weights_ones = torch.ones_like(weights)

        # Log all non-loss metrics
        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights_ones, lt_mask, gt_mask)
            metric_alias = getattr(m, 'alias', None) or getattr(type(m), 'alias', None)
            if metric_alias is not None:
                self.log(f"val/{metric_alias}", m, batch_size=batch_size,
                         logger=True, prog_bar=False)

        # Log val_loss using train_step (for scaled loss computation)
        Z, _, _, _, _, _ = self._unpack_batch(batch)
        preds_train = self.predictor.train_step(Z)
        self.metrics[-1](preds_train, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True,
                 on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch, label: str) -> None:
        preds, targets, weights, lt_mask, gt_mask, batch_size = self._unpack_batch_for_pred(batch)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        weights = torch.ones_like(weights)

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            metric_alias = getattr(m, 'alias', None) or getattr(type(m), 'alias', None)
            if metric_alias is not None:
                self.log(f"{label}/{metric_alias}", m, batch_size=batch_size,
                         logger=True, prog_bar=False)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        if self._is_multi_monomer_batch(batch):
            bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d, *_ = batch
            return self.forward_multi_monomer(bmg_A, bmg_B, fracs_A, fracs_B, counts_A, counts_B, X_d)
        bmg_A, bmg_B, fracA, fracB, X_d, *_ = batch
        return self(bmg_A, bmg_B, fracA, fracB, X_d)

    # ------------------------------------------------------------------ optimizer
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.init_lr)
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
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}

    def get_batch_size(self, batch) -> int:
        if self._is_multi_monomer_batch(batch):
            return batch[4].size(0)  # counts_A tensor length = batch size
        return len(batch[2])  # fracA tensor length = batch size
