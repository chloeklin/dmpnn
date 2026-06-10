"""Stage 2D: Architecture-aware polymer aggregation.

Implements architecture-conditioned residual corrections on top of the
fraction-weighted mixture baseline (h_mix = f_A * h_A + f_B * h_B).

Variants
--------
frac          : h_poly = h_mix  (pure baseline, no architecture signal)
2d0_fixed     : h_poly = h_mix + alpha * e_arch            (fixed scalar alpha)
2d0_arch      : h_poly = h_mix + alpha_arch * e_arch       (per-architecture alpha)
2d0_gate      : h_poly = h_mix + gate(h_mix, e_arch, f) * e_arch  (gated alpha)
2d1_fixed     : h_poly = h_mix + alpha * r_arch            (fixed alpha, MLP residual)
2d1_arch      : h_poly = h_mix + alpha_arch * r_arch       (per-arch alpha, MLP residual)
2d1_gate      : h_poly = h_mix + gate(z) * r_arch          (gated alpha, MLP residual)

Architecture labels:  alternating → 0,  random → 1,  block → 2
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Canonical mapping from architecture string to integer index.
ARCH_LABEL_MAP: Dict[str, int] = {"alternating": 0, "random": 1, "block": 2}
NUM_ARCHITECTURES = 3

VALID_STAGE2_VARIANTS = (
    "frac",
    "2d0_fixed",
    "2d0_arch",
    "2d0_gate",
    "2d1_fixed",
    "2d1_arch",
    "2d1_gate",
)


class Stage2Aggregator(nn.Module):
    """Architecture-aware Stage 2D aggregator.

    Computes h_poly from monomer embeddings (h_A, h_B), fractions (f_A, f_B),
    and architecture label, then predicts EA and IP through separate MLP heads.

    Parameters
    ----------
    d : int
        Monomer embedding dimension (output of Stage 1 GNN encoder).
    variant : str
        One of :data:`VALID_STAGE2_VARIANTS`.
    arch_emb_dim : int | None
        Architecture embedding dimension. Defaults to ``d`` if None.
    hidden_dim : int
        Hidden dimension for MLP layers (gate MLP, 2D-1 interaction MLP).
    dropout : float
        Dropout probability in MLPs.
    alpha_init : float
        Initial value for the learnable alpha scalar(s). Default 0.1 breaks
        the zero-gradient symmetry with zero-initialized residual MLPs.
    n_targets : int
        Number of prediction targets (typically 2 for EA + IP).
    """

    def __init__(
        self,
        d: int,
        variant: str = "frac",
        arch_emb_dim: int | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
        n_targets: int = 2,
    ):
        super().__init__()
        if variant not in VALID_STAGE2_VARIANTS:
            raise ValueError(
                f"Invalid stage2_variant={variant!r}. "
                f"Choose from {VALID_STAGE2_VARIANTS}"
            )

        self.d = d
        self.variant = variant
        self.arch_emb_dim = arch_emb_dim if arch_emb_dim is not None else d
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.alpha_init_val = alpha_init
        self.n_targets = n_targets

        # ── Architecture embedding (shared across 2D-0 and 2D-1 variants) ──
        if variant != "frac":
            self.arch_embedding = nn.Embedding(NUM_ARCHITECTURES, self.arch_emb_dim)
            nn.init.normal_(self.arch_embedding.weight, mean=0.0, std=0.02)
        else:
            self.arch_embedding = None

        # ── Projection: if arch_emb_dim != d, project embedding to d ──
        if variant != "frac" and self.arch_emb_dim != d:
            self.emb_proj = nn.Linear(self.arch_emb_dim, d, bias=False)
        else:
            self.emb_proj = None

        # ── Alpha parameters ──
        if variant in ("2d0_fixed", "2d1_fixed"):
            # Single learnable scalar alpha
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        elif variant in ("2d0_arch", "2d1_arch"):
            # Per-architecture alpha: [NUM_ARCHITECTURES]
            self.alpha = nn.Parameter(
                torch.full((NUM_ARCHITECTURES,), alpha_init)
            )
        else:
            self.alpha = None

        # ── Gate MLP (for gated variants) ──
        if variant == "2d0_gate":
            # Input: h_mix [d] + e_arch [d] + f_A [1] + f_B [1]
            gate_in = d + d + 2
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            # Bias gate to output ~0 at init (sigmoid(−3) ≈ 0.047)
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, -3.0)
        elif variant == "2d1_gate":
            # Input: z (same as MLP_2D1 input) → gate
            # z = [h_A, h_B, |h_A-h_B|, h_A*h_B, f_A, f_B, e_arch]
            z_dim = 4 * d + 2 + d  # last d is projected arch embedding
            self.gate_mlp = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, -3.0)
        else:
            self.gate_mlp = None

        # ── 2D-1 interaction MLP ──
        if variant.startswith("2d1"):
            # Input z = [h_A, h_B, |h_A-h_B|, h_A*h_B, f_A, f_B, e_arch]
            z_dim = 4 * d + 2 + d  # last d is projected arch embedding
            self.mlp_2d1 = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d),
            )
            # Zero-init output layer → r_arch = 0 at start
            nn.init.zeros_(self.mlp_2d1[-1].weight)
            nn.init.zeros_(self.mlp_2d1[-1].bias)
        else:
            self.mlp_2d1 = None

        # ── Prediction heads: separate MLP for each target ──
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_targets)
        ])

    def _get_arch_emb(self, arch: Tensor) -> Tensor:
        """Look up architecture embedding and project if needed.

        Parameters
        ----------
        arch : Tensor [B] (long)
            Architecture index per sample.

        Returns
        -------
        Tensor [B, d]
        """
        e = self.arch_embedding(arch)  # [B, arch_emb_dim]
        if self.emb_proj is not None:
            e = self.emb_proj(e)  # [B, d]
        return e

    def forward(
        self,
        h_A: Tensor,
        h_B: Tensor,
        f_A: Tensor,
        f_B: Tensor,
        arch: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute polymer embedding and predictions.

        Parameters
        ----------
        h_A : Tensor [B, d]
            Monomer A embedding from Stage 1.
        h_B : Tensor [B, d]
            Monomer B embedding from Stage 1.
        f_A : Tensor [B] or [B, 1]
            Fraction of monomer A.
        f_B : Tensor [B] or [B, 1]
            Fraction of monomer B.
        arch : Tensor [B] (long)
            Architecture label index (0=alt, 1=rand, 2=block).

        Returns
        -------
        preds : Tensor [B, n_targets]
            Predictions from the per-target heads.
        aux : dict[str, Tensor]
            Auxiliary diagnostics:
            - "h_poly": [B, d]
            - "alpha": scalar or [B] effective alpha values
            - "emb_norm": [B] architecture embedding L2 norms
        """
        # Ensure fractions are [B, 1]
        if f_A.ndim == 1:
            f_A = f_A.unsqueeze(1)
        if f_B.ndim == 1:
            f_B = f_B.unsqueeze(1)

        # ── h_mix: fraction-weighted mixture ──
        h_mix = f_A * h_A + f_B * h_B  # [B, d]

        aux: Dict[str, Tensor] = {}

        if self.variant == "frac":
            h_poly = h_mix
            aux["alpha"] = torch.zeros(1, device=h_mix.device)
            aux["emb_norm"] = torch.zeros(h_mix.size(0), device=h_mix.device)

        elif self.variant.startswith("2d0"):
            e_arch = self._get_arch_emb(arch)  # [B, d]
            aux["emb_norm"] = e_arch.detach().norm(dim=1)

            if self.variant == "2d0_fixed":
                alpha = self.alpha  # scalar
                h_poly = h_mix + alpha * e_arch
                aux["alpha"] = alpha.detach().expand(h_mix.size(0))

            elif self.variant == "2d0_arch":
                alpha = self.alpha[arch]  # [B]
                h_poly = h_mix + alpha.unsqueeze(1) * e_arch
                aux["alpha"] = alpha.detach()

            else:  # 2d0_gate
                gate_input = torch.cat([h_mix, e_arch, f_A, f_B], dim=1)
                alpha = torch.sigmoid(self.gate_mlp(gate_input))  # [B, 1]
                h_poly = h_mix + alpha * e_arch
                aux["alpha"] = alpha.detach().squeeze(1)

        else:  # 2d1_*
            e_arch = self._get_arch_emb(arch)  # [B, d]
            aux["emb_norm"] = e_arch.detach().norm(dim=1)

            # Build interaction feature z
            z = torch.cat([
                h_A,
                h_B,
                (h_A - h_B).abs(),
                h_A * h_B,
                f_A,
                f_B,
                e_arch,
            ], dim=1)  # [B, 4d + 2 + d]

            r_arch = self.mlp_2d1(z)  # [B, d]

            if self.variant == "2d1_fixed":
                alpha = self.alpha  # scalar
                h_poly = h_mix + alpha * r_arch
                aux["alpha"] = alpha.detach().expand(h_mix.size(0))

            elif self.variant == "2d1_arch":
                alpha = self.alpha[arch]  # [B]
                h_poly = h_mix + alpha.unsqueeze(1) * r_arch
                aux["alpha"] = alpha.detach()

            else:  # 2d1_gate
                alpha = torch.sigmoid(self.gate_mlp(z))  # [B, 1]
                h_poly = h_mix + alpha * r_arch
                aux["alpha"] = alpha.detach().squeeze(1)

        aux["h_poly"] = h_poly.detach()

        # ── Per-target prediction heads ──
        preds = torch.cat([head(h_poly) for head in self.heads], dim=1)  # [B, n_targets]

        return preds, aux

    def log_diagnostics(self, aux: Dict[str, Tensor], prefix: str = "stage2d") -> Dict[str, float]:
        """Extract scalar diagnostics for logging.

        Parameters
        ----------
        aux : dict
            Output from forward().
        prefix : str
            Logging prefix.

        Returns
        -------
        dict[str, float]
            Scalars suitable for logger.log_metrics().
        """
        metrics = {}
        if "alpha" in aux:
            alpha_val = aux["alpha"]
            if alpha_val.numel() == 1:
                metrics[f"{prefix}/alpha_mean"] = alpha_val.item()
            else:
                metrics[f"{prefix}/alpha_mean"] = alpha_val.mean().item()
                metrics[f"{prefix}/alpha_std"] = alpha_val.std().item()
        if "emb_norm" in aux:
            metrics[f"{prefix}/emb_norm_mean"] = aux["emb_norm"].mean().item()
        return metrics
