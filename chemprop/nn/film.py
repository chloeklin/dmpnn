"""FiLM (Feature-wise Linear Modulation) conditioning for GNN message passing.

Integrates global descriptors (e.g., DoP, Molality, Density) into the GNN encoder
by modulating hidden representations at each message passing layer.

Reference:
    Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" AAAI 2018
    https://arxiv.org/abs/1709.07871

Usage:
    conditioner = FilmConditioner(d_descriptor=10, d_hidden=300, n_layers=3)
    # During message passing at layer l:
    H = conditioner(H, X_d, layer_idx=l, graph_ids=bmg.batch[bmg.edge_index[0]])
"""

import torch
import torch.nn as nn
from torch import Tensor


class FilmConditioner(nn.Module):
    """Per-layer FiLM conditioning that modulates hidden states using global descriptors.

    For each message passing layer l, computes:
        (gamma_l, beta_l) = MLP_l(d)
        h_l = (1 + tanh(gamma_l)) * h_l + beta_l

    where d is the standardized descriptor vector [B, D] and h_l is the hidden
    representation [N, H] (N = total nodes or edges in the batch).

    Architecture: shared trunk MLP + per-layer linear heads.
        trunk:  Linear(D -> film_hidden_dim), ReLU
        head_l: Linear(film_hidden_dim -> 2*H), split into gamma_l, beta_l

    Parameters
    ----------
    d_descriptor : int
        Dimension of the global descriptor vector (D).
    d_hidden : int
        Dimension of the hidden representation to modulate (H).
    n_layers : int
        Number of message passing layers that will use FiLM conditioning.
    film_hidden_dim : int | None
        Hidden dimension of the FiLM MLP trunk. Defaults to d_hidden if None.
    film_layers_mode : str
        Which layers to apply FiLM to: "all" or "last".
    """

    def __init__(
        self,
        d_descriptor: int,
        d_hidden: int,
        n_layers: int,
        film_hidden_dim: int | None = None,
        film_layers_mode: str = "all",
    ):
        super().__init__()
        self.d_descriptor = d_descriptor
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.film_hidden_dim = film_hidden_dim if film_hidden_dim is not None else d_hidden
        self.film_layers_mode = film_layers_mode

        # Shared trunk: Linear(D -> film_hidden_dim), ReLU
        self.trunk = nn.Sequential(
            nn.Linear(d_descriptor, self.film_hidden_dim),
            nn.ReLU(),
        )

        # Per-layer heads: Linear(film_hidden_dim -> 2*H) for gamma and beta
        if film_layers_mode == "all":
            n_heads = n_layers
        elif film_layers_mode == "last":
            n_heads = 1
        else:
            raise ValueError(f"film_layers_mode must be 'all' or 'last', got '{film_layers_mode}'")

        self.heads = nn.ModuleList([
            nn.Linear(self.film_hidden_dim, 2 * d_hidden) for _ in range(n_heads)
        ])

        # Initialize heads so that initial gamma ≈ 0 (i.e., scale ≈ 1) and beta ≈ 0
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _get_head_index(self, layer_idx: int) -> int | None:
        """Map a message passing layer index to the corresponding head index.

        Returns None if this layer should not be modulated.
        """
        if self.film_layers_mode == "all":
            if layer_idx < len(self.heads):
                return layer_idx
            return None
        elif self.film_layers_mode == "last":
            # Only modulate the last layer (layer_idx == n_layers - 1)
            if layer_idx == self.n_layers - 1:
                return 0
            return None
        return None

    def forward(
        self,
        H: Tensor,
        X_d: Tensor,
        layer_idx: int,
        graph_ids: Tensor,
    ) -> Tensor:
        """Apply FiLM modulation to hidden states.

        Parameters
        ----------
        H : Tensor
            Hidden representations of shape [N, d_hidden] where N is the total
            number of nodes or directed edges in the batch.
        X_d : Tensor
            Global descriptor vector of shape [B, D] where B is the number of
            graphs in the batch.
        layer_idx : int
            Current message passing layer index (0-indexed).
        graph_ids : Tensor
            Tensor of shape [N] mapping each node/edge to its graph index in [0, B).

        Returns
        -------
        Tensor
            Modulated hidden representations of shape [N, d_hidden].
        """
        head_idx = self._get_head_index(layer_idx)
        if head_idx is None:
            return H

        # Shared trunk: [B, D] -> [B, film_hidden_dim]
        trunk_out = self.trunk(X_d)

        # Per-layer head: [B, film_hidden_dim] -> [B, 2*H]
        params = self.heads[head_idx](trunk_out)

        # Split into gamma and beta: each [B, H]
        gamma, beta = params.chunk(2, dim=-1)

        # Broadcast to per-node/edge: [N, H]
        gamma_expanded = gamma[graph_ids]
        beta_expanded = beta[graph_ids]

        # FiLM modulation: h = (1 + tanh(gamma)) * h + beta
        H = (1.0 + torch.tanh(gamma_expanded)) * H + beta_expanded

        return H
