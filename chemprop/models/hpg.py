"""HPG-GAT model implemented as a PyTorch Lightning module.

Follows chemprop's MPNN conventions (training_step, validation_step, etc.)
so it can be used with the same Trainer infrastructure.

Architecture (matching the original HPG-GAT paper):
  1. HPGMessagePassing: stack of edge-aware GAT layers on the hierarchical graph
  2. Sum-pooling over all nodes → graph-level embedding [B, d_h]
  3. Linear projection d_h → d_ffn
  4. Optional concatenation of scalar features (X_d)
  5. FFN → prediction
"""

from __future__ import annotations

import logging
from typing import Iterable

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from chemprop.data.hpg import BatchHPGMolGraph
from chemprop.nn.hpg import HPGMessagePassing
from chemprop.nn.metrics import ChempropMetric
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

VALID_HPG_VARIANTS = ("HPG_baseline", "HPG_frac", "HPG_frac_polytype")


class HPGMPNN(pl.LightningModule):
    """Hierarchical Polymer Graph model with GAT message passing.

    Parameters
    ----------
    d_v : int
        Input node feature dimension (must match featurizer output).
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
    hpg_variant : str
        One of ``VALID_HPG_VARIANTS``:
        - ``"HPG_baseline"``: sum-pool over all nodes (original HPG).
        - ``"HPG_frac"``: fraction-weighted pool over fragment nodes.
        - ``"HPG_frac_polytype"``: fraction-weighted pool + polytype via X_d.
    """

    def __init__(
        self,
        d_v: int = 49,
        d_h: int = 128,
        d_ffn: int = 64,
        depth: int = 6,
        num_heads: int = 8,
        dropout_mp: float = 0.0,
        dropout_ffn: float = 0.2,
        n_tasks: int = 1,
        d_xd: int = 0,
        task_type: str = "regression",
        metrics: Iterable[ChempropMetric] | None = None,
        criterion: ChempropMetric | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        hpg_variant: str = "HPG_baseline",
    ):
        super().__init__()
        if hpg_variant not in VALID_HPG_VARIANTS:
            raise ValueError(
                f"Unknown hpg_variant={hpg_variant!r}. "
                f"Valid: {VALID_HPG_VARIANTS}"
            )
        self.save_hyperparameters()
        self.hpg_variant = hpg_variant

        # ── 1. Encoder: GAT message passing ──
        self.message_passing = HPGMessagePassing(
            d_v=d_v, d_h=d_h, d_e=1, depth=depth,
            num_heads=num_heads, dropout=dropout_mp,
        )

        # ── 2. Polymer pooling → linear projection ──
        self.linear_pool = nn.Linear(d_h, d_ffn)

        # ── 3. Optional metadata concatenation ──
        # d_xd covers any global scalars (e.g. polytype one-hot for
        # HPG_frac_polytype, or legacy [fracA, fracB] for old HPG+incl_desc).
        if d_xd > 0:
            self.xd_transform = nn.Identity()
        else:
            self.xd_transform = None

        # ── 4. Predictor: FFN ──
        ffn_in = d_ffn + d_xd
        self.ffn = nn.Sequential(
            nn.Linear(ffn_in, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(512, n_tasks),
        )

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
    #  Polymer-level pooling (modular – extend here for Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _standard_pooling(H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Sum-pool over **all** nodes per graph (original HPG readout).

        Parameters
        ----------
        H : Tensor [N, d_h]
            Node embeddings after message passing.
        bmg : BatchHPGMolGraph

        Returns
        -------
        Tensor [B, d_h]
        """
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        idx = bmg.batch.unsqueeze(-1).expand(-1, H.size(1))
        H_graph = torch.zeros(B, H.size(1), dtype=H.dtype, device=H.device)
        H_graph.scatter_add_(0, idx, H)
        return H_graph

    @staticmethod
    def _fraction_weighted_pooling(H: Tensor, bmg: BatchHPGMolGraph) -> Tensor:
        """Fraction-weighted pool over **fragment nodes** only.

        Computes  h_poly = sum_i  f_i * h_i'  per polymer, where h_i' are
        the polymer-contextualized fragment (monomer) embeddings and f_i are
        the corresponding monomer fractions.

        Parameters
        ----------
        H : Tensor [N, d_h]
            Node embeddings after message passing (all nodes).
        bmg : BatchHPGMolGraph
            Must have ``frag_fracs`` populated (shape [N_frag_total]).

        Returns
        -------
        Tensor [B, d_h]
        """
        assert bmg.frag_fracs is not None, (
            "frag_fracs must be provided for fraction-weighted pooling "
            "(HPG_frac / HPG_frac_polytype)"
        )
        frag_fracs = bmg.frag_fracs  # [N_frag_total]

        # Lightweight sanity checks (training-time only via assert)
        assert (frag_fracs >= 0).all(), "Fractions must be non-negative"

        # Select fragment-node embeddings and their batch indices
        H_frag = H[bmg.frag_mask]          # [N_frag_total, d_h]
        batch_frag = bmg.batch[bmg.frag_mask]  # [N_frag_total]

        # Weight each fragment embedding by its fraction
        H_weighted = frag_fracs.unsqueeze(-1) * H_frag  # [N_frag_total, d_h]

        # Scatter-add per graph
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        H_graph = torch.zeros(B, H.size(1), dtype=H.dtype, device=H.device)
        idx = batch_frag.unsqueeze(-1).expand_as(H_weighted)
        H_graph.scatter_add_(0, idx, H_weighted)
        return H_graph

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def fingerprint(self, bmg: BatchHPGMolGraph, X_d: Tensor | None = None) -> Tensor:
        """Compute graph-level fingerprint.

        Pipeline:
          1. Encoder  → node embeddings H  [N, d_h]
          2. Pooling  → graph embedding    [B, d_h]
          3. Project  → [B, d_ffn]
          4. Concat X_d (polytype / legacy descriptors) if present

        Returns
        -------
        Tensor [B, d_ffn] or [B, d_ffn + d_xd]
        """
        # ── 1. Encoder ──
        H = self.message_passing(bmg)  # [N, d_h]

        # ── 2. Polymer pooling ──
        if self.hpg_variant in ("HPG_frac", "HPG_frac_polytype"):
            H_graph = self._fraction_weighted_pooling(H, bmg)
        else:
            H_graph = self._standard_pooling(H, bmg)

        # ── 3. Linear projection ──
        H_graph = F.leaky_relu(self.linear_pool(H_graph))  # [B, d_ffn]

        # ── 4. Optional metadata concatenation ──
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
