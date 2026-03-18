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
    ):
        super().__init__()
        self.save_hyperparameters()

        # Message passing
        self.message_passing = HPGMessagePassing(
            d_v=d_v, d_h=d_h, d_e=1, depth=depth,
            num_heads=num_heads, dropout=dropout_mp,
        )

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
    #  Forward
    # ------------------------------------------------------------------

    def fingerprint(self, bmg: BatchHPGMolGraph, X_d: Tensor | None = None) -> Tensor:
        """Compute graph-level fingerprint.

        Returns
        -------
        Tensor [B, d_ffn] or [B, d_ffn + d_xd]
        """
        H = self.message_passing(bmg)  # [N, d_h]

        # Sum-pooling over nodes per graph (matching original dgl.sum_nodes)
        B = int(bmg.batch.max().item()) + 1 if bmg.batch.numel() else 1
        idx = bmg.batch.unsqueeze(-1).expand(-1, H.size(1))
        H_graph = torch.zeros(B, H.size(1), dtype=H.dtype, device=H.device)
        H_graph.scatter_add_(0, idx, H)

        H_graph = F.leaky_relu(self.linear_pool(H_graph))  # [B, d_ffn]

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
