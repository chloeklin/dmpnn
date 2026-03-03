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


class CopolymerMPNN(pl.LightningModule):
    """Two-monomer copolymer model with shared GNN encoder.

    Encodes monomer A and monomer B independently through a **shared** message-passing
    encoder + aggregation, then combines their graph-level embeddings using one of
    five integration modes:

    **Mix family** (fraction-weighted mean-field embedding):

    * **mix**: ``z = fracA * z_A + fracB * z_B`` → head input: ``z``
    * **mix_meta**: same ``z`` → head input: ``[z || meta]``
    * **mix_frac**: same ``z`` → head input: ``[z || fracA || fracB]``
    * **mix_frac_meta**: same ``z`` → head input: ``[z || fracA || fracB || meta]``

    **Interact family** (interaction-aware concatenation):

    * **interact**: head input: ``[z_A || z_B || |z_A-z_B| || z_A⊙z_B || fracA || fracB]``
    * **interact_meta**: head input: ``[z_A || z_B || |z_A-z_B| || z_A⊙z_B || fracA || fracB || meta]``

    where ``meta`` = additional scalar descriptors (e.g. RDKit, dataset-specific).
    """

    VALID_MODES = ("mix", "mix_meta", "mix_frac", "mix_frac_meta", "interact", "interact_meta")

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        copolymer_mode: str = "mix",
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
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
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

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

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

        # Keep fracA/fracB as column vectors for broadcasting
        fA = fracA.unsqueeze(1)  # [B, 1]
        fB = fracB.unsqueeze(1)  # [B, 1]

        mode = self.copolymer_mode

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

    def forward(
        self,
        bmg_A: BatchMolGraph,
        bmg_B: BatchMolGraph,
        fracA: Tensor,
        fracB: Tensor,
        X_d: Tensor | None = None,
    ) -> Tensor:
        return self.predictor(self.fingerprint(bmg_A, bmg_B, fracA, fracB, X_d))

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_idx):
        bmg_A, bmg_B, fracA, fracB, X_d, targets, weights, lt_mask, gt_mask = batch
        batch_size = len(fracA)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg_A, bmg_B, fracA, fracB, X_d)
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
        bmg_A, bmg_B, fracA, fracB, X_d, targets, weights, lt_mask, gt_mask = batch
        batch_size = len(fracA)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        # Compute predictions for metric logging
        preds = self(bmg_A, bmg_B, fracA, fracB, X_d)
        weights_ones = torch.ones_like(weights)

        # Log all non-loss metrics
        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights_ones, lt_mask, gt_mask)
            metric_alias = getattr(m, 'alias', None) or getattr(type(m), 'alias', None)
            if metric_alias is not None:
                self.log(f"val/{metric_alias}", m, batch_size=batch_size,
                         logger=True, prog_bar=False)

        # Log val_loss using train_step (for scaled loss computation)
        Z = self.fingerprint(bmg_A, bmg_B, fracA, fracB, X_d)
        preds_train = self.predictor.train_step(Z)
        self.metrics[-1](preds_train, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True,
                 on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch, label: str) -> None:
        bmg_A, bmg_B, fracA, fracB, X_d, targets, weights, lt_mask, gt_mask = batch
        batch_size = len(fracA)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self(bmg_A, bmg_B, fracA, fracB, X_d)
        weights = torch.ones_like(weights)

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            metric_alias = getattr(m, 'alias', None) or getattr(type(m), 'alias', None)
            if metric_alias is not None:
                self.log(f"{label}/{metric_alias}", m, batch_size=batch_size,
                         logger=True, prog_bar=False)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
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
        return len(batch[2])  # fracA tensor length = batch size
