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

    **Mean family** (unweighted average):

    * **mean**: ``z = (z_A + z_B) / 2`` → head input: ``z``
    * **mean_meta**: same ``z`` → head input: ``[z || meta]``

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

    VALID_MODES = ("mean", "mean_meta", "mix", "mix_meta", "mix_frac", "mix_frac_meta", "interact", "interact_meta")

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
