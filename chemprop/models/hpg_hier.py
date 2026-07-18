from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import Tensor, nn

from chemprop.data.hpg_hier import BatchTwoStageHPG
from chemprop.nn.message_passing.mol_atom_bond import MABBondMessagePassing


def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    result = values.new_zeros((size, values.size(-1)))
    result.scatter_add_(0, index.unsqueeze(-1).expand_as(values), values)
    return result


class Stage2Layer(nn.Module):
    def __init__(self, d_h: int, d_e: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(d_h + d_e, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.update = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        source, target = edge_index
        messages = self.message(torch.cat([h[source], edge_features], dim=-1))
        aggregate = _scatter_sum(messages, target, h.size(0))
        return self.norm(h + self.update(torch.cat([h, aggregate], dim=-1)))


class HPGHierMPNN(pl.LightningModule):
    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        d_h: int = 128,
        stage1_depth: int = 4,
        stage1_pool: str = "sum",
        stage2_depth: int = 2,
        stage2_edge_dim: int = 17,
        dropout: float = 0.2,
        init_lr: float = 1e-3,
    ):
        super().__init__()
        if stage1_pool not in {"sum", "mean", "attention"}:
            raise ValueError(f"Unknown stage1_pool={stage1_pool!r}")
        self.save_hyperparameters()
        self.stage1_pool = stage1_pool
        self.stage1 = MABBondMessagePassing(
            d_v=atom_fdim, d_e=bond_fdim, d_h=d_h, depth=stage1_depth,
            return_vertex_embeddings=True, return_edge_embeddings=False,
        )
        self.atom_attention = nn.Linear(d_h, 1) if stage1_pool == "attention" else None
        self.stage2_input = nn.Linear(d_h + 1, d_h)
        self.stage2 = nn.ModuleList([Stage2Layer(d_h, stage2_edge_dim) for _ in range(stage2_depth)])
        self.head = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_h, 1))
        self._output_transform = None

    def _pool_monomers(self, atom_embeddings: Tensor, atom_monomer_batch: Tensor, n_monomers: int) -> Tensor:
        pooled = _scatter_sum(atom_embeddings, atom_monomer_batch, n_monomers)
        if self.stage1_pool == "sum":
            return pooled
        counts = torch.bincount(atom_monomer_batch, minlength=n_monomers).to(atom_embeddings.dtype).unsqueeze(-1)
        if self.stage1_pool == "mean":
            return pooled / counts.clamp_min(1.0)
        scores = self.atom_attention(atom_embeddings).squeeze(-1)
        max_scores = torch.full((n_monomers,), float("-inf"), device=scores.device, dtype=scores.dtype)
        max_scores.scatter_reduce_(0, atom_monomer_batch, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_scores[atom_monomer_batch])
        normalizer = torch.zeros(n_monomers, device=scores.device, dtype=scores.dtype)
        normalizer.scatter_add_(0, atom_monomer_batch, exp_scores)
        return _scatter_sum(atom_embeddings * (exp_scores / normalizer[atom_monomer_batch]).unsqueeze(-1), atom_monomer_batch, n_monomers)

    def forward(self, batch: BatchTwoStageHPG, _unused: Tensor | None = None) -> Tensor:
        atom_embeddings, _ = self.stage1(batch.atom_graph)
        monomers = self._pool_monomers(atom_embeddings, batch.atom_graph.batch, len(batch.monomer_batch))
        h = self.stage2_input(torch.cat([monomers, batch.monomer_fracs.unsqueeze(-1)], dim=-1))
        for layer in self.stage2:
            h = layer(h, batch.stage2_edge_index, batch.stage2_edge_features)
        polymers = _scatter_sum(h * batch.monomer_fracs.unsqueeze(-1), batch.polymer_batch, len(batch))
        return self.head(polymers)

    def _loss(self, batch) -> Tensor:
        graph, _, targets, _, _, _ = batch
        predictions = self(graph)
        return torch.mean((predictions - targets) ** 2)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        self.log("val_loss", self._loss(batch), on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        predictions = self(batch[0])
        return self._output_transform(predictions) if self._output_transform is not None else predictions

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.init_lr)
