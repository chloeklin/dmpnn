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
    """Single Stage-2 message-passing layer.

    mode='feature'     : message = MLP([h_src; e_full])               (default, identical to original)
    mode='multiplier'  : message = w * MLP([h_src; e_port])           (Variant 1 wedge)
    mode='both'        : message = w * MLP([h_src; e_full])           (Variant 1 both)

    e_full is the full d_e-dim edge feature vector; w = e_full[:, -1] (transition weight scalar);
    e_port = e_full[:, :-1] (port-pair features, dim d_e-1).
    """
    def __init__(self, d_h: int, d_e: int, mode: str = "feature"):
        super().__init__()
        if mode not in {"feature", "multiplier", "both"}:
            raise ValueError(f"Unknown Stage2Layer mode={mode!r}")
        self.mode = mode
        d_msg_in = (d_h + d_e - 1) if mode == "multiplier" else (d_h + d_e)
        self.message = nn.Sequential(nn.Linear(d_msg_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.update = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        source, target = edge_index
        if self.mode == "feature":
            messages = self.message(torch.cat([h[source], edge_features], dim=-1))
        elif self.mode == "multiplier":
            w = edge_features[:, -1:]
            messages = w * self.message(torch.cat([h[source], edge_features[:, :-1]], dim=-1))
        else:  # both
            w = edge_features[:, -1:]
            messages = w * self.message(torch.cat([h[source], edge_features], dim=-1))
        aggregate = _scatter_sum(messages, target, h.size(0))
        return self.norm(h + self.update(torch.cat([h, aggregate], dim=-1)))


class OctamerPathLayer(nn.Module):
    """One step of bidirectional path message passing for the octamer encoder (no edge features)."""
    def __init__(self, d_h: int):
        super().__init__()
        self.msg = nn.Linear(d_h, d_h)
        self.update = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: Tensor, edge_index: Tensor, n_nodes: int) -> Tensor:
        src, dst = edge_index
        agg = _scatter_sum(self.msg(h[src]), dst, n_nodes)
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))


class AttentionReadout(nn.Module):
    """Learned attention pooling shared by octamer and transition-graph stage-2 readouts."""
    def __init__(self, d_h: int):
        super().__init__()
        self.pool_score = nn.Linear(d_h, 1)

    def forward(self, embeddings: Tensor, group_index: Tensor | None = None, n_groups: int | None = None) -> Tensor:
        if group_index is None:
            attention = torch.softmax(self.pool_score(embeddings).squeeze(-1), dim=1)
            return torch.sum(embeddings * attention.unsqueeze(-1), dim=1)
        if n_groups is None:
            raise ValueError("n_groups is required for grouped attention pooling")
        scores = self.pool_score(embeddings).squeeze(-1)
        max_scores = torch.full((n_groups,), float("-inf"), device=scores.device, dtype=scores.dtype)
        max_scores.scatter_reduce_(0, group_index, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_scores[group_index])
        normalizer = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
        normalizer.scatter_add_(0, group_index, exp_scores)
        weights = exp_scores / normalizer[group_index]
        return _scatter_sum(embeddings * weights.unsqueeze(-1), group_index, n_groups)


class OctamerEncoder(nn.Module):
    """Encode polymers as explicit octamer sequences; average head(encode(seq_k)) over K samples.

    monomer_embeds: (2*n_polymers, d_h) -- h[2p]=A, h[2p+1]=B for polymer p
    octamer_sequences: (n_polymers*K, octamer_len) long  -- values 0 (A) or 1 (B)
    octamer_polymer_batch: (n_polymers*K,) long  -- which polymer each replicate belongs to
    """
    def __init__(self, d_h: int, octamer_len: int, n_layers: int, use_position_embeddings: bool = True):
        super().__init__()
        self.d_h = d_h
        self.octamer_len = octamer_len
        self.layers = nn.ModuleList([OctamerPathLayer(d_h) for _ in range(n_layers)])
        self.attention_readout = AttentionReadout(d_h)
        if use_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.empty(octamer_len, d_h))
            nn.init.normal_(self.position_embeddings, std=0.02)

    @staticmethod
    def _make_path_edge_index(n_reps: int, L: int, device: torch.device) -> Tensor:
        fwd_src = torch.arange(L - 1, device=device)
        fwd_dst = fwd_src + 1
        tmpl_src = torch.cat([fwd_src, fwd_dst])   # fwd + bwd
        tmpl_dst = torch.cat([fwd_dst, fwd_src])
        offsets = (torch.arange(n_reps, device=device) * L).unsqueeze(1)  # (n_reps, 1)
        src = (tmpl_src.unsqueeze(0) + offsets).reshape(-1)
        dst = (tmpl_dst.unsqueeze(0) + offsets).reshape(-1)
        return torch.stack([src, dst])  # (2, n_reps * 2*(L-1))

    def forward(self, monomer_embeds: Tensor, octamer_sequences: Tensor, octamer_polymer_batch: Tensor) -> Tensor:
        n_reps, L = octamer_sequences.shape
        # Map sequence values (0/1) to global monomer indices 2p or 2p+1
        global_mon = 2 * octamer_polymer_batch.unsqueeze(1) + octamer_sequences  # (n_reps, L)
        h = monomer_embeds[global_mon.reshape(-1)].reshape(n_reps, L, self.d_h)
        if getattr(self, "position_embeddings", None) is not None:
            h = h + self.position_embeddings.unsqueeze(0)
        h_flat = h.reshape(n_reps * L, self.d_h)
        edge_index = self._make_path_edge_index(n_reps, L, device=h_flat.device)
        n_nodes = n_reps * L
        for layer in self.layers:
            h_flat = layer(h_flat, edge_index, n_nodes)
        h = h_flat.reshape(n_reps, L, self.d_h)
        return self.attention_readout(h)


class JunctionCouplingLayer(nn.Module):
    """One step of weighted message passing on the combined (intra + junction) atom graph."""
    def __init__(self, d_h: int):
        super().__init__()
        self.msg = nn.Linear(d_h, d_h)
        self.update = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.norm = nn.LayerNorm(d_h)

    def forward(self, h: Tensor, edge_index: Tensor, weights: Tensor) -> Tensor:
        src, dst = edge_index
        msgs = weights.unsqueeze(-1) * self.msg(h[src])
        agg = _scatter_sum(msgs, dst, h.size(0))
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1)))


class HPGHierMPNN(pl.LightningModule):
    """Two-stage hierarchical MPG-MPNN with Phase-1 variant toggles.

    Defaults reproduce the original hpg_hier exactly.

    Variant 1 — hpg_hier_wedge  : stage2_edge_weight in {"feature", "multiplier", "both"}
    Variant 2 — hpg_hier_octamer: stage2_mode in {"transition_graph", "octamer_sequence"}
    Variant 3 — hpg_hier_junction: junction_coupling in {"off", "on"}
    """
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
        stage2_edge_weight: str = "feature",
        stage2_mode: str = "transition_graph",
        stage2_readout: str | None = None,
        octamer_len: int = 8,
        n_random_samples: int = 16,
        junction_coupling: str = "off",
        n_coupling_steps: int = 2,
        use_position_embeddings: bool = True,
    ):
        super().__init__()
        if stage1_pool not in {"sum", "mean", "attention"}:
            raise ValueError(f"Unknown stage1_pool={stage1_pool!r}")
        if stage2_edge_weight not in {"feature", "multiplier", "both"}:
            raise ValueError(f"Unknown stage2_edge_weight={stage2_edge_weight!r}")
        if stage2_mode not in {"transition_graph", "octamer_sequence"}:
            raise ValueError(f"Unknown stage2_mode={stage2_mode!r}")
        if junction_coupling not in {"off", "on"}:
            raise ValueError(f"Unknown junction_coupling={junction_coupling!r}")
        if stage2_readout is None:
            stage2_readout = "attention" if stage2_mode == "octamer_sequence" else "stoich_weighted"
        if stage2_readout not in {"stoich_weighted", "attention"}:
            raise ValueError(f"Unknown stage2_readout={stage2_readout!r}")
        if stage2_mode == "octamer_sequence" and stage2_readout != "attention":
            raise ValueError(
                f"octamer_sequence with readout '{stage2_readout}' is not implemented — "
                "OctamerEncoder is only constructed for the attention readout (hpg_hier.py:201-203). "
                "This configuration would silently train the 2-node baseline. "
                "See HANDOFF §7 (arm D)."
            )
        self.save_hyperparameters()
        self.stage1_pool = stage1_pool
        self.stage2_edge_weight = stage2_edge_weight
        self.stage2_mode = stage2_mode
        self.stage2_readout = stage2_readout
        self.junction_coupling = junction_coupling
        self.n_random_samples = n_random_samples
        self.stage1 = MABBondMessagePassing(
            d_v=atom_fdim, d_e=bond_fdim, d_h=d_h, depth=stage1_depth,
            return_vertex_embeddings=True, return_edge_embeddings=False,
        )
        self.atom_attention = nn.Linear(d_h, 1) if stage1_pool == "attention" else None
        self.stage2_input = nn.Linear(d_h + 1, d_h)
        self.stage2 = nn.ModuleList([
            Stage2Layer(d_h, stage2_edge_dim, mode=stage2_edge_weight) for _ in range(stage2_depth)
        ])
        self.octamer_encoder = (
            OctamerEncoder(
                d_h, octamer_len=octamer_len, n_layers=stage2_depth,
                use_position_embeddings=use_position_embeddings,
            )
            if stage2_mode == "octamer_sequence" and stage2_readout == "attention" else None
        )
        self.stage2_attention_readout = (
            AttentionReadout(d_h)
            if stage2_mode == "transition_graph" and stage2_readout == "attention" else None
        )
        self.junction_layers = (
            nn.ModuleList([JunctionCouplingLayer(d_h) for _ in range(n_coupling_steps)])
            if junction_coupling == "on" else None
        )
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

        # Variant 3: junction coupling — spread info across A/B before pooling
        if self.junction_coupling == "on" and batch.junction_edge_index is not None:
            intra_ei = batch.atom_graph.edge_index  # (2, n_intra)
            intra_w = torch.ones(intra_ei.size(1), device=atom_embeddings.device)
            combined_ei = torch.cat([intra_ei, batch.junction_edge_index], dim=1)
            combined_w = torch.cat([intra_w, batch.junction_edge_weights])
            for layer in self.junction_layers:
                atom_embeddings = layer(atom_embeddings, combined_ei, combined_w)

        monomers = self._pool_monomers(atom_embeddings, batch.atom_graph.batch, len(batch.monomer_batch))
        h = self.stage2_input(torch.cat([monomers, batch.monomer_fracs.unsqueeze(-1)], dim=-1))

        # Variant 2: octamer sequence — yhat = mean_k( head( encode(octamer_k) ) )
        if self.stage2_mode == "octamer_sequence" and self.stage2_readout == "attention" and batch.octamer_sequences is not None:
            n_polymers = len(batch)
            oct_embeds = self.octamer_encoder(h, batch.octamer_sequences, batch.octamer_polymer_batch)
            all_preds = self.head(oct_embeds)
            pred_sum = _scatter_sum(all_preds, batch.octamer_polymer_batch, n_polymers)
            replica_counts = torch.bincount(batch.octamer_polymer_batch, minlength=n_polymers)
            return pred_sum / replica_counts.to(all_preds.dtype).unsqueeze(-1)

        # Transition graph, or octamer construction with stoichiometry-weighted readout.
        for layer in self.stage2:
            h = layer(h, batch.stage2_edge_index, batch.stage2_edge_features)
        if self.stage2_readout == "attention":
            polymers = self.stage2_attention_readout(h, batch.polymer_batch, len(batch))
        else:
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
