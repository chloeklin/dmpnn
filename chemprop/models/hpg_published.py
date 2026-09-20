"""Mathematically faithful published HPG-GAT predictor.

The published task is a six-component electrolyte prediction: six HPG graphs,
a six-value component weight-ratio vector, and normalized temperature. This
module preserves that interface and parameter naming so the released reference
state dictionary can be loaded directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chemprop.data.hpg import BatchHPGMolGraph
from chemprop.featurizers.molgraph.hpg_published import HPG_PUBLISHED_ATOM_FDIM


HPG_PUBLISHED_COMPONENT_COUNT = 6
HPG_PUBLISHED_TEMPERATURE_MEAN = 33.923080
HPG_PUBLISHED_TEMPERATURE_STD = 40.216135
HPG_PUBLISHED_TARGET_MEAN = -3.981499
HPG_PUBLISHED_TARGET_STD = 1.836202


class HPGPublishedGATLayer(nn.Module):
    """DGL-free equivalent of the published edge-aware GAT layer."""

    def __init__(self, in_feats: int, out_feats: int, edge_feats: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.W_node = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.W_edge = nn.Linear(edge_feats, num_heads, bias=False)
        self.attention_src = nn.Parameter(torch.empty(num_heads, out_feats))
        self.attention_dst = nn.Parameter(torch.empty(num_heads, out_feats))
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attention_src)
        nn.init.xavier_uniform_(self.attention_dst)

    @staticmethod
    def _edge_softmax(attention: Tensor, destinations: Tensor, num_nodes: int) -> Tensor:
        heads = attention.size(1)
        destination_index = destinations.unsqueeze(-1).expand(-1, heads)
        maxima = torch.full(
            (num_nodes, heads),
            float("-inf"),
            dtype=attention.dtype,
            device=attention.device,
        )
        maxima.scatter_reduce_(0, destination_index, attention, reduce="amax", include_self=True)
        exponentials = torch.exp(attention - maxima[destinations])
        denominators = torch.zeros(
            num_nodes,
            heads,
            dtype=attention.dtype,
            device=attention.device,
        )
        denominators.scatter_add_(0, destination_index, exponentials)
        return exponentials / denominators[destinations].clamp_min(1e-12)

    def forward(self, hidden: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        sources, destinations = edge_index
        node_count = hidden.size(0)
        transformed = self.W_node(hidden).view(node_count, self.num_heads, self.out_feats)
        source_features = transformed[sources]
        destination_features = transformed[destinations]
        transformed_edges = self.W_edge(edge_features)
        attention = F.leaky_relu(
            (source_features * self.attention_src).sum(dim=-1)
            + (destination_features * self.attention_dst).sum(dim=-1)
            + transformed_edges
        )
        alpha = self._edge_softmax(attention, destinations, node_count)
        messages = source_features * alpha.unsqueeze(-1)
        aggregated = torch.zeros(
            node_count,
            self.num_heads,
            self.out_feats,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        aggregated.scatter_add_(
            0,
            destinations.unsqueeze(-1).unsqueeze(-1).expand_as(messages),
            messages,
        )
        return F.leaky_relu(aggregated.mean(dim=1))


class HPGPublishedPolymerEncoder(nn.Module):
    """Reusable 64D polymer encoder from the published HPG architecture."""

    output_dim = 64

    def __init__(self):
        super().__init__()
        dims = [HPG_PUBLISHED_ATOM_FDIM] + [128] * 6
        self.layers = nn.ModuleList([
            HPGPublishedGATLayer(
                in_feats=dims[index],
                out_feats=dims[index + 1],
                edge_feats=1,
                num_heads=8,
            )
            for index in range(6)
        ])
        self.linear_g1 = nn.Linear(128, self.output_dim)

    @staticmethod
    def _sum_nodes(hidden: Tensor, graph: BatchHPGMolGraph) -> Tensor:
        batch_size = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        index = graph.batch.unsqueeze(-1).expand(-1, hidden.size(1))
        pooled = torch.zeros(
            batch_size,
            hidden.size(1),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        pooled.scatter_add_(0, index, hidden)
        return pooled

    def forward(self, graph: BatchHPGMolGraph) -> Tensor:
        hidden = graph.V
        for layer in self.layers:
            hidden = layer(hidden, graph.edge_index, graph.E)
        return self.linear_g1(self._sum_nodes(hidden, graph))


class HPGPublishedGATNet(nn.Module):
    """Published six-component HPG-GAT architecture."""

    def __init__(self, pred_dim: int = 1, hidden_mode: bool = False):
        super().__init__()
        self.pred_dim = pred_dim
        self.depth = 6
        self.dims = [HPG_PUBLISHED_ATOM_FDIM] + [128] * self.depth
        self.hidden_mode = hidden_mode
        self.GAT_list_1 = nn.ModuleList([
            HPGPublishedGATLayer(
                in_feats=self.dims[index],
                out_feats=self.dims[index + 1],
                edge_feats=1,
                num_heads=8,
            )
            for index in range(self.depth)
        ])
        self.linear_g1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0)
        self.linear_ratio = nn.Linear(6, 6)
        self.linear_temp = nn.Linear(1, 16)
        self.linear1 = nn.Linear(64 * 6 + 6 + 16, 512)
        self.linear2 = nn.Linear(512, pred_dim)
        self.dropout_pred = nn.Dropout(0.2)

        self.linear_ratio.apply(self._init_predictor_weights)
        self.linear_temp.apply(self._init_predictor_weights)
        self.linear1.apply(self._init_predictor_weights)
        self.linear2.apply(self._init_predictor_weights)

    @staticmethod
    def _init_predictor_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0)

    @staticmethod
    def normalize_temperature(temperature: Tensor) -> Tensor:
        return (temperature - HPG_PUBLISHED_TEMPERATURE_MEAN) / HPG_PUBLISHED_TEMPERATURE_STD

    @staticmethod
    def unscale_log_conductivity(value: Tensor) -> Tensor:
        return value * HPG_PUBLISHED_TARGET_STD + HPG_PUBLISHED_TARGET_MEAN

    @staticmethod
    def _sum_nodes(hidden: Tensor, graph: BatchHPGMolGraph) -> Tensor:
        batch_size = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        index = graph.batch.unsqueeze(-1).expand(-1, hidden.size(1))
        pooled = torch.zeros(
            batch_size,
            hidden.size(1),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        pooled.scatter_add_(0, index, hidden)
        return pooled

    def _encode_component(self, graph: BatchHPGMolGraph) -> Tensor:
        hidden = graph.V
        for layer in self.GAT_list_1:
            hidden = self.dropout(layer(hidden, graph.edge_index, graph.E))
        return self.dropout(self.linear_g1(self._sum_nodes(hidden, graph)))

    def forward(
        self,
        graphs: Sequence[BatchHPGMolGraph],
        ratio: Tensor,
        normalized_temperature: Tensor,
    ) -> Tensor:
        if len(graphs) != HPG_PUBLISHED_COMPONENT_COUNT:
            raise ValueError("HPG-published requires exactly six component graphs")
        if ratio.ndim != 2 or ratio.shape[1] != HPG_PUBLISHED_COMPONENT_COUNT:
            raise ValueError("HPG-published ratio must have shape [batch, 6]")
        if normalized_temperature.ndim != 2 or normalized_temperature.shape[1] != 1:
            raise ValueError("HPG-published normalized_temperature must have shape [batch, 1]")

        component_embeddings = [self._encode_component(graph) for graph in graphs]
        hidden_input = torch.cat(
            component_embeddings
            + [F.leaky_relu(self.linear_ratio(ratio)),
               F.leaky_relu(self.linear_temp(normalized_temperature))],
            dim=1,
        )
        hidden = F.leaky_relu(self.linear1(hidden_input))
        if self.hidden_mode:
            return hidden
        return self.dropout_pred(self.linear2(hidden))
