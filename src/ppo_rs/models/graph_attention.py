"""Pure-PyTorch heterogeneous edge-aware graph attention layers."""

from __future__ import annotations

import math

import torch
from torch import nn


class HeterogeneousGraphAttentionLayer(nn.Module):
    """One multi-head message-passing layer with relation-specific K/V maps."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        relation_count: int,
        heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.keys = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(relation_count)
        )
        self.values = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(relation_count)
        )
        self.edge_biases = nn.ModuleList(
            nn.Linear(edge_dim, heads, bias=False)
            for _ in range(relation_count)
        )
        self.edge_values = nn.ModuleList(
            nn.Linear(edge_dim, hidden_dim, bias=False)
            for _ in range(relation_count)
        )
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.Tanh()

    @staticmethod
    def _destination_softmax(
        scores: torch.Tensor,
        destinations: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        heads = scores.shape[1]
        destination_index = destinations.unsqueeze(-1).expand(-1, heads)
        maxima = torch.full(
            (node_count, heads),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        maxima.scatter_reduce_(
            0,
            destination_index,
            scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(scores - maxima[destinations])
        denominators = torch.zeros(
            (node_count, heads), dtype=scores.dtype, device=scores.device
        ).index_add(0, destinations, exponentials)
        return exponentials / denominators[destinations].clamp_min(1e-9)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
        relation_indices: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sources, destinations = edge_index
        node_count = nodes.shape[0]
        queries = self.query(nodes).view(node_count, self.heads, self.head_dim)
        aggregated = torch.zeros_like(nodes)

        if relation_indices is None:
            relation_indices = {
                relation: torch.where(edge_types == relation)[0]
                for relation in range(len(self.keys))
            }
        for relation, (key_layer, value_layer, bias_layer, edge_value_layer) in enumerate(
            zip(self.keys, self.values, self.edge_biases, self.edge_values)
        ):
            relation_index = relation_indices.get(relation)
            if relation_index is None or relation_index.numel() == 0:
                continue
            relation_sources = sources.index_select(0, relation_index)
            relation_destinations = destinations.index_select(0, relation_index)
            relation_edges = edge_features.index_select(0, relation_index)
            keys = key_layer(nodes[relation_sources]).view(
                -1, self.heads, self.head_dim
            )
            values = value_layer(nodes[relation_sources]).view(
                -1, self.heads, self.head_dim
            )
            values = values + edge_value_layer(relation_edges).view(
                -1, self.heads, self.head_dim
            )
            scores = (
                queries[relation_destinations] * keys
            ).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = scores + bias_layer(relation_edges)
            attention = self._destination_softmax(
                scores, relation_destinations, node_count
            )
            messages = (attention.unsqueeze(-1) * values).reshape(
                -1, self.hidden_dim
            )
            aggregated = aggregated.index_add(
                0, relation_destinations, messages
            )

        updated = self.self_projection(nodes) + self.message_projection(aggregated)
        return self.norm(nodes + self.activation(updated))
