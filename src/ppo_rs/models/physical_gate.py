"""Physical-state-conditioned, relation-group multi-head message gates."""
from __future__ import annotations

import math

import torch
from torch import nn


class PhysicalEdgeGate(nn.Module):
    """One shared 6->16->16 encoder and three 16->8->heads gate heads.

    Parameters are shared by both message-passing layers and by forward/reverse
    relations. Structural and stage-inapplicable edges receive exact ones.
    """

    def __init__(self, heads: int, initial_gate: float = 0.9):
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("physical_gate_initial must be strictly between zero and one")
        self.heads = heads
        self.encoder = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.gate_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, heads)) for _ in range(3)
        )
        for head in self.gate_heads:
            # Small nonzero weights allow gradients into the physical encoder
            # on the first update; a zero final matrix would initially block it.
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(head[-1].bias, math.log(initial_gate / (1.0 - initial_gate)))

    def forward(self, graph):
        if "physical_node_features" not in graph or "physical_edge_mask" not in graph:
            raise ValueError("physical gating requires a graph generated with include_physics=True")
        nodes, edge_types = graph["physical_node_features"], graph["edge_types"]
        if nodes.shape != (graph["node_features"].shape[0], 6):
            raise ValueError("physical_node_features must have shape [nodes, 6]")
        mask = graph["physical_edge_mask"]
        if mask.shape != edge_types.shape:
            raise ValueError("physical_edge_mask must have shape [edges]")
        encoded = self.encoder(nodes)
        gates = nodes.new_ones((edge_types.numel(), self.heads))
        sources, destinations = graph["edge_index"]
        for group, head in enumerate(self.gate_heads):
            forward_relation = 4 + 2 * group
            indices = torch.where(mask & ((edge_types == forward_relation) | (edge_types == forward_relation + 1)))[0]
            if indices.numel() == 0:
                continue
            # In all three groups the forward direction starts at operation;
            # reverse edges must obtain the same operation's physical context.
            operation_nodes = torch.where(
                edge_types[indices] == forward_relation, sources[indices], destinations[indices],
            )
            gates = gates.index_copy(0, indices, torch.sigmoid(head(encoded[operation_nodes])))
        return gates
