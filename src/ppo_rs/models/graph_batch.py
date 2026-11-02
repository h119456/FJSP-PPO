"""Utilities for batching variable-size scheduling graphs without PyG."""

from __future__ import annotations

from typing import Any

import torch


GraphState = dict[str, Any]
GraphBatch = dict[str, torch.Tensor]


def batch_graph_states(
    states: list[GraphState],
    *,
    action_slots: int,
    device: torch.device,
) -> GraphBatch:
    """Concatenate graphs and pad their action-to-node lookup tables."""

    if not states:
        raise ValueError("at least one graph state is required")
    node_features: list[torch.Tensor] = []
    node_types: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    node_batches: list[torch.Tensor] = []
    action_nodes = torch.full(
        (len(states), action_slots, 3),
        -1,
        dtype=torch.long,
        device=device,
    )

    node_offset = 0
    for batch_index, state in enumerate(states):
        nodes = torch.as_tensor(
            state["node_features"], dtype=torch.float32, device=device
        )
        types = torch.as_tensor(
            state["node_types"], dtype=torch.long, device=device
        )
        edges = torch.as_tensor(
            state["edge_index"], dtype=torch.long, device=device
        )
        relation_types = torch.as_tensor(
            state["edge_types"], dtype=torch.long, device=device
        )
        relation_features = torch.as_tensor(
            state["edge_features"], dtype=torch.float32, device=device
        )
        lookup = torch.as_tensor(
            state["action_nodes"], dtype=torch.long, device=device
        )
        if lookup.shape[0] > action_slots:
            raise ValueError("graph action count exceeds padded action slots")

        shifted_lookup = lookup.clone()
        shifted_lookup[shifted_lookup >= 0] += node_offset
        action_nodes[batch_index, : lookup.shape[0]] = shifted_lookup
        node_features.append(nodes)
        node_types.append(types)
        edge_indices.append(edges + node_offset)
        edge_types.append(relation_types)
        edge_features.append(relation_features)
        node_batches.append(
            torch.full(
                (nodes.shape[0],),
                batch_index,
                dtype=torch.long,
                device=device,
            )
        )
        node_offset += nodes.shape[0]

    return {
        "node_features": torch.cat(node_features, dim=0),
        "node_types": torch.cat(node_types, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_types": torch.cat(edge_types, dim=0),
        "edge_features": torch.cat(edge_features, dim=0),
        "node_batch": torch.cat(node_batches, dim=0),
        "action_nodes": action_nodes,
    }
