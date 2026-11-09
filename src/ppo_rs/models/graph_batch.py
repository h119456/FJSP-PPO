"""Utilities for batching variable-size scheduling graphs without PyG."""

from __future__ import annotations

from typing import Any

import torch


GraphState = dict[str, Any]
GraphBatch = dict[str, Any]


def _cpu_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Materialize one graph field on CPU before the single batch transfer."""

    return torch.as_tensor(value, dtype=dtype).cpu()


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
    physical_nodes: list[torch.Tensor] = []
    physical_masks: list[torch.Tensor] = []
    include_physics = "physical_node_features" in states[0]
    action_nodes = torch.full(
        (len(states), action_slots, 3),
        -1,
        dtype=torch.long,
    )

    node_offset = 0
    for batch_index, state in enumerate(states):
        nodes = _cpu_tensor(state["node_features"], torch.float32)
        types = _cpu_tensor(state["node_types"], torch.long)
        edges = _cpu_tensor(state["edge_index"], torch.long)
        relation_types = _cpu_tensor(state["edge_types"], torch.long)
        relation_features = _cpu_tensor(state["edge_features"], torch.float32)
        lookup = _cpu_tensor(state["action_nodes"], torch.long)
        if ("physical_node_features" in state) != include_physics or ("physical_edge_mask" in state) != include_physics:
            raise ValueError("all graph states must use the same physical feature schema")
        if include_physics:
            physical = _cpu_tensor(state["physical_node_features"], torch.float32)
            applicable = _cpu_tensor(state["physical_edge_mask"], torch.bool)
            if physical.shape != (nodes.shape[0], 6) or applicable.shape != relation_types.shape:
                raise ValueError("physical graph fields have incompatible shapes")
            physical_nodes.append(physical)
            physical_masks.append(applicable)
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
            )
        )
        node_offset += nodes.shape[0]

    all_node_types = torch.cat(node_types, dim=0)
    all_edge_types = torch.cat(edge_types, dim=0)
    # These CPU-built indices avoid a GPU boolean mask and its Python control
    # flow for every node/relation type during every PPO mini-batch.
    node_type_indices = {
        node_type: torch.where(all_node_types == node_type)[0]
        for node_type in torch.unique(all_node_types).tolist()
    }
    edge_type_indices = {
        relation: torch.where(all_edge_types == relation)[0]
        for relation in torch.unique(all_edge_types).tolist()
    }
    fields = {
        "node_features": torch.cat(node_features, dim=0),
        "node_types": all_node_types,
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_types": all_edge_types,
        "edge_features": torch.cat(edge_features, dim=0),
        "node_batch": torch.cat(node_batches, dim=0),
        "action_nodes": action_nodes,
    }
    result = {key: value.to(device=device) for key, value in fields.items()}
    if include_physics:
        result["physical_node_features"] = torch.cat(physical_nodes).to(device=device)
        result["physical_edge_mask"] = torch.cat(physical_masks).to(device=device)
    # Transfer all small type-index arrays together, then expose device views.
    # Absence remains CPU metadata, so skipped modules keep grad=None in DDP.
    grouped = (node_type_indices, edge_type_indices)
    index_fields = [*node_type_indices.values(), *edge_type_indices.values()]
    packed = (
        torch.cat(index_fields) if index_fields else torch.empty(0, dtype=torch.long)
    ).to(device=device)
    offset = 0
    for key, indices in zip(("node_type_indices", "edge_type_indices"), grouped):
        result[key] = {}
        for type_id, value in indices.items():
            result[key][type_id] = packed[offset : offset + value.numel()]
            offset += value.numel()
    return result
