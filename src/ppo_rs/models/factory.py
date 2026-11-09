"""Model construction shared by training and checkpoint evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .actor_critic import ActorCritic
from .graph_actor_critic import GraphActorCritic


def build_actor_critic(
    model_type: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    **kwargs: Any,
) -> nn.Module:
    if model_type == "mlp":
        if kwargs.get("shared_kv_film", False):
            raise ValueError("shared_kv_film is available only for gnn_attention")
        if kwargs.get("physical_gating", False):
            raise ValueError("physical_gating is available only for gnn_attention")
        return ActorCritic(state_dim, action_dim, hidden_dim)
    if model_type == "gnn_attention":
        if kwargs.get("shared_kv_film_schema_version", 1) != 1:
            raise ValueError("unsupported shared K/V FiLM schema version")
        if kwargs.get("physical_gate_schema_version", 1) != 1:
            raise ValueError("unsupported physical gate schema version")
        return GraphActorCritic(
            state_dim,
            action_dim,
            hidden_dim,
            node_feature_dim=int(kwargs["node_feature_dim"]),
            edge_feature_dim=int(kwargs["edge_feature_dim"]),
            node_type_count=int(kwargs["node_type_count"]),
            edge_type_count=int(kwargs["edge_type_count"]),
            gnn_layers=int(kwargs.get("gnn_layers", 2)),
            attention_heads=int(kwargs.get("attention_heads", 4)),
            shared_kv_film=kwargs.get("shared_kv_film", False),
            physical_gating=kwargs.get("physical_gating", False),
            physical_gate_initial=float(kwargs.get("physical_gate_initial", 0.9)),
        )
    raise ValueError(
        f"unknown model_type {model_type!r}; choose 'mlp' or 'gnn_attention'"
    )


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> nn.Module:
    model_type = str(checkpoint.get("model_type", "mlp"))
    model = build_actor_critic(
        model_type,
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_feature_dim"]),
        int(checkpoint["hidden_dim"]),
        **dict(checkpoint.get("model_kwargs", {})),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
