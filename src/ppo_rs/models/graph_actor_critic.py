"""Two-layer attention GNN actor-critic for heterogeneous FJSP graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .graph_attention import HeterogeneousGraphAttentionLayer
from .graph_batch import GraphBatch, GraphState, batch_graph_states
from .physical_gate import PhysicalEdgeGate


class GraphActorCritic(nn.Module):
    """Shared graph encoder with candidate-edge Actor and graph-level Critic."""

    model_type = "gnn_attention"
    requires_graph = True

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        node_type_count: int,
        edge_type_count: int,
        gnn_layers: int = 2,
        attention_heads: int = 4,
        shared_kv_film: bool = False,
        physical_gating: bool = False,
        physical_gate_initial: float = 0.9,
    ) -> None:
        super().__init__()
        if gnn_layers != 2:
            raise ValueError("this project GNN design requires exactly two layers")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.node_type_count = node_type_count
        self.edge_type_count = edge_type_count
        self.gnn_layers = gnn_layers
        self.attention_heads = attention_heads
        if not isinstance(shared_kv_film, bool):
            raise ValueError("shared_kv_film must be boolean")
        if not isinstance(physical_gating, bool):
            raise ValueError("physical_gating must be boolean")
        if shared_kv_film and physical_gating:
            raise ValueError("shared_kv_film and physical_gating are separate experiments; enable only one")
        self.shared_kv_film = shared_kv_film
        self.physical_gating = physical_gating
        self.physical_gate_initial = physical_gate_initial
        self.node_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(node_feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            for _ in range(node_type_count)
        )
        self.message_layers = nn.ModuleList(
            HeterogeneousGraphAttentionLayer(
                hidden_dim,
                edge_feature_dim,
                edge_type_count,
                attention_heads,
                shared_kv_film=shared_kv_film,
            )
            for _ in range(gnn_layers)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim * node_type_count + state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        if physical_gating:
            if edge_type_count != 10:
                raise ValueError("physical gating requires the ten-relation FJSP schema")
            self.physical_gate = PhysicalEdgeGate(attention_heads, physical_gate_initial)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def checkpoint_kwargs(self) -> dict:
        kwargs = {
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "node_type_count": self.node_type_count,
            "edge_type_count": self.edge_type_count,
            "gnn_layers": self.gnn_layers,
            "attention_heads": self.attention_heads,
        }
        # Preserve the exact legacy schema when both extensions are off.
        if self.shared_kv_film:
            kwargs.update(shared_kv_film=True, shared_kv_film_schema_version=1)
        if self.physical_gating:
            kwargs.update(physical_gating=True, physical_gate_initial=self.physical_gate_initial,
                          physical_gate_schema_version=1)
        return kwargs

    def get_edge_gates(self, graph: GraphBatch) -> torch.Tensor:
        """Return [edges, heads] without retaining diagnostic tensors in PPO."""
        if self.physical_gating:
            return self.physical_gate(graph)
        return graph["node_features"].new_ones((graph["edge_types"].numel(), self.attention_heads))

    def _encode_nodes(self, graph: GraphBatch) -> torch.Tensor:
        features = graph["node_features"]
        types = graph["node_types"]
        encoded = torch.zeros(
            (features.shape[0], self.hidden_dim),
            dtype=features.dtype,
            device=features.device,
        )
        indices = graph.get("node_type_indices")
        if indices is None:
            indices = {
                value: torch.where(types == value)[0]
                for value in range(len(self.node_encoders))
            }
        for node_type, encoder in enumerate(self.node_encoders):
            type_indices = indices.get(node_type)
            if type_indices is not None and type_indices.numel() != 0:
                encoded.index_copy_(
                    0, type_indices, encoder(features.index_select(0, type_indices))
                )
        gates = self.get_edge_gates(graph) if self.physical_gating else None
        for layer in self.message_layers:
            encoded = layer(
                encoded,
                graph["edge_index"],
                graph["edge_types"],
                graph["edge_features"],
                graph.get("edge_type_indices"),
                **({"edge_gates": gates} if gates is not None else {}),
            )
        return encoded

    def _pool_graphs(
        self,
        nodes: torch.Tensor,
        graph: GraphBatch,
        states: torch.Tensor,
    ) -> torch.Tensor:
        node_batches = graph["node_batch"]
        node_types = graph["node_types"]
        batch_size = states.shape[0]
        pooled_types: list[torch.Tensor] = []
        type_indices = graph.get("node_type_indices")
        if type_indices is None:
            type_indices = {
                value: torch.where(node_types == value)[0]
                for value in range(self.node_type_count)
            }
        for node_type in range(self.node_type_count):
            indices = type_indices.get(node_type)
            sums = torch.zeros(
                (batch_size, self.hidden_dim),
                dtype=nodes.dtype,
                device=nodes.device,
            )
            counts = torch.zeros(
                (batch_size, 1), dtype=nodes.dtype, device=nodes.device
            )
            if indices is not None and indices.numel() != 0:
                batches = node_batches.index_select(0, indices)
                sums = sums.index_add(0, batches, nodes.index_select(0, indices))
                counts = counts.index_add(
                    0,
                    batches,
                    torch.ones(
                        (indices.numel(), 1),
                        dtype=nodes.dtype,
                        device=nodes.device,
                    ),
                )
            pooled_types.append(sums / counts.clamp_min(1.0))
        return self.global_encoder(torch.cat([*pooled_types, states], dim=-1))

    @staticmethod
    def _gather_action_nodes(
        nodes: torch.Tensor, action_nodes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = action_nodes >= 0
        gathered = nodes[action_nodes.clamp_min(0)]
        gathered = gathered * valid.unsqueeze(-1)
        return gathered[:, :, 0], gathered[:, :, 1], gathered[:, :, 2]

    def forward(
        self,
        states: torch.Tensor,
        action_features: torch.Tensor,
        graph: GraphBatch,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self._encode_nodes(graph)
        graph_embedding = self._pool_graphs(nodes, graph, states)
        jobs, operations, machines = self._gather_action_nodes(
            nodes, graph["action_nodes"]
        )
        action_embedding = self.action_encoder(action_features)
        global_actions = graph_embedding.unsqueeze(1).expand(
            -1, action_features.shape[1], -1
        )
        logits = self.actor_scorer(
            torch.cat(
                [jobs, operations, machines, action_embedding, global_actions],
                dim=-1,
            )
        ).squeeze(-1)
        if action_mask is not None:
            if not torch.all(action_mask.any(dim=-1)):
                raise ValueError("every policy row needs at least one legal action")
            logits = logits.masked_fill(
                ~action_mask, torch.finfo(logits.dtype).min
            )
        values = self.critic(graph_embedding).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act(
        self,
        observation: Sequence[float],
        action_features: Sequence[Sequence[float]],
        *,
        graph_state: GraphState | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        if not action_features:
            raise ValueError("cannot act on an empty candidate set")
        if graph_state is None:
            raise ValueError("GNN policy requires a graph state")
        state = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        features = torch.tensor(
            action_features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        graph = batch_graph_states(
            [graph_state], action_slots=len(action_features), device=self.device
        )
        logits, values = self.forward(state, features, graph)
        distribution = Categorical(logits=logits[0])
        choice = (
            torch.argmax(logits[0])
            if deterministic
            else distribution.sample()
        )
        values_to_return = torch.stack(
            (choice.to(dtype=torch.float32), distribution.log_prob(choice), values[0])
        ).cpu().tolist()
        return int(values_to_return[0]), float(values_to_return[1]), float(values_to_return[2])

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        action_features: torch.Tensor,
        action_masks: torch.Tensor,
        choices: torch.Tensor,
        *,
        graph_batch: GraphBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if graph_batch is None:
            raise ValueError("GNN policy requires a graph batch")
        logits, values = self.forward(
            observations, action_features, graph_batch, action_masks
        )
        distribution = Categorical(logits=logits)
        return distribution.log_prob(choices), distribution.entropy(), values
