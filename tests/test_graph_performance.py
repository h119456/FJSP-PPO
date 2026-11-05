from __future__ import annotations

from copy import deepcopy
from types import MethodType
import math

import pytest
import torch

from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic
from ppo_rs.models.graph_batch import batch_graph_states

from tests.helpers import tiny_instance


def _legacy_layer(layer, nodes, edge_index, edge_types, edge_features, *_):
    """Reference pre-optimization operations, including absent relation skips."""
    sources, destinations = edge_index
    node_count = nodes.shape[0]
    queries = layer.query(nodes).view(node_count, layer.heads, layer.head_dim)
    aggregated = torch.zeros_like(nodes)
    for relation, modules in enumerate(zip(layer.keys, layer.values, layer.edge_biases, layer.edge_values)):
        mask = edge_types == relation
        if not torch.any(mask):
            continue
        key_layer, value_layer, bias_layer, edge_value_layer = modules
        src, dst, edges = sources[mask], destinations[mask], edge_features[mask]
        keys = key_layer(nodes[src]).view(-1, layer.heads, layer.head_dim)
        values = value_layer(nodes[src]).view(-1, layer.heads, layer.head_dim)
        values = values + edge_value_layer(edges).view(-1, layer.heads, layer.head_dim)
        scores = (queries[dst] * keys).sum(dim=-1) / math.sqrt(layer.head_dim)
        scores = scores + bias_layer(edges)
        attention = layer._destination_softmax(scores, dst, node_count)
        messages = (attention.unsqueeze(-1) * values).reshape(-1, layer.hidden_dim)
        aggregated = aggregated.index_add(0, dst, messages)
    updated = layer.self_projection(nodes) + layer.message_projection(aggregated)
    return layer.norm(nodes + layer.activation(updated))


def _legacy_encode(model, graph):
    features, types = graph["node_features"], graph["node_types"]
    encoded = torch.zeros((features.shape[0], model.hidden_dim), dtype=features.dtype)
    for node_type, encoder in enumerate(model.node_encoders):
        mask = types == node_type
        if torch.any(mask):
            encoded[mask] = encoder(features[mask])
    for layer in model.message_layers:
        encoded = layer(encoded, graph["edge_index"], graph["edge_types"], graph["edge_features"])
    return encoded


def _legacy_pool(model, nodes, graph, states):
    pooled = []
    for node_type in range(model.node_type_count):
        mask = graph["node_types"] == node_type
        sums = torch.zeros((states.shape[0], model.hidden_dim), dtype=nodes.dtype)
        counts = torch.zeros((states.shape[0], 1), dtype=nodes.dtype)
        if torch.any(mask):
            sums = sums.index_add(0, graph["node_batch"][mask], nodes[mask])
            counts = counts.index_add(0, graph["node_batch"][mask], torch.ones((int(mask.sum().item()), 1)))
        pooled.append(sums / counts.clamp_min(1.0))
    return model.global_encoder(torch.cat([*pooled, states], dim=-1))


def _fixture(mode: str):
    env = ResidualStressFJSPEnv(tiny_instance(jobs=2))
    state, info = env.reset(seed=7)
    graph = env.get_graph_state(info["valid_actions"])
    if mode != "full":
        keep = graph["edge_types"] < 2 if mode == "missing" else graph["edge_types"] < 0
        graph["edge_index"] = graph["edge_index"][:, keep]
        graph["edge_types"] = graph["edge_types"][keep]
        graph["edge_features"] = graph["edge_features"][keep]
        graph["node_types"] = graph["node_types"].copy()
        graph["node_types"][:] = 0
    model = build_actor_critic(
        "gnn_attention", env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8,
        node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
        node_type_count=env.GRAPH_NODE_TYPE_COUNT,
        edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
    )
    return model, state, info["action_features"], graph


@pytest.mark.parametrize("mode", ["full", "missing", "empty_edges"])
def test_indexed_graph_forward_and_gradients_match_original(mode):
    torch.manual_seed(17)
    model, state, features, graph_state = _fixture(mode)
    reference = deepcopy(model)
    reference._encode_nodes = MethodType(_legacy_encode, reference)
    reference._pool_graphs = MethodType(_legacy_pool, reference)
    for layer in reference.message_layers:
        layer.forward = MethodType(_legacy_layer, layer)
    graph = batch_graph_states([graph_state, graph_state], action_slots=len(features), device=torch.device("cpu"))
    states = torch.tensor([state, state], dtype=torch.float32)
    actions = torch.tensor([features, features], dtype=torch.float32)
    actual = model(states, actions, graph)
    expected = reference(states, actions, graph)
    for result, baseline in zip(actual, expected):
        torch.testing.assert_close(result, baseline, rtol=1e-6, atol=1e-7)
    sum(tensor.square().sum() for tensor in actual).backward()
    sum(tensor.square().sum() for tensor in expected).backward()
    assert model.checkpoint_kwargs == reference.checkpoint_kwargs
    assert model.state_dict().keys() == reference.state_dict().keys()
    for (name, parameter), (_, baseline) in zip(model.named_parameters(), reference.named_parameters()):
        assert (parameter.grad is None) == (baseline.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, baseline.grad, rtol=2e-5, atol=1e-6, msg=name)


def test_batch_metadata_matches_offsets_and_preserves_source_graphs():
    _, _, features, graph = _fixture("missing")
    original = deepcopy(graph)
    batch = batch_graph_states([graph, graph], action_slots=len(features) + 2, device=torch.device("cpu"))
    count = len(graph["node_features"])
    torch.testing.assert_close(batch["edge_index"][:, :graph["edge_index"].shape[1]], torch.as_tensor(graph["edge_index"]))
    torch.testing.assert_close(batch["edge_index"][:, graph["edge_index"].shape[1]:], torch.as_tensor(graph["edge_index"]) + count)
    assert batch["action_nodes"][:, -2:].eq(-1).all()
    for metadata, field in (("node_type_indices", "node_types"), ("edge_type_indices", "edge_types")):
        for type_id, indices in batch[metadata].items():
            torch.testing.assert_close(indices, torch.where(batch[field] == type_id)[0])
    for field in graph:
        torch.testing.assert_close(torch.as_tensor(graph[field]), torch.as_tensor(original[field]))


def test_forward_with_metadata_needs_no_tensor_item_or_torch_any(monkeypatch):
    model, state, features, graph_state = _fixture("missing")
    graph = batch_graph_states([graph_state], action_slots=len(features), device=torch.device("cpu"))

    def unexpected(*args, **kwargs):
        raise AssertionError("indexed forward must not read tensor scalars or use torch.any branches")

    monkeypatch.setattr(torch, "any", unexpected)
    monkeypatch.setattr(torch.Tensor, "item", unexpected)
    model(torch.tensor([state]), torch.tensor([features]), graph)


def test_manual_batch_without_metadata_and_legacy_act_are_compatible():
    model, state, features, graph_state = _fixture("full")
    graph = batch_graph_states([graph_state], action_slots=len(features), device=torch.device("cpu"))
    manual = {key: value for key, value in graph.items() if not key.endswith("_indices")}
    states, actions = torch.tensor([state]), torch.tensor([features])
    for indexed, compatible in zip(model(states, actions, graph), model(states, actions, manual)):
        torch.testing.assert_close(indexed, compatible)
    logits, values = model(states, actions, graph)
    choice = int(logits[0].argmax().item())
    distribution = torch.distributions.Categorical(logits=logits[0])
    expected = (choice, float(distribution.log_prob(torch.tensor(choice)).item()), float(values[0].item()))
    assert model.act(state, features, graph_state=graph_state, deterministic=True) == expected
