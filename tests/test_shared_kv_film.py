from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ppo_rs.algorithms import PPO, PPOConfig, collect_episode
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic, build_model_from_checkpoint, batch_graph_states
from ppo_rs.models.graph_attention import HeterogeneousGraphAttentionLayer
from ppo_rs.train import _validate_transfer_schema
from tests.helpers import tiny_instance


def _model(env, **options):
    return build_actor_critic(
        "gnn_attention", env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM,
        options.pop("hidden_dim", 8), node_feature_dim=13, edge_feature_dim=2,
        node_type_count=3, edge_type_count=10, **options,
    )


def _reference(layer, nodes, edge_index, edge_types, features):
    """Independent per-edge implementation, without shared node precomputation."""
    source, destination = edge_index
    queries = layer.query(nodes).view(-1, layer.heads, layer.head_dim)
    aggregate = torch.zeros_like(nodes)
    for relation in range(layer.relation_count):
        keep = edge_types == relation
        if not keep.any():
            continue
        src, dst, edges = source[keep], destination[keep], features[keep]
        keys = F.linear(nodes[src], layer.shared_key.weight)
        keys = (keys * layer.film_key_scale[relation]).view(-1, layer.heads, layer.head_dim)
        values = F.linear(nodes[src], layer.shared_value.weight)
        values = values * layer.film_value_scale[relation] + layer.film_value_bias[relation]
        values = (values + layer.edge_values[relation](edges)).view(-1, layer.heads, layer.head_dim)
        scores = (queries[dst] * keys).sum(-1) / math.sqrt(layer.head_dim)
        scores = scores + layer.edge_biases[relation](edges)
        attention = torch.zeros_like(scores)
        for node in dst.unique():
            group = dst == node
            attention[group] = scores[group].softmax(dim=0)
        aggregate = aggregate.index_add(0, dst, (attention.unsqueeze(-1) * values).flatten(1))
    updated = layer.self_projection(nodes) + layer.message_projection(aggregate)
    return layer.norm(nodes + layer.activation(updated))


@pytest.mark.parametrize("indexed", [False, True])
def test_film_forward_and_all_gradients_match_per_edge_reference(indexed):
    torch.manual_seed(31)
    layer = HeterogeneousGraphAttentionLayer(8, 2, 3, 2, shared_kv_film=True)
    with torch.no_grad():
        for table in (layer.film_key_scale, layer.film_value_scale, layer.film_value_bias):
            table.normal_()  # Includes negative scales, unlike a sigmoid gate.
    reference = deepcopy(layer)
    nodes = torch.randn(5, 8, requires_grad=True)
    other_nodes = nodes.detach().clone().requires_grad_()
    edges = torch.tensor([[0, 1, 2, 0, 3, 4], [3, 3, 3, 4, 4, 1]])
    types = torch.tensor([0, 0, 1, 1, 1, 0])  # Relation 2 intentionally absent.
    features = torch.randn(6, 2)
    indices = {r: torch.where(types == r)[0] for r in range(3)} if indexed else None
    actual = layer(nodes, edges, types, features, indices)
    expected = _reference(reference, other_nodes, edges, types, features)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(nodes.grad, other_nodes.grad, rtol=2e-5, atol=2e-6)
    for (name, parameter), (_, other) in zip(layer.named_parameters(), reference.named_parameters()):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-6, msg=name)
    assert layer.film_key_scale.grad[:2].abs().sum() > 0
    assert layer.film_key_scale.grad[2].eq(0).all()
    assert layer.film_value_scale.grad[2].eq(0).all()
    assert layer.film_value_bias.grad[2].eq(0).all()


def test_identity_film_matches_tied_legacy_and_layers_are_independent():
    torch.manual_seed(7)
    layer = HeterogeneousGraphAttentionLayer(8, 2, 3, 2, shared_kv_film=True)
    legacy = HeterogeneousGraphAttentionLayer(8, 2, 3, 2)
    common = {key: value for key, value in layer.state_dict().items() if key in legacy.state_dict()}
    legacy.load_state_dict(common, strict=False)
    with torch.no_grad():
        for key, value in zip(legacy.keys, legacy.values):
            key.weight.copy_(layer.shared_key.weight)
            value.weight.copy_(layer.shared_value.weight)
    nodes = torch.randn(4, 8)
    edges = torch.tensor([[0, 1, 2, 3], [2, 2, 3, 1]])
    types, features = torch.tensor([0, 0, 1, 2]), torch.randn(4, 2)
    torch.testing.assert_close(layer(nodes, edges, types, features), legacy(nodes, edges, types, features))
    model = _model(ResidualStressFJSPEnv(tiny_instance()), shared_kv_film=True)
    first, second = model.message_layers
    assert first.shared_key.weight is not first.shared_value.weight
    assert first.shared_key.weight is not second.shared_key.weight
    assert first.film_key_scale is not second.film_key_scale


def test_node_film_does_not_scale_or_shift_edge_value_projection():
    torch.manual_seed(19)
    layer = HeterogeneousGraphAttentionLayer(4, 2, 1, 1, shared_kv_film=True)
    with torch.no_grad():
        layer.shared_value.weight.zero_()
        layer.film_value_scale.fill_(17.)
    nodes, features = torch.randn(2, 4), torch.randn(1, 2)
    edges, types = torch.tensor([[0], [1]]), torch.tensor([0])
    actual = layer(nodes, edges, types, features)
    aggregate = torch.zeros_like(nodes)
    aggregate[1] = layer.edge_values[0](features)[0]
    expected = layer.norm(nodes + layer.activation(layer.self_projection(nodes) + layer.message_projection(aggregate)))
    torch.testing.assert_close(actual, expected)
    # V bias enters the aggregated message even for a singleton relation.
    with torch.no_grad():
        layer.film_value_bias.copy_(torch.tensor([[1., -2., 3., -4.]]))
    assert not torch.allclose(actual[1], layer(nodes, edges, types, features)[1])


def test_static_type_key_bias_cancels_but_key_scaling_changes_attention():
    query = torch.tensor([.5, -.8])
    keys = torch.tensor([[1., 2.], [-2., .5]])
    scores = keys @ query
    shifted = (keys + torch.tensor([3., -1.])) @ query
    torch.testing.assert_close(scores.softmax(0), shifted.softmax(0))
    scaled = (keys * torch.tensor([2., -.5])) @ query
    assert not torch.allclose(scores.softmax(0), scaled.softmax(0))


def test_schema_reload_legacy_defaults_and_invalid_combinations():
    env = ResidualStressFJSPEnv(tiny_instance())
    state, info = env.reset(seed=7)
    torch.manual_seed(13)
    legacy = _model(env)
    torch.manual_seed(13)
    explicit = _model(env, shared_kv_film=False)
    assert legacy.checkpoint_kwargs == explicit.checkpoint_kwargs
    assert 'shared_kv_film' not in legacy.checkpoint_kwargs
    assert not any('film' in key or 'shared_key' in key for key in legacy.state_dict())
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    model = _model(env, shared_kv_film=True)
    assert not model.physical_gating and not hasattr(model, 'physical_gate')
    checkpoint = dict(model_type='gnn_attention', observation_dim=env.OBSERVATION_DIM,
                      action_feature_dim=env.ACTION_FEATURE_DIM, hidden_dim=8,
                      model_kwargs=model.checkpoint_kwargs, model_state=model.state_dict(),
                      environment_semantics_version=env.ENVIRONMENT_SEMANTICS_VERSION,
                      observation_profile=env.observation_profile, action_feature_profile=env.action_feature_profile)
    loaded = build_model_from_checkpoint(checkpoint, torch.device('cpu'))
    graph = env.get_graph_state(info['valid_actions'])
    assert model.act(state, info['action_features'], graph_state=graph, deterministic=True) == loaded.act(state, info['action_features'], graph_state=graph, deterministic=True)
    with pytest.raises(ValueError, match='graph architecture'):
        _validate_transfer_schema(checkpoint, legacy, env.observation_profile, env.action_feature_profile)
    with pytest.raises(ValueError, match='schema version'):
        build_model_from_checkpoint(dict(checkpoint, model_kwargs=dict(model.checkpoint_kwargs, shared_kv_film_schema_version=2)), torch.device('cpu'))
    with pytest.raises(ValueError, match='only one'):
        _model(env, shared_kv_film=True, physical_gating=True)
    with pytest.raises(ValueError, match='boolean'):
        _model(env, shared_kv_film='true')
    with pytest.raises(ValueError, match='only for gnn_attention'):
        build_actor_critic('mlp', 23, 25, 8, shared_kv_film=True)


def test_rollout_log_probs_and_ppo_update_without_physical_fields():
    torch.manual_seed(7)
    env = ResidualStressFJSPEnv(tiny_instance(jobs=2))
    model = _model(env, shared_kv_film=True)
    buffer, metrics = collect_episode(env, model, seed=7)
    assert metrics['terminated'] and not metrics['truncated']
    assert all('physical_node_features' not in graph for graph in buffer.graph_states)
    data = buffer.tensors(device=torch.device('cpu'), gamma=1., gae_lambda=.95)
    graph = batch_graph_states(buffer.graph_states, action_slots=data['action_features'].shape[1], device=torch.device('cpu'))
    with torch.no_grad():
        logs, _, _ = model.evaluate_actions(data['observations'], data['action_features'], data['action_masks'], data['choices'], graph_batch=graph)
    torch.testing.assert_close(logs, data['old_log_probs'], atol=2e-6, rtol=2e-6)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    losses = PPO(model, PPOConfig(ppo_epochs=1, mini_batch_size=64)).update(buffer)
    assert all(math.isfinite(value) for value in losses.values())
    for suffix in ('shared_key.weight', 'shared_value.weight', 'film_key_scale', 'film_value_scale', 'film_value_bias'):
        assert any(not torch.equal(before[key], value) for key, value in model.named_parameters() if key.endswith(suffix)), suffix


def test_empty_edges_and_missing_metadata():
    layer = HeterogeneousGraphAttentionLayer(8, 2, 3, 2, shared_kv_film=True)
    nodes = torch.randn(2, 8)
    result = layer(nodes, torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, 2))
    expected = layer.norm(nodes + layer.activation(layer.self_projection(nodes) + layer.message_projection(torch.zeros_like(nodes))))
    torch.testing.assert_close(result, expected)


def test_configs_and_full_state_parameter_count():
    for phase, episodes in (('static', 20000), ('dynamic', 10000)):
        config = json.loads(Path(f'configs/shared_kv_film/full_gnn_{phase}.json').read_text())
        assert config['shared_kv_film'] is True and config['physical_gating'] is False
        assert config['episodes'] == episodes and config['hidden_dim'] == 128
        assert config['observation_profile'] == 'unified23' and config['action_feature_profile'] == 'enhanced25'
        assert config['output_dir_mode'] == 'unique'
    env = ResidualStressFJSPEnv(tiny_instance(), observation_profile='unified23', action_feature_profile='enhanced25')
    legacy = _model(env, hidden_dim=128)
    film = _model(env, hidden_dim=128, shared_kv_film=True)
    assert sum(p.numel() for p in legacy.parameters()) == 1_002_274
    assert sum(p.numel() for p in film.parameters()) == 420_130
