import math
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic, build_model_from_checkpoint, batch_graph_states
from ppo_rs.models.graph_attention import HeterogeneousGraphAttentionLayer
from ppo_rs.train import _validate_transfer_schema
from scripts import train_shared_kv_only as launcher
from tests.helpers import tiny_instance


def model(env, **options):
    return build_actor_critic('gnn_attention', env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM,
                             options.pop('hidden_dim', 8), node_feature_dim=13, edge_feature_dim=2,
                             node_type_count=3, edge_type_count=10, **options)


def test_shared_forward_and_gradients_equal_tied_relation_projections():
    torch.manual_seed(19)
    shared = HeterogeneousGraphAttentionLayer(8, 2, 3, 2, shared_kv_only=True)
    legacy = HeterogeneousGraphAttentionLayer(8, 2, 3, 2)
    legacy.load_state_dict({k: v for k, v in shared.state_dict().items() if k in legacy.state_dict()}, strict=False)
    with torch.no_grad():
        for key, value in zip(legacy.keys, legacy.values):
            key.weight.copy_(shared.shared_key.weight); value.weight.copy_(shared.shared_value.weight)
    nodes = torch.randn(5, 8, requires_grad=True); other = nodes.detach().clone().requires_grad_()
    edges = torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 0]])
    types = torch.tensor([0, 0, 1, 1, 0]); features = torch.randn(5, 2)
    actual = shared(nodes, edges, types, features); expected = legacy(other, edges, types, features)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward(); (expected * weights).sum().backward()
    torch.testing.assert_close(nodes.grad, other.grad, atol=2e-6, rtol=2e-5)
    for layers, parameter in [(legacy.keys, shared.shared_key.weight), (legacy.values, shared.shared_value.weight)]:
        total = sum(layer.weight.grad for layer in layers if layer.weight.grad is not None)
        torch.testing.assert_close(parameter.grad, total, atol=2e-6, rtol=2e-5)
        assert parameter.grad.abs().sum() > 0
    assert not any('film' in name for name in shared.state_dict())


def test_shared_kv_retains_per_relation_softmax_and_edge_value_parameters():
    layer = HeterogeneousGraphAttentionLayer(8,2,2,2,shared_kv_only=True)
    with torch.no_grad():
        layer.query.weight.zero_();layer.shared_key.weight.zero_()
        layer.shared_value.weight.copy_(torch.eye(8))
        for projection in [*layer.edge_values,*layer.edge_biases]:projection.weight.zero_()
    nodes=torch.zeros(3,8);nodes[0,0]=1;nodes[1,1]=2
    edges=torch.tensor([[0,1],[2,2]]);features=torch.zeros(2,2)
    captured=[]
    handle=layer.message_projection.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    layer(nodes,edges,torch.tensor([0,1]),features)
    # One neighbor in each relation has weight one: relation sums, not a
    # single cross-relation softmax that would halve both messages.
    torch.testing.assert_close(captured[-1][2],nodes[0]+nodes[1])
    with torch.no_grad():layer.edge_values[1].weight[2,0]=.4
    feature=torch.tensor([[1.,0.]])
    layer(nodes,edges[:,:1],torch.tensor([0]),feature);without=captured[-1]
    layer(nodes,edges[:,:1],torch.tensor([1]),feature);with_type=captured[-1]
    expected=torch.zeros_like(with_type);expected[2,2]=.4
    torch.testing.assert_close(with_type-without,expected)
    handle.remove()


def test_zero_feature_singleton_relations_have_direction_aliasing_without_film():
    # Documents an expressivity limit of this ablation: splitting singleton
    # neighbors by relation cannot distinguish them after shared-V summation.
    torch.manual_seed(53)
    layer=HeterogeneousGraphAttentionLayer(8,2,4,2,shared_kv_only=True)
    film=HeterogeneousGraphAttentionLayer(8,2,4,2,shared_kv_film=True)
    film.load_state_dict(layer.state_dict(),strict=False)
    with torch.no_grad():film.film_value_scale[3].fill_(2.)
    nodes=torch.randn(3,8);edges=torch.tensor([[0,1],[2,2]]);features=torch.zeros(2,2)
    forward=torch.tensor([2,3]);reverse=torch.tensor([3,2])
    torch.testing.assert_close(layer(nodes,edges,forward,features),layer(nodes,edges,reverse,features),atol=1e-6,rtol=1e-6)
    assert not torch.allclose(film(nodes,edges,forward,features),film(nodes,edges,reverse,features))


def test_shared_kv_batching_does_not_mix_scheduling_instances():
    envs=[ResidualStressFJSPEnv(tiny_instance(jobs=n),observation_profile='unified23',action_feature_profile='enhanced25') for n in (2,3)]
    pairs=[env.reset(seed=7) for env in envs]
    graphs=[env.get_graph_state(info['valid_actions']) for env,(_,info) in zip(envs,pairs)]
    network=model(envs[0],shared_kv_only=True).eval()
    slots=max(len(info['action_features']) for _,info in pairs)
    states=torch.tensor([state for state,_ in pairs],dtype=torch.float32)
    features=torch.zeros(2,slots,25);mask=torch.zeros(2,slots,dtype=torch.bool)
    for i,(_,info) in enumerate(pairs):
        count=len(info['action_features']);features[i,:count]=torch.as_tensor(info['action_features']);mask[i,:count]=True
    batch=batch_graph_states(graphs,action_slots=slots,device=torch.device('cpu'))
    alone=batch_graph_states(graphs[:1],action_slots=slots,device=torch.device('cpu'))
    logits,values=network(states,features,batch,mask)
    single_logits,single_values=network(states[:1],features[:1],alone,mask[:1])
    torch.testing.assert_close(logits[0][mask[0]],single_logits[0][mask[0]],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(values[0],single_values[0],atol=1e-6,rtol=1e-6)


def test_parameters_checkpoint_roundtrip_and_incompatible_variants():
    env = ResidualStressFJSPEnv(tiny_instance(), observation_profile='unified23', action_feature_profile='enhanced25')
    state, info = env.reset(seed=7)
    shared = model(env, hidden_dim=128, shared_kv_only=True)
    assert sum(p.numel() for p in shared.parameters()) == 412450
    assert not shared.shared_kv_film and not shared.physical_gating
    assert not any('film' in name or 'physical_gate' in name for name in shared.state_dict())
    checkpoint = dict(model_type='gnn_attention', observation_dim=env.OBSERVATION_DIM,
                      action_feature_dim=env.ACTION_FEATURE_DIM, hidden_dim=128,
                      model_kwargs=shared.checkpoint_kwargs, model_state=shared.state_dict(),
                      environment_semantics_version=env.ENVIRONMENT_SEMANTICS_VERSION,
                      observation_profile=env.observation_profile, action_feature_profile=env.action_feature_profile)
    assert checkpoint['model_kwargs']['shared_kv_only'] is True
    loaded = build_model_from_checkpoint(checkpoint, torch.device('cpu'))
    graph = env.get_graph_state(info['valid_actions'])
    assert shared.act(state, info['action_features'], graph_state=graph, deterministic=True) == loaded.act(state, info['action_features'], graph_state=graph, deterministic=True)
    for other in [model(env, hidden_dim=128), model(env, hidden_dim=128, shared_kv_film=True)]:
        with pytest.raises(ValueError, match='graph architecture'):
            _validate_transfer_schema(checkpoint, other, env.observation_profile, env.action_feature_profile)
    with pytest.raises(ValueError, match='schema version'):
        build_model_from_checkpoint(dict(checkpoint, model_kwargs=dict(shared.checkpoint_kwargs, shared_kv_only_schema_version=2)), torch.device('cpu'))


def test_invalid_shared_variants_are_rejected():
    env = ResidualStressFJSPEnv(tiny_instance())
    with pytest.raises(ValueError, match='only one'):
        model(env, shared_kv_only=True, shared_kv_film=True)
    with pytest.raises(ValueError, match='only one'):
        model(env, shared_kv_only=True, physical_gating=True)
    with pytest.raises(ValueError, match='boolean'):
        model(env, shared_kv_only='true')
    with pytest.raises(ValueError, match='only for gnn_attention'):
        build_actor_critic('mlp', 23, 25, 8, shared_kv_only=True)


def test_cpu_training_saves_and_transfers_shared_only_checkpoint(tmp_path, monkeypatch):
    import importlib
    training = importlib.import_module('ppo_rs.train')
    monkeypatch.setattr(training, 'load_instances', lambda source: (tiny_instance(jobs=2),))
    config = dict(train_instances='mock_train', eval_instances='mock_validation', model_type='gnn_attention',
                  shared_kv_only=True, shared_kv_film=False, physical_gating=False, hidden_dim=8,
                  gnn_layers=2, attention_heads=2, observation_profile='unified23', action_feature_profile='enhanced25',
                  episodes=2, episodes_per_update=2, validation_interval=2, latest_checkpoint_interval=2,
                  checkpoint_interval=2, ppo_epochs=1, mini_batch_size=4, device='cpu', distributed_world_size=1,
                  enable_plots=False, seed=7, output_dir=str(tmp_path/'static'), output_dir_mode='existing')
    original, history, _ = training.train(config)
    assert len(history) == 2 and original.shared_kv_only
    path = tmp_path/'static/checkpoints/best.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['model_kwargs']['shared_kv_only']
    dynamic, history, _ = training.train(dict(config, output_dir=str(tmp_path/'dynamic')), initialize_from=path)
    assert len(history) == 2 and dynamic.shared_kv_only
    assert (tmp_path/'dynamic/checkpoints/final.pt').exists()


@pytest.mark.parametrize('dataset,devices', [(1, (4,)), (2, (5,)), (3, (6, 7))])
def test_launch_configs_match_requested_cards_and_static_source(dataset, devices):
    stages = launcher.configs(Path.cwd(), dataset)
    assert launcher.GPU_PRESETS[dataset] == devices
    for phase, c in stages.items():
        assert c['distributed_world_size'] == len(devices)
        assert c['shared_kv_only'] and not c['shared_kv_film'] and not c['physical_gating']
        assert c['mini_batch_size'] == 64 and c['observation_profile'] == 'unified23'
        assert c['action_feature_profile'] == 'enhanced25'
        assert c['episodes'] == (20000 if phase == 'static' else 10000)
        assert c['train_instances'] == str(Path.cwd()/f'data/dataset{dataset}_cost0/{phase}/train')
        assert c['output_dir'].endswith(f'GNN+sharedKV_set{dataset}/{phase}')
    assert 'initialize_from' not in stages['static']
    assert stages['dynamic']['initialize_from'] == str(Path(stages['static']['output_dir'])/'checkpoints/best.pt')


@pytest.mark.parametrize('dataset', [1, 2, 3])
@pytest.mark.parametrize('failure', [False, True])
def test_serial_stages_use_requested_devices_and_stop_on_failure(tmp_path, monkeypatch, dataset, failure):
    (tmp_path/'outputs').mkdir()
    stages = {p: dict(episodes=2) for p in ('static', 'dynamic')}
    monkeypatch.setattr(launcher, 'configs', lambda root, dataset: stages)
    monkeypatch.setattr(launcher, 'preflight', lambda configs: {p: dict(passed=True) for p in configs})
    monkeypatch.setattr(launcher, 'audit', lambda config: dict(passed=True))
    probe = Mock(return_value=[]); monkeypatch.setattr(launcher, 'gpu_processes', probe)
    monkeypatch.setattr(launcher.signal, 'signal', Mock())
    verify = Mock(); monkeypatch.setattr(launcher, 'verify_complete', verify)
    child = Mock(pid=123); child.wait.return_value = 1 if failure else 0; child.poll.return_value = child.wait.return_value
    spawn = Mock(return_value=child); monkeypatch.setattr(launcher.subprocess, 'Popen', spawn)
    if failure:
        with pytest.raises(RuntimeError, match='exited'):
            launcher.run(dataset, tmp_path)
    else:
        assert launcher.run(dataset, tmp_path) == 0
    assert spawn.call_count == (1 if failure else 2)
    assert verify.call_count == (0 if failure else 2)
    expected = ','.join(map(str, launcher.GPU_PRESETS[dataset]))
    assert spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == expected
    assert f'--nproc_per_node={len(launcher.GPU_PRESETS[dataset])}' in spawn.call_args.args[0]


def test_launcher_refuses_occupied_cards_or_existing_run(tmp_path, monkeypatch):
    (tmp_path/'outputs').mkdir()
    monkeypatch.setattr(launcher, 'gpu_processes', lambda ids: ['occupied'])
    with pytest.raises(RuntimeError, match='occupied'):
        launcher.run(1, tmp_path)
    (tmp_path/'outputs/GNN+sharedKV_set1').mkdir()
    with pytest.raises(RuntimeError, match='overwrite'):
        launcher.run(1, tmp_path)
