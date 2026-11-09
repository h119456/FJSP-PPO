from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ppo_rs.algorithms import PPO, PPOConfig, collect_episode
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic, build_model_from_checkpoint, batch_graph_states
from ppo_rs.models.graph_attention import HeterogeneousGraphAttentionLayer
from ppo_rs.models.physical_gate import PhysicalEdgeGate
from ppo_rs.train import train, _validate_transfer_schema
from tests.helpers import choose, tiny_instance


def interval_env():
    env = ResidualStressFJSPEnv(tiny_instance())
    env.step(choose(env, "NORMAL", machine="M1"))
    while env.jobs["J1"].busy:
        env.step(choose(env, "WAIT"))
    return env


def model_for(env, enabled=True):
    return build_actor_critic(
        "gnn_attention", env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8,
        node_feature_dim=env.GRAPH_NODE_FEATURE_DIM, edge_feature_dim=2,
        node_type_count=3, edge_type_count=10, physical_gating=enabled,
    )


def test_six_signals_stage_mapping_and_snapshot():
    env = interval_env()
    graph = env.get_graph_state(include_physics=True)
    old = env.get_graph_state()
    assert set(graph) - set(old) == {"physical_node_features", "physical_edge_mask"}
    for key in old:
        np.testing.assert_array_equal(graph[key], old[key])
    previous = env._graph_operation_nodes["J1", "O1"]
    upcoming = env._graph_operation_nodes["J1", "O2"]
    scale = max(1.0, env._duration_scale)
    signal = graph["physical_node_features"][previous]
    np.testing.assert_allclose(signal, [0., 3./scale, 1., (3.-.65)/scale, -1., 0.])
    np.testing.assert_array_equal(signal, graph["physical_node_features"][upcoming])
    sources, destinations = graph["edge_index"]
    for index, relation in enumerate(graph["edge_types"]):
        operation = sources[index] if relation % 2 == 0 else destinations[index]
        expected = (relation in (4,5) and operation == upcoming) or (relation >= 6 and operation == previous)
        assert bool(graph["physical_edge_mask"][index]) == expected
    snapshot = graph["physical_node_features"].copy()
    # Known resource contention makes the gain negative; it must not be clipped.
    env.ledger.available_at["R"] = env.current_time + 10.
    later = env.get_graph_state(include_physics=True)
    assert later["physical_node_features"][previous,3] == pytest.approx((3.-10.-.65)/scale)
    np.testing.assert_array_equal(graph["physical_node_features"], snapshot)


def test_relief_changes_physical_margin_and_current_interval_flag():
    env = interval_env()
    env.step(choose(env,"STRESS_RELIEF"))
    while env.jobs['J1'].busy:
        env.step(choose(env,'WAIT'))
    graph = env.get_graph_state(include_physics=True)
    signal = graph['physical_node_features'][env._graph_operation_nodes['J1','O1']]
    assert signal[0] == pytest.approx(1.)  # M3 stabilizes at A, not zero.
    assert signal[1] == 0.
    assert signal[2] == pytest.approx(-1.)
    assert signal[5] == 1.
    env.step(choose(env,'COMPENSATION'))
    while env.jobs['J1'].busy:
        env.step(choose(env,'WAIT'))
    signal = env.get_graph_state(include_physics=True)['physical_node_features'][env._graph_operation_nodes['J1','O1']]
    assert signal[0] == pytest.approx(.1)
    assert signal[2] == pytest.approx(.8)
    assert signal[5] == 1.


def test_gate_groups_reverse_edges_and_inapplicable_edges():
    env = interval_env()
    state = env.get_graph_state(include_physics=True)
    graph = batch_graph_states([state], action_slots=len(state['action_nodes']),device=torch.device('cpu'))
    gate = PhysicalEdgeGate(4)
    for head,value in zip(gate.gate_heads,(.2,.5,.8)):
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.constant_(head[-1].bias, math.log(value/(1-value)))
    result = gate(graph)
    for index,relation in enumerate(state['edge_types']):
        expected = (.2,.5,.8)[(relation-4)//2] if state['physical_edge_mask'][index] else 1.
        torch.testing.assert_close(result[index],torch.full((4,),expected))
    result.sum().backward()
    assert all(head[-1].bias.grad is not None for head in gate.gate_heads)
    # Other variants must fail explicitly rather than silently dropping physics.
    legacy = batch_graph_states([env.get_graph_state()],action_slots=len(state['action_nodes']),device=torch.device('cpu'))
    with pytest.raises(ValueError,match='include_physics'):
        gate(legacy)
    with pytest.raises(ValueError,match='same physical'):
        batch_graph_states([state,env.get_graph_state()],action_slots=len(state['action_nodes']),device=torch.device('cpu'))


def test_gate_batch_alignment_empty_edges_and_encoder_gradient():
    env = interval_env()
    state = env.get_graph_state(include_physics=True)
    graph = batch_graph_states([state,state],action_slots=len(state['action_nodes']),device=torch.device('cpu'))
    gate = PhysicalEdgeGate(4)
    result = gate(graph)
    edges = len(state['edge_types'])
    torch.testing.assert_close(result[:edges],result[edges:])
    result.sum().backward()
    assert gate.encoder[0].weight.grad.abs().sum() > 0
    empty = deepcopy(state)
    empty['edge_index'] = empty['edge_index'][:,:0]
    for key in ('edge_types','edge_features','physical_edge_mask'):
        empty[key] = empty[key][:0]
    graph = batch_graph_states([empty],action_slots=len(state['action_nodes']),device=torch.device('cpu'))
    assert gate(graph).shape == (0,4)


def test_post_softmax_gate_controls_singleton_and_complete_edge_value():
    torch.manual_seed(5)
    layer = HeterogeneousGraphAttentionLayer(4,2,1,1)
    with torch.no_grad():
        layer.values[0].weight.zero_()  # Isolate the edge-feature value path.
        layer.edge_values[0].weight.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]))
        layer.self_projection.weight.zero_(); layer.self_projection.bias.zero_()
        layer.message_projection.weight.copy_(torch.eye(4)); layer.message_projection.bias.zero_()
    nodes = torch.tensor([[.1,.3,-.2,.5],[.5,-.1,.2,.4]])
    edges = torch.tensor([[0],[1]])
    types = torch.zeros(1,dtype=torch.long)
    features = torch.tensor([[.4,.2]])
    base = layer(nodes,edges,types,features)
    ones = layer(nodes,edges,types,features,edge_gates=torch.ones(1,1))
    torch.testing.assert_close(base,ones,rtol=0,atol=0)
    zero = layer(nodes,edges,types,features,edge_gates=torch.zeros(1,1))
    torch.testing.assert_close(zero,layer.norm(nodes),rtol=0,atol=0)
    assert not torch.allclose(base[1],zero[1])


def test_default_disabled_is_legacy_rng_schema_and_reload(tmp_path):
    env = interval_env()
    torch.manual_seed(13);legacy = model_for(env,False)
    torch.manual_seed(13);explicit = model_for(env,False)
    assert not any('physical' in k for k in legacy.state_dict())
    assert 'physical_gating' not in legacy.checkpoint_kwargs
    graph = env.get_graph_state()
    args = (env.get_observation(),env.get_action_features())
    assert legacy.act(*args,graph_state=graph,deterministic=True) == explicit.act(*args,graph_state=graph,deterministic=True)
    # Real old checkpoints from baseline runs must still be evaluable.
    checkpoint = dict(model_type='gnn_attention',observation_dim=23,action_feature_dim=21,hidden_dim=8,
                      model_kwargs=legacy.checkpoint_kwargs,model_state=legacy.state_dict())
    loaded = build_model_from_checkpoint(checkpoint,torch.device('cpu'))
    assert loaded.act(*args,graph_state=graph,deterministic=True) == legacy.act(*args,graph_state=graph,deterministic=True)
    gated = model_for(env,True)
    assert sum(p.numel() for p in gated.parameters())-sum(p.numel() for p in legacy.parameters()) == 900
    with pytest.raises(ValueError,match='graph architecture'):
        _validate_transfer_schema(dict(checkpoint,environment_semantics_version=7,observation_profile='unified23',action_feature_profile='compact21'),gated,'unified23','compact21')


def test_physics_never_reads_hidden_jobs_or_future_events():
    first = tiny_instance(jobs=2)
    hidden = replace(first.jobs[1],operations=tuple(replace(o,deformation_scale=100.,deformation_rate=100.) for o in first.jobs[1].operations))
    event = {'type':'job_release','time':10.,'job_id':'J2'}
    left = replace(first,dynamic_events={'mode':'fixed','events':[event]})
    right = replace(first,jobs=(first.jobs[0],hidden),dynamic_events={'mode':'fixed','events':[dict(event,time=20.),{'type':'machine_breakdown','time':30.,'machine_id':'M1','repair_duration':9.,'interruption_policy':'pause_resume'}]})
    envs = [ResidualStressFJSPEnv(i) for i in (left,right)]
    for env in envs:
        env.step(choose(env,'NORMAL',job_id='J1',machine='M1'))
        while env.jobs['J1'].busy:
            env.step(choose(env,'WAIT'))
    graphs = [e.get_graph_state(include_physics=True) for e in envs]
    for key in graphs[0]:
        np.testing.assert_array_equal(graphs[0][key],graphs[1][key],err_msg=key)


def test_disabled_compensation_has_neutral_physical_gates():
    instance = tiny_instance()
    env = ResidualStressFJSPEnv(replace(instance,compensation=replace(instance.compensation,enabled=False,trigger_deformation=None)))
    graph = env.get_graph_state(include_physics=True)
    assert not graph['physical_edge_mask'].any()
    assert np.isfinite(graph['physical_node_features']).all()


def test_rollout_reevaluation_and_optimizer_gradients():
    torch.manual_seed(7)
    env = ResidualStressFJSPEnv(tiny_instance(jobs=2))
    model = model_for(env)
    buffer,metrics = collect_episode(env,model,seed=7)
    assert metrics['terminated'] and not metrics['truncated']
    assert any(state['physical_edge_mask'].any() for state in buffer.graph_states)
    data = buffer.tensors(device=torch.device('cpu'),gamma=1.,gae_lambda=.95)
    graph = batch_graph_states(buffer.graph_states,action_slots=data['action_features'].shape[1],device=torch.device('cpu'))
    with torch.no_grad():
        logs,_,_ = model.evaluate_actions(data['observations'],data['action_features'],data['action_masks'],data['choices'],graph_batch=graph)
    torch.testing.assert_close(logs,data['old_log_probs'],atol=2e-6,rtol=2e-6)
    before = {k:p.detach().clone() for k,p in model.physical_gate.named_parameters()}
    losses = PPO(model,PPOConfig(ppo_epochs=1,mini_batch_size=64)).update(buffer)
    assert all(math.isfinite(v) for v in losses.values())
    assert any(not torch.equal(before[k],p) for k,p in model.physical_gate.named_parameters())


def test_gated_training_checkpoint_reload_and_static_dynamic_transfer(tmp_path):
    config = json.loads(Path('configs/state_ablation/full_gnn_static.json').read_text())
    config.update(train_instances='data/examples/synthetic_train.json',eval_instances='data/examples/synthetic_test.json',
                  test_instances='data/examples/synthetic_test.json',output_dir=str(tmp_path/'static'),device='cpu',
                  distributed_world_size=1,hidden_dim=8,episodes=2,episodes_per_update=2,ppo_epochs=1,
                  validation_interval=2,latest_checkpoint_interval=2,checkpoint_interval=2,
                  physical_gating=True,physical_gate_initial=.9)
    _,history,_ = train(config)
    assert len(history) == 2
    run, = (tmp_path/'static').iterdir()
    checkpoint = torch.load(run/'checkpoints/final.pt',weights_only=False)
    loaded = build_model_from_checkpoint(checkpoint,torch.device('cpu'))
    assert loaded.physical_gating and loaded.checkpoint_kwargs['physical_gate_schema_version'] == 1
    config.update(output_dir=str(tmp_path/'dynamic'))
    raw = json.loads(Path('data/examples/synthetic_train.json').read_text())
    raw['dynamic_events'] = {'mode':'fixed','events':[{'type':'machine_breakdown','time':.5,'machine_id':'M1','repair_duration':1.,'interruption_policy':'pause_resume'}]}
    dynamic = tmp_path/'events.json';dynamic.write_text(json.dumps(raw))
    config.update(train_instances=str(dynamic),eval_instances=str(dynamic))
    _,history,_ = train(config,initialize_from=run/'checkpoints/final.pt')
    assert [row['episode'] for row in history] == [1.,2.]
