from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic
from scripts import train_simplestate_pipeline as pipeline


def small_configs(tmp_path, number=2):
    configs = pipeline.build_configs(number, tmp_path / 'outputs')
    raw = json.loads(Path('data/examples/synthetic_train.json').read_text())
    paths = {}
    for index, split in enumerate(('train', 'validation', 'test')):
        content = deepcopy(raw)
        content['jobs'][0]['operations'][0]['machines']['M1'] += index * .1
        for phase in ('static', 'dynamic'):
            path = tmp_path / 'data' / phase / split / 'instance.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
            paths[phase, split] = str(path.parent)
    for phase, pair in configs.items():
        for config in pair.values():
            config.update(hidden_dim=8, episodes=10, validation_interval=10,
                          latest_checkpoint_interval=10, checkpoint_interval=10)
            for key, split in (('train_instances','train'),('eval_instances','validation'),('test_instances','test')):
                config[key] = paths[phase, split]
    return configs


def test_three_gpu_audit_preserves_global_batch_and_defers_only_sources(tmp_path):
    configs = small_configs(tmp_path)
    report = pipeline.audit_pair(configs['static'])
    assert report['passed'], report['errors']
    model = report['models']['gnn']
    assert model['global_mini_batch_size'] == 64
    assert model['local_mini_batch_size'] == 22
    assert model['full_batch_valid_samples_per_rank'] == [22,22,20]
    assert model['observation_dim'] == 18 and model['action_feature_dim'] == 21
    assert pipeline.audit_pair(configs['dynamic'], pending_sources=True)['passed']
    assert not pipeline.audit_pair(configs['dynamic'])['passed']
    configs['static']['mlp']['mini_batch_size'] = 0
    assert not pipeline.audit_pair(configs['static'])['passed']


@pytest.mark.parametrize('number,gpus', [(1,[0,1]),(2,[2,3,4]),(3,[5,6,7])])
def test_config_routes_cost0_and_matching_static_best(tmp_path, number, gpus):
    configs = pipeline.build_configs(number, tmp_path)
    assert pipeline.GPU_GROUPS[number] == gpus
    for phase, pair in configs.items():
        for model, config in pair.items():
            assert config['distributed_world_size'] == len(gpus)
            assert f'dataset{number}_cost0/{phase}/train' in config['train_instances']
            assert config['output_dir'] == str(tmp_path / f'simplestate_{"jingtai" if phase == "static" else "dongtai"}_set{number}' / model.upper())
            assert config['episodes'] == (20000 if phase == 'static' else 10000)
            if phase == 'dynamic':
                assert config['initialize_from'] == str(tmp_path / f'simplestate_jingtai_set{number}' / model.upper() / 'checkpoints/best.pt')
            else:
                assert 'initialize_from' not in config


@pytest.mark.parametrize('failure', [False,True])
def test_pipeline_orders_transfer_and_stops_after_failure(tmp_path, monkeypatch, failure):
    configs = small_configs(tmp_path)
    monkeypatch.setattr(pipeline,'require_fullstate_finished', lambda: None)
    monkeypatch.setattr(pipeline,'build_configs', lambda *args: configs)
    monkeypatch.setattr(pipeline,'preflight', lambda *args: {'passed':True})
    monkeypatch.setattr(torch.cuda,'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda,'device_count', lambda: 3)
    monkeypatch.setattr(pipeline,'STOP',None)
    monkeypatch.setattr(pipeline,'CHILDREN',[])
    calls = []

    class TrainingProcess:
        pid = 999999

        def __init__(self, command, **kwargs):
            assert '--nproc_per_node=3' in command
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2,3,4'
            config = json.loads(Path(command[-1]).read_text())
            calls.append(config)
            self.config = config

        def wait(self):
            if failure:
                return 1
            config = self.config
            d = Path(config['output_dir'])
            env = ResidualStressFJSPEnv(load_instance(Path(config['train_instances'])/'instance.json'),
                                       observation_profile='compact18',action_feature_profile='compact21')
            model = build_actor_critic(config['model_type'],18,21,8,
                                      node_feature_dim=13,edge_feature_dim=2,node_type_count=3,edge_type_count=10)
            checkpoint = dict(config=config,episode=10,environment_semantics_version=env.ENVIRONMENT_SEMANTICS_VERSION,
                              dynamic_input_semantics_version=env.DYNAMIC_INPUT_SEMANTICS_VERSION,
                              observation_profile='compact18',action_feature_profile='compact21',
                              observation_dim=18,action_feature_dim=21,hidden_dim=8,
                              model_type=config['model_type'],model_kwargs=getattr(model,'checkpoint_kwargs',{}),
                              model_state=model.state_dict())
            (d/'checkpoints').mkdir()
            for name in ('best','latest','final'):
                torch.save(checkpoint,d/f'checkpoints/{name}.pt')
            (d/'training_history.json').write_text(json.dumps([{'episode':i+1,'total_environment_steps':100} for i in range(10)]))
            (d/'checkpoint_summary.json').write_text(json.dumps({'best_validation_objective':2.0}))
            return 0

        def poll(self):
            return 1 if failure else 0

    monkeypatch.setattr(pipeline.subprocess,'Popen',TrainingProcess)
    if failure:
        with pytest.raises(RuntimeError,match='exited 1'):
            pipeline.run_dataset(2,tmp_path/'outputs',7)
        assert len(calls) == 1
    else:
        pipeline.run_dataset(2,tmp_path/'outputs',7)
        assert [(c['model_type'], 'initialize_from' in c) for c in calls] == [
            ('mlp',False),('gnn_attention',False),('mlp',True),('gnn_attention',True)]
    state = json.loads((tmp_path/'outputs/simplestate_jingtai_set2/pipeline_status.json').read_text())
    assert state['status'] == ('failed' if failure else 'complete')
    if failure:
        assert state['jobs']['dynamic/MLP']['status'] == 'queued'
    else:
        sources = json.loads((tmp_path/'outputs/simplestate_dongtai_set2/initialization_sources.json').read_text())
        assert sources['GNN']['optimizer_reset']
    with pytest.raises(FileExistsError,match='overwrite'):
        pipeline.run_dataset(2,tmp_path/'outputs',7)


def test_check_does_not_launch_or_create_outputs(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(pipeline,'preflight',lambda *args: {'passed':True})
    monkeypatch.setattr(pipeline.subprocess,'Popen',lambda *args,**kwargs: pytest.fail('must not launch'))
    assert pipeline.main(['--check','--output-root',str(tmp_path/'unused')]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
    assert not (tmp_path/'unused').exists()


def test_existing_fullstate_chain_must_finish(tmp_path, monkeypatch):
    path = tmp_path/'outputs/fullstate_jingtai_set2/pipeline_status.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status':'training'}))
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='has not completed'):
        pipeline.require_fullstate_finished()


def test_all_launches_three_independent_dataset_supervisors(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'require_fullstate_finished',lambda: None)
    monkeypatch.setattr(pipeline,'STOP',None)
    monkeypatch.setattr(pipeline,'CHILDREN',[])
    calls = []

    class Supervisor:
        pid = 999999

        def __init__(self, command, **kwargs):
            calls.append(command)
            assert kwargs['start_new_session']

        def poll(self):
            return 0

    monkeypatch.setattr(pipeline.subprocess,'Popen',Supervisor)
    pipeline.run_all(tmp_path,7)
    assert [c[c.index('--dataset')+1] for c in calls] == ['1','2','3']
    assert not pipeline.CHILDREN
