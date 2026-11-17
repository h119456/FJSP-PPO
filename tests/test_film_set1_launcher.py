import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import train_gnn_film_set1 as launcher


def test_two_card_config_and_matching_transfer():
    stages = launcher.configs(Path.cwd())
    for phase,c in stages.items():
        assert c['shared_kv_film'] and not c['physical_gating']
        assert c['distributed_world_size']==2 and c['mini_batch_size']==64
        assert c['observation_profile']=='unified23' and c['action_feature_profile']=='enhanced25'
        assert c['episodes']==(20000 if phase=='static' else 10000)
        assert c['output_dir'].endswith(f'GNN+FiLM_set1/{phase}')
    assert stages['dynamic']['initialize_from']==str(Path(stages['static']['output_dir'])/'checkpoints/best.pt')


def test_completion_requires_full_history_and_checkpoints(tmp_path):
    (tmp_path/'training_history.json').write_text(json.dumps([{'episode':1}]))
    with pytest.raises(RuntimeError,match='Incomplete'):
        launcher.verify_complete(tmp_path,2)
    (tmp_path/'training_history.json').write_text(json.dumps([{'episode':1},{'episode':2}]))
    with pytest.raises(RuntimeError,match='Missing'):
        launcher.verify_complete(tmp_path,2)
    (tmp_path/'checkpoints').mkdir()
    for name in ('best','latest','final'):
        (tmp_path/f'checkpoints/{name}.pt').touch()
    launcher.verify_complete(tmp_path,2)


@pytest.mark.parametrize('fails',[False,True])
def test_static_then_dynamic_or_stop_on_failure(tmp_path,monkeypatch,fails):
    from scripts import wait_train_gnn_gate_set1 as waiter
    (tmp_path/'outputs').mkdir()
    stages={p:dict(episodes=2,output_dir=str(tmp_path/'outputs/GNN+FiLM_set1'/p)) for p in ('static','dynamic')}
    monkeypatch.setattr(launcher,'configs',lambda root,dataset=1,world_size=2:stages)
    monkeypatch.setattr(waiter,'gpu_processes',lambda gpu_ids=(0,1):[])
    monkeypatch.setattr(launcher,'audit',lambda config:dict(passed=True,errors=[]))
    monkeypatch.setattr(launcher.signal,'signal',Mock())
    completed=[]
    monkeypatch.setattr(launcher,'verify_complete',lambda directory,target:completed.append(directory.name))
    child=Mock(pid=123);child.wait.return_value=1 if fails else 0;child.poll.return_value=child.wait.return_value
    spawn=Mock(return_value=child);monkeypatch.setattr(launcher.subprocess,'Popen',spawn)
    if fails:
        with pytest.raises(RuntimeError,match='exited'):
            launcher.run(tmp_path)
        assert spawn.call_count==1 and completed==[]
    else:
        assert launcher.run(tmp_path)==0
        assert spawn.call_count==2 and completed==['static','dynamic']
    status=json.loads((tmp_path/'outputs/GNN+FiLM_set1/pipeline_status.json').read_text())
    assert status['status']==('failed' if fails else 'complete')


def test_existing_output_refused(tmp_path):
    (tmp_path/'outputs/GNN+FiLM_set1').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='overwrite'):
        launcher.run(tmp_path)


def test_dataset2_uses_matching_splits_and_its_own_static_checkpoint():
    stages = launcher.configs(Path.cwd(), dataset=2)
    for phase, config in stages.items():
        for key, split in (('train_instances','train'),('eval_instances','validation'),('test_instances','test')):
            assert config[key] == str(Path.cwd()/f'data/dataset2_cost0/{phase}/{split}')
        assert config['shared_kv_film'] and not config['physical_gating']
        assert config['distributed_world_size'] == 2 and config['mini_batch_size'] == 64
        assert config['output_dir'].endswith(f'GNN+FiLM_set2/{phase}')
    assert 'initialize_from' not in stages['static']
    assert stages['dynamic']['initialize_from'] == str(Path(stages['static']['output_dir'])/'checkpoints/best.pt')


@pytest.mark.parametrize('fail',[False,True])
def test_paused_evaluator_resumes_on_success_and_failure(monkeypatch,fail):
    monkeypatch.setattr(launcher,'_wait_evaluator_stopped',lambda pid:None)
    monkeypatch.setattr(launcher,'_evaluator_identity',lambda pid:'same-start-time')
    monkeypatch.setattr(launcher,'_evaluator_children',lambda pid:'')
    kill=Mock();monkeypatch.setattr(launcher.os,'kill',kill)
    if fail:
        with pytest.raises(ValueError):
            with launcher.paused_evaluator(42):
                raise ValueError('training failed')
    else:
        with launcher.paused_evaluator(42):
            pass
    assert [call.args for call in kill.call_args_list] == [(42,launcher.signal.SIGSTOP),(42,launcher.signal.SIGCONT)]


def test_active_evaluator_is_not_paused(monkeypatch):
    monkeypatch.setattr(launcher,'_evaluator_identity',lambda pid:'same-start-time')
    monkeypatch.setattr(launcher,'_evaluator_children',lambda pid:'123')
    kill=Mock();monkeypatch.setattr(launcher.os,'kill',kill)
    with pytest.raises(RuntimeError,match='active workers'):
        with launcher.paused_evaluator(42):
            pytest.fail('must not run training')
    kill.assert_not_called()


def test_evaluator_spawn_race_resumes_and_refuses_training(monkeypatch):
    monkeypatch.setattr(launcher,'_wait_evaluator_stopped',lambda pid:None)
    monkeypatch.setattr(launcher,'_evaluator_identity',lambda pid:'same-start-time')
    children=iter(['','123'])
    monkeypatch.setattr(launcher,'_evaluator_children',lambda pid:next(children))
    kill=Mock();monkeypatch.setattr(launcher.os,'kill',kill)
    with pytest.raises(RuntimeError,match='active workers'):
        with launcher.paused_evaluator(42):
            pytest.fail('must not run training')
    assert [call.args for call in kill.call_args_list] == [(42,launcher.signal.SIGSTOP),(42,launcher.signal.SIGCONT)]


def test_reused_pid_is_not_resumed(monkeypatch):
    monkeypatch.setattr(launcher,'_wait_evaluator_stopped',lambda pid:None)
    identities=iter(['old-start-time','new-start-time'])
    monkeypatch.setattr(launcher,'_evaluator_identity',lambda pid:next(identities))
    monkeypatch.setattr(launcher,'_evaluator_children',lambda pid:'')
    kill=Mock();monkeypatch.setattr(launcher.os,'kill',kill)
    with launcher.paused_evaluator(42):
        pass
    kill.assert_called_once_with(42,launcher.signal.SIGSTOP)


def test_dataset3_three_card_config_and_matching_transfer():
    stages = launcher.configs(Path.cwd(), dataset=3, world_size=3)
    for phase, config in stages.items():
        assert config['shared_kv_film'] and not config['physical_gating']
        assert config['distributed_world_size'] == 3 and config['mini_batch_size'] == 64
        assert config['observation_profile'] == 'unified23' and config['action_feature_profile'] == 'enhanced25'
        assert config['episodes'] == (20000 if phase == 'static' else 10000)
        assert config['train_instances'] == str(Path.cwd()/f'data/dataset3_cost0/{phase}/train')
        assert config['output_dir'].endswith(f'GNN+FiLM_set3/{phase}')
    assert 'initialize_from' not in stages['static']
    assert stages['dynamic']['initialize_from'] == str(Path(stages['static']['output_dir'])/'checkpoints/best.pt')


@pytest.mark.parametrize('selection',['2,2,4','0','0,1,2,3','-1,2','2,8','2,x,4'])
def test_invalid_gpu_selection_is_rejected(selection):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        launcher.gpu_selection(selection)


@pytest.mark.parametrize('fails',[False,True])
def test_set3_uses_three_requested_gpus_and_stops_after_static_failure(tmp_path,monkeypatch,fails):
    from scripts import wait_train_gnn_gate_set1 as waiter
    (tmp_path/'outputs').mkdir()
    stages={p:dict(episodes=2) for p in ('static','dynamic')}
    build=Mock(return_value=stages);monkeypatch.setattr(launcher,'configs',build)
    probe=Mock(return_value=[]);monkeypatch.setattr(waiter,'gpu_processes',probe)
    monkeypatch.setattr(launcher,'audit',lambda config:dict(passed=True,errors=[]))
    monkeypatch.setattr(launcher.signal,'signal',Mock())
    verify=Mock();monkeypatch.setattr(launcher,'verify_complete',verify)
    child=Mock(pid=123);child.wait.return_value=1 if fails else 0;child.poll.return_value=child.wait.return_value
    spawn=Mock(return_value=child);monkeypatch.setattr(launcher.subprocess,'Popen',spawn)
    if fails:
        with pytest.raises(RuntimeError,match='exited'):
            launcher.run(tmp_path,dataset=3,gpu_ids=(2,3,4))
    else:
        assert launcher.run(tmp_path,dataset=3,gpu_ids=(2,3,4)) == 0
    assert spawn.call_count == (1 if fails else 2)
    probe.assert_called_once_with((2,3,4))
    build.assert_called_once_with(tmp_path,3,3)
    assert '--nproc_per_node=3' in spawn.call_args.args[0]
    assert spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2,3,4'
    status=json.loads((tmp_path/'outputs/GNN+FiLM_set3/pipeline_status.json').read_text())
    assert status['gpu_ids'] == [2,3,4]
    assert status['status'] == ('failed' if fails else 'complete')


def test_set3_busy_card_refuses_launch_before_output_creation(tmp_path,monkeypatch):
    from scripts import wait_train_gnn_gate_set1 as waiter
    (tmp_path/'outputs').mkdir()
    monkeypatch.setattr(waiter,'gpu_processes',lambda ids:['GPU-card4, 123'])
    with pytest.raises(RuntimeError,match='occupied'):
        launcher.run(tmp_path,dataset=3,gpu_ids=(2,3,4))
    assert not (tmp_path/'outputs/GNN+FiLM_set3').exists()
