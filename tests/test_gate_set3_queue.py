import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import queue_gnn_gate_set3 as queue


def test_set3_three_card_config():
    configs=queue.build_configs(Path.cwd())
    assert queue.GPU_IDS==(5,6,7)
    for phase,c in configs.items():
        assert c['physical_gating'] and not c['shared_kv_film']
        assert c['mini_batch_size']==64 and c['distributed_world_size']==3
        assert c['episodes']==(20000 if phase=='static' else 10000)
        assert c['observation_profile']=='unified23' and c['action_feature_profile']=='enhanced25'
        assert f'dataset3_cost0/{phase}/train' in c['train_instances']
        assert c['output_dir'].endswith(f'GNN+gate_set3/{phase}')
    assert configs['dynamic']['initialize_from']==str(Path(configs['static']['output_dir'])/'checkpoints/best.pt')


def test_waits_for_whole_chain_and_card7(tmp_path,monkeypatch):
    p=tmp_path/'outputs/simplestate_jingtai_set3/pipeline_status.json';p.parent.mkdir(parents=True)
    s=dict(status='training',active_job='dynamic/GNN',jobs={j:dict(status='complete') for j in queue.JOBS})
    p.write_text(json.dumps(s));probe=Mock(return_value=[]);monkeypatch.setattr(queue,'gpu_processes',probe)
    assert not queue.readiness(tmp_path)[0];probe.assert_not_called()
    s['status']='complete';p.write_text(json.dumps(s))
    probe.return_value=['GPU-card7, 123']
    assert not queue.readiness(tmp_path)[0];probe.assert_called_with((5,6,7))
    probe.return_value=[];assert queue.readiness(tmp_path)[0]
    s['status']='failed';p.write_text(json.dumps(s))
    with pytest.raises(RuntimeError,match='stopped'):queue.readiness(tmp_path)


def test_audit_preserves_real_64_batch(monkeypatch):
    from scripts import check_comparison as check
    model=Mock(return_value={});monkeypatch.setattr(check,'_model_summary',model)
    monkeypatch.setattr(check,'_dataset_summary',lambda c:({},[]))
    c=queue.build_configs(Path.cwd())['dynamic']
    result=queue.audit(c,pending_source=True)
    assert result['passed']
    assert result['effective_config']['mini_batch_size']==64
    assert model.call_args.args[0]['mini_batch_size']==64
    assert 'initialize_from' not in model.call_args.args[0]
    assert result['model']['full_batch_valid_samples_per_rank']==[22,22,20]


@pytest.mark.parametrize('fail',[False,True])
def test_two_idle_probes_and_static_dynamic_failure_stop(tmp_path,monkeypatch,fail):
    (tmp_path/'outputs').mkdir()
    stages={p:dict(episodes=2) for p in ('static','dynamic')}
    monkeypatch.setattr(queue,'build_configs',lambda r:stages)
    monkeypatch.setattr(queue,'audit',lambda c,**k:dict(passed=True,errors=[]))
    probes=iter([(False,'busy'),(True,'idle'),(True,'idle')]);monkeypatch.setattr(queue,'readiness',lambda r:next(probes))
    sleep=Mock();monkeypatch.setattr(queue.time,'sleep',sleep);monkeypatch.setattr(queue.signal,'signal',Mock())
    verify=Mock();monkeypatch.setattr(queue,'verify_complete',verify)
    child=Mock(pid=123);child.wait.return_value=1 if fail else 0;child.poll.return_value=child.wait.return_value
    launch=Mock(return_value=child);monkeypatch.setattr(queue.subprocess,'Popen',launch)
    if fail:
        with pytest.raises(RuntimeError,match='exited'):queue.run(tmp_path)
        assert launch.call_count==1 and verify.call_count==0
    else:
        assert queue.run(tmp_path)==0
        assert launch.call_count==2 and verify.call_count==2
        assert '--nproc_per_node=3' in launch.call_args.args[0]
    assert sleep.call_count==2
    s=json.loads((tmp_path/'outputs/gnn_gate_set3_queue_status.json').read_text())
    assert s['status']==('failed' if fail else 'complete')


def test_no_overwrite(tmp_path):
    (tmp_path/'outputs/GNN+gate_set3').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='overwrite'):queue.run(tmp_path)


def test_gpu_probe_selects_three_cards(monkeypatch):
    from scripts import wait_train_gnn_gate_set1 as waiter
    outputs=iter(['0, GPU-a\n5, GPU-c\n6, GPU-d\n7, GPU-e\n','GPU-a, 1\nGPU-e, 4\n'])
    monkeypatch.setattr(waiter.subprocess,'check_output',lambda *a,**k:next(outputs))
    assert waiter.gpu_processes((5,6,7))==['GPU-e, 4']


def test_explicit_start_bypasses_failed_predecessor_but_still_checks_gpus(tmp_path,monkeypatch):
    p=tmp_path/'outputs/simplestate_jingtai_set3/pipeline_status.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(status='failed')))
    probe=Mock(return_value=['GPU-card7, 123']);monkeypatch.setattr(queue,'gpu_processes',probe)
    assert not queue.readiness(tmp_path,start_now=True)[0]
    probe.assert_called_with((5,6,7))
    probe.return_value=[]
    assert queue.readiness(tmp_path,start_now=True)[0]
    with pytest.raises(RuntimeError,match='stopped'):
        queue.readiness(tmp_path)


def test_explicit_start_keeps_no_overwrite_guard(tmp_path):
    (tmp_path/'outputs/GNN+gate_set3').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='overwrite'):
        queue.readiness(tmp_path,start_now=True)
