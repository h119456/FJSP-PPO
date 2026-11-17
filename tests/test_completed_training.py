import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import test_completed_training as runner


def test_jobs_use_training_gpu_groups_and_film_waits_for_pipeline():
    jobs = runner.jobs(Path('/project'))
    assert len(jobs) == len({j['id'] for j in jobs}) == 5
    for job in jobs:
        assert job['gpu_ids'] == ([0, 1] if job['model'] == 'GNN+FiLM'
                                  else [2, 3, 4] if job['model'] == 'GNN+gate' else [5, 6, 7])
        assert bool(job.get('wait_pipeline')) == (job['model'] == 'GNN+FiLM')


def test_film_static_best_waits_until_dynamic_is_complete(tmp_path, monkeypatch):
    run = tmp_path / 'FiLM/static'
    run.mkdir(parents=True)
    job = dict(run=str(run), wait_pipeline=True, id='film_static')
    monkeypatch.setattr(runner, 'training_outcome', lambda job: 'complete')
    path = run.parent / 'pipeline_status.json'
    path.write_text(json.dumps(dict(status='training')))
    assert runner.readiness(job) == (False, 'complete')
    path.write_text(json.dumps(dict(status='complete')))
    assert runner.readiness(job) == (True, 'complete')
    path.write_text(json.dumps(dict(status='failed')))
    with pytest.raises(RuntimeError, match='pipeline stopped'):
        runner.readiness(job)


def test_training_failure_is_not_evaluated(monkeypatch):
    monkeypatch.setattr(runner, 'training_outcome', lambda job: 'failed')
    with pytest.raises(RuntimeError, match='not complete'):
        runner.readiness(dict(id='gate_static'))


@pytest.mark.parametrize('corruption', [None, 'missing_instance', 'wrong_checkpoint', 'wrong_seed', 'wrong_order', 'wrong_test_hash'])
def test_verify_requires_exact_full_test_and_checkpoint(tmp_path, corruption):
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/best.pt').write_bytes(b'checkpoint')
    source = tmp_path / 'test'
    source.mkdir()
    for i in range(200):
        (source / f'instance_{i:04d}.json').write_text(json.dumps(dict(name=f'dataset2_test_{i:04d}')))
    (tmp_path / 'training_config.json').write_text(json.dumps(dict(test_instances=str(source))))
    result = dict(checkpoint_sha256=runner.digest(tmp_path / 'checkpoints/best.pt'), seed=runner.SEED,
                  deterministic=True, test_instances=str(source), aggregate=dict(instance_count=200),
                  instances=[dict(name=f'dataset2_test_{i:04d}') for i in range(200)],
                  test_manifest=[dict(name=p.name, sha256=runner.digest(p)) for p in sorted(source.glob('*.json'))])
    if corruption == 'missing_instance':
        result['instances'].pop()
    elif corruption == 'wrong_checkpoint':
        result['checkpoint_sha256'] = 'different'
    elif corruption == 'wrong_seed':
        result['seed'] += 1
    elif corruption == 'wrong_order':
        result['instances'].reverse()
    elif corruption == 'wrong_test_hash':
        (source / 'instance_0000.json').write_text(json.dumps(dict(name='dataset2_test_0000', changed=True)))
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(result))
    job = dict(run=str(tmp_path), result=str(path), id='test_job')
    if corruption:
        with pytest.raises(RuntimeError, match='mismatched'):
            runner.verify_result(job)
    else:
        runner.verify_result(job)


@pytest.mark.parametrize('worker_fails', [False, True])
def test_parallel_workers_use_matching_free_cards_and_report_exit(tmp_path, monkeypatch, worker_fails):
    (tmp_path / 'outputs').mkdir()
    selected = [j for j in runner.jobs(tmp_path) if not j.get('wait_pipeline')]
    for job in selected:
        Path(job['run']).mkdir(parents=True)
    monkeypatch.setattr(runner, 'jobs', lambda root, scope='completed': selected)
    monkeypatch.setattr(runner, 'readiness', lambda job: (True, 'complete'))
    monkeypatch.setattr(runner, 'verify_result', lambda job: None)
    monkeypatch.setattr(runner, 'export_summary', lambda output, state: None)
    monkeypatch.setattr(runner, 'gpu_processes', lambda ids: [])
    monkeypatch.setattr(runner.signal, 'signal', Mock())
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    children = [Mock(pid=100 + i) for i in range(3)]
    for i, child in enumerate(children):
        child.poll.return_value = 1 if worker_fails and i == 0 else 0
    spawn = Mock(side_effect=children)
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    output = tmp_path / 'outputs/test_evaluation_test'
    assert runner.run(output, tmp_path) == (1 if worker_fails else 0)
    devices = [call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in spawn.call_args_list]
    assert devices == ['5', '2', '3']
    assert all(call.kwargs['env']['OMP_NUM_THREADS'] == '2' for call in spawn.call_args_list)
    state = json.loads((output / 'status.json').read_text())
    assert state['counts']['complete'] == (2 if worker_fails else 3)
    assert state['status'] == ('complete_with_failures' if worker_fails else 'complete')


def test_existing_worker_is_monitored_without_restart(tmp_path):
    destination = tmp_path / 'results.json'
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                                '--worker', 'job.json', '--result', str(destination)])
    try:
        worker = runner.ExistingWorker(dict(pid=process.pid, result=str(destination)))
        assert worker.poll() is None
        destination.write_text('{}')
        process.terminate()
        process.wait(timeout=5)
        assert worker.poll() == 0
        assert worker.wait(timeout=1) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_resume_recovers_valid_results_without_spawning(tmp_path, monkeypatch):
    output = tmp_path / 'outputs/test_evaluation_test'
    (output / 'logs').mkdir(parents=True)
    selected = [dict(j, status='failed') for j in runner.jobs(tmp_path) if not j.get('wait_pipeline')]
    for i, job in enumerate(selected):
        p = output / f'result{i}.json'
        p.write_text('{}')
        job['result'] = str(p)
        job['error'] = 'old validation error'
    (output / 'status.json').write_text(json.dumps(dict(supervisor_pid=123, jobs=selected)))
    verify = Mock()
    spawn = Mock()
    monkeypatch.setattr(runner, 'verify_result', verify)
    monkeypatch.setattr(runner, 'export_summary', lambda output, state: None)
    monkeypatch.setattr(runner.signal, 'signal', Mock())
    monkeypatch.setattr(runner.subprocess, 'Popen', spawn)
    assert runner.run(output, tmp_path, resume=True) == 0
    assert verify.call_count == 3
    spawn.assert_not_called()
    state = json.loads((output / 'status.json').read_text())
    assert state['counts'] == {'complete': 3}
    assert all('error' not in j for j in state['jobs'])


def test_film_set3_scope_matches_dataset_phase_and_original_cards():
    selected = runner.jobs(Path('/project'), scope='film-set3')
    assert len(selected) == 2
    assert {j['id'] for j in selected} == {'GNN_FiLM_set3_static', 'GNN_FiLM_set3_dynamic'}
    for job in selected:
        assert job['gpu_ids'] == [2, 3, 4]
        assert job['dataset'] == 3 and job['model'] == 'GNN+FiLM'
        assert job['run'] == f"/project/outputs/GNN+FiLM_set3/{job['phase']}"
        assert job['wait_pipeline'] is True


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match='Unknown evaluation scope'):
        runner.jobs(scope='unknown')


def test_static_mlp_dynamic_test_keeps_static_checkpoint_source():
    selected=runner.jobs(Path('/project'),scope='fullstate-mlp-set1-static-to-dynamic')
    assert len(selected)==1
    job=selected[0]
    assert job['run']=='/project/outputs/fullstate_jingtai_set1/MLP'
    assert job['phase']=='static' and job['evaluation_phase']=='dynamic'
    assert job['model']=='MLP' and job['state']=='fullstate' and job['dataset']==1
    assert job['gpu_ids']==[0,1]


def test_cross_phase_scope_can_run_while_original_evaluation_lock_is_held(tmp_path,monkeypatch):
    import fcntl
    output=tmp_path/'outputs/test_evaluation_cross'
    output.parent.mkdir()
    selected=runner.jobs(tmp_path,scope='fullstate-mlp-set1-static-to-dynamic')
    Path(selected[0]['run']).mkdir(parents=True)
    monkeypatch.setattr(runner,'jobs',lambda root,scope='completed':selected)
    monkeypatch.setattr(runner,'readiness',lambda job:(True,'complete'))
    monkeypatch.setattr(runner,'verify_result',lambda job:None)
    monkeypatch.setattr(runner,'export_summary',lambda output,state:None)
    monkeypatch.setattr(runner,'gpu_processes',lambda ids:[])
    monkeypatch.setattr(runner.signal,'signal',Mock())
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:None)
    child=Mock(pid=123);child.poll.return_value=0
    spawn=Mock(return_value=child);monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    with (output.parent/'.test_completed_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert runner.run(output,tmp_path,scope='fullstate-mlp-set1-static-to-dynamic')==0
    assert spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES']=='0'
