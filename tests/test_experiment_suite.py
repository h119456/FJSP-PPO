import json

import pytest

from scripts.evaluate_experiment_suite import evaluation_source, export_summary, planned_runs, summarize, training_outcome


def metrics(objective, **kwargs):
    return dict(objective=objective, terminated=True, truncated=False,
                schedule_violation_count=0, process_readiness_violations=0, **kwargs)


def test_successful_mean_keeps_large_valid_objective_and_reports_failures():
    rows = [metrics(10), metrics(1000), metrics(20)]
    rows[-1].update(terminated=False, truncated=True, schedule_violation_count=1)
    result = summarize(rows)
    assert result['successful_mean_objective'] == 505
    assert result['mean_objective'] == pytest.approx(1030 / 3)
    assert result['valid_instance_count'] == 2
    assert result['completion_rate'] == pytest.approx(2 / 3)
    assert result['total_schedule_violations'] == 1


def test_no_success_is_explicitly_empty():
    row = metrics(999)
    row.update(process_readiness_violations=1)
    result = summarize([row])
    assert result['successful_mean_objective'] is None
    assert result['valid_instance_count'] == 0
    assert result['mean_objective'] == 999


def test_all_experiments_have_unique_ids_and_matching_test_phase(tmp_path):
    jobs = planned_runs(tmp_path)
    assert len(jobs) == len({j['id'] for j in jobs}) == 32
    assert sum(j['state'] == 'simplestate' for j in jobs) == 12
    assert all(str(tmp_path) in j['run'] for j in jobs)
    assert {j['model'] for j in jobs} == {'MLP', 'GNN', 'GNN+gate', 'GNN+FiLM'}


def test_running_best_is_not_complete_and_independent_status_overrides_failed_chain(tmp_path):
    run = tmp_path / 'dynamic/GNN'
    run.mkdir(parents=True)
    (run / 'training_config.json').write_text('{}')
    (run / 'checkpoints').mkdir()
    (run / 'checkpoints/best.pt').touch()
    (run.parent / 'pipeline_status.json').write_text(json.dumps(
        {'jobs': {'dynamic/GNN': {'status': 'queued'}}, 'status': 'failed'}))
    job = dict(run=str(run), phase='dynamic', model='GNN')
    assert training_outcome(job) == 'queued'
    (run / 'independent_status.json').write_text('{"status":"training"}')
    assert training_outcome(job) == 'training'
    (run / 'checkpoints/final.pt').touch()
    (run / 'checkpoint_summary.json').write_text('{}')
    assert training_outcome(job) == 'complete'


def test_static_checkpoint_can_explicitly_target_matching_dynamic_test(tmp_path):
    config = dict(test_instances=str(tmp_path/'data/dataset1_cost0/static/test'))
    job = dict(dataset=1, phase='static')
    assert evaluation_source(job, config, tmp_path) == tmp_path/'data/dataset1_cost0/static/test'
    job['evaluation_phase'] = 'dynamic'
    assert evaluation_source(job, config, tmp_path) == tmp_path/'data/dataset1_cost0/dynamic/test'
    assert config['test_instances'].endswith('/static/test')


@pytest.mark.parametrize('phase,target',[('dynamic','static'),('static','train')])
def test_other_cross_phase_evaluations_are_rejected(tmp_path,phase,target):
    with pytest.raises(RuntimeError,match='static-to-dynamic'):
        evaluation_source(dict(dataset=1,phase=phase,evaluation_phase=target),{},tmp_path)


def test_cross_phase_summary_identifies_static_to_dynamic(tmp_path):
    job=dict(id='static_mlp_on_dynamic',dataset=1,phase='static',evaluation_phase='dynamic',
             model='MLP',state='fullstate',status='waiting_training')
    export_summary(tmp_path,dict(jobs=[job]))
    assert '跨阶段对照' in (tmp_path/'summary.md').read_text()
