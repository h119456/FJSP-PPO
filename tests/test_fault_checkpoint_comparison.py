import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import compare_fault_checkpoints as comparison
from scripts.evaluate_experiment_suite import SEED, cached_evaluation, evaluation_indices, summarize


def metrics(objective, valid=True):
    return dict(objective=objective, terminated=valid, truncated=not valid,
                schedule_violation_count=0, process_readiness_violations=0)


def test_pairs_cover_all_18_model_state_dataset_combinations():
    selected = comparison.pairs(Path('/project'))
    assert len(selected) == len({p['id'] for p in selected}) == 18
    assert sum(p['state'] == 'simplestate' for p in selected) == 6
    assert {p['dataset'] for p in selected} == {1, 2, 3}
    assert {p['model'] for p in selected} == {'MLP', 'GNN', 'GNN+gate', 'GNN+FiLM'}
    assert selected[0]['label'] == 'MLP，完整state，set1'
    assert all(p['static'] != p['dynamic'] for p in selected)


def test_fault_selection_keeps_full_test_positions(tmp_path):
    paths = []
    for index in range(200):
        scenario = 'breakdown_only' if 20 <= index < 40 else 'both' if 180 <= index < 190 else 'none'
        p = tmp_path / f'{index:03d}.json'
        p.write_text(json.dumps(dict(dynamic_events=dict(scenario=scenario))))
        paths.append(p)
    indices = evaluation_indices(dict(fault_only=True), paths)
    assert indices == list(range(20, 40)) + list(range(180, 190))
    assert SEED + indices[0] == 1027
    assert evaluation_indices({}, paths) == list(range(200))
    with pytest.raises(RuntimeError, match='exactly 30'):
        evaluation_indices(dict(fault_only=True), paths[:40])


@pytest.mark.parametrize('corruption', [None, 'renumbered_seed', 'wrong_scenario', 'duplicate_name', 'different_manifest'])
def test_cached_subset_is_paired_by_global_index_and_validated(tmp_path, corruption):
    indices = list(range(20, 40)) + list(range(180, 190))
    scenarios = ['breakdown_only' if i in indices[:20] else 'both' if i in indices[20:] else 'none' for i in range(200)]
    names = [f'job{i}' for i in range(200)]
    manifest = [dict(name=f'{i}.json', sha256=str(i)) for i in range(200)]
    data = dict(checkpoint_sha256='checkpoint', seed=SEED, deterministic=True, test_manifest=manifest,
                test_instances='/test', observation_profile='unified23', action_feature_profile='enhanced25',
                instances=[dict(name=names[i], seed=SEED+i, scenario=scenarios[i], metrics=metrics(i+1)) for i in range(200)])
    if corruption == 'renumbered_seed':
        data['instances'][20]['seed'] = SEED
    elif corruption == 'wrong_scenario':
        data['instances'][20]['scenario'] = 'none'
    elif corruption == 'duplicate_name':
        data['instances'][0]['name'] = data['instances'][1]['name']
    elif corruption == 'different_manifest':
        data['test_manifest'] = []
    p = tmp_path / 'results.json'; p.write_text(json.dumps(data))
    found = cached_evaluation(p, checkpoint_hash='checkpoint', source=Path('/test'), manifest=manifest,
                             profiles=dict(observation_profile='unified23', action_feature_profile='enhanced25'),
                             names=names, scenarios=scenarios, indices=indices)
    if corruption:
        assert found is None
    else:
        assert len(found[1]) == 30
        assert found[1][0]['seed'] == SEED + 20
        assert found[1][-1]['name'] == 'job189'


def test_report_has_exact_four_columns_and_keeps_failures(tmp_path):
    (tmp_path / 'results').mkdir()
    pair = comparison.pairs(Path('/project'))[0]
    jobs = []
    for phase, value in [('static', 100), ('dynamic', 110)]:
        rows = [dict(name=f'i{i}', seed=SEED+i, scenario='breakdown_only' if i < 20 else 'both',
                     metrics=metrics(value, valid=not (phase == 'static' and i == 0))) for i in range(30)]
        path = tmp_path / 'results' / f'{phase}.json'
        path.write_text(json.dumps(dict(instances=rows, aggregate=summarize([x['metrics'] for x in rows]), checkpoint_episode=1)))
        jobs.append(dict(pair_id=pair['id'], phase=phase, status='complete', training_status='complete', result=str(path)))
    state = dict(pairs=[pair], jobs=jobs)
    comparison.save_reports(tmp_path, state)
    with (tmp_path / 'summary.csv').open() as stream:
        table = list(csv.reader(stream))
    assert table[0] == ['类型', '静态ckpt均值', '动态ckpt均值', '改善百分比']
    assert table[1] == [pair['label'], '100.000000', '110.000000', '-10.0000%']
    detail = json.loads((tmp_path / 'details.json').read_text())
    assert detail['pairs'][0]['static_valid'] == 29
    assert '失败惩罚' in (tmp_path / 'summary.md').read_text()


@pytest.mark.parametrize('fail_worker', [False, True])
def test_each_gpu_has_one_worker_and_automatically_takes_next_task(tmp_path, monkeypatch, fail_worker):
    (tmp_path / 'outputs').mkdir()
    selected = comparison.pairs(tmp_path)[:6]  # Twelve tasks: eight active, then four replacements.
    monkeypatch.setattr(comparison, 'pairs', lambda root: selected)
    monkeypatch.setattr(comparison, 'test_context', lambda dataset, root: dict(
        source=Path('/test'), manifest=[], indices=[], names=[], scenarios=[]))
    monkeypatch.setattr(comparison, 'result_cache', lambda specs, root: {})
    monkeypatch.setattr(comparison, 'prepare_job', lambda job, context, cache: job.update(status='queued'))
    monkeypatch.setattr(comparison, 'verify_fault_result', lambda job, context: None)
    monkeypatch.setattr(comparison, 'gpu_processes', lambda ids: [])
    monkeypatch.setattr(comparison.signal, 'signal', Mock())
    monkeypatch.setattr(comparison.time, 'sleep', lambda seconds: None)

    def save(output, state):
        state['counts'] = {s: sum(j['status'] == s for j in state['jobs']) for s in {j['status'] for j in state['jobs']}}
        (output / 'status.json').write_text(json.dumps(state))

    monkeypatch.setattr(comparison, 'save_reports', save)
    children = [Mock(pid=100+i) for i in range(12)]
    for i, child in enumerate(children):
        child.poll.return_value = 1 if fail_worker and i == 0 else 0
    spawn = Mock(side_effect=children); monkeypatch.setattr(comparison.subprocess, 'Popen', spawn)
    # The manifest is separately validated; this scheduler fixture has no actual dataset.
    monkeypatch.setattr(comparison, 'write', lambda path, value: None)
    output = tmp_path / 'outputs/comparison'
    assert comparison.run(output, tmp_path) == (1 if fail_worker else 0)
    devices = [call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in spawn.call_args_list]
    assert devices == [str(i) for i in range(8)] + [str(i) for i in range(4)]
    state = json.loads((output / 'status.json').read_text())
    assert state['counts']['complete'] == (11 if fail_worker else 12)
    assert state['status'] == ('complete_with_failures' if fail_worker else 'complete')
