"""Compare every static/dynamic best checkpoint on the same 30 fault cases."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.evaluate_experiment_suite import (  # noqa: E402
    SEED, cached_evaluation, digest, evaluation_indices, now, read, successful, summarize,
    training_outcome, write,
)
from scripts.wait_train_gnn_gate_set1 import gpu_processes  # noqa: E402

HEADERS = ['类型', '静态ckpt均值', '动态ckpt均值', '改善百分比']


def pairs(root=ROOT):
    result = []
    for state, label in [('fullstate', '完整state'), ('simplestate', '简化state')]:
        for dataset in (1, 2, 3):
            for model in ('MLP', 'GNN'):
                result.append(dict(id=f'{state}_set{dataset}_{model}', model=model, state=state,
                                   dataset=dataset, label=f'{model}，{label}，set{dataset}',
                                   static=str(root / f'outputs/{state}_jingtai_set{dataset}/{model}'),
                                   dynamic=str(root / f'outputs/{state}_dongtai_set{dataset}/{model}')))
    for variant, label in [('gate', 'GNN＋物理门控'), ('FiLM', 'GNN＋共享K/V＋类型FiLM')]:
        for dataset in (1, 2, 3):
            result.append(dict(id=f'GNN_{variant}_set{dataset}', model=f'GNN+{variant}',
                               state='fullstate', dataset=dataset, label=f'{label}，完整state，set{dataset}',
                               static=str(root / f'outputs/GNN+{variant}_set{dataset}/static'),
                               dynamic=str(root / f'outputs/GNN+{variant}_set{dataset}/dynamic')))
    return result


def test_context(dataset, root=ROOT):
    from ppo_rs import resolve_instance_paths
    source = root / f'data/dataset{dataset}_cost0/dynamic/test'
    paths = resolve_instance_paths(source)
    if len(paths) != 200:
        raise RuntimeError(f'Expected 200 full test cases: {source}')
    payloads = [read(path) for path in paths]
    indices = evaluation_indices(dict(fault_only=True), paths)
    scenarios = [x.get('dynamic_events', {}).get('scenario', 'none') for x in payloads]
    if ([scenarios[i] for i in indices].count('breakdown_only') != 20
            or [scenarios[i] for i in indices].count('both') != 10):
        raise RuntimeError('Expected 20 breakdown-only and 10 combined-event cases')
    return dict(source=source, paths=paths, manifest=[dict(name=p.name, sha256=digest(p)) for p in paths],
                names=[x['name'] for x in payloads], scenarios=scenarios, indices=indices)


def result_cache(specs, root=ROOT):
    candidates = set(root.glob('outputs/test_evaluation*/results/*.json'))
    for pair in specs:
        for phase in ('static', 'dynamic'):
            candidates.update(Path(pair[phase]).glob('test_evaluation*/results.json'))
    index = {}
    for path in sorted(candidates):
        try:
            data = read(path)
            if 'instances' in data and 'checkpoint_sha256' in data:
                index.setdefault((data['checkpoint_sha256'], data.get('test_instances')), []).append(path)
        except (OSError, ValueError, KeyError):
            continue
    return index


def prepare_job(job, context, cache):
    run = Path(job['run'])
    job['training_status'] = training_outcome(job)
    if job['training_status'] not in ('complete', 'early_stopped', 'failed'):
        raise RuntimeError(f"Checkpoint is not from an ended run: {job['training_status']}")
    config = read(run / 'training_config.json')
    checkpoint = run / 'checkpoints/best.pt'
    checkpoint_hash = digest(checkpoint)
    valid = [v for v in read(run / 'validation_history.json') if v['terminated'] and not v['truncated']]
    best = min(valid, key=lambda row: row['objective'])
    job.update(checkpoint_sha256=checkpoint_hash, checkpoint_episode=best['episode'],
               best_validation_objective=best['objective'])
    profiles = {key: config[key] for key in ('observation_profile', 'action_feature_profile')}
    for path in cache.get((checkpoint_hash, str(context['source'])), []):
        cached = cached_evaluation(path, checkpoint_hash=checkpoint_hash, source=context['source'],
                                   manifest=context['manifest'], profiles=profiles, names=context['names'],
                                   scenarios=context['scenarios'], indices=context['indices'])
        if cached is None:
            continue
        old, rows = cached
        if (old.get('checkpoint_episode') != best['episode']
                or not math.isclose(old.get('best_validation_objective', math.inf), best['objective'], abs_tol=1e-8)):
            continue
        metadata = {key: value for key, value in old.items()
                    if key not in ('instances', 'aggregate', 'scenario_aggregates')}
        metadata.update(job=dict(job), checkpoint=str(checkpoint), checkpoint_training_phase=job['phase'],
                        evaluation_phase='dynamic', fault_only=True, evaluated_indices=context['indices'],
                        reused_from=str(path), physical_gpu=None, device='cached',
                        started_at=now(), completed_at=now())
        aggregate = summarize([row['metrics'] for row in rows])
        scenario_aggregates = {s: summarize([row['metrics'] for row in rows if row['scenario'] == s])
                               for s in ('breakdown_only', 'both')}
        write(Path(job['result']), dict(**metadata, aggregate=aggregate,
                                       scenario_aggregates=scenario_aggregates, instances=rows))
        job.update(status='complete', reused_from=str(path), ended_at=now())
        return
    job['status'] = 'queued'


def verify_fault_result(job, context):
    data = read(Path(job['result']))
    if (data['checkpoint_sha256'] != digest(Path(job['run']) / 'checkpoints/best.pt')
            or data['checkpoint_sha256'] != job['checkpoint_sha256']
            or data['checkpoint_episode'] != job['checkpoint_episode']
            or not math.isclose(data['best_validation_objective'], job['best_validation_objective'], abs_tol=1e-8)
            or data['test_manifest'] != context['manifest'] or data['test_instances'] != str(context['source'])
            or data['seed'] != SEED or data['deterministic'] is not True
            or data.get('evaluated_indices') != context['indices'] or len(data['instances']) != 30):
        raise RuntimeError(f"Mismatched fault evaluation: {job['id']}")
    for index, row in zip(context['indices'], data['instances']):
        if (row['name'] != context['names'][index] or row['scenario'] != context['scenarios'][index]
                or row['seed'] != SEED + index or not math.isfinite(float(row['metrics']['objective']))):
            raise RuntimeError(f"Mismatched per-instance fault evaluation: {job['id']}")
        successful(row['metrics'])
    aggregate = summarize([row['metrics'] for row in data['instances']])
    if (data['aggregate']['instance_count'] != 30
            or not math.isclose(data['aggregate']['mean_objective'], aggregate['mean_objective'], abs_tol=1e-8)):
        raise RuntimeError(f"Wrong fault aggregate: {job['id']}")
    return data


def improvement(static, dynamic):
    return (static - dynamic) / static * 100 if static != 0 else None


def save_reports(output, state):
    state['updated_at'] = now()
    state['counts'] = {s: sum(j['status'] == s for j in state['jobs'])
                       for s in sorted({j['status'] for j in state['jobs']})}
    lookup = {(j['pair_id'], j['phase']): j for j in state['jobs']}
    table, details, paired_rows, notes = [], [], [], []
    for pair in state['pairs']:
        jobs_by_phase = {p: lookup[(pair['id'], p)] for p in ('static', 'dynamic')}
        record = dict(pair)
        record['statuses'] = {p: j['status'] for p, j in jobs_by_phase.items()}
        row = [pair['label'], '', '', '']
        for phase, job in jobs_by_phase.items():
            if job.get('training_status') not in (None, 'complete'):
                notes.append(f"{pair['label']}：{phase}训练状态为{job['training_status']}，使用其已保存的验证集最佳模型。")
        if all(j['status'] == 'complete' for j in jobs_by_phase.values()):
            a, b = [read(Path(jobs_by_phase[p]['result'])) for p in ('static', 'dynamic')]
            av, bv = a['aggregate']['mean_objective'], b['aggregate']['mean_objective']
            gain = improvement(av, bv)
            row = [pair['label'], f'{av:.6f}', f'{bv:.6f}', f'{gain:.4f}%' if gain is not None else '未定义']
            record.update(static_mean=av, dynamic_mean=bv, improvement_percent=gain,
                          static_valid=a['aggregate']['valid_instance_count'],
                          dynamic_valid=b['aggregate']['valid_instance_count'],
                          static_checkpoint_episode=a['checkpoint_episode'],
                          dynamic_checkpoint_episode=b['checkpoint_episode'],
                          static_result=jobs_by_phase['static']['result'], dynamic_result=jobs_by_phase['dynamic']['result'])
            if record['static_valid'] < 30 or record['dynamic_valid'] < 30:
                notes.append(f"{pair['label']}：静态／动态有效完成{record['static_valid']}/30、{record['dynamic_valid']}/30；均值保留全部失败惩罚。")
            for x, y in zip(a['instances'], b['instances']):
                if (x['name'], x['seed'], x['scenario']) != (y['name'], y['seed'], y['scenario']):
                    raise RuntimeError(f"Unpaired fault rows: {pair['id']}")
                paired_rows.append(dict(type=pair['label'], name=x['name'], seed=x['seed'], scenario=x['scenario'],
                                        static_objective=x['metrics']['objective'], dynamic_objective=y['metrics']['objective'],
                                        static_valid=successful(x['metrics']), dynamic_valid=successful(y['metrics'])))
        table.append(row)
        details.append(record)
    state['completed_pairs'] = sum('static_mean' in row for row in details)
    write(output / 'status.json', state)
    with (output / 'summary.csv.tmp').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(HEADERS); writer.writerows(table)
    (output / 'summary.csv.tmp').replace(output / 'summary.csv')
    lines = ['# 故障场景：静态与动态最佳checkpoint对照', '',
             '每行使用对应动态测试集同样30例（仅故障20例＋故障与新作业同时发生10例）。',
             '均值包含全部30例及失败惩罚；改善百分比=(静态均值−动态均值)/静态均值×100%，正值表示动态模型更优。',
             f"原测试序号不变，逐例seed={SEED}+原序号，确定性动作；已完成{state['completed_pairs']}/{len(details)}组。", '',
             '| ' + ' | '.join(HEADERS) + ' |', '|---|---:|---:|---:|']
    lines += ['| ' + ' | '.join(value if value != '' else '待完成' for value in row) + ' |' for row in table]
    if notes:
        lines += ['', '补充说明：', ''] + ['- ' + note for note in dict.fromkeys(notes)]
    temporary = output / 'summary.md.tmp'; temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(output / 'summary.md')
    write(output / 'details.json', dict(pairs=details, notes=list(dict.fromkeys(notes))))
    if paired_rows:
        with (output / 'paired_instances.csv.tmp').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
        (output / 'paired_instances.csv.tmp').replace(output / 'paired_instances.csv')


def run(output, root=ROOT):
    with (root / 'outputs/.fault_checkpoint_comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        for name in ('results', 'jobs', 'logs'):
            (output / name).mkdir()
        specs = pairs(root)
        state = dict(status='preflight', supervisor_pid=os.getpid(), started_at=now(), gpu_ids=list(range(8)),
                     seed=SEED, pairs=specs, jobs=[])
        for pair in specs:
            for phase in ('static', 'dynamic'):
                identifier = f"{pair['id']}_{phase}"
                state['jobs'].append(dict(id=identifier, pair_id=pair['id'], dataset=pair['dataset'], phase=phase,
                                          evaluation_phase='dynamic', fault_only=True, model=pair['model'], state=pair['state'],
                                          run=pair[phase], status='pending', result=str(output / 'results' / f'{identifier}.json')))
        running = {}

        def stop(signum, _frame):
            raise InterruptedError(f'Received signal {signum}')

        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        save_reports(output, state)
        try:
            contexts = {dataset: test_context(dataset, root) for dataset in (1, 2, 3)}
            write(output / 'fault_case_manifest.json', {str(d): dict(
                source=str(c['source']), full_manifest=c['manifest'], evaluated_indices=c['indices'],
                selected=[dict(name=c['names'][i], scenario=c['scenarios'][i], seed=SEED+i) for i in c['indices']])
                for d, c in contexts.items()})
            cache = result_cache(specs, root)
            for job in state['jobs']:
                try:
                    prepare_job(job, contexts[job['dataset']], cache)
                    if job['status'] == 'complete':
                        verify_fault_result(job, contexts[job['dataset']])
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    job.update(status='failed', error=str(exc))
            state['status'] = 'running'; save_reports(output, state)
            print(f"{now()} Prepared {len(state['jobs'])} tasks: {state['counts']}", flush=True)
            while True:
                for gpu, (process, job, log) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close(); del running[gpu]
                    job.update(returncode=code, ended_at=now())
                    try:
                        if code:
                            raise RuntimeError(f'Worker exited with code {code}')
                        verify_fault_result(job, contexts[job['dataset']])
                        job['status'] = 'complete'
                        print(f"{now()} GPU{gpu} complete: {job['id']}", flush=True)
                    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                        job.update(status='failed', error=str(exc))
                for gpu in range(8):
                    if gpu in running:
                        continue
                    job = next((j for j in state['jobs'] if j['status'] == 'queued'), None)
                    if job is None:
                        break
                    if gpu_processes((gpu,)):
                        continue
                    job_path = output / 'jobs' / f"{job['id']}.json"; write(job_path, job)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               PYTHONUNBUFFERED='1', PYTHONPATH=str(root / 'src') + ':' + str(root))
                    command = [sys.executable, '-u', str(root / 'scripts/evaluate_experiment_suite.py'),
                               '--worker', str(job_path), '--result', job['result']]
                    log = (output / 'logs' / f"{job['id']}.log").open('x')
                    try:
                        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close(); raise
                    job.update(status='evaluating', gpu=gpu, pid=process.pid, started_at=now())
                    running[gpu] = process, job, log
                    print(f"{now()} GPU{gpu} start: {job['id']}; PID={process.pid}", flush=True)
                save_reports(output, state)
                if all(j['status'] in ('complete', 'failed') for j in state['jobs']):
                    state.update(status='complete' if all(j['status'] == 'complete' for j in state['jobs'])
                                 else 'complete_with_failures', completed_at=now())
                    save_reports(output, state)
                    return 0 if state['status'] == 'complete' else 1
                time.sleep(2)
        except BaseException as exc:
            for process, job, log in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
                log.close(); job['status'] = 'interrupted'
            state.update(status='interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                         else 'failed', error=str(exc))
            save_reports(output, state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        contexts = {d: test_context(d) for d in (1, 2, 3)}
        print(json.dumps(dict(pair_count=len(pairs()), task_count=len(pairs())*2,
                              fault_cases={d: len(c['indices']) for d, c in contexts.items()},
                              gpu_ids=list(range(8))), ensure_ascii=False, indent=2))
    elif args.output is None:
        parser.error('--output is required')
    else:
        raise SystemExit(run(args.output.resolve()))
