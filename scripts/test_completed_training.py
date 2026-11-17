"""Evaluate finished training on its original GPUs; wait for the FiLM pipeline."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.evaluate_experiment_suite import (  # noqa: E402
    SEED, digest, evaluation_source, export_summary, now, planned_runs, read, training_outcome, write,
)
from scripts.wait_train_gnn_gate_set1 import gpu_processes  # noqa: E402


def jobs(root=ROOT, scope='completed'):
    if scope == 'fullstate-mlp-set1-static-to-dynamic':
        return [dict(id='fullstate_set1_static_MLP_on_dynamic_test', dataset=1,
                     phase='static', evaluation_phase='dynamic', state='fullstate', model='MLP',
                     run=str(root / 'outputs/fullstate_jingtai_set1/MLP'), gpu_ids=[0, 1])]
    if scope == 'film-set3':
        return [dict(id=f'GNN_FiLM_set3_{phase}', dataset=3, phase=phase, state='fullstate',
                     model='GNN+FiLM', run=str(root / f'outputs/GNN+FiLM_set3/{phase}'),
                     gpu_ids=[2, 3, 4], wait_pipeline=True) for phase in ('static', 'dynamic')]
    if scope != 'completed':
        raise ValueError(f'Unknown evaluation scope: {scope}')
    selected = {'GNN_gate_set2_static', 'GNN_gate_set2_dynamic', 'simplestate_set3_dynamic_GNN'}
    result = [dict(job, gpu_ids=[2, 3, 4] if job['model'] == 'GNN+gate' else [5, 6, 7])
              for job in planned_runs(root) if job['id'] in selected]
    result += [dict(id=f'GNN_FiLM_set2_{phase}', dataset=2, phase=phase, state='fullstate',
                    model='GNN+FiLM', run=str(root / f'outputs/GNN+FiLM_set2/{phase}'),
                    gpu_ids=[0, 1], wait_pipeline=True) for phase in ('static', 'dynamic')]
    return result


def readiness(job):
    """Do not evaluate a mutable checkpoint, or interrupt static-to-dynamic transfer."""
    outcome = training_outcome(job)
    if job.get('wait_pipeline'):
        path = Path(job['run']).parent / 'pipeline_status.json'
        if not path.exists():
            return False, 'not_started'
        pipeline = read(path)
        if pipeline['status'] in ('failed', 'interrupted', 'stopped'):
            raise RuntimeError(f"FiLM training pipeline stopped: {pipeline['status']}")
        if pipeline['status'] != 'complete':
            return False, outcome
    if outcome in ('failed', 'interrupted', 'stopped'):
        raise RuntimeError(f"Training is not complete: {job['id']} ({outcome})")
    return outcome == 'complete', outcome


def verify_result(job):
    """Only mark a full, matching 200-instance test as complete."""
    result = read(Path(job['result']))
    config = read(Path(job['run']) / 'training_config.json')
    from ppo_rs import resolve_instance_paths
    source = evaluation_source(job, config)
    paths = resolve_instance_paths(source)
    manifest = [{'name': path.name, 'sha256': digest(path)} for path in paths]
    names = [read(path)['name'] for path in paths]
    if (result['checkpoint_sha256'] != digest(Path(job['run']) / 'checkpoints/best.pt')
            or result['seed'] != SEED or result['deterministic'] is not True
            or result['test_instances'] != str(source)
            or result['aggregate']['instance_count'] != 200
            or len(result['instances']) != 200 or len(paths) != 200
            or result['test_manifest'] != manifest
            or [row['name'] for row in result['instances']] != names):
        raise RuntimeError(f"Incomplete or mismatched test result: {job['id']}")


class ExistingWorker:
    """Monitor a worker that survived a supervisor restart without rerunning it."""

    def __init__(self, job):
        self.pid = job['pid']
        self.result = Path(job['result'])
        try:
            args = (Path('/proc') / str(self.pid) / 'cmdline').read_bytes().split(b'\0')
            if (b'--worker' not in args or b'--result' not in args
                    or os.fsencode(str(self.result)) not in args):
                raise RuntimeError(f'Worker PID {self.pid} has different arguments')
            self.start_ticks = self._identity()[1]
        except FileNotFoundError:
            self.start_ticks = None

    def _identity(self):
        fields = (Path('/proc') / str(self.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]

    def poll(self):
        try:
            state, ticks = self._identity()
            if self.start_ticks is not None and ticks == self.start_ticks and state not in ('Z', 'X'):
                return None
        except FileNotFoundError:
            pass
        return 0 if self.result.is_file() else 1

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.1)


def run(output, root=ROOT, resume=False, scope='completed'):
    lock_name = ('.test_static_mlp_set1_on_dynamic.lock'
                 if scope == 'fullstate-mlp-set1-static-to-dynamic' else '.test_completed_training.lock')
    with (root / 'outputs' / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = {}
        if resume:
            state = read(output / 'status.json')
            state.update(status='running', previous_supervisor_pid=state['supervisor_pid'],
                         supervisor_pid=os.getpid(), resumed_at=now())
            for job in state['jobs']:
                if job.get('result') and Path(job['result']).is_file():
                    try:
                        verify_result(job)
                        job.update(status='complete', recovered_at=now())
                        job.pop('error', None)
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        job.update(status='failed', error=str(exc))
                elif job['status'] == 'evaluating':
                    process = ExistingWorker(job)
                    log = (output / 'logs' / f"{job['id']}.log").open('a')
                    running[job['gpu']] = process, job, log
                    job['adopted_worker'] = True
        else:
            output.mkdir(exist_ok=False)
            (output / 'jobs').mkdir()
            (output / 'logs').mkdir()
            state = dict(status='running', supervisor_pid=os.getpid(), started_at=now(), seed=SEED,
                         scope=scope, jobs=[dict(job, status='waiting_training') for job in jobs(root, scope=scope)])
        stamp = output.name

        def save():
            state['updated_at'] = now()
            state['counts'] = {s: sum(j['status'] == s for j in state['jobs'])
                               for s in sorted({j['status'] for j in state['jobs']})}
            write(output / 'status.json', state)
            export_summary(output, state)

        def stop(signum, _frame):
            raise InterruptedError(f'Received signal {signum}')

        def stop_worker(process):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        save()
        try:
            while True:
                for gpu, (process, job, log) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del running[gpu]
                    job.update(returncode=code, ended_at=now())
                    if job.get('adopted_worker'):
                        job['returncode_source'] = 'inferred_from_process_end_and_result_presence'
                    try:
                        if code:
                            raise RuntimeError(f'Evaluation exited with code {code}')
                        verify_result(job)
                        job['status'] = 'complete'
                        print(f"{now()} Complete: {job['id']}", flush=True)
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        job.update(status='failed', error=str(exc))
                for job in state['jobs']:
                    if job['status'] in ('complete', 'failed', 'evaluating'):
                        continue
                    try:
                        ready, outcome = readiness(job)
                        job['training_status'] = outcome
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        job.update(status='failed', error=str(exc))
                        continue
                    if not ready:
                        job['status'] = 'waiting_training'
                        continue
                    job['status'] = 'ready'
                    gpu = next((gpu for gpu in job['gpu_ids']
                                if gpu not in running and not gpu_processes((gpu,))), None)
                    if gpu is None:
                        continue
                    destination = Path(job['run']) / stamp
                    destination.mkdir(exist_ok=False)
                    job_path = output / 'jobs' / f"{job['id']}.json"
                    job['result'] = str(destination / 'results.json')
                    write(job_path, job)
                    command = [sys.executable, '-u', str(root / 'scripts/evaluate_experiment_suite.py'),
                               '--worker', str(job_path), '--result', job['result']]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                               PYTHONPATH=str(root / 'src') + ':' + str(root))
                    log = (output / 'logs' / f"{job['id']}.log").open('x')
                    try:
                        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    job.update(status='evaluating', gpu=gpu, pid=process.pid, started_at=now())
                    running[gpu] = process, job, log
                    print(f"{now()} GPU{gpu}: {job['id']}; PID={process.pid}", flush=True)
                save()
                if all(j['status'] in ('complete', 'failed') for j in state['jobs']):
                    state.update(status='complete' if all(j['status'] == 'complete' for j in state['jobs'])
                                 else 'complete_with_failures', completed_at=now())
                    save()
                    return 0 if state['status'] == 'complete' else 1
                time.sleep(5)
        except BaseException as exc:
            for process, job, log in running.values():
                stop_worker(process)
                log.close()
                job['status'] = 'interrupted'
            state.update(status='interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                         else 'failed', error=str(exc))
            save()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Recover existing results and monitor surviving workers')
    parser.add_argument('--scope', choices=('completed', 'film-set3', 'fullstate-mlp-set1-static-to-dynamic'),
                        default='completed')
    args = parser.parse_args()
    if args.check:
        for job in jobs(scope=args.scope):
            ready, outcome = readiness(job)
            print(job['id'], job['gpu_ids'], outcome, 'ready' if ready else 'waiting')
    elif args.output is None or not args.output.name.startswith('test_evaluation_'):
        parser.error('--output must have a unique test_evaluation_... directory name')
    else:
        raise SystemExit(run(args.output.resolve(), resume=args.resume, scope=args.scope))
