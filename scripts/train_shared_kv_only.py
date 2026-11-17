"""Train full-state shared K/V without FiLM: static then matching dynamic."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.train_gnn_film_set1 import audit, now, verify_complete  # noqa: E402
from scripts.wait_train_gnn_gate_set1 import gpu_processes, write_status  # noqa: E402

GPU_PRESETS = {1: (4,), 2: (5,), 3: (6, 7)}


def configs(root, dataset):
    if dataset not in GPU_PRESETS:
        raise ValueError('dataset must be 1, 2 or 3')
    output = root / f'outputs/GNN+sharedKV_set{dataset}'
    stages = {}
    for phase in ('static', 'dynamic'):
        config = json.loads((root / f'configs/shared_kv_film/full_gnn_{phase}.json').read_text())
        config.update(shared_kv_only=True, shared_kv_film=False, physical_gating=False,
                      distributed_world_size=len(GPU_PRESETS[dataset]), output_dir=str(output / phase),
                      output_dir_mode='existing')
        for key, split in [('train_instances', 'train'), ('eval_instances', 'validation'), ('test_instances', 'test')]:
            config[key] = str(root / f'data/dataset{dataset}_cost0/{phase}/{split}')
        config.pop('initialize_from', None)
        if phase == 'dynamic':
            config['initialize_from'] = str(output / 'static/checkpoints/best.pt')
        stages[phase] = config
    return stages


def preflight(stages):
    reports = {}
    for phase, config in stages.items():
        inspected = dict(config)
        inspected.pop('initialize_from', None)
        reports[phase] = audit(inspected)
    return reports


def run(dataset, root=ROOT):
    gpu_ids = GPU_PRESETS[dataset]
    output = root / f'outputs/GNN+sharedKV_set{dataset}'
    with (root / f'outputs/.gnn_shared_kv_set{dataset}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite {output}')
        if gpu_processes(gpu_ids):
            raise RuntimeError(f'GPUs {gpu_ids} are occupied')
        stages = configs(root, dataset)
        reports = preflight(stages)
        output.mkdir(exist_ok=False)
        state = dict(status='preflight', supervisor_pid=os.getpid(), started_at=now(), dataset=dataset,
                     gpu_ids=list(gpu_ids), variant='shared_kv_only', order=['static', 'dynamic'],
                     jobs={p: dict(status='queued', episodes_target=c['episodes']) for p, c in stages.items()})
        child = None

        def save():
            write_status(output / 'pipeline_status.json', state)

        def stop(signum, _frame):
            raise InterruptedError(f'Received signal {signum}')

        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        save()
        try:
            for phase, config in stages.items():
                directory = output / phase; directory.mkdir()
                (directory / 'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
                (directory / 'queued_preflight.json').write_text(json.dumps(reports[phase], indent=2) + '\n')
            for phase, config in stages.items():
                directory = output / phase
                state.update(status='preflight', active_job=phase); save()
                report = audit(config)
                (directory / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
                while gpu_processes(gpu_ids):
                    state.update(status='waiting_gpu'); save(); time.sleep(5)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_ids)),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                           PYTHONPATH=str(root / 'src') + ':' + str(root))
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                           f'--nproc_per_node={len(gpu_ids)}', '-m', 'ppo_rs.train',
                           '--config', str(directory / 'training_config.json')]
                with (directory / 'training.log').open('x') as stream:
                    child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='training')
                    state['jobs'][phase].update(status='training', pid=child.pid, started_at=now(), command=command)
                    save(); print(f'{now()} Starting {phase}: child PID={child.pid}; GPUs={gpu_ids}', flush=True)
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{phase} exited with code {code}')
                verify_complete(directory, config['episodes'])
                state['jobs'][phase].update(status='complete', completed_at=now()); save()
            state.update(status='complete', active_job=None, completed_at=now()); save()
            return 0
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
            state.update(status='interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                         else 'failed', error=str(exc))
            if state.get('active_job'):
                state['jobs'][state['active_job']]['status'] = state['status']
            save(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    if args.check:
        reports = preflight(configs(ROOT, args.dataset))
        print(json.dumps(dict(dataset=args.dataset, gpu_ids=GPU_PRESETS[args.dataset],
                              passed=all(r['passed'] for r in reports.values()),
                              parameters={p: r['models']['gnn']['parameter_count'] for p, r in reports.items()},
                              errors={p: r['errors'] for p, r in reports.items()},
                              dynamic_transfer_check='deferred_until_static_completion'), indent=2))
    else:
        raise SystemExit(run(args.dataset))
