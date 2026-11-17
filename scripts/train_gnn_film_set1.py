"""Run full-state shared K/V + FiLM, static then dynamic, on selected GPUs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def configs(root, dataset=1, world_size=2):
    if dataset not in (1, 2, 3):
        raise ValueError('dataset must be 1, 2 or 3')
    if world_size not in (2, 3):
        raise ValueError('world_size must be 2 or 3')
    output = root/f'outputs/GNN+FiLM_set{dataset}'
    result = {}
    for phase in ('static', 'dynamic'):
        config = json.loads((root/f'configs/shared_kv_film/full_gnn_{phase}.json').read_text())
        config.update(distributed_world_size=world_size, output_dir=str(output/phase), output_dir_mode='existing',
                      shared_kv_film=True, physical_gating=False)
        for key, split in (('train_instances', 'train'), ('eval_instances', 'validation'), ('test_instances', 'test')):
            config[key] = str(root/f'data/dataset{dataset}_cost0/{phase}/{split}')
        if phase == 'dynamic':
            config['initialize_from'] = str(output/'static/checkpoints/best.pt')
        else:
            config.pop('initialize_from', None)
        result[phase] = config
    return result


def verify_complete(directory, target):
    history = json.loads((directory/'training_history.json').read_text())
    if len(history) != target or int(history[-1]['episode']) != target:
        raise RuntimeError(f'Incomplete training history: {directory}')
    for name in ('best', 'latest', 'final'):
        if not (directory/f'checkpoints/{name}.pt').is_file():
            raise RuntimeError(f'Missing {name} checkpoint: {directory}')


def audit(config):
    if config.get('distributed_world_size') == 3:
        # Reuse the audit for masked, uneven 64-sample batches over three ranks.
        from scripts.queue_gnn_gate_set3 import audit as audit_three_cards
        report = audit_three_cards(config)
    else:
        from scripts.train_state_comparison import audit_run_configs
        report = audit_run_configs({'gnn': config})
    if not report['passed']:
        raise RuntimeError(str(report['errors']))
    return report


def _evaluator_identity(pid):
    process = Path('/proc')/str(pid)
    arguments = (process/'cmdline').read_bytes().split(b'\0')
    if not any(Path(os.fsdecode(arg)).name == 'evaluate_experiment_suite.py' for arg in arguments if arg):
        raise RuntimeError(f'PID {pid} is not an evaluation supervisor')
    if b'--output' not in arguments or b'--worker' in arguments:
        raise RuntimeError(f'PID {pid} is not an evaluation supervisor')
    return (process/'stat').read_text().rsplit(')', 1)[1].split()[19]


def _evaluator_children(pid):
    return (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().strip()


def _wait_evaluator_stopped(pid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
        if state in ('T','t'):
            return
        time.sleep(0.01)
    raise RuntimeError(f'Evaluation supervisor {pid} did not stop')


@contextmanager
def paused_evaluator(pid):
    """Pause only an idle evaluation supervisor; restore it on exit."""
    if pid is None:
        yield
        return
    identity = _evaluator_identity(pid)
    if _evaluator_children(pid):
        raise RuntimeError(f'Evaluation supervisor {pid} still has active workers')
    os.kill(pid, signal.SIGSTOP)
    try:
        _wait_evaluator_stopped(pid)
        # Check again after stopping the scheduler, closing the spawn race.
        if _evaluator_children(pid):
            raise RuntimeError(f'Evaluation supervisor {pid} still has active workers')
        print(f'{now()} Paused idle evaluation supervisor PID={pid}', flush=True)
        yield
    finally:
        try:
            same_process = _evaluator_identity(pid) == identity
        except (OSError, RuntimeError):
            same_process = False
        if same_process:
            os.kill(pid, signal.SIGCONT)
            print(f'{now()} Resumed evaluation supervisor PID={pid}', flush=True)


def gpu_selection(value):
    try:
        selected = tuple(int(index) for index in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('GPUs must be comma-separated integers') from exc
    if len(selected) not in (2, 3) or len(set(selected)) != len(selected) or any(i < 0 or i > 7 for i in selected):
        raise argparse.ArgumentTypeError('Select two or three distinct GPU indices from 0 to 7')
    return selected


def run(root=ROOT, dataset=1, gpu_ids=(0, 1)):
    from scripts.wait_train_gnn_gate_set1 import gpu_processes
    gpu_ids = gpu_selection(','.join(str(index) for index in gpu_ids))
    output = root/f'outputs/GNN+FiLM_set{dataset}'
    with (root/f'outputs/.gnn_film_set{dataset}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another FiLM dataset{dataset} launcher is running')
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite: {output}')
        busy = gpu_processes(gpu_ids)
        if busy:
            raise RuntimeError(f'GPUs {gpu_ids} are occupied: {busy}')
        stages = configs(root, dataset, len(gpu_ids))
        output.mkdir()
        state = dict(status='preflight', supervisor_pid=os.getpid(), gpu_ids=list(gpu_ids), started_at=now(),
                     order=['static','dynamic'], jobs={p:dict(status='queued',episodes_target=c['episodes']) for p,c in stages.items()})

        def save():
            state['updated_at'] = now()
            temporary = output/'pipeline_status.tmp'
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
            temporary.replace(output/'pipeline_status.json')

        child = None

        def stop(signum, frame):
            raise InterruptedError(f'Received signal {signum}')

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        save()
        try:
            for phase,config in stages.items():
                directory = output/phase
                directory.mkdir()
                (directory/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
            for phase,config in stages.items():
                directory = output/phase
                state.update(active_job=phase,status='preflight');save()
                report = audit(config)
                (directory/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
                command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
                           f'--nproc_per_node={len(gpu_ids)}','-m','ppo_rs.train','--config',str(directory/'training_config.json')]
                print(f'{now()} Starting {phase}: {command}',flush=True)
                with (directory/'training.log').open('w') as log:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,gpu_ids)),
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                               PYTHONPATH=str(root/'src')+':'+str(root))
                    child = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL,start_new_session=True)
                    state.update(status='training')
                    state['jobs'][phase].update(status='training',pid=child.pid,started_at=now(),command=command);save()
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{phase} exited with code {code}; see {directory}/training.log')
                verify_complete(directory,config['episodes'])
                state['jobs'][phase].update(status='complete',completed_at=now());save()
            state.update(status='complete',active_job=None,completed_at=now());save()
            print(f'{now()} FiLM dataset{dataset} static and dynamic completed',flush=True)
            return 0
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
            state.update(status='interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',error=str(exc))
            active = state.get('active_job')
            if active:
                state['jobs'][active]['status'] = state['status']
            save()
            raise


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    os.environ['PYTHONPATH'] = str(ROOT/'src')+':'+str(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--gpus',type=gpu_selection,default=(0,1),help='Two or three physical GPUs, e.g. 2,3,4')
    parser.add_argument('--pause-evaluator',type=int,help='Pause an idle evaluation supervisor until this pipeline exits')
    parser.add_argument('--check',action='store_true',help='Audit both configs without creating outputs or launching')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,args.gpus))
    if args.check:
        reports = {}
        for phase, config in configs(ROOT,args.dataset,len(args.gpus)).items():
            inspected = dict(config)
            inspected.pop('initialize_from',None)
            reports[phase] = audit(inspected)
        print(json.dumps(dict(passed=all(r['passed'] for r in reports.values()),
                              errors={p:r['errors'] for p,r in reports.items()},
                              dynamic_source_check='deferred_until_static_completion'),indent=2))
        return 0
    with paused_evaluator(args.pause_evaluator):
        return run(ROOT,args.dataset,args.gpus)


if __name__ == '__main__':
    raise SystemExit(main())
