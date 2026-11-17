"""Queue full-state dataset3 GNN+gate on GPUs 5,6,7, static then dynamic."""
from __future__ import annotations

import argparse
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
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from scripts.wait_train_gnn_gate_set1 import JOBS, gpu_processes, now, write_status
from scripts.train_gnn_film_set1 import verify_complete

GPU_IDS = (5, 6, 7)


def build_configs(root):
    output = root/'outputs/GNN+gate_set3'
    result = {}
    for phase in ('static', 'dynamic'):
        config = json.loads((root/f'configs/physical_gate/full_gnn_{phase}.json').read_text())
        config.update(distributed_world_size=3, output_dir=str(output/phase), output_dir_mode='existing',
                      shared_kv_film=False, physical_gating=True)
        for key, split in (('train_instances','train'), ('eval_instances','validation'), ('test_instances','test')):
            config[key] = str(root/f'data/dataset3_cost0/{phase}/{split}')
        if phase == 'dynamic':
            config['initialize_from'] = str(output/'static/checkpoints/best.pt')
        else:
            config.pop('initialize_from', None)
        result[phase] = config
    return result


def audit(config, pending_source=False):
    # Preserve actual batch64; only bypass the legacy checker's divisibility
    # rule. PPO already masks the two padding slots for three ranks.
    from scripts.check_comparison import _constraints, _dataset_summary, _model_summary, effective_config
    config = effective_config(config)
    inspected = dict(config)
    batch, world = config['mini_batch_size'], config['distributed_world_size']
    inspected['mini_batch_size'] = math.ceil(batch/world)*world
    errors = _constraints(inspected)
    datasets, data_errors = _dataset_summary(config)
    errors.extend(data_errors)
    model_config = dict(config)
    if pending_source:
        model_config.pop('initialize_from', None)
    model = _model_summary(model_config) if not errors else {}
    model.update(local_mini_batch_size=math.ceil(batch/world),
                 full_batch_valid_samples_per_rank=[22,22,20],
                 padded_global_mini_batch_size=66, padding_contributes_to_loss=False)
    return dict(passed=not errors, errors=errors, effective_config=config,
                model=model, datasets=datasets,
                source_compatibility='pending_static_completion' if pending_source else 'checked_if_present')


def readiness(root, start_now=False):
    if (root/'outputs/GNN+gate_set3').exists():
        raise RuntimeError('GNN+gate_set3 already exists; refusing to overwrite')
    if start_now:
        busy = gpu_processes(GPU_IDS)
        return ((False, f'Waiting for GPUs 5-7 compute processes: {busy}') if busy
                else (True, 'Explicit start requested; GPUs 5-7 idle'))
    path = root/'outputs/simplestate_jingtai_set3/pipeline_status.json'
    if not path.exists():
        return False, 'Waiting for simplestate dataset3 status'
    state = json.loads(path.read_text())
    if state['status'] in ('failed','error','interrupted','cancelled','stopped'):
        raise RuntimeError(f"Previous pipeline stopped: {state['status']}")
    if state['status'] != 'complete':
        return False, f"Waiting for simplestate dataset3: {state.get('active_job',state['status'])}"
    if not all(state.get('jobs',{}).get(job,{}).get('status')=='complete' for job in JOBS):
        raise RuntimeError('Previous pipeline reports complete but four jobs are incomplete')
    busy = gpu_processes(GPU_IDS)
    return (False,f'Waiting for GPUs 5-7 compute processes: {busy}') if busy else (True,'Previous chain complete; GPUs 5-7 idle')


def run(root=ROOT, interval=60, start_now=False):
    outputs = root/'outputs'
    output = outputs/'GNN+gate_set3'
    state_path = outputs/'gnn_gate_set3_queue_status.json'
    with (outputs/'.gnn_gate_set3_queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another GNN+gate set3 queue is running')
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite {output}')
        configs = build_configs(root)
        state = dict(status='waiting',supervisor_pid=os.getpid(),started_at=now(),gpu_ids=list(GPU_IDS),
                     poll_seconds=interval,output_dir=str(output),start_now=start_now,
                     order=['static','dynamic'],jobs={p:dict(status='queued',episodes_target=c['episodes']) for p,c in configs.items()})
        child = None

        def save():
            write_status(state_path,state)
            if output.exists():
                write_status(output/'pipeline_status.json',state)

        def stop(signum, frame):
            raise InterruptedError(f'Received signal {signum}')

        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        save()
        try:
            # Concrete, reviewable queued settings and CPU preflight; no GPU allocation.
            preflight = {p:audit(c,pending_source=p=='dynamic') for p,c in configs.items()}
            (outputs/'gnn_gate_set3_queued_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
            (outputs/'gnn_gate_set3_queue_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
            if not all(p['passed'] for p in preflight.values()):
                raise RuntimeError(f'Queue preflight failed: {preflight}')
            state['preflight_passed'] = True
            consecutive = 0
            previous = None
            while True:
                try:
                    ready,reason = readiness(root,start_now=True) if start_now else readiness(root)
                except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                    ready,reason = False,f'Cannot establish readiness: {exc}'
                consecutive = consecutive+1 if ready else 0
                state.update(reason=reason,consecutive_idle=consecutive);save()
                if reason != previous:
                    print(f'{now()} {reason}',flush=True);previous=reason
                if consecutive>=2:
                    break
                time.sleep(interval)
            output.mkdir()
            for phase,config in configs.items():
                directory = output/phase;directory.mkdir()
                (directory/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
            for phase,config in configs.items():
                directory = output/phase
                state.update(status='preflight',active_job=phase);save()
                report = audit(config)
                (directory/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
                if not report['passed']:
                    raise RuntimeError(f'{phase} preflight failed: {report["errors"]}')
                command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
                           '--nproc_per_node=3','-m','ppo_rs.train','--config',str(directory/'training_config.json')]
                with (directory/'training.log').open('w') as log:
                    child = subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='training')
                    state['jobs'][phase].update(status='training',pid=child.pid,started_at=now(),command=command);save()
                    print(f'{now()} Starting {phase}; child PID={child.pid}',flush=True)
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{phase} exited with code {code}; see {directory}/training.log')
                verify_complete(directory,config['episodes'])
                state['jobs'][phase].update(status='complete',completed_at=now());save()
            state.update(status='complete',active_job=None,completed_at=now());save()
            return 0
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
            state.update(status='interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',error=str(exc))
            if state.get('active_job'):
                state['jobs'][state['active_job']]['status']=state['status']
            save()
            raise


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    os.environ['PYTHONPATH']=str(ROOT/'src')+':'+str(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='CPU-only config/readiness audit; no output creation')
    parser.add_argument('--start-now',action='store_true',help='Explicitly start when GPUs are idle, without the previous-chain dependency')
    parser.add_argument('--poll-seconds',type=int,default=60)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be between 1 and 60')
    if args.check:
        reports={p:audit(c,pending_source=p=='dynamic') for p,c in build_configs(ROOT).items()}
        ready,reason=readiness(ROOT,start_now=args.start_now)
        print(json.dumps(dict(passed=all(r['passed'] for r in reports.values()),ready=ready,reason=reason,
                             errors={p:r['errors'] for p,r in reports.items()},gpu_ids=list(GPU_IDS)),indent=2))
        return 0 if all(r['passed'] for r in reports.values()) else 1
    return run(interval=args.poll_seconds,start_now=args.start_now)


if __name__=='__main__':
    raise SystemExit(main())
