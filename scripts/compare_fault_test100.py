"""Queue all 18 static/dynamic comparisons on the independent fault tests."""
from __future__ import annotations

import argparse
from collections import Counter
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
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.compare_fault_checkpoints import pairs, HEADERS
from scripts.evaluate_experiment_suite import SEED, now, read, write, digest, successful, summarize, training_outcome

GPUS = (5, 6, 7)


def process_identity(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


class AdoptedProcess:
    """Monitor an existing worker after replacing only its supervisor."""
    def __init__(self,pid,identity):
        self.pid=pid;self.identity=identity

    def poll(self):
        try:
            fields=(Path('/proc')/str(self.pid)/'stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:
            return 0  # A result must still pass verify_result before completion.
        if self.identity is None or fields[19]!=self.identity:
            return 0
        if fields[0] in ('Z','X'):
            return os.waitstatus_to_exitcode(int(fields[49])) if len(fields)>49 else 0
        return None

    def wait(self,timeout):
        deadline=time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic()>=deadline:
                raise subprocess.TimeoutExpired(f'adopted worker {self.pid}',timeout)
            time.sleep(.1)
        return self.poll()


def gpu_selection(value):
    try:result=tuple(int(x) for x in value.split(','))
    except ValueError as exc:raise argparse.ArgumentTypeError('GPUs must be comma-separated integers') from exc
    if not result or len(result)!=len(set(result)) or any(x<0 or x>7 for x in result):
        raise argparse.ArgumentTypeError('GPUs must be unique indices from 0 to 7')
    return result


def context(dataset, root=ROOT):
    from ppo_rs import resolve_instance_paths
    source = root/f'data/dynamic_fault_test100_cost0/set{dataset}/test'
    paths = resolve_instance_paths(source)
    manifest = read(source.parents[1]/'manifest.json')
    hashes = {p.name: digest(p) for p in paths}
    payloads = [read(p) for p in paths]
    if len(paths) != 100 or hashes != manifest['sets'][f'set{dataset}']['files_sha256']:
        raise RuntimeError('Expected unchanged independent 100-case test manifest')
    scenarios = [p['dynamic_events']['scenario'] for p in payloads]
    if Counter(scenarios) != {'breakdown_only': 67, 'both': 33}:
        raise RuntimeError('Expected 67 breakdown_only + 33 both cases')
    return dict(source=str(source), hashes=hashes, names=[p['name'] for p in payloads], scenarios=scenarios)


def validate_rows(rows, ctx):
    if len(rows) != 100:
        raise RuntimeError('Expected exactly 100 evaluated instances')
    for i, row in enumerate(rows):
        if (row['name'] != ctx['names'][i] or row['seed'] != SEED+i
                or row['scenario'] != ctx['scenarios'][i]
                or not math.isfinite(float(row['metrics']['objective']))):
            raise RuntimeError('Incorrect instance, seed, scenario or objective')
        successful(row['metrics'])


def verify_result(job, ctx):
    data = read(Path(job['result']))
    for key in ('checkpoint_sha256','checkpoint_episode','best_validation_objective','profiles'):
        if data[key] != job[key]:
            raise RuntimeError(f'Result checkpoint metadata mismatch: {key}')
    if (digest(Path(job['run'])/'checkpoints/best.pt') != job['checkpoint_sha256']
            or data['source'] != ctx['source'] or data['test_manifest'] != ctx['hashes']
            or data['seed'] != SEED or data['deterministic'] is not True):
        raise RuntimeError('Result source, checkpoint or evaluation settings mismatch')
    validate_rows(data['instances'], ctx)
    aggregate = summarize([r['metrics'] for r in data['instances']])
    if data['aggregate']['instance_count'] != 100 or not math.isclose(data['aggregate']['mean_objective'], aggregate['mean_objective'], abs_tol=1e-9):
        raise RuntimeError('Incorrect aggregate')
    return data


def cached_pair(job, ctx, root=ROOT):
    """Only reuse the completed new-100-case pair with identical provenance."""
    if job['pair_id'] != 'fullstate_set2_MLP':
        return None
    for path in sorted(root.glob('outputs/fault100_fullstate_set2_MLP_*/results.json')):
        try:
            old = read(path); ckpt = old['checkpoints'][job['phase']]
            if (old['source'] != ctx['source'] or old['test_manifest'] != ctx['hashes']
                    or old['seed'] != SEED or old['deterministic'] is not True
                    or ckpt['sha256'] != job['checkpoint_sha256'] or ckpt['episode'] != job['checkpoint_episode']
                    or ckpt['best_validation_objective'] != job['best_validation_objective'] or ckpt['profiles'] != job['profiles']):
                continue
            rows = [dict(name=r['name'],scenario=r['scenario'],seed=r['seed'],metrics=r[job['phase']]) for r in old['instances']]
            validate_rows(rows, ctx)
            return rows, str(path)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
    return None


def prepare(job, ctx, root=ROOT):
    import torch
    from scripts.evaluate_dataset import _checkpoint_profiles
    from ppo_rs.environment import ResidualStressFJSPEnv
    from ppo_rs.environment.checkpoint_semantics import check_dynamic_input_semantics
    from ppo_rs.models import build_model_from_checkpoint
    run = Path(job['run']); path = run/'checkpoints/best.pt'
    status = training_outcome(job)
    if (run/'early_stop.json').exists():
        status = 'early_stopped'
    if status not in ('complete','early_stopped','failed'):
        raise RuntimeError(f'Training is not ended: {status}')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('environment_semantics_version') != ResidualStressFJSPEnv.ENVIRONMENT_SEMANTICS_VERSION:
        raise RuntimeError('Incompatible environment semantics')
    check_dynamic_input_semantics(ckpt, dynamic=True)
    profile = _checkpoint_profiles(ckpt)
    expected = ('compact18','compact21') if job['state']=='simplestate' else ('unified23','enhanced25')
    if (profile['observation_profile'], profile['action_feature_profile']) != expected:
        raise RuntimeError('Unexpected state/action profile')
    model_type = 'mlp' if job['model']=='MLP' else 'gnn_attention'
    if ckpt.get('model_type') != model_type:
        raise RuntimeError('Unexpected model type')
    architecture = ckpt.get('model_kwargs', {})
    if (bool(architecture.get('physical_gating')) != (job['model']=='GNN+gate')
            or bool(architecture.get('shared_kv_film')) != (job['model']=='GNN+FiLM')
            or architecture.get('shared_kv_only', False)):
        raise RuntimeError('Unexpected graph architecture')
    model = build_model_from_checkpoint(ckpt, torch.device('cpu'))
    eligible = [v for v in read(run/'validation_history.json') if v['terminated'] and not v['truncated']]
    best = min(eligible, key=lambda v:v['objective'])
    if int(ckpt['episode']) != int(best['episode']) or not math.isclose(ckpt['best_validation_objective'],best['objective'],abs_tol=1e-8):
        raise RuntimeError('Checkpoint is not validation best')
    job.update(training_status=status, checkpoint_sha256=digest(path), checkpoint_episode=int(ckpt['episode']),
               best_validation_objective=ckpt['best_validation_objective'], profiles=profile,
               parameter_count=sum(p.numel() for p in model.parameters()), status='queued')
    cached = cached_pair(job, ctx, root)
    if cached:
        rows, source = cached
        result = {k:job[k] for k in ('checkpoint_sha256','checkpoint_episode','best_validation_objective','profiles')}
        result.update(source=ctx['source'], test_manifest=ctx['hashes'], seed=SEED, deterministic=True,
                      reused_from=source, completed_at=now(), instances=rows,
                      aggregate=summarize([r['metrics'] for r in rows]))
        write(Path(job['result']), result); verify_result(job,ctx)
        job.update(status='complete', reused_from=source, completed_at=now())


def worker(job_path):
    import torch
    from ppo_rs import load_instances
    from ppo_rs.cli_utils import set_seed
    from ppo_rs.environment import ResidualStressFJSPEnv
    from ppo_rs.evaluation import evaluate_model
    from scripts.evaluate_dataset import _load_model
    job = read(job_path); ctx = context(job['dataset'])
    torch.set_num_threads(2); set_seed(SEED)
    model, ckpt = _load_model(Path(job['run'])/'checkpoints/best.pt', torch.device('cuda:0'))
    model.eval()
    if digest(Path(job['run'])/'checkpoints/best.pt') != job['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed since preflight')
    rows = []; destination = Path(job['result'])
    with destination.with_suffix('.instances.jsonl').open('x') as stream:
        for i, instance in enumerate(load_instances(ctx['source'])):
            metrics,_ = evaluate_model(ResidualStressFJSPEnv(instance, **job['profiles']),model,seed=SEED+i,deterministic=True)
            row = dict(name=instance.name, scenario=ctx['scenarios'][i], seed=SEED+i, metrics=metrics)
            if not math.isfinite(float(metrics['objective'])):
                raise RuntimeError('Nonfinite objective')
            rows.append(row); stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
            write(destination.with_suffix('.progress.json'),dict(completed=len(rows),total=100,updated_at=now()))
            if len(rows)%10==0:
                print(f'{now()} {job["id"]}: {len(rows)}/100',flush=True)
    validate_rows(rows,ctx)
    if ctx != context(job['dataset']):
        raise RuntimeError('Test data changed during evaluation')
    result={k:job[k] for k in ('checkpoint_sha256','checkpoint_episode','best_validation_objective','profiles')}
    result.update(source=ctx['source'],test_manifest=ctx['hashes'],seed=SEED,deterministic=True,
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],completed_at=now(),instances=rows,
                  aggregate=summarize([r['metrics'] for r in rows]))
    write(destination,result); verify_result(job,ctx)


def four_columns(label, rows):
    if not rows:
        return [label,'','','']
    a=sum(r['static_objective'] for r in rows)/len(rows)
    b=sum(r['dynamic_objective'] for r in rows)/len(rows)
    return [label,f'{a:.6f}',f'{b:.6f}',f'{100*(a-b)/a:.4f}%']


def reports(output,state):
    lookup={(j['pair_id'],j['phase']):j for j in state['jobs']}
    tables={key:[] for key in ('summary','all_cases_summary','breakdown_only_paired_valid_summary','both_paired_valid_summary')}
    details=[]; paired=[]; notes=[]
    for pair in state['pairs']:
        jobs={p:lookup[(pair['id'],p)] for p in ('static','dynamic')}
        record=dict(pair, jobs=jobs); rows=[]
        for p,j in jobs.items():
            if j.get('training_status') not in (None,'complete'):
                notes.append(f'{pair["label"]}：{p}训练状态{j["training_status"]}，使用已保存且通过校验的验证best。')
        if all(j['status']=='complete' for j in jobs.values()):
            a,b=[read(Path(jobs[p]['result']))['instances'] for p in ('static','dynamic')]
            for x,y in zip(a,b):
                if (x['name'],x['seed'],x['scenario']) != (y['name'],y['seed'],y['scenario']):
                    raise RuntimeError('Unpaired result rows')
                rows.append(dict(type=pair['label'],name=x['name'],seed=x['seed'],scenario=x['scenario'],
                                 static_objective=x['metrics']['objective'],dynamic_objective=y['metrics']['objective'],
                                 static_valid=successful(x['metrics']),dynamic_valid=successful(y['metrics'])))
                if not rows[-1]['static_valid'] or not rows[-1]['dynamic_valid']:
                    notes.append(f'{pair["label"]} / {x["name"]}：配对剔除；静态={x["metrics"]["objective"]:.6f}，动态={y["metrics"]["objective"]:.6f}，静态有效={rows[-1]["static_valid"]}，动态有效={rows[-1]["dynamic_valid"]}。')
            paired.extend(rows)
        valid=[r for r in rows if r['static_valid'] and r['dynamic_valid']]
        for key,selected in [('summary',valid),('all_cases_summary',rows),
                             ('breakdown_only_paired_valid_summary',[r for r in valid if r['scenario']=='breakdown_only']),
                             ('both_paired_valid_summary',[r for r in valid if r['scenario']=='both'])]:
            tables[key].append(four_columns(pair['label'],selected))
        record.update(evaluated_count=len(rows),paired_valid_count=len(valid))
        if rows:
            notes.append(f'{pair["label"]}：双侧有效{len(valid)}/{len(rows)}。')
        details.append(record)
    state['completed_pairs']=sum(d['evaluated_count']==100 for d in details)
    state['counts']=dict(Counter(j['status'] for j in state['jobs'])); state['updated_at']=now()
    write(output/'status.json',state); write(output/'details.json',dict(pairs=details,notes=notes))
    for key,table in tables.items():
        with (output/f'{key}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(HEADERS);writer.writerows(table)
        text=['# 新100例故障动态测试：静态／动态验证best对照','',
              f'已完成{state["completed_pairs"]}/{len(details)}组；每组对应set的100例（67仅故障＋33故障与插单）。',
              '主表summary使用双侧共同有效实例；all_cases_summary保留全部实例及失败惩罚；其余文件为对应场景的双侧有效子集。',
              '有效条件：正常结束、无截断、无调度／就绪违规；不按目标值高低筛选。改善=(静态−动态)/静态×100%，保留负数。','',
              '| '+' | '.join(HEADERS)+' |','|---|---:|---:|---:|']
        text += ['| '+' | '.join(v or '待完成／无有效配对' for v in r)+' |' for r in table]
        text += ['','样本数与训练状态：','']+['- '+n for n in notes]
        (output/f'{key}.md').write_text('\n'.join(text)+'\n')
    if paired:
        with (output/'paired_instances.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)


def run(output,root=ROOT,*,resume=False,gpus=None):
    output=Path(output)
    gpu_ids=tuple(GPUS if gpus is None else gpus)
    with (root/'outputs/.fault_test100_comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        running={}
        if resume:
            state=read(output/'status.json')
            if state['pairs']!=pairs(root):raise RuntimeError('Resume pair specification changed')
            previous=state['supervisor_pid']
            previous_identity=process_identity(previous)
            if previous_identity is not None and AdoptedProcess(previous,previous_identity).poll() is None:
                raise RuntimeError('Previous supervisor still exists; refusing concurrent control')
            for job in state['jobs']:
                if job['status']=='evaluating':
                    gpu=job['gpu']
                    if gpu not in gpu_ids or gpu in running:raise RuntimeError('Invalid adopted GPU mapping')
                    identity=job.get('process_start_ticks') or process_identity(job['pid'])
                    process=AdoptedProcess(job['pid'],identity)
                    log=(output/'logs'/f'{job["id"]}.log').open('a')
                    running[gpu]=(process,job,log)
            state.setdefault('supervisor_history',[]).append(dict(pid=previous,replaced_at=now()))
            state.update(supervisor_pid=os.getpid(),gpu_ids=list(gpu_ids),resumed_at=now())
        else:
            output.mkdir(exist_ok=False)
            for directory in ('results','jobs','logs'):(output/directory).mkdir()
            specs=pairs(root); state=dict(status='preflight',supervisor_pid=os.getpid(),gpu_ids=list(gpu_ids),started_at=now(),pairs=specs,jobs=[])
            for pair in specs:
                for phase in ('static','dynamic'):
                    identifier=f'{pair["id"]}_{phase}'
                    state['jobs'].append(dict(id=identifier,pair_id=pair['id'],phase=phase,dataset=pair['dataset'],
                                              model=pair['model'],state=pair['state'],run=pair[phase],status='pending',
                                              result=str(output/'results'/f'{identifier}.json')))
        def stop(signum,_frame):raise InterruptedError(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        reports(output,state)
        try:
            contexts={d:context(d,root) for d in (1,2,3)}
            if resume:
                if read(output/'test_case_manifest.json')!={str(k):v for k,v in contexts.items()}:
                    raise RuntimeError('Resume test data differs from original manifest')
                for job in state['jobs']:
                    if job['status']=='complete':verify_result(job,contexts[job['dataset']])
                    elif job['status'] in ('queued','evaluating') and digest(Path(job['run'])/'checkpoints/best.pt')!=job['checkpoint_sha256']:
                        raise RuntimeError('Resume checkpoint changed')
            else:
                write(output/'test_case_manifest.json',contexts)
                for job in state['jobs']:
                    try:prepare(job,contexts[job['dataset']],root)
                    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:job.update(status='failed',error=str(exc))
            state['status']='running';reports(output,state)
            print(f'{now()} Preflight: {state["counts"]}',flush=True)
            while True:
                changed=False
                for gpu,(process,job,log) in list(running.items()):
                    code=process.poll()
                    if code is None:continue
                    log.close();del running[gpu];changed=True
                    job.update(returncode=code,completed_at=now())
                    try:
                        if code:raise RuntimeError(f'Worker exited with code {code}')
                        verify_result(job,contexts[job['dataset']]);job['status']='complete'
                        print(f'{now()} GPU{gpu} complete {job["id"]}',flush=True)
                    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:job.update(status='failed',error=str(exc))
                for gpu in gpu_ids:
                    if gpu in running:continue
                    queued=[j for j in state['jobs'] if j['status']=='queued']
                    if not queued:break
                    # Longest scales first reduce the final single-worker tail.
                    job=max(queued,key=lambda j:(j['dataset'],j['model']!='MLP'))
                    path=output/'jobs'/f'{job["id"]}.json';write(path,job)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                    log=(output/'logs'/f'{job["id"]}.log').open('x')
                    try:
                        process=subprocess.Popen(['nice','-n','10',sys.executable,'-u',str(root/'scripts/compare_fault_test100.py'),'--worker',str(path)],
                                                 cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    except BaseException:log.close();raise
                    job.update(status='evaluating',gpu=gpu,pid=process.pid,started_at=now());running[gpu]=(process,job,log);changed=True
                    job['process_start_ticks']=process_identity(process.pid)
                    print(f'{now()} GPU{gpu} start {job["id"]}; PID={process.pid}',flush=True)
                if changed:reports(output,state)
                if all(j['status'] in ('complete','failed') for j in state['jobs']):
                    state.update(status='complete' if all(j['status']=='complete' for j in state['jobs']) else 'complete_with_failures',completed_at=now())
                    reports(output,state);return 0 if state['status']=='complete' else 1
                time.sleep(2)
        except BaseException as exc:
            for process,job,log in running.values():
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                log.close();job['status']='interrupted'
            state.update(status='interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',error=str(exc))
            reports(output,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gpus',type=gpu_selection,default=GPUS)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.output:raise SystemExit(run(args.output.resolve(),resume=args.resume,gpus=args.gpus))
    else:parser.error('--output or --worker is required')
