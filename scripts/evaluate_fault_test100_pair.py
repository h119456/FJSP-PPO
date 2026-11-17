"""Evaluate full-state static/dynamic validation-best on fixed dynamic tests."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from scripts.evaluate_experiment_suite import now, read, write, successful, summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(rows):
    static = sum(r['static']['objective'] for r in rows)/len(rows)
    dynamic = sum(r['dynamic']['objective'] for r in rows)/len(rows)
    return dict(count=len(rows), static_mean=static, dynamic_mean=dynamic,
                improvement_percent=100*(static-dynamic)/static,
                dynamic_wins=sum(r['dynamic']['objective'] < r['static']['objective']-1e-9 for r in rows),
                ties=sum(abs(r['dynamic']['objective']-r['static']['objective']) <= 1e-9 for r in rows),
                dynamic_losses=sum(r['dynamic']['objective'] > r['static']['objective']+1e-9 for r in rows))


def run(output, dataset, *, model_name='MLP', test_source='new_fault100'):
    import torch
    from ppo_rs import load_instances, resolve_instance_paths
    from ppo_rs.cli_utils import set_seed
    from ppo_rs.environment import ResidualStressFJSPEnv
    from ppo_rs.environment.checkpoint_semantics import check_dynamic_input_semantics
    from ppo_rs.evaluation import evaluate_model
    from scripts.evaluate_dataset import _load_model, _checkpoint_profiles

    output = Path(output); output.mkdir(exist_ok=False)
    total = 100 if test_source=='new_fault100' else 200
    state = dict(status='preflight', pid=os.getpid(), dataset=dataset, model=model_name, test_source=test_source,
                 physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started_at=now(), completed=0, total=total)
    def save():
        state['updated_at'] = now(); write(output/'status.json', state)
    save()
    try:
        source_root = ROOT/'data/dynamic_fault_test100_cost0' if test_source=='new_fault100' else ROOT/f'data/dataset{dataset}_cost0/dynamic'
        source = source_root/f'set{dataset}/test' if test_source=='new_fault100' else source_root/'test'
        manifest = read(source_root/'manifest.json')
        paths = resolve_instance_paths(source)
        declared = manifest['sets'][f'set{dataset}']['files_sha256'] if test_source=='new_fault100' else {p.name:digest(p) for p in paths}
        hashes = {p.name: digest(p) for p in paths}
        if len(paths) != total or hashes != declared:
            raise RuntimeError('Test set differs from its expected manifest')
        if test_source!='new_fault100' and set(manifest['splits']['test'])!={f'test/{p.name}' for p in paths}:
            raise RuntimeError('Original test file inventory differs from manifest')
        scenarios = [read(p)['dynamic_events']['scenario'] for p in paths]
        expected_mix={'breakdown_only':67,'both':33} if test_source=='new_fault100' else {'none':130,'job_release_only':40,'breakdown_only':20,'both':10}
        if Counter(scenarios) != expected_mix:
            raise RuntimeError('Unexpected scenario mix')
        instances = load_instances(source)
        torch.set_num_threads(2); set_seed(1007)
        models, checkpoints = {}, {}
        for phase, label in (('static','jingtai'), ('dynamic','dongtai')):
            if model_name in ('MLP','GNN'):
                run_dir = ROOT/f'outputs/fullstate_{label}_set{dataset}/{model_name}'
            else:
                run_dir = ROOT/f'outputs/{model_name}_set{dataset}/{phase}'
            path = run_dir/'checkpoints/best.pt'
            model, ckpt = _load_model(path, torch.device('cuda:0'))
            profiles = _checkpoint_profiles(ckpt)
            expected_type='mlp' if model_name=='MLP' else 'gnn_attention'
            kwargs=ckpt.get('model_kwargs',{})
            if (ckpt.get('model_type') != expected_type or profiles != dict(observation_profile='unified23', action_feature_profile='enhanced25')
                    or bool(kwargs.get('shared_kv_only'))!=(model_name=='GNN+sharedKV')
                    or bool(kwargs.get('shared_kv_film'))!=(model_name=='GNN+FiLM') or kwargs.get('physical_gating',False)):
                raise RuntimeError('Expected matching full-state model and architecture')
            check_dynamic_input_semantics(ckpt, dynamic=True)
            eligible = [r for r in read(run_dir/'validation_history.json') if r['terminated'] and not r['truncated']]
            best = min(eligible, key=lambda r: r['objective'])
            if int(ckpt['episode']) != int(best['episode']) or not math.isclose(ckpt['best_validation_objective'], best['objective'], abs_tol=1e-8):
                raise RuntimeError('Checkpoint disagrees with validation best')
            models[phase] = model.eval()
            checkpoints[phase] = dict(path=str(path), sha256=digest(path), episode=int(ckpt['episode']),
                                      best_validation_objective=best['objective'], profiles=profiles,
                                      parameter_count=sum(p.numel() for p in model.parameters()))
        state.update(status='evaluating', source=str(source), checkpoints=checkpoints); save()
        rows = []
        with (output/'instances.jsonl').open('x') as stream:
            for index, (instance, scenario) in enumerate(zip(instances, scenarios)):
                row = dict(name=instance.name, file=paths[index].name, scenario=scenario, seed=1007+index)
                for phase, model in models.items():
                    metrics, _ = evaluate_model(ResidualStressFJSPEnv(instance, **checkpoints[phase]['profiles']),
                                                model, seed=1007+index, deterministic=True)
                    if not math.isfinite(float(metrics['objective'])):
                        raise RuntimeError(f'Nonfinite objective: {instance.name}/{phase}')
                    row[phase] = metrics
                rows.append(row); stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                state['completed'] = len(rows); save()
                if len(rows)%5 == 0:
                    print(f'{now()} paired progress {len(rows)}/{total}', flush=True)
        if hashes != {p.name: digest(p) for p in paths} or any(digest(Path(v['path'])) != v['sha256'] for v in checkpoints.values()):
            raise RuntimeError('Test files or checkpoints changed during evaluation')
        groups = dict(all=rows,**{scenario:[r for r in rows if r['scenario']==scenario] for scenario in sorted(set(scenarios))})
        valid = {k:[r for r in v if successful(r['static']) and successful(r['dynamic'])] for k,v in groups.items()}
        summary = {k:comparison(v) for k,v in groups.items()}
        valid_summary = {k:comparison(v) if v else None for k,v in valid.items()}
        result = dict(dataset=dataset, model=model_name, state='fullstate', source=str(source),
                      physical_gpu=state['physical_gpu'], seed=1007, deterministic=True,
                      checkpoints=checkpoints, test_manifest=hashes, started_at=state['started_at'],
                      completed_at=now(), summary=summary, paired_valid_summary=valid_summary,
                      aggregates={p:summarize([r[p] for r in rows]) for p in models}, instances=rows)
        write(output/'results.json', result)
        with (output/'paired_instances.csv').open('x') as stream:
            fields=['name','scenario','static_objective','dynamic_objective','improvement_percent','static_valid','dynamic_valid']
            writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for r in rows:
                a,b=r['static']['objective'],r['dynamic']['objective']
                writer.writerow(dict(name=r['name'],scenario=r['scenario'],static_objective=a,dynamic_objective=b,
                                     improvement_percent=100*(a-b)/a,static_valid=successful(r['static']),dynamic_valid=successful(r['dynamic'])))
        lines=[f'# {model_name}，完整state，动态测试集配对评估','',f'set{dataset}，cost0，GPU{state["physical_gpu"]}，确定性动作；静态与动态均使用各自验证best。',
               '', '| 场景 | 数量 | 静态均值 | 动态均值 | 改善 |', '|---|---:|---:|---:|---:|']
        for name,r in summary.items():
            lines.append(f'| {name} | {r["count"]} | {r["static_mean"]:.6f} | {r["dynamic_mean"]:.6f} | {r["improvement_percent"]:+.4f}% |')
        lines += ['', '改善=(静态−动态)/静态，正数表示动态更优；主表包含全部实例及失败惩罚。',
                  f'双侧有效配对：{len(valid["all"])}/{total}；排除无效后的配对统计另存results.json，不按目标值高低剔除。',
                  f'动态胜／平／负：{summary["all"]["dynamic_wins"]}／{summary["all"]["ties"]}／{summary["all"]["dynamic_losses"]}。']
        (output/'summary.md').write_text('\n'.join(lines)+'\n')
        state.update(status='complete', completed_at=now(), summary=summary); save()
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return result
    except BaseException as exc:
        state.update(status='failed', error=str(exc)); save(); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=int, choices=(1,2,3), default=2)
    parser.add_argument('--model',choices=('MLP','GNN','GNN+FiLM','GNN+sharedKV'),default='MLP')
    parser.add_argument('--test-source',choices=('new_fault100','original_dynamic200'),default='new_fault100')
    args=parser.parse_args()
    run(args.output, args.dataset,model_name=args.model,test_source=args.test_source)
