"""Export explicitly labelled, outcome-filtered sensitivity tables for 19 pairs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.compare_fault_test100 import context, validate_rows
from scripts.compare_fault_checkpoints import HEADERS
from scripts.evaluate_experiment_suite import digest, read, successful, write

ORIGINAL=ROOT/'outputs/fault_test100_comparison_20261116_173132'
SHARED=ROOT/'outputs/sharedKV_set1_audit_fault100_20261117_101617/results.json'


def select(rows,multiplier=1.5):
    if not math.isfinite(multiplier) or multiplier<=0:
        raise ValueError('IQR multiplier must be finite and positive')
    a=np.asarray([r['static'] for r in rows],dtype=float)
    b=np.asarray([r['dynamic'] for r in rows],dtype=float)
    if not len(rows) or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<=0).any():
        raise ValueError('Expected finite objectives with positive static values')
    delta=b-a
    def fence(values):
        q1,q3=np.quantile(values,[.25,.75],method='linear')
        return dict(q1=float(q1),q3=float(q3),iqr=float(q3-q1),upper=float(q3+multiplier*(q3-q1)))
    dynamic_fence,delta_fence=fence(b),fence(delta)
    high=b>dynamic_fence['upper']
    deterioration=(delta>0)&(delta>delta_fence['upper'])
    retained=~(high|deterioration)
    def stats(mask):
        if not mask.any():return None
        x,y=float(a[mask].mean()),float(b[mask].mean())
        return dict(count=int(mask.sum()),static_mean=x,dynamic_mean=y,improvement_percent=100*(x-y)/x)
    details=[]
    for i,r in enumerate(rows):
        reasons=[]
        if high[i]:reasons.append('dynamic_high_iqr')
        if deterioration[i]:reasons.append('deterioration_high_iqr')
        details.append(dict(**r,delta=float(delta[i]),retained=bool(retained[i]),reasons=';'.join(reasons)))
    return dict(original=stats(np.ones(len(rows),dtype=bool)),filtered=stats(retained),
                retained_count=int(retained.sum()),excluded_count=int((~retained).sum()),
                dynamic_high_count=int(high.sum()),deterioration_high_count=int(deterioration.sum()),
                dynamic_fence=dynamic_fence,delta_fence=delta_fence,instances=details)


def pairs_from_results():
    ctx={d:context(d) for d in (1,2,3)}
    result=[];sources={}
    for pair in read(ORIGINAL/'details.json')['pairs']:
        loaded={phase:read(Path(pair['jobs'][phase]['result'])) for phase in ('static','dynamic')}
        for phase,data in loaded.items():
            path=Path(pair['jobs'][phase]['result']);sources[str(path)]=digest(path)
            if data['source']!=ctx[pair['dataset']]['source'] or data['test_manifest']!=ctx[pair['dataset']]['hashes']:
                raise ValueError('Result is not from the matching new 100-case test')
            validate_rows(data['instances'],ctx[pair['dataset']])
        rows=[]
        for a,b in zip(loaded['static']['instances'],loaded['dynamic']['instances']):
            if (a['name'],a['seed'],a['scenario'])!=(b['name'],b['seed'],b['scenario']):raise ValueError('Unpaired data')
            if not successful(a['metrics']) or not successful(b['metrics']):raise ValueError('Expected all 100 original cases to be valid')
            rows.append(dict(name=a['name'],scenario=a['scenario'],seed=a['seed'],static=a['metrics']['objective'],dynamic=b['metrics']['objective']))
        result.append(dict(id=pair['id'],label=pair['label'],dataset=pair['dataset'],rows=rows,
                           training_status={phase:pair['jobs'][phase]['training_status'] for phase in ('static','dynamic')}))
    data=read(SHARED);sources[str(SHARED)]=digest(SHARED)
    if data['source']!=ctx[1]['source'] or data['test_manifest']!=ctx[1]['hashes']:raise ValueError('Shared-KV test source mismatch')
    for phase in ('static','dynamic'):
        validate_rows([dict(name=r['name'],scenario=r['scenario'],seed=r['seed'],metrics=r[phase]) for r in data['instances']],ctx[1])
    rows=[]
    for r in data['instances']:
        if not successful(r['static']) or not successful(r['dynamic']):raise ValueError('Invalid shared-KV case')
        rows.append(dict(name=r['name'],scenario=r['scenario'],seed=r['seed'],static=r['static']['objective'],dynamic=r['dynamic']['objective']))
    result.append(dict(id='GNN_sharedKV_set1',label='GNN＋仅共享K/V，完整state，set1',dataset=1,rows=rows,
                       training_status=dict(static='complete',dynamic='complete')))
    return result,sources


def run(output,multiplier=1.5):
    specs,sources=pairs_from_results()
    records=[dict(id=p['id'],label=p['label'],dataset=p['dataset'],training_status=p['training_status'],**select(p['rows'],multiplier)) for p in specs]
    output=Path(output);output.mkdir(exist_ok=False)
    report=dict(status='complete',created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                analysis_type='outcome_filtered_sensitivity_not_primary_test_result',pair_count=len(records),
                rule=dict(multiplier=multiplier,quantile_method='linear',reference='each original 100-case pair, one pass',
                          excluded_if='dynamic > Q3_dynamic + k*IQR_dynamic OR (dynamic-static > 0 AND dynamic-static > Q3_delta + k*IQR_delta)',
                          paired_deletion=True,iterative=False),source_result_sha256=sources,pairs=records)
    table=[];counts=[];all_rows=[]
    for r in records:
        s=r['filtered'];label=f'{r["label"]}（n={r["retained_count"]}）'
        table.append([label,f'{s["static_mean"]:.6f}' if s else '',f'{s["dynamic_mean"]:.6f}' if s else '',f'{s["improvement_percent"]:+.4f}%' if s else '无保留样本'])
        counts.append(dict(type=r['label'],input_count=100,excluded=r['excluded_count'],retained=r['retained_count'],
                           dynamic_high=r['dynamic_high_count'],large_deterioration=r['deterioration_high_count'],
                           original_gain_percent=r['original']['improvement_percent'],filtered_gain_percent=s['improvement_percent'] if s else None,
                           dynamic_upper=r['dynamic_fence']['upper'],delta_upper=r['delta_fence']['upper']))
        all_rows.extend(dict(type=r['label'],**row) for row in r['instances'])
    write(output/'details.json',report)
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(HEADERS);writer.writerows(table)
    for name,rows in [('sample_counts.csv',counts),('paired_instances_with_filter.csv',all_rows),('excluded_instances.csv',[r for r in all_rows if not r['retained']])]:
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list((rows or all_rows)[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 新100例动态测试：剔除统计高值与较大退化后的敏感性结果','',
           f'共{len(records)}组。各组从原始100例一次性计算Q1／Q3，分位数使用线性插值，k={multiplier}。',
           '剔除规则：动态目标值超过Q3＋k×IQR，或者正退化差值（动态−静态）超过其Q3＋k×IQR；两侧共同删除同一实例。',
           '第一列n为保留样本数；重叠命中只删除一次。改善=(静态−动态)/静态×100%，保留负值。',
           '**本表是按测试表现筛选后的敏感性分析，原始合法100例主结果保持不变。不同组保留实例可能不同，不用于直接比较架构排名。**','',
           '| '+' | '.join(HEADERS)+' |','|---|---:|---:|---:|']
    lines+=['| '+' | '.join(row)+' |' for row in table]
    lines+=['','原训练状态：GNN完整state set1静态早停；MLP简化state set3动态曾中断，使用已保存验证best。',
            'sample_counts.csv记录原始与筛选后改善率、阈值及样本数；excluded_instances.csv保留逐例目标值与剔除原因。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    if any(digest(Path(path))!=sha for path,sha in sources.items()):raise RuntimeError('Original results changed during analysis')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--multiplier',type=float,default=1.5)
    args=parser.parse_args();result=run(args.output,args.multiplier)
    print(json.dumps(dict(output=str(args.output),pair_count=result['pair_count'],retained_counts={r['id']:r['retained_count'] for r in result['pairs']}),ensure_ascii=False,indent=2))
