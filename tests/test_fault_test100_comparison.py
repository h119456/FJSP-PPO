import csv
import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import compare_fault_test100 as suite


def metrics(value, valid=True):
    return dict(objective=value,terminated=valid,truncated=not valid,schedule_violation_count=0,process_readiness_violations=0)


def rows():
    return [dict(name=f'case{i}',seed=1007+i,scenario='breakdown_only' if i<67 else 'both',metrics=metrics(10)) for i in range(100)]


def ctx():
    selected=rows()
    return dict(source='/new_test',hashes={'file':'hash'},names=[r['name'] for r in selected],scenarios=[r['scenario'] for r in selected])


@pytest.mark.parametrize('corruption',[None,'seed','source','hash','profile','scenario','duplicate','checkpoint'])
def test_cache_requires_exact_new_data_checkpoint_profile_and_pairing(tmp_path,corruption):
    p=tmp_path/'outputs/fault100_fullstate_set2_MLP_previous';p.mkdir(parents=True)
    profile=dict(observation_profile='unified23',action_feature_profile='enhanced25')
    job=dict(pair_id='fullstate_set2_MLP',phase='static',checkpoint_sha256='ckpt',checkpoint_episode=20000,best_validation_objective=1,profiles=profile)
    metadata=dict(sha256='ckpt',episode=20000,best_validation_objective=1,profiles=dict(profile))
    data=dict(source='/new_test',test_manifest={'file':'hash'},seed=1007,deterministic=True,checkpoints={'static':metadata},
              instances=[dict(name=r['name'],seed=r['seed'],scenario=r['scenario'],static=r['metrics']) for r in rows()])
    if corruption=='seed':data['instances'][2]['seed']=1
    if corruption=='source':data['source']='/old200'
    if corruption=='hash':data['test_manifest']={}
    if corruption=='profile':metadata['profiles']['observation_profile']='compact18'
    if corruption=='scenario':data['instances'][0]['scenario']='none'
    if corruption=='duplicate':data['instances'][2]['name']='case1'
    if corruption=='checkpoint':metadata['sha256']='wrong'
    (p/'results.json').write_text(json.dumps(data))
    result=suite.cached_pair(job,ctx(),tmp_path)
    assert (result is None)==bool(corruption)
    if result:assert len(result[0])==100


def test_report_excludes_invalid_pairs_keeps_valid_high_values_and_negative_gain(tmp_path):
    (tmp_path/'results').mkdir()
    pair=suite.pairs(Path('/project'))[0];jobs=[]
    for phase,base in [('static',100),('dynamic',110)]:
        selected=rows()
        for i,r in enumerate(selected):
            r['metrics']=metrics(base*2 if i==2 else base,valid=not ((phase=='static' and i==0) or (phase=='dynamic' and i==1)))
        path=tmp_path/'results'/f'{phase}.json';path.write_text(json.dumps(dict(instances=selected)))
        jobs.append(dict(pair_id=pair['id'],phase=phase,status='complete',training_status='failed' if phase=='dynamic' else 'complete',result=str(path)))
    state=dict(pairs=[pair],jobs=jobs);suite.reports(tmp_path,state)
    table=list(csv.reader((tmp_path/'summary.csv').open()))
    assert table[0]==['类型','静态ckpt均值','动态ckpt均值','改善百分比']
    assert len(table[1])==4 and table[1][-1]=='-10.0000%'
    assert float(table[1][1])==pytest.approx((97*100+200)/98,abs=1e-6)
    detail=json.loads((tmp_path/'details.json').read_text())['pairs'][0]
    assert detail['paired_valid_count']==98 and detail['evaluated_count']==100
    assert 'failed' in (tmp_path/'summary.md').read_text()
    assert len(list(csv.DictReader((tmp_path/'paired_instances.csv').open())))==100


@pytest.mark.parametrize('failure',[False,True])
def test_only_three_requested_cards_one_worker_each_and_automatic_replacement(tmp_path,monkeypatch,failure):
    (tmp_path/'outputs').mkdir()
    selected=suite.pairs(tmp_path)[:3]
    monkeypatch.setattr(suite,'pairs',lambda root:selected)
    monkeypatch.setattr(suite,'context',lambda dataset,root:ctx())
    monkeypatch.setattr(suite,'prepare',lambda job,context,root:job.update(status='queued'))
    monkeypatch.setattr(suite,'verify_result',lambda job,context:None)
    monkeypatch.setattr(suite.signal,'signal',Mock())
    monkeypatch.setattr(suite.time,'sleep',lambda seconds:None)
    def report(output,state):
        state['counts']=dict(Counter(j['status'] for j in state['jobs']))
        (output/'status.json').write_text(json.dumps(state))
    monkeypatch.setattr(suite,'reports',report)
    children=[Mock(pid=100+i) for i in range(6)]
    for i,p in enumerate(children):p.poll.return_value=int(failure and i==0)
    spawn=Mock(side_effect=children);monkeypatch.setattr(suite.subprocess,'Popen',spawn)
    output=tmp_path/'outputs/test'
    assert suite.run(output,tmp_path)==int(failure)
    assert [c.kwargs['env']['CUDA_VISIBLE_DEVICES'] for c in spawn.call_args_list]==['5','6','7','5','6','7']
    state=json.loads((output/'status.json').read_text())
    assert state['counts']['complete']==6-int(failure)
    assert state['status']==('complete_with_failures' if failure else 'complete')


@pytest.mark.parametrize('state,identity,expected',[('R','123',None),('R','different',0),('Z','123',0)])
def test_adoption_tracks_process_identity_and_completion(monkeypatch,state,identity,expected):
    fields=['0']*50;fields[0]=state;fields[19]='123';fields[49]='0'
    monkeypatch.setattr(Path,'read_text',lambda self:'42 (worker name) '+' '.join(fields))
    assert suite.AdoptedProcess(42,identity).poll()==expected


def test_resume_adds_gpu4_without_restarting_existing_workers(tmp_path,monkeypatch):
    (tmp_path/'outputs').mkdir();output=tmp_path/'outputs/test';output.mkdir()
    for name in ('jobs','results','logs'):(output/name).mkdir()
    selected=suite.pairs(tmp_path)[:2];jobs=[]
    for pair in selected:
        for phase in ('static','dynamic'):
            index=len(jobs)
            jobs.append(dict(id=f'{pair["id"]}_{phase}',pair_id=pair['id'],dataset=pair['dataset'],
                             model=pair['model'],state=pair['state'],
                             phase=phase,run=pair[phase],checkpoint_sha256='ckpt',
                             status='evaluating' if index<3 else 'queued',gpu=5+index if index<3 else None,
                             pid=100+index,result=str(output/'results'/f'{index}.json')))
    (output/'status.json').write_text(json.dumps(dict(supervisor_pid=999,gpu_ids=[5,6,7],pairs=selected,jobs=jobs)))
    (output/'test_case_manifest.json').write_text(json.dumps({str(d):ctx() for d in (1,2,3)}))
    monkeypatch.setattr(suite,'pairs',lambda root:selected)
    monkeypatch.setattr(suite,'context',lambda dataset,root:ctx())
    monkeypatch.setattr(suite,'process_identity',lambda pid:None if pid==999 else 'ticks')
    monkeypatch.setattr(suite,'digest',lambda path:'ckpt')
    monkeypatch.setattr(suite,'verify_result',lambda job,context:None)
    monkeypatch.setattr(suite.signal,'signal',Mock());monkeypatch.setattr(suite.time,'sleep',lambda seconds:None)
    existing=[Mock(pid=100+i) for i in range(3)]
    for p in existing:p.poll.side_effect=[None,0]
    adopt=Mock(side_effect=existing);monkeypatch.setattr(suite,'AdoptedProcess',adopt)
    child=Mock(pid=200);child.poll.return_value=0
    spawn=Mock(return_value=child);monkeypatch.setattr(suite.subprocess,'Popen',spawn)
    def report(output,state):
        state['counts']=dict(Counter(j['status'] for j in state['jobs']))
        (output/'status.json').write_text(json.dumps(state))
    monkeypatch.setattr(suite,'reports',report)
    assert suite.run(output,tmp_path,resume=True,gpus=(4,5,6,7))==0
    assert adopt.call_count==3 and spawn.call_count==1
    assert spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES']=='4'
    state=json.loads((output/'status.json').read_text())
    assert state['gpu_ids']==[4,5,6,7] and state['counts']=={'complete':4}
