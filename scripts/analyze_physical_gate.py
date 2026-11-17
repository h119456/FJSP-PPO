"""Read-only gate diagnostics and fixed-policy validation interventions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ppo_rs import load_instance, resolve_instance_paths
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_model_from_checkpoint, batch_graph_states


def fixed_gates(graph, table):
    """Replace only applicable gates; structural/inapplicable edges stay one."""
    gates = graph['node_features'].new_ones((graph['edge_types'].numel(), table.shape[1]))
    for group in range(3):
        mask = graph['physical_edge_mask'] & ((graph['edge_types'] == 4+2*group) | (graph['edge_types'] == 5+2*group))
        gates[mask] = table[group]
    return gates


def gate_summary(features, gates, groups, module):
    result = {}
    for group in range(3):
        x = features[groups == group].astype(np.float64)
        y = gates[groups == group].astype(np.float64)
        if not len(x):
            continue
        # Bound autograd memory and preserve coverage over the complete trace.
        sample = np.linspace(0, len(x)-1, min(4096, len(x)), dtype=int)
        inputs = torch.tensor(x[sample], dtype=torch.float32, requires_grad=True)
        values = torch.sigmoid(module.gate_heads[group](module.encoder(inputs)))
        with torch.no_grad():
            encoded = module.encoder(inputs)
            head_hidden = module.gate_heads[group][1](module.gate_heads[group][0](encoded))
        jac = torch.stack([torch.autograd.grad(values[:, h].sum(), inputs, retain_graph=True)[0]
                           for h in range(values.shape[1])], dim=1).detach().numpy()
        corr = np.zeros((y.shape[1], x.shape[1]))
        for head in range(y.shape[1]):
            for feature in range(x.shape[1]):
                if x[:, feature].std() > 1e-8 and y[:, head].std() > 1e-8:
                    corr[head, feature] = np.corrcoef(x[:, feature], y[:, head])[0, 1]
        # Constant-feature sensitivity is zero on the observed distribution.
        result[str(group)] = dict(
            count=len(x), mean=y.mean(0).tolist(), std=y.std(0).tolist(),
            quantiles=np.quantile(y, [.01,.1,.5,.9,.99], axis=0).tolist(),
            minimum=y.min(0).tolist(), maximum=y.max(0).tolist(),
            fraction_below_005=(y < .05).mean(0).tolist(), fraction_above_095=(y > .95).mean(0).tolist(),
            feature_mean=x.mean(0).tolist(), feature_std=x.std(0).tolist(),
            correlation_head_by_feature=corr.tolist(),
            rms_local_derivative_head_by_feature=np.sqrt((jac**2).mean(0)).tolist(),
            standardized_sensitivity_head_by_feature=(np.sqrt((jac**2).mean(0))*x.std(0)).tolist(),
            fraction_encoder_tanh_abs_above_095=float((encoded.abs() > .95).float().mean()),
            fraction_head_tanh_abs_above_095=float((head_hidden.abs() > .95).float().mean()),
            mean_by_relief_used={str(flag): y[x[:,5] == flag].mean(0).tolist()
                                for flag in (0,1) if (x[:,5] == flag).any()},
        )
    return result


def evaluate_task(task):
    torch.set_num_threads(1)
    label, checkpoint_path, source, output, table = task
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = build_model_from_checkpoint(checkpoint, torch.device('cpu'))
    config = checkpoint['config']
    natural = table is None
    if not natural:
        table_tensor = torch.tensor(table, dtype=torch.float32)
        model.get_edge_gates = MethodType(lambda self, graph: fixed_gates(graph, table_tensor), model)
    features, gates, groups = [], [], []
    counts = dict(total_edges=0, applicable_edges=0, decisions=0)
    rows = []
    for index, path in enumerate(resolve_instance_paths(source)):
        env = ResidualStressFJSPEnv(load_instance(path), observation_profile=config['observation_profile'],
                                   action_feature_profile=config['action_feature_profile'])
        state, info = env.reset(seed=int(config.get('eval_seed', 1007))+index)
        terminated = truncated = False
        while not (terminated or truncated):
            actions = info['valid_actions']
            if not actions:
                state, _, terminated, truncated, info = env.step(0)
                continue
            graph_state = env.get_graph_state(actions, include_physics=True)
            graph = batch_graph_states([graph_state], action_slots=len(actions), device=torch.device('cpu'))
            with torch.no_grad():
                edge_gates = model.get_edge_gates(graph)
                # Capture diagnostics before message passing; avoid duplicate gate inference.
                if natural:
                    types = graph['edge_types']; mask = graph['physical_edge_mask']
                    counts['total_edges'] += len(types)
                    counts['applicable_edges'] += int(mask.sum())
                    for group in range(3):
                        # One observation per operation/group, excluding reverse and duplicate equipment edges.
                        indices = torch.where(mask & (types == 4+2*group))[0]
                        if not len(indices):
                            continue
                        operations = graph['edge_index'][0, indices]
                        unique, inverse = torch.unique(operations, return_inverse=True)
                        first = torch.stack([torch.where(inverse == j)[0][0] for j in range(len(unique))])
                        features.append(graph['physical_node_features'][unique].numpy())
                        gates.append(edge_gates[indices[first]].numpy())
                        groups.append(np.full(len(unique), group, dtype=np.int64))
                original = model.get_edge_gates
                model.get_edge_gates = MethodType(lambda self, graph: edge_gates, model)
                logits, _ = model(torch.tensor([state], dtype=torch.float32),
                                  torch.tensor([info['action_features']], dtype=torch.float32), graph)
                model.get_edge_gates = original
                choice = int(logits[0].argmax())
            counts['decisions'] += 1
            state, _, terminated, truncated, info = env.step(choice)
        metrics = env.get_metrics(0.)
        # Classify by the fixed input trace, not whether a policy finishes
        # before a late event occurs; keep subgroups paired across policies.
        events = (env.instance.dynamic_events or {}).get('events', [])
        release = any(e['type'] == 'job_release' for e in events)
        failure = any(e['type'] == 'machine_breakdown' for e in events)
        scenario = 'both' if release and failure else 'release' if release else 'breakdown' if failure else 'none'
        rows.append(dict(name=env.instance.name, objective=metrics['objective'], scenario=scenario,
                         terminated=bool(terminated), truncated=bool(truncated),
                         violations=metrics['schedule_violation_count']))
        if (index+1) % 25 == 0:
            print(f'{label}: {index+1} instances', flush=True)
    summary = dict(label=label, checkpoint=str(checkpoint_path), checkpoint_episode=checkpoint['episode'],
                   validation_source=str(source), mode='learned' if natural else 'fixed', instances=rows,
                   objective=float(np.mean([r['objective'] for r in rows])), counts=counts)
    if natural:
        x, y, g = np.concatenate(features), np.concatenate(gates), np.concatenate(groups)
        np.savez_compressed(Path(output)/f'{label}_samples.npz', features=x, gates=y, groups=g)
        summary['gate_statistics'] = gate_summary(x, y, g, model.physical_gate)
    else:
        summary['fixed_table'] = table
    summary['by_scenario'] = {
        s: dict(count=sum(r['scenario']==s for r in rows),
                objective=float(np.mean([r['objective'] for r in rows if r['scenario']==s])))
        for s in sorted({r['scenario'] for r in rows})
    }
    (Path(output)/f'{label}.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--reuse-natural', action='store_true', help='Reuse saved natural traces; refresh changed/missing interventions')
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT/'outputs/GNN+gate_set1'/f'diagnostics_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(parents=True, exist_ok=args.reuse_natural)
    base = ROOT/'outputs/GNN+gate_set1'
    checkpoints = {phase: base/phase/'checkpoints/best.pt' for phase in ('static','dynamic')}
    data = ROOT/'data/dataset1_cost0'
    initial = [
        ('static_on_static', checkpoints['static'], data/'static/validation', output, None),
        ('static_on_dynamic', checkpoints['static'], data/'dynamic/validation', output, None),
        ('dynamic_on_dynamic', checkpoints['dynamic'], data/'dynamic/validation', output, None),
    ]
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn')) as pool:
        natural = ([json.loads((output/f'{task[0]}.json').read_text()) for task in initial]
                   if args.reuse_natural else list(pool.map(evaluate_task, initial)))
        tasks = []
        for result in natural[1:]:
            label = result['label']
            means = [result['gate_statistics'][str(g)]['mean'] for g in range(3)]
            for mode, table in [('constant_mean', means), ('ones', np.ones((3,4)).tolist())]:
                target = output/f'{label}_{mode}.json'
                if args.reuse_natural and target.exists() and json.loads(target.read_text()).get('fixed_table') == table:
                    continue
                tasks.append((label+'_'+mode, Path(result['checkpoint']), data/'dynamic/validation', output, table))
        list(pool.map(evaluate_task, tasks))
    interventions = [json.loads((output/f"{r['label']}_{mode}.json").read_text())
                     for r in natural[1:] for mode in ('constant_mean','ones')]
    # Read persisted per-run results as the canonical, reviewable artifacts.
    natural = [json.loads((output/f"{r['label']}.json").read_text()) for r in natural]
    for result in natural[1:]:
        baseline = {row['name']: row['objective'] for row in result['instances']}
        for variant in (row for row in interventions if row['label'].startswith(result['label'])):
            variant['paired_changes'] = dict(
                count=sum(abs(row['objective']-baseline[row['name']]) > 1e-8 for row in variant['instances']),
                mean_difference=variant['objective']-result['objective'],
                relative_percent=(variant['objective']/result['objective']-1)*100,
            )
    report = dict(natural=natural, interventions=interventions,
                  feature_names=list(ResidualStressFJSPEnv.PHYSICAL_GATE_FEATURE_NAMES),
                  group_names=['ordinary','stress_relief','compensation'],
                  limitations=['Fixed-policy interventions are not retrained ablations.',
                               'Statistics use one operation/group per decision; temporal samples are correlated.',
                               'Local derivatives describe gate output sensitivity, not PPO training gradients.',
                               'CPU reevaluation can differ slightly from original GPU validation; compare within this report.',
                               'Event subgroups use the fixed input trace, independent of policy finishing time.',
                               'Static and dynamic policies use the same dynamic validation instances and CPU device.'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('REPORT', output/'report.json', flush=True)


if __name__ == '__main__':
    main()
