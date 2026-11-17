"""Same-GPU, fixed-input timing of baseline and shared-K/V FiLM GNNs."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch  # noqa: E402
from ppo_rs import load_instances  # noqa: E402
from ppo_rs.algorithms import PPO, PPOConfig, RolloutBuffer  # noqa: E402
from ppo_rs.cli_utils import set_seed  # noqa: E402
from ppo_rs.environment import ResidualStressFJSPEnv  # noqa: E402
from scripts.evaluate_dataset import _checkpoint_profiles, _load_model  # noqa: E402
from scripts.evaluate_experiment_suite import digest, now, write  # noqa: E402
from scripts.wait_train_gnn_gate_set1 import gpu_processes  # noqa: E402


def workload(model, instance, profiles):
    env = ResidualStressFJSPEnv(instance, **profiles)
    observation, info = env.reset(seed=1007)
    buffer = RolloutBuffer()
    while len(buffer) < 64:
        if not info['valid_actions']:
            observation, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                observation, info = env.reset(seed=1007 + len(buffer))
            continue
        graph = copy.deepcopy(env.get_graph_state(info['valid_actions']))
        choice, log_prob, value = model.act(observation, info['action_features'], graph_state=graph, deterministic=True)
        next_observation, reward, terminated, truncated, next_info = env.step(choice)
        buffer.add(observation, info['action_features'], choice, log_prob, value, reward,
                   terminated or truncated or len(buffer) == 63, graph_state=graph)
        observation, info = next_observation, next_info
        if terminated or truncated:
            observation, info = env.reset(seed=1007 + len(buffer))
    return buffer


def inference_round(model, buffer, deterministic):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for i in range(64):
        model.act(buffer.observations[i], buffer.action_features[i], graph_state=buffer.graph_states[i],
                  deterministic=deterministic)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / 64 * 1000


def run(output):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != '4':
        raise RuntimeError('This benchmark requires the explicitly selected physical GPU4')
    if gpu_processes((4,)):
        raise RuntimeError('GPU4 is occupied')
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    result = dict(started_at=now(), physical_gpu=4, gpu=torch.cuda.get_device_name(device),
                  torch_version=torch.__version__, cpu_threads=2, dtype='float32',
                  repeats=3, inference_calls_per_repeat=64, ppo_samples=64, ppo_epochs=4,
                  notes=['Fixed 64-step workload from the baseline policy on the first static test instance.',
                         'Identical observations, action candidates, graph states and rollout supplied to both architectures.',
                         'Native act includes CPU collation, host/device transfers and output synchronization.',
                         'PPO is a fixed-workload compute benchmark; reused rollout is not a convergence experiment.',
                         'Single GPU only, no DDP communication; other GPUs may have unrelated jobs.',
                         'No original weights, configuration or training outputs are modified.'], datasets=[])
    for dataset in (1, 2, 3):
        paths = dict(base=ROOT / f'outputs/fullstate_jingtai_set{dataset}/GNN/checkpoints/best.pt',
                     film=ROOT / f'outputs/GNN+FiLM_set{dataset}/static/checkpoints/best.pt')
        models, checkpoints = {}, {}
        for name, path in paths.items():
            models[name], checkpoints[name] = _load_model(path, device)
        assert not models['base'].shared_kv_film and models['film'].shared_kv_film
        assert not models['base'].physical_gating and not models['film'].physical_gating
        profiles = _checkpoint_profiles(checkpoints['base'])
        assert profiles == _checkpoint_profiles(checkpoints['film'])
        set_seed(1007)
        instance = load_instances(ROOT / f'data/dataset{dataset}_cost0/static/test')[0]
        buffer = workload(models['base'], instance, profiles)
        record = dict(dataset=dataset, instance=instance.name, profiles=profiles,
                      checkpoint_hashes={name: digest(path) for name, path in paths.items()},
                      parameters={name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
                      mean_graph_nodes=statistics.mean(len(g['node_features']) for g in buffer.graph_states),
                      mean_graph_edges=statistics.mean(len(g['edge_types']) for g in buffer.graph_states), timings={})
        for deterministic, label in [(True, 'validation_act_ms'), (False, 'sampling_act_ms')]:
            for model in models.values():
                for i in range(16):
                    model.act(buffer.observations[i], buffer.action_features[i], graph_state=buffer.graph_states[i],
                              deterministic=deterministic)
            measured = dict(base=[], film=[])
            for repeat in range(3):
                for name in (('base', 'film') if repeat % 2 == 0 else ('film', 'base')):
                    set_seed(20261116 + repeat)
                    measured[name].append(inference_round(models[name], buffer, deterministic))
            record['timings'][label] = {name: dict(median=statistics.median(values), samples=values)
                                        for name, values in measured.items()}
        algorithms = {name: PPO(model, PPOConfig(ppo_epochs=4, mini_batch_size=64)) for name, model in models.items()}
        for algorithm in algorithms.values():
            set_seed(20261116); algorithm.update(buffer)
        measured = dict(base=[], film=[])
        for repeat in range(3):
            for name in (('base', 'film') if repeat % 2 == 0 else ('film', 'base')):
                set_seed(20261116 + repeat)
                torch.cuda.synchronize(); started = time.perf_counter()
                algorithms[name].update(buffer)
                torch.cuda.synchronize(); measured[name].append(time.perf_counter() - started)
        record['timings']['ppo_update_seconds'] = {name: dict(median=statistics.median(values), samples=values)
                                                  for name, values in measured.items()}
        for metric, values in record['timings'].items():
            assert values['base']['median'] > 0 and values['film']['median'] > 0
            values['film_time_reduction_percent'] = (1 - values['film']['median'] / values['base']['median']) * 100
        result['datasets'].append(record)
        write(output / 'results.json', result)
        print(json.dumps(dict(dataset=dataset, timings=record['timings']), ensure_ascii=False), flush=True)
        del algorithms, models, checkpoints, buffer
        torch.cuda.empty_cache()
    result['completed_at'] = now(); write(output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
