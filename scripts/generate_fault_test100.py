"""Create independent cost0 fault-only evaluation sets, 100 cases per scale."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import generate_dataset as small, generate_dataset2 as medium, generate_dataset3 as large
from scripts.generate_dynamic_dataset import SCENARIO_PROBABILITIES, _fixed_spec
from scripts.make_zero_cost_dataset import _zero_cost
from scripts.validate_dynamic_dataset import _validate_events
from ppo_rs import load_instance
from ppo_rs.environment import materialize_dynamic_events

SIZE = 100
SEED = 2026111601
TARGET = ROOT / 'data/dynamic_fault_test100_cost0'
SCENARIOS = ('breakdown_only', 'both')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fingerprint(raw):
    # Compare the underlying scheduling instance, independently of event/name.
    base = {k: v for k, v in _zero_cost(raw).items()
            if k not in {'name', 'dynamic_events', 'synthetic_test_data'}}
    return hashlib.sha256(json.dumps(base, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def quotas(size=SIZE):
    total = sum(SCENARIO_PROBABILITIES[s] for s in SCENARIOS)
    exact = {s: size * SCENARIO_PROBABILITIES[s] / total for s in SCENARIOS}
    result = {s: math.floor(exact[s]) for s in SCENARIOS}
    for s in sorted(SCENARIOS, key=lambda s: -(exact[s] - result[s]))[:size-sum(result.values())]:
        result[s] += 1
    return result


def inventory(data_root):
    seen, manifests, file_count, mixes = set(), {}, 0, {}
    for dataset in (1, 2, 3):
        for phase in ('static', 'dynamic'):
            root = data_root / f'dataset{dataset}_cost0' / phase
            manifest = read(root / 'manifest.json')
            manifests[str((root / 'manifest.json').relative_to(data_root))] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
            for split in ('train', 'validation', 'test'):
                files = sorted((root / split).glob('instance_*.json'))
                if len(files) != manifest['split_sizes'][split]:
                    raise ValueError(f'incomplete source: {root / split}')
                counts = Counter()
                for path in files:
                    raw = read(path); seen.add(fingerprint(raw)); file_count += 1
                    if phase == 'dynamic' and split == 'test':
                        counts[raw['dynamic_events']['scenario']] += 1
                if counts:
                    mixes[f'set{dataset}'] = dict(counts)
                    if counts['breakdown_only'] != 2 * counts['both'] or not counts['both']:
                        raise ValueError('source fault scenario ratio changed; expected 2:1')
    return seen, dict(source_instance_files=file_count, unique_source_bases=len(seen),
                      source_manifest_sha256=manifests, source_test_scenario_counts=mixes)


def candidate(dataset, rng, number):
    module = {1: small, 2: medium, 3: large}[dataset]
    jobs = module.MIN_JOBS + (number - 1) % (module.MAX_JOBS - module.MIN_JOBS + 1)
    if dataset == 1:
        raw = small._instance(rng, 'fault_test100', number, jobs)
    else:
        machines = module.MIN_NORMAL_MACHINES + (number - 1) % (module.MAX_NORMAL_MACHINES - module.MIN_NORMAL_MACHINES + 1)
        raw = medium._instance(rng, 'fault_test100', number, jobs, machines,
                               operation_count_range=(module.MIN_OPERATIONS, module.MAX_OPERATIONS),
                               normal_machine_pool=module.NORMAL_MACHINE_POOL, name_prefix=f'dataset{dataset}')
    raw['name'] = f'dataset{dataset}_cost0_fault_test100_{number:04d}'
    return _zero_cost(raw)


def validate(target, data_root=ROOT / 'data', *, existing=None):
    target = Path(target)
    if existing is None:
        existing, _ = inventory(Path(data_root))
    manifest = read(target / 'manifest.json')
    if manifest['size_per_set'] != SIZE or manifest['scenario_counts_per_set'] != quotas():
        raise ValueError('incorrect size or scenario quotas')
    seen, names, seeds, report = set(), set(), set(), {}
    for dataset in (1, 2, 3):
        key = f'set{dataset}'; directory = target / key / 'test'
        files = sorted(directory.glob('*.json'))
        listed = manifest['sets'][key]['files_sha256']
        if len(files) != SIZE or {p.name for p in files} != set(listed):
            raise ValueError(f'{key}: file inventory mismatch')
        counts, jobs, machines = Counter(), Counter(), Counter()
        for number, path in enumerate(files, 1):
            raw = read(path); dynamic = raw['dynamic_events']
            if hashlib.sha256(path.read_bytes()).hexdigest() != listed[path.name]:
                raise ValueError(f'{path}: checksum mismatch')
            digest = fingerprint(raw)
            if digest in existing or digest in seen:
                raise ValueError(f'{path}: overlapping scheduling instance')
            seen.add(digest)
            if raw['name'] in names or dynamic['scenario_seed'] in seeds:
                raise ValueError('duplicate name or event seed')
            names.add(raw['name']); seeds.add(dynamic['scenario_seed'])
            if raw['stress_relief']['cost'] != 0 or raw['compensation']['cost'] != 0:
                raise ValueError('expected cost0')
            if dynamic['mode'] != 'fixed' or dynamic['future_events_visible_to_agent'] is not False or dynamic['unreleased_jobs_visible_to_agent'] is not False:
                raise ValueError('expected fixed hidden events')
            scenario = dynamic['scenario']; counts[scenario] += 1
            types = Counter(e['type'] for e in dynamic['events'])
            if not 1 <= types['machine_breakdown'] <= 2 or types['job_release'] not in ((0,) if scenario == 'breakdown_only' else (1, 2)):
                raise ValueError('event types do not match scenario')
            instance = load_instance(path)
            _validate_events(path, instance, dynamic['events'])
            if materialize_dynamic_events(instance, seed=7) != materialize_dynamic_events(instance, seed=999):
                raise ValueError('fixed trace depends on evaluation seed')
            # Recreate every case from its recorded generation seeds. This also
            # checks all process, physics, cost and scale parameters together.
            spec = manifest['sets'][key]
            expected = candidate(dataset, random.Random(spec['base_seed'] + number), number)
            expected['dynamic_events'] = _fixed_spec(expected, scenario_seed=spec['event_seed'] + number, scenario=scenario)
            if raw != expected:
                raise ValueError('instance differs from reproducible generation rules')
            jobs[len(instance.jobs)] += 1; machines[len(instance.machines)] += 1
        if dict(counts) != quotas():
            raise ValueError(f'{key}: incorrect scenario counts')
        report[key] = dict(instances=len(files), scenario_counts=dict(counts), jobs=dict(sorted(jobs.items())),
                           total_machines=dict(sorted(machines.items())), overlaps_with_existing=0)
    return dict(passed=True, total_instances=len(seen), sets=report)


def generate(target=TARGET, *, data_root=ROOT / 'data', seed=SEED):
    target, data_root = Path(target), Path(data_root)
    if target.exists():
        raise FileExistsError(f'refusing to overwrite {target}')
    existing, provenance = inventory(data_root)
    manifest = dict(schema_version=1, seed=seed, size_per_set=SIZE, variant='cost0',
                    purpose='independent_fault_conditioned_test_only', scenario_counts_per_set=quotas(),
                    source_scenario_probabilities=SCENARIO_PROBABILITIES,
                    conditional_scenario_probabilities={'breakdown_only': 2/3, 'both': 1/3},
                    environment_semantics_version=7, dynamic_input_semantics_version=2,
                    source_audit=provenance, sets={})
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fault-test100-', dir=target.parent) as temporary:
        stage = Path(temporary) / 'dataset'; stage.mkdir()
        for dataset in (1, 2, 3):
            key = f'set{dataset}'; base_seed = seed + dataset * 1_000_003
            event_seed = base_seed + 400_000
            scenarios = [s for s, n in quotas().items() for _ in range(n)]
            random.Random(base_seed + 200_000).shuffle(scenarios)
            directory = stage / key / 'test'; directory.mkdir(parents=True)
            hashes = {}
            for number, scenario in enumerate(scenarios, 1):
                raw = candidate(dataset, random.Random(base_seed + number), number)
                raw['dynamic_events'] = _fixed_spec(raw, scenario_seed=event_seed + number, scenario=scenario)
                path = directory / f'instance_{number:04d}.json'; write(path, raw)
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['sets'][key] = dict(base_seed=base_seed, event_seed=event_seed,
                                         test_directory=f'{key}/test', files_sha256=hashes)
        write(stage / 'manifest.json', manifest)
        report = validate(stage, data_root, existing=existing)
        write(stage / 'validation_report.json', report)
        (stage / 'README.md').write_text(
            '# 独立故障动态测试集（cost0）\n\n'
            'set1/test、set2/test、set3/test各100例，共300例。每组仅故障67例、故障＋插单33例，'
            '由原动态场景10%／5%的条件比例2:1按最大余数法取整。\n\n'
            '基础调度实例全新生成，与现有三组cost0的全部训练、验证、测试基础实例逐一去重；'
            '不复用旧测试实例，也不补入无事件或仅插单场景。\n\n'
            '沿用各规模原生成器、物理参数和事件规则；仅辅助费用为0，目标权重与时长不变。'
            '插单指延迟释放既有工件，故障采用pause-repair-resume；每类事件1～2个。'
            '事件轨迹固定，未来事件和未释放工件隐藏。\n\n'
            'manifest.json记录种子、配额、源manifest和每例SHA256；validation_report.json记录完整性校验。'
            '评估时直接传入各set的test子目录，避免将元数据JSON当作实例。\n\n'
            '复现：`python scripts/generate_fault_test100.py --target-root /tmp/fault_test100_copy`。'
            '复检：`python scripts/generate_fault_test100.py --check`。\n\n'
            '这是故障条件下的独立测试集，均值不代表原含65%无事件的混合动态分布。\n')
        stage.rename(target)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', type=Path, default=TARGET)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = validate(args.target_root) if args.check else generate(args.target_root, seed=args.seed)
    print(json.dumps(dict(root=str(args.target_root), **report), ensure_ascii=False, indent=2))
