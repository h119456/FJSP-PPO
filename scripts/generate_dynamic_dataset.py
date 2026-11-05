"""Generate dataset1-scale instances with stochastic/fixed dynamic-event scenarios."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import math
import shutil
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "dataset1" / "static"
TARGET_ROOT = ROOT / "data" / "dataset1" / "dynamic"
SEED = 20261103
JOB_RELEASE_COUNT_RANGE = (1, 2)
JOB_RELEASE_COUNT_PROBABILITIES = {1: 0.90, 2: 0.10}
JOB_RELEASE_TIME_FRACTION = (0.05, 0.30)
BREAKDOWN_COUNT_RANGE = (1, 2)
BREAKDOWN_COUNT_PROBABILITIES = {1: 0.95, 2: 0.05}
BREAKDOWN_TIME_FRACTION = (0.15, 0.70)
REPAIR_DURATION_FACTOR = (0.50, 1.50)
SCENARIO_PROBABILITIES = {
    "none": 0.65,
    "job_release_only": 0.20,
    "breakdown_only": 0.10,
    "both": 0.05,
}



def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _normal_machines(instance: dict[str, Any]) -> tuple[str, ...]:
    auxiliary = set(instance["stress_relief"]["machines"]) | set(instance["compensation"]["machines"])
    machines = tuple(machine for machine in instance["machines"] if machine not in auxiliary)
    if not machines:
        raise ValueError("dynamic generation requires a normal processing machine")
    return machines


def fixed_scenario_counts(size: int) -> dict[str, int]:
    """Use largest-remainder allocation, exact at multiples of twenty."""
    if size <= 0:
        raise ValueError("evaluation split size must be positive")
    exact = {scenario: size * probability for scenario, probability in SCENARIO_PROBABILITIES.items()}
    counts = {scenario: math.floor(value) for scenario, value in exact.items()}
    remainder = size - sum(counts.values())
    order = sorted(exact, key=lambda scenario: -(exact[scenario] - counts[scenario]))
    for scenario in order[:remainder]:
        counts[scenario] += 1
    return counts


def _base_payload(instance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if key not in {"name", "dynamic_events"}}


def _minimum_duration(operation: dict[str, Any]) -> float:
    return min(float(value) for value in operation["machines"].values())


def _reference_horizon(instance: dict[str, Any]) -> float:
    workloads = [
        sum(_minimum_duration(operation) for operation in job["operations"])
        for job in instance["jobs"]
    ]
    return sum(workloads) / len(_normal_machines(instance)) + max(workloads)


def _mean_processing_duration(instance: dict[str, Any]) -> float:
    values = [
        float(duration)
        for job in instance["jobs"]
        for operation in job["operations"]
        for duration in operation["machines"].values()
    ]
    return sum(values) / len(values)


def _stochastic_spec(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "stochastic",
        "resample_each_reset": True,
        "event_seed_source": "environment_reset_seed",
        "future_events_visible_to_agent": False,
        "unreleased_jobs_visible_to_agent": False,
        "scenario_probabilities": SCENARIO_PROBABILITIES,
        "job_release": {
            "enabled": True,
            "delayed_job_count_range": list(JOB_RELEASE_COUNT_RANGE),
            "count_probabilities": {
                str(key): value
                for key, value in JOB_RELEASE_COUNT_PROBABILITIES.items()
            },
            "release_time_fraction_of_reference_horizon": list(
                JOB_RELEASE_TIME_FRACTION
            ),
            "keep_at_least_one_job_released_at_time_zero": True,
        },
        "machine_breakdown": {
            "enabled": True,
            "event_count_range": list(BREAKDOWN_COUNT_RANGE),
            "count_probabilities": {
                str(key): value
                for key, value in BREAKDOWN_COUNT_PROBABILITIES.items()
            },
            "machine_pool": list(_normal_machines(instance)),
            "breakdown_time_fraction_of_reference_horizon": list(
                BREAKDOWN_TIME_FRACTION
            ),
            "repair_duration_factor_of_mean_processing_time": list(
                REPAIR_DURATION_FACTOR
            ),
            "interruption_policy": "pause_resume",
            "allow_overlapping_failures_on_same_machine": False,
        },
        "reference_horizon": round(_reference_horizon(instance), 3),
    }


def _fixed_spec(
    instance: dict[str, Any], *, scenario_seed: int, scenario: str
) -> dict[str, Any]:
    rng = random.Random(scenario_seed)
    horizon = _reference_horizon(instance)
    mean_duration = _mean_processing_duration(instance)
    jobs = [str(job["id"]) for job in instance["jobs"]]
    events: list[dict[str, Any]] = []
    if scenario in {"job_release_only", "both"}:
        delayed_count = rng.choices(
            list(JOB_RELEASE_COUNT_PROBABILITIES),
            weights=list(JOB_RELEASE_COUNT_PROBABILITIES.values()),
            k=1,
        )[0]
        delayed_count = min(delayed_count, len(jobs) - 1)
        for job_id in rng.sample(jobs, delayed_count):
            events.append(
                {
                    "time": round(
                        rng.uniform(*JOB_RELEASE_TIME_FRACTION) * horizon, 3
                    ),
                    "type": "job_release",
                    "job_id": job_id,
                    "known_to_agent_before_occurrence": False,
                }
            )

    if scenario in {"breakdown_only", "both"}:
        breakdown_count = rng.choices(
            list(BREAKDOWN_COUNT_PROBABILITIES),
            weights=list(BREAKDOWN_COUNT_PROBABILITIES.values()),
            k=1,
        )[0]
        for machine_id in rng.sample(list(_normal_machines(instance)), min(breakdown_count, len(_normal_machines(instance)))):
            events.append(
                {
                    "time": round(
                        rng.uniform(*BREAKDOWN_TIME_FRACTION) * horizon, 3
                    ),
                    "type": "machine_breakdown",
                    "machine_id": machine_id,
                    "repair_duration": round(
                        rng.uniform(*REPAIR_DURATION_FACTOR) * mean_duration, 3
                    ),
                    "interruption_policy": "pause_resume",
                    "known_to_agent_before_occurrence": False,
                }
            )
    events.sort(key=lambda event: (float(event["time"]), str(event["type"])))
    return {
        "mode": "fixed",
        "scenario_seed": scenario_seed,
        "scenario": scenario,
        "future_events_visible_to_agent": False,
        "unreleased_jobs_visible_to_agent": False,
        "reference_horizon": round(horizon, 3),
        "events": events,
    }


def _readme(manifest: dict[str, Any]) -> str:
    sizes = manifest["split_sizes"]
    return f"""# 动态数据集

本目录与 `{manifest['base_dataset']}` 的静态实例逐文件配对，工件、工序、机器和物理参数一致，仅增加动态事件配置。

- 容量：{sizes['train']} train、{sizes['validation']} validation、{sizes['test']} test。
- 训练事件每次 reset 依环境种子重新采样；验证与测试使用固定事件轨迹。
- 场景概率：无事件65%、插单20%、故障10%、混合5%。固定集合按此比例分层，保留旧轨迹后按差额补足。
- 事件为延迟工件释放与普通生产机器故障，故障采用 pause-repair-resume；普通机器池按每个实例的公开设备清单扣除辅助设备，参考时间也使用实际普通机器数。
- 沿用每类1～2个事件及原时间／维修参数；大实例的相对扰动比例因此可能较低，未暗中加强事件。
- 未释放工件及未来事件对策略隐藏，释放时图才增加对应节点。

## 动态输入版本 2

归一化只使用已释放工件。WAIT 的候选存在性和期限由公开状态计算，真实事件可以提前中断；故障比例不使用隐藏工件路线。环境静态语义为7，动态输入版本为2。旧动态成绩需在修复后重新训练和评价，旧动态断点不能静默续训到新协议。

`manifest.json` 记录容量、来源、事件种子、固定场景配额、文件清单和原样保留的旧文件SHA256。
"""


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def generate_dataset(
    source_root: Path,
    target_root: Path,
    *,
    seed: int = SEED,
    preserved_root: Path | None = None,
) -> dict[str, Any]:
    """Generate paired dynamic data without overwriting an existing dataset."""
    source_root, target_root = Path(source_root), Path(target_root)
    preserved_root = Path(preserved_root) if preserved_root is not None else None
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty dynamic dataset: {target_root}")
    source_manifest = _read_json(source_root / "manifest.json")
    preserved_manifest = _read_json(preserved_root / "manifest.json") if preserved_root else None
    sizes = {split: int(source_manifest["split_sizes"][split]) for split in ("train", "validation", "test")}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("all source split sizes must be positive")
    source_files: dict[str, list[Path]] = {}
    preserved: dict[str, dict[str, Path]] = {}
    fixed_counts: dict[str, dict[str, int]] = {}
    added_scenarios: dict[str, list[str]] = {}
    # Validate sources before writing. Legacy data must belong to the same split.
    for split, size in sizes.items():
        files = sorted((source_root / split).glob("instance_*.json"),
                       key=lambda path: int(path.stem.rsplit("_", 1)[1]))
        if len(files) != size:
            raise ValueError(f"{split}: expected {size} source files, got {len(files)}")
        expected_paths = {f"{split}/{path.name}" for path in files}
        if set(source_manifest["splits"][split]) != expected_paths:
            raise ValueError(f"{split}: source manifest file set differs from directory")
        source_files[split] = files
        preserved[split] = {path.name: path for path in (preserved_root / split).glob("instance_*.json")} if preserved_root else {}
        if preserved_manifest is not None:
            listed = preserved_manifest["splits"][split]
            if (int(preserved_manifest["split_sizes"][split]) != len(preserved[split])
                    or len(listed) != len(set(listed))
                    or set(listed) != {f"{split}/{name}" for name in preserved[split]}):
                raise ValueError(f"{split}: preserved manifest file set differs from directory")
        available = {path.name: path for path in files}
        if not set(preserved[split]) <= set(available):
            raise ValueError(f"{split}: preserved filenames are not present in source")
        counts: Counter[str] = Counter()
        for name, path in preserved[split].items():
            old = _read_json(path)
            if _base_payload(old) != _base_payload(_read_json(available[name])):
                raise ValueError(f"{path}: preserved dynamic instance does not match static source")
            dynamic = old.get("dynamic_events", {})
            if dynamic.get("mode") != ("stochastic" if split == "train" else "fixed"):
                raise ValueError(f"{path}: incorrect preserved event mode")
            if split != "train":
                counts[str(dynamic.get("scenario"))] += 1
        if split != "train":
            desired = fixed_scenario_counts(size)
            if any(counts[scenario] > desired.get(scenario, 0) for scenario in counts):
                raise ValueError(f"{split}: preserved scenarios exceed target quotas")
            fixed_counts[split] = desired
            scenarios = [scenario for scenario, count in desired.items() for _ in range(count - counts[scenario])]
            random.Random(seed + (10_000 if split == "validation" else 20_000)).shuffle(scenarios)
            added_scenarios[split] = scenarios

    manifest: dict[str, Any] = {
        "seed": seed,
        "base_dataset": _display_path(source_root),
        "split_sizes": sizes,
        "training_event_mode": "stochastic_resampled_each_reset",
        "evaluation_event_mode": "fixed_trace",
        "future_events_visible_to_agent": False,
        "unreleased_jobs_visible_to_agent": False,
        "dynamic_input_semantics_version": 2,
        "event_semantics": {"job_release": "predeclared job becomes dispatchable", "machine_breakdown": "pause_repair_resume"},
        "scenario_probabilities": SCENARIO_PROBABILITIES,
        "fixed_scenario_counts_per_evaluation_split": fixed_counts,
        "event_ranges": {
            "delayed_job_count": list(JOB_RELEASE_COUNT_RANGE),
            "delayed_job_count_probabilities": JOB_RELEASE_COUNT_PROBABILITIES,
            "job_release_time_fraction": list(JOB_RELEASE_TIME_FRACTION),
            "breakdown_count": list(BREAKDOWN_COUNT_RANGE),
            "breakdown_count_probabilities": BREAKDOWN_COUNT_PROBABILITIES,
            "breakdown_time_fraction": list(BREAKDOWN_TIME_FRACTION),
            "repair_duration_factor": list(REPAIR_DURATION_FACTOR),
        },
        "preserved_from": _display_path(preserved_root) if preserved_root else None,
        "preserved_files": {},
        "splits": {},
    }
    for key in ("job_count_range", "operation_count_range", "normal_machine_count_range", "normal_machines", "normal_machine_pool"):
        if key in source_manifest:
            manifest[key] = source_manifest[key]
    for split, files in source_files.items():
        directory = target_root / split
        directory.mkdir(parents=True, exist_ok=True)
        scenario_iter = iter(added_scenarios.get(split, []))
        written = []
        for index, source in enumerate(files, start=1):
            target = directory / source.name
            if source.name in preserved[split]:
                original = preserved[split][source.name]
                shutil.copy2(original, target)
                manifest["preserved_files"][f"{split}/{source.name}"] = hashlib.sha256(target.read_bytes()).hexdigest()
            else:
                data = copy.deepcopy(_read_json(source))
                data["name"] = f"dynamic_{data['name']}"
                if split == "train":
                    data["dynamic_events"] = _stochastic_spec(data)
                else:
                    offset = 10_000 if split == "validation" else 20_000
                    data["dynamic_events"] = _fixed_spec(data, scenario_seed=seed + offset + index,
                                                          scenario=next(scenario_iter))
                target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written.append(f"{split}/{target.name}")
        manifest["splits"][split] = written
    (target_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target_root / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--preserved-root", type=Path)
    args = parser.parse_args()
    manifest = generate_dataset(args.source_root, args.target_root, seed=args.seed, preserved_root=args.preserved_root)
    print(json.dumps({"target": str(args.target_root), "split_sizes": manifest["split_sizes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
