"""Validate dataset2 schema, ranges, split sizes, and scheduling readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs.baselines import run_baseline  # noqa: E402
from ppo_rs.config import load_instance  # noqa: E402
from ppo_rs.environment import ResidualStressFJSPEnv  # noqa: E402


DATASET_ROOT = ROOT / "data" / "dataset2" / "static"
EXPECTED_SPLITS = {"train": 1000, "validation": 100, "test": 200}
EXPECTED_RANGES = {
    "jobs": (12, 20),
    "operations_per_job": (5, 10),
    "normal_machines": (6, 10),
    "candidate_machines": (2, 5),
    "normal_duration": (1.5, 5.0),
    "deformation_scale": (0.18, 0.45),
    "deformation_rate": (0.080, 0.250),
    "stress_relief_duration": (0.26, 0.78),
    "stress_relief_cost": (0.12, 0.48),
    "compensation_duration": (0.6, 1.2),
    "compensation_cost": (0.1, 0.4),
    "compensation_factor": (0.10, 0.20),
}


def _check_range(name: str, value: float) -> None:
    low, high = EXPECTED_RANGES[name]
    if not low <= value <= high:
        raise AssertionError(f"{name}={value} is outside [{low}, {high}]")


def _record(observed: dict[str, list[float]], name: str, value: float) -> None:
    _check_range(name, value)
    observed[name].append(value)


def validate_dataset(dataset_root: Path = DATASET_ROOT) -> dict[str, Any]:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["split_sizes"] != EXPECTED_SPLITS:
        raise AssertionError(f"unexpected manifest split sizes: {manifest['split_sizes']}")

    observed = {name: [] for name in EXPECTED_RANGES}
    split_summary: dict[str, dict[str, int]] = {}
    names: set[str] = set()
    smoke_paths: list[Path] = []

    for split, expected_count in EXPECTED_SPLITS.items():
        paths = sorted((dataset_root / split).glob("*.json"))
        if len(paths) != expected_count:
            raise AssertionError(
                f"{split}: expected {expected_count} files, found {len(paths)}"
            )
        smoke_paths.extend((paths[0], paths[len(paths) // 2], paths[-1]))

        split_jobs = 0
        split_operations = 0
        for path in paths:
            instance = load_instance(path)
            if instance.name in names:
                raise AssertionError(f"duplicate instance name: {instance.name}")
            names.add(instance.name)

            job_count = len(instance.jobs)
            _record(observed, "jobs", job_count)
            normal_machines = set(instance.machines) - {"M3", "M4"}
            _record(observed, "normal_machines", len(normal_machines))
            split_jobs += job_count

            if instance.compensation.trigger_deformation != 0.30:
                raise AssertionError(f"{path}: compensation threshold must be 0.30")
            if instance.physics.minimum_deformation_ratio != 0.5:
                raise AssertionError(f"{path}: minimum deformation ratio must be 0.5")
            if not instance.compensation.allow_original_machine:
                raise AssertionError(f"{path}: original-machine compensation is disabled")

            _record(
                observed,
                "stress_relief_duration",
                next(iter(instance.stress_relief.machines.values())),
            )
            _record(observed, "stress_relief_cost", instance.stress_relief.cost)
            _record(
                observed,
                "compensation_duration",
                next(iter(instance.compensation.machines.values())),
            )
            _record(observed, "compensation_cost", instance.compensation.cost)
            _record(
                observed,
                "compensation_factor",
                instance.physics.compensation_deformation_factor,
            )

            for job in instance.jobs:
                operation_count = len(job.operations)
                _record(observed, "operations_per_job", operation_count)
                split_operations += operation_count
                for operation in job.operations:
                    candidates = set(operation.machines)
                    if not candidates <= normal_machines:
                        raise AssertionError(
                            f"{path}: {job.job_id}/{operation.operation_id} uses "
                            "an auxiliary or unknown machine"
                        )
                    _record(observed, "candidate_machines", len(candidates))
                    _record(
                        observed,
                        "deformation_scale",
                        float(operation.deformation_scale),
                    )
                    _record(
                        observed,
                        "deformation_rate",
                        float(operation.deformation_rate),
                    )
                    for duration in operation.machines.values():
                        _record(observed, "normal_duration", duration)

        split_summary[split] = {
            "instances": len(paths),
            "jobs": split_jobs,
            "operations": split_operations,
        }

    smoke_results = []
    for path in smoke_paths:
        env = ResidualStressFJSPEnv(load_instance(path))
        metrics, _ = run_baseline(env, "ordinary_fjsp", seed=0)
        if not metrics["terminated"] or metrics["truncated"]:
            raise AssertionError(f"smoke scheduling did not terminate normally: {path}")
        if metrics["schedule_violation_count"] != 0:
            raise AssertionError(
                f"smoke scheduling violations in {path}: "
                f"{metrics['schedule_violations']}"
            )
        smoke_results.append(
            {
                "instance": path.relative_to(dataset_root).as_posix(),
                "makespan": metrics["makespan"],
                "readiness_violations": metrics["process_readiness_violations"],
            }
        )

    summary: dict[str, Any] = {
        "status": "ok",
        "unique_instances": len(names),
        "splits": split_summary,
        "observed_ranges": {
            name: [min(values), max(values)] for name, values in observed.items()
        },
        "scheduling_smoke": smoke_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
