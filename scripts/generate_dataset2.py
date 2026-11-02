"""Generate the deterministic factory-scale synthetic dataset2."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "dataset2"
SEED = 20261030

NORMAL_MACHINE_POOL = (
    "M1",
    "M2",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11",
    "M12",
)
STRESS_RELIEF_MACHINE = "M3"
COMPENSATION_MACHINE = "M4"

MIN_JOBS = 12
MAX_JOBS = 20
MIN_OPERATIONS = 5
MAX_OPERATIONS = 10
MIN_NORMAL_MACHINES = 6
MAX_NORMAL_MACHINES = 10
MIN_CANDIDATE_MACHINES = 2
MAX_CANDIDATE_MACHINES = 5

STRESS_RELIEF_DURATION_FACTOR = 1.30
STRESS_RELIEF_COST_FACTOR = 1.20


def _rounded(rng: random.Random, low: float, high: float, digits: int) -> float:
    return round(rng.uniform(low, high), digits)


def _operation(
    rng: random.Random,
    operation_index: int,
    normal_machines: tuple[str, ...],
) -> dict[str, Any]:
    candidate_count = rng.randint(
        MIN_CANDIDATE_MACHINES,
        min(MAX_CANDIDATE_MACHINES, len(normal_machines)),
    )
    candidate_machines = rng.sample(normal_machines, candidate_count)
    return {
        "id": f"O{operation_index}",
        "machines": {
            machine: _rounded(rng, 1.5, 5.0, 1)
            for machine in candidate_machines
        },
        "deformation_scale": _rounded(rng, 0.18, 0.45, 3),
        "deformation_rate": _rounded(rng, 0.080, 0.250, 4),
    }


def _instance(
    rng: random.Random,
    split: str,
    number: int,
    job_count: int,
    normal_machine_count: int,
) -> dict[str, Any]:
    normal_machines = NORMAL_MACHINE_POOL[:normal_machine_count]
    jobs = []
    for job_index in range(1, job_count + 1):
        operation_count = rng.randint(MIN_OPERATIONS, MAX_OPERATIONS)
        jobs.append(
            {
                "id": f"J{job_index}",
                "operations": [
                    _operation(rng, operation_index, normal_machines)
                    for operation_index in range(1, operation_count + 1)
                ],
            }
        )

    return {
        "name": f"dataset2_{split}_{number:04d}",
        "synthetic_test_data": True,
        "machines": [
            *normal_machines,
            STRESS_RELIEF_MACHINE,
            COMPENSATION_MACHINE,
        ],
        "jobs": jobs,
        "stress_relief": {
            "enabled": True,
            "machines": {
                STRESS_RELIEF_MACHINE: round(
                    _rounded(rng, 0.2, 0.6, 1)
                    * STRESS_RELIEF_DURATION_FACTOR,
                    2,
                )
            },
            "cost": round(
                _rounded(rng, 0.1, 0.4, 1) * STRESS_RELIEF_COST_FACTOR,
                2,
            ),
            "max_per_job": MAX_OPERATIONS - 1,
            "allowed_after_operations": [
                f"O{operation_index}"
                for operation_index in range(1, MAX_OPERATIONS)
            ],
        },
        "compensation": {
            "enabled": True,
            "machines": {COMPENSATION_MACHINE: _rounded(rng, 0.6, 1.2, 1)},
            "cost": _rounded(rng, 0.1, 0.4, 1),
            "max_per_job": MAX_OPERATIONS - 1,
            "allowed_after_operations": [
                f"O{operation_index}"
                for operation_index in range(1, MAX_OPERATIONS)
            ],
            "trigger_deformation": 0.30,
            "allow_original_machine": True,
            "original_machine_duration_factor": 1.0,
        },
        "physics": {
            "default_deformation_scale": 1.0,
            "default_deformation_rate": 0.04,
            "minimum_deformation_ratio": 0.5,
            "compensation_deformation_factor": _rounded(rng, 0.10, 0.20, 2),
        },
        "objective": {
            "makespan_weight": 1.0,
            "relief_cost_weight": 0.5,
            "compensation_cost_weight": 0.5,
            "waiting_cost_weight": 0.0,
            "waiting_unit_cost": 0.0,
        },
        "environment": {
            "max_decisions": 5000,
            "max_time": 1000.0,
            "readiness_scan_step": 0.25,
        },
    }


def main() -> None:
    rng = random.Random(SEED)
    split_sizes = {"train": 1000, "validation": 100, "test": 200}
    manifest: dict[str, Any] = {
        "seed": SEED,
        "split_sizes": split_sizes,
        "job_count_range": [MIN_JOBS, MAX_JOBS],
        "operation_count_range": [MIN_OPERATIONS, MAX_OPERATIONS],
        "normal_machine_count_range": [
            MIN_NORMAL_MACHINES,
            MAX_NORMAL_MACHINES,
        ],
        "candidate_machine_count_range": [
            MIN_CANDIDATE_MACHINES,
            MAX_CANDIDATE_MACHINES,
        ],
        "normal_machine_pool": list(NORMAL_MACHINE_POOL),
        "stress_relief_machine": STRESS_RELIEF_MACHINE,
        "compensation_machine": COMPENSATION_MACHINE,
        "parameter_ranges": {
            "normal_duration": [1.5, 5.0],
            "deformation_scale": [0.18, 0.45],
            "deformation_rate": [0.080, 0.250],
            "stress_relief_duration": [0.26, 0.78],
            "stress_relief_cost": [0.12, 0.48],
            "compensation_duration": [0.6, 1.2],
            "compensation_cost": [0.1, 0.4],
            "compensation_factor": [0.10, 0.20],
            "mandatory_compensation_threshold": 0.30,
        },
        "objective_weights": {
            "makespan": 1.0,
            "stress_relief_cost": 0.5,
            "compensation_cost": 0.5,
            "waiting_cost": 0.0,
        },
        "splits": {},
    }

    for split, size in split_sizes.items():
        directory = DATASET_ROOT / split
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for number in range(1, size + 1):
            filename = f"instance_{number:04d}.json"
            job_count = MIN_JOBS + (number - 1) % (MAX_JOBS - MIN_JOBS + 1)
            normal_machine_count = MIN_NORMAL_MACHINES + (
                (number - 1) % (MAX_NORMAL_MACHINES - MIN_NORMAL_MACHINES + 1)
            )
            data = _instance(
                rng,
                split,
                number,
                job_count,
                normal_machine_count,
            )
            (directory / filename).write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            files.append(f"{split}/{filename}")
        manifest["splits"][split] = files

    (DATASET_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
