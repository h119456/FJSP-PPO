"""Generate the deterministic synthetic 100/20/20 FJSP dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "dataset"
SEED = 20261028
NORMAL_MACHINES = ("M1", "M2", "M5", "M6")
STRESS_RELIEF_MACHINE = "M3"
COMPENSATION_MACHINE = "M4"
MIN_JOBS = 6
MAX_JOBS = 10
MIN_OPERATIONS = 4
MAX_OPERATIONS = 7
STRESS_RELIEF_DURATION_FACTOR = 1.30
STRESS_RELIEF_COST_FACTOR = 1.20


def _rounded(rng: random.Random, low: float, high: float, digits: int) -> float:
    return round(rng.uniform(low, high), digits)


def _operation(
    rng: random.Random,
    operation_index: int,
) -> dict[str, Any]:
    candidate_count = rng.choice([2, 2, 3, 3, 4])
    candidate_machines = rng.sample(NORMAL_MACHINES, candidate_count)
    machines = {
        machine: _rounded(rng, 1.5, 5.0, 1)
        for machine in candidate_machines
    }

    operation: dict[str, Any] = {
        "id": f"O{operation_index}",
        "machines": machines,
        "deformation_scale": _rounded(rng, 0.18, 0.45, 3),
        "deformation_rate": _rounded(rng, 0.080, 0.250, 4),
    }
    return operation


def _instance(
    rng: random.Random,
    split: str,
    number: int,
    job_count: int,
) -> dict[str, Any]:
    jobs = []
    for job_index in range(1, job_count + 1):
        operation_count = rng.choice([4, 5, 5, 6, 6, 7])
        jobs.append(
            {
                "id": f"J{job_index}",
                "operations": [
                    _operation(rng, operation_index)
                    for operation_index in range(1, operation_count + 1)
                ],
            }
        )

    return {
        "name": f"{split}_{number:02d}",
        "synthetic_test_data": True,
        "machines": [*NORMAL_MACHINES, STRESS_RELIEF_MACHINE, COMPENSATION_MACHINE],
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
            "max_decisions": 1500,
            "max_time": 500.0,
            "readiness_scan_step": 0.25,
        },
    }


def main() -> None:
    rng = random.Random(SEED)
    split_sizes = {"train": 100, "validation": 20, "test": 20}
    manifest: dict[str, Any] = {
        "seed": SEED,
        "split_sizes": split_sizes,
        "job_count_range": [MIN_JOBS, MAX_JOBS],
        "operation_count_range": [MIN_OPERATIONS, MAX_OPERATIONS],
        "normal_machines": list(NORMAL_MACHINES),
        "stress_relief_machine": STRESS_RELIEF_MACHINE,
        "compensation_machine": COMPENSATION_MACHINE,
        "stress_relief_duration_factor": STRESS_RELIEF_DURATION_FACTOR,
        "stress_relief_cost_factor": STRESS_RELIEF_COST_FACTOR,
        "splits": {},
    }
    for split, size in split_sizes.items():
        directory = DATASET_ROOT / split
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for number in range(1, size + 1):
            filename = f"instance_{number:02d}.json"
            job_count = MIN_JOBS + (number - 1) % (MAX_JOBS - MIN_JOBS + 1)
            data = _instance(rng, split, number, job_count)
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
