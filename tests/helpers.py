"""Small synthetic fixtures. These values are software tests, not material data."""

from __future__ import annotations

import math

from ppo_rs.config import InstanceSpec, instance_from_dict


def tiny_instance(
    *,
    jobs: int = 1,
    min_dwell: float = 3.0,
    compensation_threshold: float = 0.5,
    deformation_rate: float = 0.1,
    relief_max: int = 1,
    compensation_max: int = 2,
) -> InstanceSpec:
    if min_dwell <= 0:
        raise ValueError("min_dwell must be positive")
    first_operation_rate = math.log(2.0) / min_dwell
    job_rows = []
    for index in range(jobs):
        job_rows.append(
            {
                "id": f"J{index + 1}",
                "operations": [
                    {
                        "id": "O1",
                        "machines": {"M1": 1.0, "M2": 1.5},
                        "deformation_scale": 1.0,
                        "deformation_rate": first_operation_rate,
                    },
                    {
                        "id": "O2",
                        "machines": {"M1": 1.0},
                        "deformation_scale": 0.8,
                        "deformation_rate": deformation_rate * 1.1,
                    },
                ],
            }
        )
    return instance_from_dict(
        {
            "name": "synthetic_unit_fixture",
            "synthetic_test_data": True,
            "machines": ["M1", "M2", "R", "C"],
            "jobs": job_rows,
            "stress_relief": {
                "enabled": True,
                "machines": {"R": 0.65},
                "cost": 0.36,
                "max_per_job": relief_max,
            },
            "compensation": {
                "enabled": True,
                "machines": {"C": 1.0},
                "cost": 0.3,
                "max_per_job": compensation_max,
                "trigger_deformation": compensation_threshold,
                "allow_original_machine": True,
                "original_machine_duration_factor": 1.0,
            },
            "physics": {
                "default_deformation_scale": 1.0,
                "default_deformation_rate": deformation_rate,
                "minimum_deformation_ratio": 0.5,
                "compensation_deformation_factor": 0.1,
            },
            "objective": {
                "makespan_weight": 1.0,
                "relief_cost_weight": 0.5,
                "compensation_cost_weight": 0.5,
                "waiting_cost_weight": 0.0,
                "waiting_unit_cost": 0.0,
            },
            "environment": {
                "max_decisions": 40,
                "max_time": 50.0,
                "readiness_scan_step": 0.25,
            },
        }
    )


def choose(env, kind: str, *, job_id: str | None = None, machine: str | None = None) -> int:
    for index, action in enumerate(env.get_valid_actions()):
        if action.kind.value != kind:
            continue
        if job_id is not None and action.job_id != job_id:
            continue
        if machine is not None and action.machine_id != machine:
            continue
        return index
    raise AssertionError(f"no action {kind} job={job_id} machine={machine}: {env.get_valid_actions()}")

