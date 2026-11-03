"""Generate dataset1-scale instances with stochastic/fixed dynamic-event scenarios."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "dataset"
TARGET_ROOT = ROOT / "data" / "dataset_dynamic"
SEED = 20261103
SPLIT_SIZES = {"train": 100, "validation": 20, "test": 20}
NORMAL_MACHINES = ("M1", "M2", "M5", "M6")
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
FIXED_SCENARIO_COUNTS = {
    "none": 12,
    "job_release_only": 5,
    "breakdown_only": 2,
    "both": 1,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _minimum_duration(operation: dict[str, Any]) -> float:
    return min(float(value) for value in operation["machines"].values())


def _reference_horizon(instance: dict[str, Any]) -> float:
    workloads = [
        sum(_minimum_duration(operation) for operation in job["operations"])
        for job in instance["jobs"]
    ]
    return sum(workloads) / len(NORMAL_MACHINES) + max(workloads)


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
            "machine_pool": list(NORMAL_MACHINES),
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
        for machine_id in rng.sample(list(NORMAL_MACHINES), breakdown_count):
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


def _readme() -> str:
    return """# Dataset Dynamic

This dataset reuses dataset1's static FJSP instances and adds exogenous dynamic-event scenarios.

- Split sizes: 100 train, 20 validation, 20 test.
- Static scale: 6-10 jobs, 4-7 operations per job, and the same six machines and parameter ranges as dataset1.
- Training instances contain stochastic event distributions that must be resampled from the environment reset seed on every episode.
- Validation and test instances contain fixed event traces for fair, reproducible policy comparison.
- Scenario mix: 65% no event, 20% insertion only, 10% breakdown only, and 5% both.
- Each 20-instance validation/test split is stratified as 12/5/2/1 across those four scenarios.
- Dynamic events are delayed job releases and normal-machine breakdowns.
- Breakdowns use pause-repair-resume semantics and are not visible to the policy before occurrence.
- Job definitions are stored in the file for reproducibility, but an unreleased job and its operations must not be exposed in the policy observation or graph. The active GNN topology grows when `job_release` occurs.

The environment executes these traces as external events. Future events and unreleased jobs remain hidden from the policy.
"""


def main() -> None:
    manifest: dict[str, Any] = {
        "seed": SEED,
        "base_dataset": "data/dataset",
        "split_sizes": SPLIT_SIZES,
        "job_count_range": [6, 10],
        "operation_count_range": [4, 7],
        "normal_machines": list(NORMAL_MACHINES),
        "training_event_mode": "stochastic_resampled_each_reset",
        "evaluation_event_mode": "fixed_trace",
        "future_events_visible_to_agent": False,
        "unreleased_jobs_visible_to_agent": False,
        "event_semantics": {
            "job_release": "predeclared job becomes dispatchable",
            "machine_breakdown": "pause_repair_resume",
        },
        "scenario_probabilities": SCENARIO_PROBABILITIES,
        "fixed_scenario_counts_per_evaluation_split": FIXED_SCENARIO_COUNTS,
        "event_ranges": {
            "delayed_job_count": list(JOB_RELEASE_COUNT_RANGE),
            "delayed_job_count_probabilities": JOB_RELEASE_COUNT_PROBABILITIES,
            "job_release_time_fraction": list(JOB_RELEASE_TIME_FRACTION),
            "breakdown_count": list(BREAKDOWN_COUNT_RANGE),
            "breakdown_count_probabilities": BREAKDOWN_COUNT_PROBABILITIES,
            "breakdown_time_fraction": list(BREAKDOWN_TIME_FRACTION),
            "repair_duration_factor": list(REPAIR_DURATION_FACTOR),
        },
        "splits": {},
    }
    for split, expected_size in SPLIT_SIZES.items():
        source_directory = SOURCE_ROOT / split
        target_directory = TARGET_ROOT / split
        target_directory.mkdir(parents=True, exist_ok=True)
        source_files = sorted(
            source_directory.glob("instance_*.json"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if len(source_files) != expected_size:
            raise ValueError(
                f"{split} expected {expected_size} source files, got {len(source_files)}"
            )
        written = []
        fixed_scenarios: list[str] = []
        if split != "train":
            for scenario, count in FIXED_SCENARIO_COUNTS.items():
                fixed_scenarios.extend([scenario] * count)
            scenario_rng = random.Random(
                SEED + (10_000 if split == "validation" else 20_000)
            )
            scenario_rng.shuffle(fixed_scenarios)
        for index, source in enumerate(source_files, start=1):
            data = copy.deepcopy(_read_json(source))
            data["name"] = f"dynamic_{split}_{index:02d}"
            if split == "train":
                data["dynamic_events"] = _stochastic_spec(data)
            else:
                split_offset = 10_000 if split == "validation" else 20_000
                data["dynamic_events"] = _fixed_spec(
                    data,
                    scenario_seed=SEED + split_offset + index,
                    scenario=fixed_scenarios[index - 1],
                )
            target = target_directory / source.name
            target.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(f"{split}/{target.name}")
        manifest["splits"][split] = written

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    (TARGET_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (TARGET_ROOT / "README.md").write_text(_readme(), encoding="utf-8")


if __name__ == "__main__":
    main()
