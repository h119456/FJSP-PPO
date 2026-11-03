"""Validate the dataset1-scale dynamic-event dataset contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "dataset_dynamic"
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs import load_instance  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def main() -> None:
    expected_sizes = {"train": 100, "validation": 20, "test": 20}
    totals = {"job_release": 0, "machine_breakdown": 0}
    expected_scenarios = {
        "none": 12,
        "job_release_only": 5,
        "breakdown_only": 2,
        "both": 1,
    }
    for split, expected_size in expected_sizes.items():
        files = sorted((DATASET_ROOT / split).glob("instance_*.json"))
        if len(files) != expected_size:
            raise ValueError(f"{split}: expected {expected_size}, got {len(files)}")
        for path in files:
            raw = _read(path)
            instance = load_instance(path)
            if not 6 <= len(instance.jobs) <= 10:
                raise ValueError(f"{path}: job count out of dataset1 range")
            if any(not 4 <= len(job.operations) <= 7 for job in instance.jobs):
                raise ValueError(f"{path}: operation count out of dataset1 range")
            dynamic = raw.get("dynamic_events")
            if not isinstance(dynamic, dict):
                raise ValueError(f"{path}: missing dynamic_events")
            if dynamic.get("unreleased_jobs_visible_to_agent") is not False:
                raise ValueError(f"{path}: unreleased jobs must be hidden")
            if split == "train":
                if dynamic.get("mode") != "stochastic":
                    raise ValueError(f"{path}: training events must be stochastic")
                if not dynamic.get("resample_each_reset"):
                    raise ValueError(f"{path}: training events must resample")
                if "events" in dynamic:
                    raise ValueError(f"{path}: training must not contain fixed trace")
                if dynamic.get("scenario_probabilities") != {
                    "none": 0.65,
                    "job_release_only": 0.2,
                    "breakdown_only": 0.1,
                    "both": 0.05,
                }:
                    raise ValueError(f"{path}: incorrect training scenario mix")
                continue

            if dynamic.get("mode") != "fixed":
                raise ValueError(f"{path}: evaluation events must be fixed")
            events = dynamic.get("events")
            if not isinstance(events, list):
                raise ValueError(f"{path}: fixed event trace missing")
            if events != sorted(
                events, key=lambda event: (float(event["time"]), str(event["type"]))
            ):
                raise ValueError(f"{path}: events are not time sorted")
            job_ids = {job.job_id for job in instance.jobs}
            for event in events:
                event_type = str(event["type"])
                if float(event["time"]) <= 0:
                    raise ValueError(f"{path}: event time must be positive")
                if event_type == "job_release":
                    if event.get("job_id") not in job_ids:
                        raise ValueError(f"{path}: unknown released job")
                elif event_type == "machine_breakdown":
                    if event.get("machine_id") not in {"M1", "M2", "M5", "M6"}:
                        raise ValueError(f"{path}: invalid breakdown machine")
                    if float(event.get("repair_duration", 0.0)) <= 0:
                        raise ValueError(f"{path}: repair duration must be positive")
                    if event.get("interruption_policy") != "pause_resume":
                        raise ValueError(f"{path}: unsupported interruption policy")
                else:
                    raise ValueError(f"{path}: unknown event type {event_type!r}")
                totals[event_type] += 1
        if split != "train":
            scenario_counts: dict[str, int] = {}
            for path in files:
                scenario = str(_read(path)["dynamic_events"]["scenario"])
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            if scenario_counts != expected_scenarios:
                raise ValueError(
                    f"{split}: scenario counts {scenario_counts}, "
                    f"expected {expected_scenarios}"
                )
    print(
        "dynamic dataset valid: "
        f"train=100 validation=20 test=20 fixed_events={totals}"
    )


if __name__ == "__main__":
    main()
