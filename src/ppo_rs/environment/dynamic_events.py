"""Materialize stochastic or fixed external-event traces from an instance."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ppo_rs.config import InstanceSpec


@dataclass(frozen=True)
class DynamicEvent:
    time: float
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "type": self.event_type, **self.payload}


def _fixed_events(config: dict[str, Any]) -> tuple[DynamicEvent, ...]:
    events = []
    for row in config.get("events", []):
        payload = {
            str(key): value
            for key, value in row.items()
            if key not in {"time", "type"}
        }
        events.append(
            DynamicEvent(float(row["time"]), str(row["type"]), payload)
        )
    return tuple(sorted(events, key=lambda event: (event.time, event.event_type)))


def _stochastic_events(
    instance: InstanceSpec,
    config: dict[str, Any],
    seed: int,
) -> tuple[DynamicEvent, ...]:
    rng = random.Random(seed)
    horizon = float(config["reference_horizon"])
    events: list[DynamicEvent] = []

    scenario_probabilities = dict(
        config.get(
            "scenario_probabilities",
            {"both": 1.0},
        )
    )
    scenarios = tuple(str(name) for name in scenario_probabilities)
    weights = tuple(float(scenario_probabilities[name]) for name in scenarios)
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("dynamic scenario probabilities must be non-negative and non-zero")
    scenario = rng.choices(scenarios, weights=weights, k=1)[0]
    if scenario not in {"none", "job_release_only", "breakdown_only", "both"}:
        raise ValueError(f"unknown stochastic dynamic scenario {scenario!r}")

    def sample_count(spec: dict[str, Any], range_key: str) -> int:
        probabilities = spec.get("count_probabilities")
        if probabilities:
            values = tuple(int(value) for value in probabilities)
            count_weights = tuple(float(probabilities[str(value)]) for value in values)
            return rng.choices(values, weights=count_weights, k=1)[0]
        low, high = (int(value) for value in spec[range_key])
        return rng.randint(low, high)

    release = dict(config.get("job_release", {}))
    if release.get("enabled", False) and scenario in {"job_release_only", "both"}:
        count = min(
            sample_count(release, "delayed_job_count_range"),
            len(instance.jobs) - 1,
        )
        time_low, time_high = (
            float(value)
            for value in release["release_time_fraction_of_reference_horizon"]
        )
        for job in rng.sample(list(instance.jobs), count):
            events.append(
                DynamicEvent(
                    round(rng.uniform(time_low, time_high) * horizon, 3),
                    "job_release",
                    {
                        "job_id": job.job_id,
                        "known_to_agent_before_occurrence": False,
                    },
                )
            )

    breakdown = dict(config.get("machine_breakdown", {}))
    if breakdown.get("enabled", False) and scenario in {"breakdown_only", "both"}:
        machine_pool = [str(value) for value in breakdown["machine_pool"]]
        count = min(
            sample_count(breakdown, "event_count_range"),
            len(machine_pool),
        )
        time_low, time_high = (
            float(value)
            for value in breakdown[
                "breakdown_time_fraction_of_reference_horizon"
            ]
        )
        repair_low, repair_high = (
            float(value)
            for value in breakdown[
                "repair_duration_factor_of_mean_processing_time"
            ]
        )
        durations = [
            duration
            for job in instance.jobs
            for operation in job.operations
            for duration in operation.machines.values()
        ]
        mean_duration = sum(durations) / len(durations)
        for machine_id in rng.sample(machine_pool, count):
            events.append(
                DynamicEvent(
                    round(rng.uniform(time_low, time_high) * horizon, 3),
                    "machine_breakdown",
                    {
                        "machine_id": machine_id,
                        "repair_duration": round(
                            rng.uniform(repair_low, repair_high) * mean_duration,
                            3,
                        ),
                        "interruption_policy": str(
                            breakdown.get(
                                "interruption_policy", "pause_resume"
                            )
                        ),
                        "known_to_agent_before_occurrence": False,
                    },
                )
            )
    return tuple(sorted(events, key=lambda event: (event.time, event.event_type)))


def materialize_dynamic_events(
    instance: InstanceSpec, *, seed: int
) -> tuple[DynamicEvent, ...]:
    """Return an episode trace; stochastic configs use seed, fixed traces ignore it."""

    config = instance.dynamic_events
    if not config:
        return ()
    mode = str(config.get("mode", "none"))
    if mode == "fixed":
        return _fixed_events(config)
    if mode == "stochastic":
        return _stochastic_events(instance, config, seed)
    if mode == "none":
        return ()
    raise ValueError(f"unknown dynamic event mode {mode!r}")
