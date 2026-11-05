"""Validate paired dynamic data, event visibility, split manifests and traces."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "dataset1" / "dynamic"
sys.path.insert(0, str(ROOT / "src"))

from ppo_rs import load_instance  # noqa: E402
from ppo_rs.environment import materialize_dynamic_events  # noqa: E402

SCENARIO_PROBABILITIES = {"none": 0.65, "job_release_only": 0.20,
                          "breakdown_only": 0.10, "both": 0.05}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _base(instance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if key not in {"name", "dynamic_events"}}


def _validate_events(path: Path, instance: Any, events: list[dict[str, Any]]) -> None:
    if events != sorted(events, key=lambda event: (float(event["time"]), str(event["type"]))):
        raise ValueError(f"{path}: events are not time sorted")
    auxiliary = set(instance.stress_relief.machines) | set(instance.compensation.machines)
    normal = set(instance.machines) - auxiliary
    job_ids = {job.job_id for job in instance.jobs}
    releases: set[str] = set()
    repairs: dict[str, float] = {}
    for event in events:
        event_type = str(event["type"])
        time = float(event["time"])
        if not math.isfinite(time) or time <= 0 or time >= instance.environment.max_time:
            raise ValueError(f"{path}: event time outside episode horizon")
        if event.get("known_to_agent_before_occurrence") is not False:
            raise ValueError(f"{path}: future event must remain hidden")
        if event_type == "job_release":
            job_id = str(event.get("job_id"))
            if job_id not in job_ids or job_id in releases:
                raise ValueError(f"{path}: invalid or repeated released job")
            releases.add(job_id)
        elif event_type == "machine_breakdown":
            machine = str(event.get("machine_id"))
            duration = float(event.get("repair_duration", 0.0))
            if machine not in normal:
                raise ValueError(f"{path}: invalid breakdown machine")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{path}: repair duration must be finite and positive")
            if time < repairs.get(machine, 0):
                raise ValueError(f"{path}: overlapping failures on one machine")
            repairs[machine] = time + duration
            if event.get("interruption_policy") != "pause_resume":
                raise ValueError(f"{path}: unsupported interruption policy")
        else:
            raise ValueError(f"{path}: unknown event type {event_type!r}")
    if len(releases) >= len(job_ids):
        raise ValueError(f"{path}: at least one job must be initially released")


def validate_dataset(root: Path) -> dict[str, Any]:
    root = Path(root)
    manifest = _read(root / "manifest.json")
    sizes = {split: int(manifest["split_sizes"][split]) for split in ("train", "validation", "test")}
    # Paired static data normally lives beside dynamic, including legacy archives.
    source = root.parent / "static"
    if not (source / "manifest.json").is_file():
        source = Path(manifest["base_dataset"])
        if not source.is_absolute():
            source = ROOT / source
    source_manifest = _read(source / "manifest.json")
    if source_manifest["split_sizes"] != sizes:
        raise ValueError("static and dynamic split sizes differ")
    desired = manifest["fixed_scenario_counts_per_evaluation_split"]
    scenario_counts = {}
    totals: Counter[str] = Counter({"job_release": 0, "machine_breakdown": 0})
    names: set[str] = set()
    content_splits: dict[str, str] = {}
    sample_traces: set[str] = set()
    ranges = {key: manifest.get(key, source_manifest.get(key))
              for key in ("job_count_range", "operation_count_range")}
    for split, size in sizes.items():
        files = sorted((root / split).glob("*.json"))
        if size <= 0 or len(files) != size:
            raise ValueError(f"{split}: expected {size}, got {len(files)}")
        actual = {f"{split}/{path.name}" for path in files}
        listed = manifest["splits"][split]
        if len(listed) != len(set(listed)) or set(listed) != actual:
            raise ValueError(f"{split}: manifest file set differs from directory")
        if actual != set(source_manifest["splits"][split]):
            raise ValueError(f"{split}: static and dynamic filenames differ")
        scenarios: Counter[str] = Counter({key: 0 for key in SCENARIO_PROBABILITIES})
        fixed_seeds: set[int] = set()
        for index, path in enumerate(files):
            raw = _read(path)
            instance = load_instance(path)
            static = _read(source / split / path.name)
            if _base(raw) != _base(static):
                raise ValueError(f"{path}: dynamic data differs from its paired static instance")
            if instance.name in names:
                raise ValueError(f"{path}: repeated instance name")
            names.add(instance.name)
            canonical = json.dumps(_base(raw), sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if digest in content_splits:
                raise ValueError(f"{path}: duplicate static payload in {content_splits[digest]}")
            content_splits[digest] = split
            if ranges["job_count_range"] and not ranges["job_count_range"][0] <= len(instance.jobs) <= ranges["job_count_range"][1]:
                raise ValueError(f"{path}: job count out of manifest range")
            if ranges["operation_count_range"] and any(
                not ranges["operation_count_range"][0] <= len(job.operations) <= ranges["operation_count_range"][1]
                for job in instance.jobs
            ):
                raise ValueError(f"{path}: operation count out of manifest range")
            dynamic = raw.get("dynamic_events")
            if not isinstance(dynamic, dict):
                raise ValueError(f"{path}: missing dynamic_events")
            if dynamic.get("unreleased_jobs_visible_to_agent") is not False or dynamic.get("future_events_visible_to_agent") is not False:
                raise ValueError(f"{path}: future events and unreleased jobs must be hidden")
            preserved_digest = manifest.get("preserved_files", {}).get(f"{split}/{path.name}")
            if preserved_digest and hashlib.sha256(path.read_bytes()).hexdigest() != preserved_digest:
                raise ValueError(f"{path}: preserved bytes have changed")
            if split == "train":
                if dynamic.get("mode") != "stochastic" or not dynamic.get("resample_each_reset") or "events" in dynamic:
                    raise ValueError(f"{path}: training events must resample stochastically")
                if dynamic.get("scenario_probabilities") != SCENARIO_PROBABILITIES:
                    raise ValueError(f"{path}: incorrect training scenario mix")
                normal = set(instance.machines) - set(instance.stress_relief.machines) - set(instance.compensation.machines)
                if set(dynamic["machine_breakdown"]["machine_pool"]) != normal:
                    raise ValueError(f"{path}: breakdown pool differs from public normal machines")
                if dynamic["machine_breakdown"].get("interruption_policy") != "pause_resume":
                    raise ValueError(f"{path}: unsupported stochastic interruption policy")
                # Check all source specs with three seeds, and explicitly prove
                # repeatability plus nonconstant traces on the first instance.
                seeds = range(40) if index == 0 else (0, 7, 19)
                for seed in seeds:
                    events = materialize_dynamic_events(instance, seed=seed)
                    _validate_events(path, instance, [event.to_dict() for event in events])
                    if index == 0:
                        sample_traces.add(repr(events))
                if materialize_dynamic_events(instance, seed=7) != materialize_dynamic_events(instance, seed=7):
                    raise ValueError(f"{path}: reset seed does not reproduce trace")
                continue
            if dynamic.get("mode") != "fixed" or not isinstance(dynamic.get("events"), list):
                raise ValueError(f"{path}: evaluation events must be fixed")
            scenario = str(dynamic.get("scenario"))
            if scenario not in SCENARIO_PROBABILITIES:
                raise ValueError(f"{path}: unknown scenario")
            scenarios[scenario] += 1
            scenario_seed = int(dynamic["scenario_seed"])
            if scenario_seed in fixed_seeds:
                raise ValueError(f"{path}: repeated scenario seed")
            fixed_seeds.add(scenario_seed)
            events = dynamic["events"]
            _validate_events(path, instance, events)
            types = {event["type"] for event in events}
            expected_types = {
                "none": set(), "job_release_only": {"job_release"},
                "breakdown_only": {"machine_breakdown"},
                "both": {"job_release", "machine_breakdown"},
            }[scenario]
            if types != expected_types:
                raise ValueError(f"{path}: event types disagree with scenario")
            if materialize_dynamic_events(instance, seed=7) != materialize_dynamic_events(instance, seed=999):
                raise ValueError(f"{path}: fixed trace varies with reset seed")
            totals.update(event["type"] for event in events)
        if split != "train":
            expected = desired.get(split, desired)
            if dict(scenarios) != expected:
                raise ValueError(f"{split}: scenario counts {dict(scenarios)}, expected {expected}")
            scenario_counts[split] = dict(scenarios)
    if len(sample_traces) < 2:
        raise ValueError("training reset seeds never change event trace")
    return {"status": "ok", "dataset_root": str(root), "split_sizes": sizes,
            "fixed_scenarios": scenario_counts, "fixed_events": dict(totals),
            "unique_instances": len(names), "paired_static": str(source)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
