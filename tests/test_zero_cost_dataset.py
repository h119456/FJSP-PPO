"""Zero auxiliary fees must preserve physical transitions and dynamic traces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from ppo_rs import load_instance
from ppo_rs.baselines.policies import relief_rule
from ppo_rs.environment import ResidualStressFJSPEnv
from scripts.make_zero_cost_dataset import generate_dataset, validate_dataset
from tests.helpers import tiny_instance


SPLITS = ("train", "validation", "test")


def _fixture_payload(name: str) -> dict:
    instance = tiny_instance(jobs=2, compensation_threshold=0.4)
    raw = asdict(instance)
    raw["name"] = name
    raw.pop("dynamic_events")
    raw["jobs"] = [
        {"id": job.job_id, "operations": [
            {"id": op.operation_id, "machines": op.machines,
             "deformation_scale": op.deformation_scale,
             "deformation_rate": op.deformation_rate}
            for op in job.operations]}
        for job in instance.jobs
    ]
    return raw


def _source_family(root: Path) -> None:
    for mode in ("static", "dynamic"):
        files = {}
        for index, split in enumerate(SPLITS):
            raw = _fixture_payload(f"tiny_{split}")
            # Independent split payloads prevent accidental duplicate fixtures.
            raw["jobs"][0]["operations"][1]["machines"]["M1"] += index * 0.1
            if mode == "dynamic":
                raw["dynamic_events"] = {
                    "mode": "fixed",
                    "future_events_visible_to_agent": False,
                    "unreleased_jobs_visible_to_agent": False,
                    "events": [
                        {"time": 0.25, "type": "machine_breakdown", "machine_id": "M1",
                         "repair_duration": 0.5, "interruption_policy": "pause_resume",
                         "known_to_agent_before_occurrence": False},
                        {"time": 0.5, "type": "job_release", "job_id": "J2",
                         "known_to_agent_before_occurrence": False},
                    ],
                }
            path = root / mode / split / "instance_01.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=2) + "\n")
            files[split] = [f"{split}/{path.name}"]
        (root / mode / "manifest.json").write_text(json.dumps({
            "split_sizes": {split: 1 for split in SPLITS}, "splits": files,
        }))


def _bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_generation_changes_only_two_costs_and_preserves_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "cost0"
    _source_family(source)
    before = _bytes(source)
    generate_dataset(source, target)
    assert _bytes(source) == before
    for mode in ("static", "dynamic"):
        manifest = json.loads((target / mode / "manifest.json").read_text())
        assert manifest["split_sizes"] == {split: 1 for split in SPLITS}
        for split in SPLITS:
            original = json.loads((source / mode / split / "instance_01.json").read_text())
            expected = deepcopy(original)
            expected["stress_relief"]["cost"] = 0.0
            expected["compensation"]["cost"] = 0.0
            assert json.loads((target / mode / split / "instance_01.json").read_text()) == expected
    validate_dataset(source, target)
    with pytest.raises(FileExistsError):
        generate_dataset(source, target)


@pytest.mark.parametrize("tamper", ["duration", "dynamic_event"])
def test_validator_rejects_changed_physical_time_or_event_trace(tmp_path, tamper):
    source, target = tmp_path / "source", tmp_path / "cost0"
    _source_family(source)
    generate_dataset(source, target)
    path = target / "dynamic/test/instance_01.json"
    raw = json.loads(path.read_text())
    if tamper == "duration":
        raw["stress_relief"]["machines"]["R"] += 0.1
    else:
        raw["dynamic_events"]["events"][0]["repair_duration"] += 0.1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        validate_dataset(source, target)


def test_missing_source_or_incomplete_manifest_cannot_generate(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        generate_dataset(tmp_path / "missing", tmp_path / "output")
    source = tmp_path / "source"
    _source_family(source)
    (source / "dynamic/test/instance_01.json").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        generate_dataset(source, tmp_path / "incomplete")


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_zero_fee_actions_preserve_time_occupation_and_completion(tmp_path, mode):
    source, target = tmp_path / "source", tmp_path / "cost0"
    _source_family(source)
    generate_dataset(source, target)
    envs = [ResidualStressFJSPEnv(load_instance(root / mode / "test/instance_01.json"))
            for root in (source, target)]
    for env in envs:
        env.reset(seed=7)
    while not (envs[0]._terminated or envs[0]._truncated):
        original, zero = (env.get_valid_actions() for env in envs)
        # Cost may affect observations and rewards, but not legal physical moves.
        assert [replace(action, cost=0.0) for action in original] == zero
        choice = relief_rule(original)
        for env in envs:
            env.step(choice)
        assert envs[0].current_time == envs[1].current_time
        assert envs[0]._terminated == envs[1]._terminated
        assert envs[0]._truncated == envs[1]._truncated
    original_metrics, zero_metrics = (env.get_metrics() for env in envs)
    assert original_metrics["terminated"] and not original_metrics["truncated"]
    assert original_metrics["stress_relief_count"] > 0
    assert original_metrics["compensation_count"] > 0
    assert original_metrics["schedule_violation_count"] == zero_metrics["schedule_violation_count"] == 0
    assert original_metrics["total_independent_cost"] > 0
    assert zero_metrics["total_independent_cost"] == 0.0
    assert zero_metrics["objective"] == pytest.approx(zero_metrics["makespan"])
    assert original_metrics["makespan"] == zero_metrics["makespan"]
    for env in envs:
        for entry in env.get_schedule():
            if entry["kind"] in {"STRESS_RELIEF", "COMPENSATION"}:
                assert entry["end"] > entry["start"]
    assert [{key: value for key, value in entry.items() if key != "independent_cost"}
            for entry in envs[0].get_schedule()] == [
        {key: value for key, value in entry.items() if key != "independent_cost"}
        for entry in envs[1].get_schedule()
    ]
    if mode == "dynamic":
        assert original_metrics["job_release_event_count"] == zero_metrics["job_release_event_count"] == 1
        assert original_metrics["machine_breakdown_event_count"] == zero_metrics["machine_breakdown_event_count"] == 1
