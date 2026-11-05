"""Guard larger dataset generation and compatibility with existing experiments."""

from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from scripts.generate_dataset2 import SEED as DATASET2_SEED, _instance as dataset2_instance
from scripts.generate_dataset3 import generate_dataset


SPLITS = ("train", "validation", "test")


def _payloads(root: Path, split: str) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted((root / split).glob("*.json"))}


def test_dataset3_generation_is_reproducible_and_refuses_overwrite(tmp_path):
    sizes = {"train": 5, "validation": 2, "test": 2}
    first, repeated, different = (tmp_path / name for name in ("first", "repeated", "different"))
    manifest = generate_dataset(first, seed=41, split_sizes=sizes)
    generate_dataset(repeated, seed=41, split_sizes=sizes)
    generate_dataset(different, seed=42, split_sizes=sizes)
    assert manifest["split_sizes"] == sizes
    for split, size in sizes.items():
        assert len(_payloads(first, split)) == size
        assert _payloads(first, split) == _payloads(repeated, split)
        assert _payloads(first, split) != _payloads(different, split)
        assert set(manifest["splits"][split]) == {
            f"{split}/{name}" for name in _payloads(first, split)
        }
    before = {split: _payloads(first, split) for split in SPLITS}
    with pytest.raises(FileExistsError):
        generate_dataset(first, seed=99, split_sizes=sizes)
    assert before == {split: _payloads(first, split) for split in SPLITS}


def test_larger_training_split_does_not_change_evaluation_instances(tmp_path):
    small, large = tmp_path / "small", tmp_path / "large"
    generate_dataset(small, seed=103, split_sizes={"train": 2, "validation": 3, "test": 3})
    generate_dataset(large, seed=103, split_sizes={"train": 3, "validation": 3, "test": 3})
    for split in ("validation", "test"):
        assert _payloads(small, split) == _payloads(large, split)


def test_requested_scale_and_original_physics_costs_are_preserved(tmp_path):
    root = tmp_path / "static"
    # 91 instances cover all 13 job counts crossed with all 7 machine counts.
    sizes = {"train": 91, "validation": 20, "test": 20}
    manifest = generate_dataset(root, seed=73, split_sizes=sizes)
    assert manifest["job_count_range"] == [18, 30]
    assert manifest["operation_count_range"] == [6, 13]
    assert manifest["normal_machine_count_range"] == [10, 16]
    observed_jobs, observed_machines, observed_ops, combinations = set(), set(), set(), set()
    names, content = set(), set()
    for split in SPLITS:
        for path in (root / split).glob("*.json"):
            raw = json.loads(path.read_text())
            assert raw["name"] not in names
            names.add(raw["name"])
            base = {key: value for key, value in raw.items() if key != "name"}
            canonical = json.dumps(base, sort_keys=True)
            assert canonical not in content
            content.add(canonical)
            assert not raw.get("dynamic_events")
            jobs, machines = len(raw["jobs"]), len(raw["machines"])
            assert 18 <= jobs <= 30
            assert 12 <= machines <= 18  # includes M3 and M4
            normal = set(raw["machines"]) - {"M3", "M4"}
            assert len(normal) == machines - 2
            observed_jobs.add(jobs)
            observed_machines.add(machines)
            if split == "train":
                combinations.add((jobs, machines))
            relief, compensation = raw["stress_relief"], raw["compensation"]
            assert set(relief["machines"]) == {"M3"}
            assert set(compensation["machines"]) == {"M4"}
            assert relief["cost"] in {0.12, 0.24, 0.36, 0.48}
            assert compensation["cost"] in {0.1, 0.2, 0.3, 0.4}
            assert 0.26 <= relief["machines"]["M3"] <= 0.78
            assert 0.6 <= compensation["machines"]["M4"] <= 1.2
            for auxiliary in (relief, compensation):
                assert auxiliary["max_per_job"] == 12
                assert auxiliary["allowed_after_operations"] == [f"O{index}" for index in range(1, 13)]
            assert compensation["trigger_deformation"] == 0.30
            assert compensation["allow_original_machine"] is True
            assert compensation["original_machine_duration_factor"] == 1.0
            assert raw["physics"]["minimum_deformation_ratio"] == 0.5
            assert 0.10 <= raw["physics"]["compensation_deformation_factor"] <= 0.20
            assert raw["objective"] == {
                "makespan_weight": 1.0,
                "relief_cost_weight": 0.5,
                "compensation_cost_weight": 0.5,
                "waiting_cost_weight": 0.0,
                "waiting_unit_cost": 0.0,
            }
            assert raw["environment"] == {
                "max_decisions": 5000,
                "max_time": 1000.0,
                "readiness_scan_step": 0.25,
            }
            for job in raw["jobs"]:
                operations = len(job["operations"])
                assert 6 <= operations <= 13
                observed_ops.add(operations)
                for operation in job["operations"]:
                    assert 2 <= len(operation["machines"]) <= 5
                    assert set(operation["machines"]) <= normal
                    assert all(1.5 <= duration <= 5.0 for duration in operation["machines"].values())
                    assert 0.18 <= operation["deformation_scale"] <= 0.45
                    assert 0.080 <= operation["deformation_rate"] <= 0.250
    assert observed_jobs == set(range(18, 31))
    assert observed_machines == set(range(12, 19))
    assert observed_ops == set(range(6, 14))
    assert len(combinations) == 91


def test_shared_factory_keeps_dataset2_default_payload_unchanged():
    expected = json.loads(Path("data/dataset2/static/train/instance_0001.json").read_text())
    actual = dataset2_instance(random.Random(DATASET2_SEED), "train", 1, 12, 6)
    assert actual == expected
