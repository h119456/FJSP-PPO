from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from collections import Counter

import pytest

from scripts.generate_dataset2 import _instance
from scripts.generate_dynamic_dataset import (
    _fixed_spec, _normal_machines, _reference_horizon,
    fixed_scenario_counts, generate_dataset,
)
from scripts.validate_dynamic_dataset import validate_dataset


def static_source(root: Path, sizes: dict[str, int]) -> None:
    files = {}
    for offset, (split, size) in enumerate(sizes.items()):
        rng = random.Random(101 + offset)
        paths = []
        for number in range(1, size + 1):
            instance = _instance(rng, split, number, 12 + number % 9, 6 + number % 5)
            path = root / split / f"instance_{number:04d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(instance) + "\n")
            paths.append(f"{split}/{path.name}")
        files[split] = paths
    (root / "manifest.json").write_text(json.dumps({
        "split_sizes": sizes, "splits": files,
        "job_count_range": [12, 20], "operation_count_range": [5, 10],
    }))


def test_capacity_and_machine_inventory_adapt_to_static_source(tmp_path):
    source, target = tmp_path / "static", tmp_path / "dynamic"
    sizes = {"train": 3, "validation": 20, "test": 40}
    static_source(source, sizes)
    manifest = generate_dataset(source, target)
    assert manifest["split_sizes"] == sizes
    assert manifest["fixed_scenario_counts_per_evaluation_split"]["test"] == {
        "none": 26, "job_release_only": 8, "breakdown_only": 4, "both": 2,
    }
    for path in (target / "train").glob("*.json"):
        raw = json.loads(path.read_text())
        expected = set(raw["machines"]) - {"M3", "M4"}
        assert set(raw["dynamic_events"]["machine_breakdown"]["machine_pool"]) == expected
        assert len(expected) > 4
        workloads = [sum(min(op["machines"].values()) for op in j["operations"]) for j in raw["jobs"]]
        assert _reference_horizon(raw) == sum(workloads) / len(expected) + max(workloads)
    result = validate_dataset(target)
    assert result["status"] == "ok"
    assert result["unique_instances"] == 63
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_dataset(source, target)


def test_legacy_bytes_are_preserved_and_quotas_completed(tmp_path):
    source, old, target = tmp_path / "static", tmp_path / "old", tmp_path / "dynamic"
    static_source(source, {"train": 2, "validation": 20, "test": 20})
    generate_dataset(source, old)
    # Legacy 20-instance evaluation used 12/5/2/1 rather than 13/4/2/1.
    for split in ("validation", "test"):
        for path in sorted((old / split).glob("*.json")):
            raw = json.loads(path.read_text())
            if raw["dynamic_events"]["scenario"] == "none":
                raw["dynamic_events"] = _fixed_spec(raw, scenario_seed=raw["dynamic_events"]["scenario_seed"], scenario="job_release_only")
                path.write_text(json.dumps(raw, indent=3) + "\n")
                break
    before = {path.relative_to(old).as_posix(): path.read_bytes() for path in old.glob("*/*.json")}
    static_source(source, {"train": 4, "validation": 40, "test": 40})
    manifest = generate_dataset(source, target, preserved_root=old)
    assert len(manifest["preserved_files"]) == 42
    for relative, content in before.items():
        assert (target / relative).read_bytes() == content
        assert manifest["preserved_files"][relative] == hashlib.sha256(content).hexdigest()
    assert validate_dataset(target)["status"] == "ok"
    for split in ("validation", "test"):
        counts = Counter(json.loads(p.read_text())["dynamic_events"]["scenario"] for p in (target / split).glob("*.json"))
        assert dict(counts) == fixed_scenario_counts(40)


def test_pair_mismatch_and_excess_legacy_scenarios_are_rejected(tmp_path):
    source, old = tmp_path / "static", tmp_path / "old"
    static_source(source, {"train": 2, "validation": 20, "test": 20})
    generate_dataset(source, old)
    path = old / "train/instance_0001.json"
    raw = json.loads(path.read_text()); raw["jobs"][0]["operations"][0]["deformation_scale"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not match"):
        generate_dataset(source, tmp_path / "bad_pair", preserved_root=old)
    assert not (tmp_path / "bad_pair").exists()
    # New evaluation data must never silently retain an incompatible quota.
    for path in (old / "validation").glob("*.json"):
        raw = json.loads(path.read_text()); raw["dynamic_events"]["scenario"] = "both";path.write_text(json.dumps(raw))
    path = old / "train/instance_0001.json"; raw = json.loads((source / "train/instance_0001.json").read_text())
    from scripts.generate_dynamic_dataset import _stochastic_spec
    raw["dynamic_events"] = _stochastic_spec(raw);path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exceed target quotas"):
        generate_dataset(source, tmp_path / "bad_quota", preserved_root=old)


def test_validation_detects_hidden_event_exposure(tmp_path):
    source, target = tmp_path / "static", tmp_path / "dynamic"
    static_source(source, {"train": 1, "validation": 20, "test": 20})
    generate_dataset(source, target)
    path = next((target / "validation").glob("*.json"))
    raw = json.loads(path.read_text());raw["dynamic_events"]["future_events_visible_to_agent"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="must be hidden"):
        validate_dataset(target)
