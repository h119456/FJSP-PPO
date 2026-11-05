"""Check dataset3 ranges, pairing, split separation and an optional cost0 variant."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ppo_rs import load_instance  # noqa: E402
from scripts.organize_datasets import static_summary  # noqa: E402
from scripts.validate_dynamic_dataset import validate_dataset as validate_dynamic  # noqa: E402
from scripts.make_zero_cost_dataset import validate_dataset as validate_cost0  # noqa: E402
from scripts.generate_dataset3 import SPLIT_SIZES  # noqa: E402


def validate_dataset(root: Path, *, cost0_root: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    result = {"static": static_summary(root / "static", SPLIT_SIZES),
              "dynamic": validate_dynamic(root / "dynamic")}
    for split in SPLIT_SIZES:
        jobs, machines, operations = Counter(), Counter(), set()
        for path in (root / "static" / split).glob("*.json"):
            raw = json.loads(path.read_text())
            instance = load_instance(path)
            if not 18 <= len(instance.jobs) <= 30 or not 12 <= len(instance.machines) <= 18:
                raise ValueError(f"{path}: job/total-machine count outside dataset3 ranges")
            jobs[len(instance.jobs)] += 1
            machines[len(instance.machines)] += 1
            if instance.compensation.trigger_deformation != 0.30 or instance.physics.minimum_deformation_ratio != 0.5:
                raise ValueError(f"{path}: changed deformation thresholds")
            if not 0.10 <= instance.physics.compensation_deformation_factor <= 0.20:
                raise ValueError(f"{path}: compensation factor outside original range")
            for auxiliary in (instance.stress_relief, instance.compensation):
                if auxiliary.max_per_job != 12 or set(auxiliary.allowed_after_operations) != {f"O{i}" for i in range(1, 13)}:
                    raise ValueError(f"{path}: auxiliary limit does not cover 13 operations")
            if instance.stress_relief.cost not in (0.12, 0.24, 0.36, 0.48) or instance.compensation.cost not in (0.1, 0.2, 0.3, 0.4):
                raise ValueError(f"{path}: auxiliary cost outside original distribution")
            if not 0.26 <= instance.stress_relief.machines["M3"] <= 0.78 or not 0.6 <= instance.compensation.machines["M4"] <= 1.2:
                raise ValueError(f"{path}: changed auxiliary duration range")
            expected_objective = {"makespan_weight": 1.0, "relief_cost_weight": 0.5,
                                  "compensation_cost_weight": 0.5, "waiting_cost_weight": 0.0,
                                  "waiting_unit_cost": 0.0}
            expected_environment = {"max_decisions": 5000, "max_time": 1000.0, "readiness_scan_step": 0.25}
            if raw["objective"] != expected_objective or raw["environment"] != expected_environment:
                raise ValueError(f"{path}: objective/environment differs from dataset2")
            if not instance.compensation.allow_original_machine or instance.compensation.original_machine_duration_factor != 1.0:
                raise ValueError(f"{path}: changed original-machine compensation semantics")
            normal = set(instance.machines) - {"M3", "M4"}
            for job in instance.jobs:
                if not 6 <= len(job.operations) <= 13:
                    raise ValueError(f"{path}: operation count outside dataset3 range")
                operations.add(len(job.operations))
                for operation in job.operations:
                    if not 2 <= len(operation.machines) <= 5 or not set(operation.machines) <= normal:
                        raise ValueError(f"{path}: invalid candidate machine pool")
                    if any(not 1.5 <= duration <= 5.0 for duration in operation.machines.values()):
                        raise ValueError(f"{path}: normal processing duration outside original range")
                    if not 0.18 <= operation.deformation_scale <= 0.45 or not 0.080 <= operation.deformation_rate <= 0.250:
                        raise ValueError(f"{path}: deformation parameters outside original range")
        if set(jobs) != set(range(18, 31)) or max(jobs.values()) - min(jobs.values()) > 1:
            raise ValueError(f"{split}: job count distribution is not balanced")
        if set(machines) != set(range(12, 19)) or max(machines.values()) - min(machines.values()) > 1:
            raise ValueError(f"{split}: total machine distribution is not balanced")
        if operations != set(range(6, 14)):
            raise ValueError(f"{split}: operation ranges are not covered")
    if cost0_root is not None:
        result["cost0"] = validate_cost0(root, Path(cost0_root))
        result["cost0_dynamic"] = validate_dynamic(Path(cost0_root) / "dynamic")
    return {"status": "ok", "dataset_root": str(root), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "dataset3")
    parser.add_argument("--cost0-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_root, cost0_root=args.cost0_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
