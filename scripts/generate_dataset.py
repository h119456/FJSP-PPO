"""Expand the small-scale static dataset while preserving archived instances."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "dataset1" / "static"
ARCHIVE_ROOT = ROOT / "data" / "dataset_test" / "static"
SPLIT_SIZES = {"train": 1000, "validation": 100, "test": 200}
EXPANSION_SEED = 20261105
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


def generate_dataset(
    target_root: Path = DATASET_ROOT,
    *,
    preserved_root: Path = ARCHIVE_ROOT,
    seed: int = EXPANSION_SEED,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Preserve each archived split byte-for-byte, then append independent samples."""
    sizes = dict(SPLIT_SIZES if split_sizes is None else split_sizes)
    if set(sizes) != {"train", "validation", "test"} or any(n <= 0 for n in sizes.values()):
        raise ValueError("split_sizes must contain three positive split counts")
    target_root, preserved_root = Path(target_root), Path(preserved_root)
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty dataset: {target_root}")
    manifest = copy.deepcopy(json.loads((preserved_root / "manifest.json").read_text()))
    manifest.update({
        "seed": seed,
        "split_sizes": sizes,
        "generation": "preserve_each_split_then_append_with_independent_split_seeds",
        "preserved_source": str(preserved_root),
        "preserved_seed": manifest.get("seed"),
        "preserved_counts": {},
        "split_seeds": {},
        "preserved_sha256": {},
        "splits": {},
    })
    seen: set[str] = set()
    for split_index, (split, size) in enumerate(sizes.items(), start=1):
        paths = sorted((preserved_root / split).glob("instance_*.json"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if len(paths) > size:
            raise ValueError(f"{split}: cannot preserve {len(paths)} files in {size} slots")
        directory = target_root / split
        directory.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, directory / path.name)
            manifest["preserved_sha256"][f"{split}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["preserved_counts"][split] = len(paths)
        rng = random.Random(seed + split_index * 1_000_003)
        manifest["split_seeds"][split] = seed + split_index * 1_000_003
        # Preserve split membership: growing train must never shift old eval samples.
        for number in range(len(paths) + 1, size + 1):
            job_count = MIN_JOBS + (number - 1) % (MAX_JOBS - MIN_JOBS + 1)
            data = _instance(rng, split, number, job_count)
            data["name"] = f"dataset1_{split}_{number:04d}"
            path = directory / f"instance_{number:02d}.json"
            if path.exists():
                raise ValueError(f"duplicate source sequence number: {path}")
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files = sorted(directory.glob("instance_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        manifest["splits"][split] = [f"{split}/{p.name}" for p in files]
        for path in files:
            raw = json.loads(path.read_text())
            raw.pop("name", None)
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            if digest in seen:
                raise ValueError(f"duplicate scheduling instance: {path}")
            seen.add(digest)
    (target_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (target_root / "README.md").write_text(
        "# dataset1 静态数据\n\n"
        f"划分：train={sizes['train']}、validation={sizes['validation']}、test={sizes['test']}。\n\n"
        "每实例6～10个工件、每件4～7道普通工序、4台生产设备，另有应力消除机M3和补偿机M4各1台。\n"
        "原dataset_test/static实例按原split逐文件保留，新增实例使用独立split随机种子，工艺和成本分布不变。\n"
        "静态和dynamic目录按相同文件名配对；两者是同一基础实例的场景变体，不应跨训练／测试划分混用。\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--preserved-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--seed", type=int, default=EXPANSION_SEED)
    args = parser.parse_args()
    manifest = generate_dataset(args.target_root, preserved_root=args.preserved_root, seed=args.seed)
    print(json.dumps({"root": str(args.target_root), "split_sizes": manifest["split_sizes"]}))


if __name__ == "__main__":
    main()
