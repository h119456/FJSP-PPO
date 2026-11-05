"""Generate dataset3 static instances using dataset2's process and cost rules."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_dataset2 import _instance  # noqa: E402

DATASET_ROOT = ROOT / "data" / "dataset3" / "static"
SEED = 20261106
SPLIT_SIZES = {"train": 1000, "validation": 100, "test": 200}
MIN_JOBS, MAX_JOBS = 18, 30
MIN_OPERATIONS, MAX_OPERATIONS = 6, 13
MIN_NORMAL_MACHINES, MAX_NORMAL_MACHINES = 10, 16
NORMAL_MACHINE_POOL = ("M1", "M2", *(f"M{number}" for number in range(5, 19)))


def generate_dataset(
    target_root: Path = DATASET_ROOT,
    *,
    seed: int = SEED,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Balance job/machine marginals, with RNG independent across splits."""
    target_root = Path(target_root).resolve()
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty dataset: {target_root}")
    sizes = dict(SPLIT_SIZES if split_sizes is None else split_sizes)
    if set(sizes) != set(SPLIT_SIZES) or any(not isinstance(n, int) or n <= 0 for n in sizes.values()):
        raise ValueError("split_sizes must contain three positive integer counts")
    # The shared template contains every process, physics and objective setting.
    template = _instance(random.Random(seed), "train", 1, MIN_JOBS, MIN_NORMAL_MACHINES,
                         operation_count_range=(MIN_OPERATIONS, MAX_OPERATIONS),
                         normal_machine_pool=NORMAL_MACHINE_POOL, name_prefix="dataset3")
    manifest: dict[str, Any] = {
        "seed": seed, "split_sizes": sizes,
        "job_count_range": [MIN_JOBS, MAX_JOBS],
        "operation_count_range": [MIN_OPERATIONS, MAX_OPERATIONS],
        "total_machine_count_range": [12, 18],
        "normal_machine_count_range": [MIN_NORMAL_MACHINES, MAX_NORMAL_MACHINES],
        "candidate_machine_count_range": [2, 5],
        "normal_machine_pool": list(NORMAL_MACHINE_POOL),
        "stress_relief_machine": "M3", "compensation_machine": "M4",
        "parameter_ranges": {
            "normal_duration": [1.5, 5.0], "deformation_scale": [0.18, 0.45],
            "deformation_rate": [0.080, 0.250], "stress_relief_duration": [0.26, 0.78],
            "stress_relief_cost": [0.12, 0.48], "compensation_duration": [0.6, 1.2],
            "compensation_cost": [0.1, 0.4], "compensation_factor": [0.10, 0.20],
            "mandatory_compensation_threshold": 0.30,
        },
        "objective": copy.deepcopy(template["objective"]),
        "environment": copy.deepcopy(template["environment"]),
        "generation": "balanced_job_and_machine_marginals_independent_split_rng",
        "machine_count_semantics": "12-18 total including M3 and M4; 10-16 normal",
        "split_seeds": {}, "observed_counts": {}, "splits": {},
    }
    target_root.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tempfile.TemporaryDirectory(prefix=".dataset3-static-", dir=target_root.parent) as temporary:
        stage = Path(temporary) / "static"
        stage.mkdir()
        for offset, (split, size) in enumerate(sizes.items(), start=1):
            split_seed = seed + offset * 1_000_003
            manifest["split_seeds"][split] = split_seed
            rng = random.Random(split_seed)
            directory = stage / split
            directory.mkdir()
            job_counts, machine_counts = Counter(), Counter()
            paths = []
            for number in range(1, size + 1):
                job_count = MIN_JOBS + (number - 1) % (MAX_JOBS - MIN_JOBS + 1)
                normal_count = MIN_NORMAL_MACHINES + (number - 1) % (MAX_NORMAL_MACHINES - MIN_NORMAL_MACHINES + 1)
                data = _instance(rng, split, number, job_count, normal_count,
                                 operation_count_range=(MIN_OPERATIONS, MAX_OPERATIONS),
                                 normal_machine_pool=NORMAL_MACHINE_POOL, name_prefix="dataset3")
                payload = {k: v for k, v in data.items() if k != "name"}
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if digest in seen:
                    raise ValueError("duplicate static scheduling instance")
                seen.add(digest)
                filename = f"instance_{number:04d}.json"
                (directory / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                paths.append(f"{split}/{filename}")
                job_counts[job_count] += 1
                machine_counts[normal_count + 2] += 1
            manifest["splits"][split] = paths
            manifest["observed_counts"][split] = {"jobs": dict(sorted(job_counts.items())),
                                                   "total_machines": dict(sorted(machine_counts.items()))}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        (stage / "README.md").write_text(
            "# dataset3 静态数据\n\n"
            f"容量：{sizes['train']}训练、{sizes['validation']}验证、{sizes['test']}测试。\n\n"
            "18～30个工件，每件6～13道普通工序，总设备12～18台（10～16台普通设备＋M3／M4各1台）。"
            "每工序2～5台候选机器，其余加工、变形、辅助费用和目标权重沿用dataset2。\n"
            "去应力／补偿每件最多12次，允许O1～O12之后执行，以覆盖13工序的全部工序间隙。"
            "保留max_decisions=5000、max_time=1000及readiness_scan_step=0.25。\n\n"
            "按工件数和机器数边际均衡生成；各split使用独立随机种子，文件名与dynamic配对。\n",
            encoding="utf-8")
        if target_root.exists():
            target_root.rmdir()
        stage.rename(target_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = generate_dataset(args.target_root, seed=args.seed)
    print(json.dumps({"root": str(args.target_root), "split_sizes": manifest["split_sizes"]}))


if __name__ == "__main__":
    main()
