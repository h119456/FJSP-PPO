"""Archive legacy small datasets and build paired static/dynamic dataset families."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.generate_dataset import generate_dataset as generate_small  # noqa: E402
from scripts.generate_dynamic_dataset import generate_dataset as generate_dynamic, _readme as dynamic_readme  # noqa: E402
from scripts.validate_dynamic_dataset import validate_dataset as validate_dynamic  # noqa: E402

SPLITS = ("train", "validation", "test")
SIZES = {"train": 1000, "validation": 100, "test": 200}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def static_summary(root: Path, expected: dict[str, int]) -> dict:
    from ppo_rs import load_instance

    manifest = read(root / "manifest.json")
    if manifest["split_sizes"] != expected:
        raise ValueError(f"{root}: manifest has unexpected split sizes")
    names, payloads = set(), set()
    job_counts, operation_counts, normal_counts = [], [], []
    preserved = manifest.get("preserved_sha256", {})
    for split, size in expected.items():
        files = sorted((root / split).glob("*.json"))
        relative = {f"{split}/{path.name}" for path in files}
        listed = manifest["splits"][split]
        if len(files) != size or len(listed) != len(set(listed)) or relative != set(listed):
            raise ValueError(f"{root}/{split}: manifest differs from files")
        for path in files:
            raw = read(path)
            instance = load_instance(path)
            if raw.get("dynamic_events"):
                raise ValueError(f"{path}: static instance contains dynamic events")
            if instance.name in names:
                raise ValueError(f"{path}: duplicate instance name")
            names.add(instance.name)
            base = {key: value for key, value in raw.items() if key not in {"name", "dynamic_events"}}
            digest = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
            if digest in payloads:
                raise ValueError(f"{path}: duplicate scheduling payload across or within splits")
            payloads.add(digest)
            key = f"{split}/{path.name}"
            if key in preserved and hashlib.sha256(path.read_bytes()).hexdigest() != preserved[key]:
                raise ValueError(f"{path}: archived bytes have changed")
            job_counts.append(len(instance.jobs))
            operation_counts.extend(len(job.operations) for job in instance.jobs)
            normal = set(instance.machines) - set(instance.stress_relief.machines) - set(instance.compensation.machines)
            normal_counts.append(len(normal))
            if not set(instance.stress_relief.machines) == {"M3"} or not set(instance.compensation.machines) == {"M4"}:
                raise ValueError(f"{path}: unexpected auxiliary resources")
    return {"split_sizes": expected, "unique_instances": len(names),
            "job_count_range": [min(job_counts), max(job_counts)],
            "operations_per_job_range": [min(operation_counts), max(operation_counts)],
            "production_machine_count_range": [min(normal_counts), max(normal_counts)]}


def validate_layout(data: Path) -> dict:
    result = {}
    for family, sizes in (("dataset_test", {"train": 100, "validation": 20, "test": 20}),
                          ("dataset1", SIZES), ("dataset2", SIZES)):
        root = data / family
        result[family] = {"static": static_summary(root / "static", sizes),
                          "dynamic": validate_dynamic(root / "dynamic")}
    # Optional legacy aliases may be removed once configurations are migrated.
    for old, new in (("dataset", "dataset_test/static"), ("dataset_dynamic", "dataset_test/dynamic")):
        path = data / old
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != (data / new).resolve():
                raise ValueError(f"{old}: compatibility link points at a different dataset")
    for part in (*SPLITS, "manifest.json"):
        if (data / "dataset2" / part).resolve() != (data / "dataset2/static" / part).resolve():
            raise ValueError(f"dataset2/{part}: incompatible legacy reference")
    return result


def organize(data: Path) -> dict:
    data = Path(data).resolve()
    if (data / "dataset_test").exists() or (data / "dataset1").exists() or (data / "dataset2/static").exists():
        raise FileExistsError("organized dataset directories already exist; use --check instead")
    legacy = {"static": data / "dataset", "dynamic": data / "dataset_dynamic", "dataset2": data / "dataset2"}
    before = {name: hashes(path) for name, path in legacy.items()}
    if not all(before.values()):
        raise ValueError("one or more original datasets are empty")
    # All expensive generation and schema checks finish before any path changes.
    with tempfile.TemporaryDirectory(prefix=".dataset-layout-", dir=data) as temporary:
        stage = Path(temporary)
        archive = stage / "dataset_test"
        archive.mkdir()
        shutil.copytree(legacy["static"], archive / "static")
        shutil.copytree(legacy["dynamic"], archive / "dynamic")
        small = stage / "dataset1"
        generate_small(small / "static", preserved_root=archive / "static")
        generate_dynamic(small / "static", small / "dynamic", seed=20261103,
                         preserved_root=archive / "dynamic")
        large = stage / "dataset2"
        large.mkdir()
        shutil.copytree(legacy["dataset2"], large / "static")
        generate_dynamic(large / "static", large / "dynamic", seed=20261105)
        # Canonical references survive publication and checkout on another host.
        for family in ("dataset1", "dataset2"):
            path = stage / family / "dynamic/manifest.json"
            manifest = read(path)
            manifest["base_dataset"] = f"data/{family}/static"
            if family == "dataset1":
                manifest["preserved_from"] = "data/dataset_test/dynamic"
            write(path, manifest)
            path = stage / family / "dynamic/README.md"
            path.write_text(dynamic_readme(manifest), encoding="utf-8")
        path = small / "static/manifest.json"
        manifest = read(path)
        manifest["preserved_source"] = "data/dataset_test/static"
        write(path, manifest)
        for family in ("dataset_test", "dataset1", "dataset2"):
            root = stage / family
            sizes = {"train": 100, "validation": 20, "test": 20} if family == "dataset_test" else SIZES
            static_summary(root / "static", sizes)
            validate_dynamic(root / "dynamic")
        if hashes(archive / "static") != before["static"] or hashes(archive / "dynamic") != before["dynamic"]:
            raise ValueError("legacy small archive hash mismatch")
        if hashes(large / "static") != before["dataset2"]:
            raise ValueError("dataset2 static bytes changed")
        # Create compatibility paths inside the staged dataset2 before publication.
        for part in (*SPLITS, "manifest.json"):
            (large / part).symlink_to(Path("static") / part, target_is_directory=part in SPLITS)
        for family in ("dataset_test", "dataset1", "dataset2"):
            sizes = {"train": 100, "validation": 20, "test": 20} if family == "dataset_test" else SIZES
            (stage / family / "README.md").write_text(
                f"# {family}\n\n"
                f"static和dynamic各含{sizes['train']}训练、{sizes['validation']}验证、{sizes['test']}测试实例。\n\n"
                "同一split内按文件名静态／动态配对；dynamic训练事件每次reset重新采样，验证和测试事件固定。\n"
                "详细参数与生成来源见两个子目录的README和manifest。\n",
                encoding="utf-8")
        report = {"date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "split_sizes": {"dataset_test": {"train": 100, "validation": 20, "test": 20},
                                  "dataset1": SIZES, "dataset2": SIZES},
                  "legacy_hashes": before,
                  "compatibility_links": {"dataset2/{train,validation,test,manifest.json}": "dataset2/static/{same}"},
                  "static_dataset2_unchanged": True, "legacy_small_unchanged": True}
        # Renames preserve original directories until the whole layout is published.
        moved = []
        published = []
        try:
            for name, path in legacy.items():
                backup = stage / f"original_{name}"
                path.rename(backup)
                moved.append((path, backup))
            for family in ("dataset_test", "dataset1", "dataset2"):
                (stage / family).rename(data / family)
                published.append(data / family)
        except BaseException:
            for old in (data / "dataset", data / "dataset_dynamic"):
                if old.is_symlink():
                    old.unlink()
            for path in reversed(published):
                path.rename(stage / path.name)
            for path, backup in reversed(moved):
                backup.rename(path)
            raise
        write(data / "dataset-layout-migration.json", report)
    # Publication is complete; perform the same checks through final public paths.
    result = validate_layout(data)
    write(data / "dataset-layout-validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = validate_layout(args.data_root) if args.check else organize(args.data_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
