"""Create paired static/dynamic cost0 datasets without changing resource durations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
MODES = ("static", "dynamic")
CHANGES = {"stress_relief.cost": 0.0, "compensation.cost": 0.0}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _inventory(root: Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    manifest = _read(root / "manifest.json")
    sizes = manifest["split_sizes"]
    if set(sizes) != set(SPLITS) or any(not isinstance(n, int) or n <= 0 for n in sizes.values()):
        raise ValueError(f"{root}: expected three positive split sizes")
    files = {}
    for split in SPLITS:
        paths = sorted((root / split).glob("*.json"))
        expected = {f"{split}/{path.name}" for path in paths}
        listed = manifest["splits"][split]
        if len(paths) != sizes[split] or len(listed) != len(set(listed)) or set(listed) != expected:
            raise ValueError(f"{root}/{split}: manifest file set or split size differs from files")
        files[split] = paths
    return manifest, files


def _zero_cost(raw: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    result["stress_relief"]["cost"] = 0.0
    result["compensation"]["cost"] = 0.0
    return result


def validate_dataset(source_root: Path, target_root: Path) -> dict[str, Any]:
    """Compare every derived instance; only the two auxiliary cost fields may change."""
    source_root, target_root = Path(source_root), Path(target_root)
    result = {}
    for mode in MODES:
        source, target = source_root / mode, target_root / mode
        original, source_files = _inventory(source)
        derived, target_files = _inventory(target)
        if original["split_sizes"] != derived["split_sizes"]:
            raise ValueError(f"{mode}: source and cost0 split sizes differ")
        if derived.get("changes") != CHANGES or derived.get("derived_from") != _display(source):
            raise ValueError(f"{mode}: incorrect cost0 provenance")
        if derived.get("source_manifest_sha256") != hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest():
            raise ValueError(f"{source}: source manifest changed since cost0 generation")
        names = set()
        source_hashes = derived.get("source_instance_sha256", {})
        for split in SPLITS:
            if {p.name for p in source_files[split]} != {p.name for p in target_files[split]}:
                raise ValueError(f"{mode}/{split}: source and cost0 filenames differ")
            for path in source_files[split]:
                relative = f"{split}/{path.name}"
                expected = _zero_cost(_read(path))
                actual = _read(target / relative)
                if actual != expected:
                    raise ValueError(f"{target / relative}: fields other than auxiliary costs changed")
                if actual["name"] in names:
                    raise ValueError(f"{target / relative}: repeated instance name")
                names.add(actual["name"])
                if source_hashes.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"{path}: source hash differs from cost0 provenance")
        result[mode] = {"split_sizes": original["split_sizes"], "instances": len(names),
                        "only_auxiliary_costs_changed": True}
    return {"status": "ok", "source": _display(source_root), "target": _display(target_root),
            "modes": result}


def generate_dataset(source_root: Path, target_root: Path) -> dict[str, Any]:
    source_root, target_root = Path(source_root).resolve(), Path(target_root).resolve()
    if target_root == source_root or source_root in target_root.parents or target_root in source_root.parents:
        raise ValueError("source and target must be separate dataset family directories")
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty dataset: {target_root}")
    sources = {mode: _inventory(source_root / mode) for mode in MODES}
    if sources["static"][0]["split_sizes"] != sources["dynamic"][0]["split_sizes"]:
        raise ValueError("source static and dynamic split sizes differ")
    # Check pairing before creating files, including fixed dynamic traces.
    for split in SPLITS:
        static_paths = {p.name: p for p in sources["static"][1][split]}
        dynamic_paths = {p.name: p for p in sources["dynamic"][1][split]}
        if static_paths.keys() != dynamic_paths.keys():
            raise ValueError(f"{split}: source static/dynamic filenames differ")
        for name, path in static_paths.items():
            static, dynamic = _read(path), _read(dynamic_paths[name])
            base = lambda raw: {k: v for k, v in raw.items() if k not in {"name", "dynamic_events"}}
            if base(static) != base(dynamic):
                raise ValueError(f"{split}/{name}: source static/dynamic payloads differ")
    before = _hashes(source_root)
    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target_root.name}-", dir=target_root.parent) as temporary:
        stage = Path(temporary) / target_root.name
        stage.mkdir()
        for mode, (source_manifest, files) in sources.items():
            source, target = source_root / mode, stage / mode
            target.mkdir()
            manifest = copy.deepcopy(source_manifest)
            # Old preserved-byte assertions concern the original costed data.
            for key in ("preserved_sha256", "preserved_files", "preserved_source", "preserved_from", "preserved_counts"):
                manifest.pop(key, None)
            manifest.update({
                "variant": "cost0", "derived_from": _display(source), "changes": CHANGES,
                "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                "source_instance_sha256": {},
                "unchanged": ["instance names and split membership", "all processing durations and resource occupancy",
                              "physics and thresholds", "objective weights", "dynamic event specs and traces"],
            })
            if "parameter_ranges" in manifest:
                manifest["parameter_ranges"]["stress_relief_cost"] = [0.0, 0.0]
                manifest["parameter_ranges"]["compensation_cost"] = [0.0, 0.0]
            if mode == "dynamic":
                manifest["base_dataset"] = _display(target_root / "static")
            for split, paths in files.items():
                (target / split).mkdir()
                for path in paths:
                    relative = f"{split}/{path.name}"
                    _write(target / relative, _zero_cost(_read(path)))
                    manifest["source_instance_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            _write(target / "manifest.json", manifest)
            (target / "README.md").write_text(
                f"# {target_root.name} / {mode}\n\n"
                f"来源：`{_display(source)}`，容量为{manifest['split_sizes']}。\n\n"
                "仅stress_relief.cost与compensation.cost置0.0。名称、工序、设备占用时长、"
                "物理参数、目标权重和动态事件配置逐字段保持不变。\n"
                "原机补偿和专用机补偿费用均为0；两者仍占用原设备和原时长。\n",
                encoding="utf-8")
        validate_dataset(source_root, stage)
        if _hashes(source_root) != before:
            raise RuntimeError("source dataset changed during generation")
        sizes = sources["static"][0]["split_sizes"]
        (stage / "README.md").write_text(
            f"# {target_root.name}\n\n"
            f"来源：`{_display(source_root)}`；static和dynamic各含{sizes['train']}训练、"
            f"{sizes['validation']}验证、{sizes['test']}测试实例。\n\n"
            "只把去应力和补偿单次额外费用设为0，时间占用不变。目标权重不变；"
            "当前等待费用为0，因此正常完成排程的目标等于makespan。\n\n"
            "本数据与有成本版本逐实例配对，可用于成本消融；不得交叉混入训练和测试划分。\n",
            encoding="utf-8")
        if target_root.exists():
            target_root.rmdir()  # only an empty user-provided destination is permitted
        stage.rename(target_root)
    result = validate_dataset(source_root, target_root)
    _write(target_root / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="source family containing static and dynamic")
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--check", action="store_true", help="validate existing derived data")
    args = parser.parse_args()
    if bool(args.source_root) != bool(args.target_root):
        parser.error("--source-root and --target-root must be supplied together")
    pairs = [(args.source_root, args.target_root)] if args.source_root else [
        (ROOT / "data" / family, ROOT / "data" / f"{family}_cost0")
        for family in ("dataset1", "dataset2")]
    operation = validate_dataset if args.check else generate_dataset
    results = [operation(source, target) for source, target in pairs]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
