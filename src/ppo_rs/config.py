"""Validated configuration models and JSON/YAML loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any


@dataclass(frozen=True)
class OperationSpec:
    operation_id: str
    machines: dict[str, float]
    deformation_scale: float | None = None
    deformation_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.machines or any(value <= 0 for value in self.machines.values()):
            raise ValueError(f"operation {self.operation_id!r} needs positive machine durations")
        if self.deformation_scale is not None and self.deformation_scale <= 0:
            raise ValueError(f"operation {self.operation_id!r} needs positive deformation_scale")
        if self.deformation_rate is not None and self.deformation_rate <= 0:
            raise ValueError(f"operation {self.operation_id!r} needs positive deformation_rate")


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    operations: tuple[OperationSpec, ...]

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError(f"job {self.job_id!r} has no operations")


@dataclass(frozen=True)
class AuxiliarySpec:
    enabled: bool
    machines: dict[str, float]
    cost: float
    max_per_job: int = 1
    allowed_after_operations: tuple[str, ...] = ()
    trigger_deformation: float | None = None
    allow_original_machine: bool = False
    original_machine_duration_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.enabled and not self.machines:
            raise ValueError("enabled auxiliary process needs at least one dedicated machine")
        if any(duration <= 0 for duration in self.machines.values()):
            raise ValueError("auxiliary machine durations must be positive")
        if self.cost < 0 or self.max_per_job < 0:
            raise ValueError("auxiliary cost and max_per_job must be non-negative")
        if self.original_machine_duration_factor <= 0:
            raise ValueError("original_machine_duration_factor must be positive")

    def allowed_after(self, operation_id: str) -> bool:
        return not self.allowed_after_operations or operation_id in self.allowed_after_operations


@dataclass(frozen=True)
class PhysicsSpec:
    default_deformation_scale: float = 1.0
    default_deformation_rate: float = 0.04
    minimum_deformation_ratio: float = 0.5
    compensation_deformation_factor: float = 0.2

    def __post_init__(self) -> None:
        if self.default_deformation_scale <= 0 or self.default_deformation_rate <= 0:
            raise ValueError("default exponential deformation parameters must be positive")
        if not 0 < self.minimum_deformation_ratio < 1:
            raise ValueError("minimum_deformation_ratio must be strictly between 0 and 1")
        if not 0 <= self.compensation_deformation_factor <= 1:
            raise ValueError("compensation_deformation_factor must be in [0, 1]")


@dataclass(frozen=True)
class ObjectiveSpec:
    makespan_weight: float = 1.0
    relief_cost_weight: float = 1.0
    compensation_cost_weight: float = 1.0
    waiting_cost_weight: float = 0.0
    waiting_unit_cost: float = 0.0


@dataclass(frozen=True)
class EnvironmentSpec:
    max_decisions: int = 500
    max_time: float = 10_000.0
    readiness_scan_step: float = 0.25


@dataclass(frozen=True)
class InstanceSpec:
    """A complete FJSP instance plus replaceable synthetic-physics parameters."""

    name: str
    synthetic_test_data: bool
    machines: tuple[str, ...]
    jobs: tuple[JobSpec, ...]
    stress_relief: AuxiliarySpec
    compensation: AuxiliarySpec
    physics: PhysicsSpec = field(default_factory=PhysicsSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)

    def __post_init__(self) -> None:
        machine_set = set(self.machines)
        if not machine_set:
            raise ValueError("at least one machine is required")
        job_ids = [job.job_id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("job IDs must be unique")
        for job in self.jobs:
            operation_ids = [operation.operation_id for operation in job.operations]
            if len(operation_ids) != len(set(operation_ids)):
                raise ValueError(f"operation IDs must be unique within job {job.job_id!r}")
            for operation in job.operations:
                if operation.deformation_scale is None or operation.deformation_rate is None:
                    raise ValueError(
                        f"operation {job.job_id}/{operation.operation_id} must define its own "
                        "deformation_scale and deformation_rate"
                    )
        referenced = set(self.stress_relief.machines) | set(self.compensation.machines)
        referenced |= {
            machine
            for job in self.jobs
            for operation in job.operations
            for machine in operation.machines
        }
        unknown = referenced - machine_set
        if unknown:
            raise ValueError(f"configuration references unknown machines: {sorted(unknown)}")
        if self.stress_relief.trigger_deformation is not None:
            raise ValueError(
                "stress_relief.trigger_deformation is not used in semantics v2; "
                "stress relief is offered while natural dwell remains"
            )
        comp_trigger = self.compensation.trigger_deformation
        if self.compensation.enabled and (
            comp_trigger is None or comp_trigger <= 0
        ):
            raise ValueError(
                "enabled compensation needs one positive trigger_deformation; "
                "deformation above it requires compensation"
            )


def _operation(data: dict[str, Any]) -> OperationSpec:
    if "min_dwell_before" in data:
        raise ValueError(
            "operation.min_dwell_before is obsolete; minimum dwell is calculated from "
            "physics.minimum_deformation_ratio and the completed operation's deformation_rate"
        )
    if "deformation_limit_before" in data or "require_compensation_if_exceeded" in data:
        raise ValueError(
            "per-operation deformation limits are obsolete; use the single mandatory "
            "compensation.trigger_deformation threshold"
        )
    return OperationSpec(
        operation_id=str(data["id"]),
        machines={str(k): float(v) for k, v in data["machines"].items()},
        deformation_scale=(
            None if data.get("deformation_scale") is None else float(data["deformation_scale"])
        ),
        deformation_rate=(
            None if data.get("deformation_rate") is None else float(data["deformation_rate"])
        ),
    )


def _auxiliary(data: dict[str, Any]) -> AuxiliarySpec:
    return AuxiliarySpec(
        enabled=bool(data.get("enabled", True)),
        machines={str(k): float(v) for k, v in data.get("machines", {}).items()},
        cost=float(data.get("cost", 0.0)),
        max_per_job=int(data.get("max_per_job", 1)),
        allowed_after_operations=tuple(str(v) for v in data.get("allowed_after_operations", [])),
        trigger_deformation=(
            None if data.get("trigger_deformation") is None else float(data["trigger_deformation"])
        ),
        allow_original_machine=bool(data.get("allow_original_machine", False)),
        original_machine_duration_factor=float(
            data.get("original_machine_duration_factor", 1.0)
        ),
    )


def instance_from_dict(data: dict[str, Any]) -> InstanceSpec:
    """Build and validate an instance from primitive configuration values."""

    jobs = tuple(
        JobSpec(str(job["id"]), tuple(_operation(op) for op in job["operations"]))
        for job in data["jobs"]
    )
    physics_data = data.get("physics", {})
    if "relief_deformation_factor" in physics_data:
        raise ValueError(
            "physics.relief_deformation_factor belongs to obsolete semantics v1; "
            "stress relief now drives deformation directly to operation saturation A"
        )
    physics = PhysicsSpec(**{k: float(v) for k, v in physics_data.items()})
    objective = ObjectiveSpec(**{k: float(v) for k, v in data.get("objective", {}).items()})
    environment = EnvironmentSpec(**data.get("environment", {}))
    return InstanceSpec(
        name=str(data.get("name", "unnamed")),
        synthetic_test_data=bool(data.get("synthetic_test_data", False)),
        machines=tuple(str(v) for v in data["machines"]),
        jobs=jobs,
        stress_relief=_auxiliary(data.get("stress_relief", {"enabled": False})),
        compensation=_auxiliary(data.get("compensation", {"enabled": False})),
        physics=physics,
        objective=objective,
        environment=environment,
    )


def load_instance(path: str | Path) -> InstanceSpec:
    """Load a JSON or YAML instance. YAML support requires PyYAML."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("YAML input requires PyYAML; install requirements.txt") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported instance format: {source.suffix}")
    if not isinstance(data, dict):
        raise ValueError("instance root must be an object/mapping")
    return instance_from_dict(data)


def resolve_instance_paths(
    source: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    """Resolve one file, one directory, or a list of both into instance files."""

    sources = (source,) if isinstance(source, (str, Path)) else tuple(source)
    paths: list[Path] = []
    for item in sources:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                child
                for child in sorted(path.iterdir())
                if child.is_file() and child.suffix.lower() in {".json", ".yaml", ".yml"}
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"instance source does not exist: {path}")
    if not paths:
        raise ValueError("instance source contains no JSON/YAML files")
    return tuple(paths)


def load_instances(
    source: str | Path | Sequence[str | Path],
) -> tuple[InstanceSpec, ...]:
    """Load a dataset from one file, a directory, or an explicit file list."""

    return tuple(load_instance(path) for path in resolve_instance_paths(source))

