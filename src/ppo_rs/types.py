"""Shared immutable scheduling types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Kinds of decisions exposed by the environment."""

    NORMAL = "NORMAL"
    STRESS_RELIEF = "STRESS_RELIEF"
    COMPENSATION = "COMPENSATION"
    WAIT = "WAIT"


@dataclass(frozen=True)
class Action:
    """One currently legal, fully specified scheduling decision."""

    kind: ActionType
    job_id: str | None = None
    operation_id: str | None = None
    machine_id: str | None = None
    duration: float = 0.0
    cost: float = 0.0


@dataclass
class ScheduleEntry:
    """A traceable non-preemptive machine occupation interval."""

    kind: ActionType
    job_id: str
    operation_id: str
    machine_id: str
    start: float
    end: float
    independent_cost: float = 0.0
    interruption_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

