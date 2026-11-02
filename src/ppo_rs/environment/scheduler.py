"""Shared non-preemptive machine resource ledger."""

from __future__ import annotations

from dataclasses import dataclass, field

from ppo_rs.types import ScheduleEntry


@dataclass
class MachineLedger:
    """Tracks machine availability and auditable occupation intervals."""

    machine_ids: tuple[str, ...]
    available_at: dict[str, float] = field(init=False)
    entries: list[ScheduleEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.available_at = {machine_id: 0.0 for machine_id in self.machine_ids}

    def is_available(self, machine_id: str, at: float) -> bool:
        if machine_id not in self.available_at:
            raise KeyError(f"unknown machine {machine_id!r}")
        return self.available_at[machine_id] <= at + 1e-9

    def reserve(self, entry: ScheduleEntry) -> None:
        if entry.end <= entry.start:
            raise ValueError("machine reservation must have positive duration")
        if not self.is_available(entry.machine_id, entry.start):
            raise ValueError(
                f"machine {entry.machine_id!r} is occupied until "
                f"{self.available_at[entry.machine_id]:.6g}"
            )
        self.available_at[entry.machine_id] = entry.end
        self.entries.append(entry)

