"""Independent validation helpers for completed schedules."""

from __future__ import annotations

import math
from collections import defaultdict

from ppo_rs.config import InstanceSpec, OperationSpec
from ppo_rs.types import ActionType, ScheduleEntry


TOLERANCE = 1e-9


def _duration(entry: ScheduleEntry) -> float:
    return entry.end - entry.start


def _natural_deformation(operation: OperationSpec, dwell: float) -> float:
    assert operation.deformation_scale is not None
    assert operation.deformation_rate is not None
    return operation.deformation_scale * (
        -math.expm1(-operation.deformation_rate * max(0.0, dwell))
    )


def _required_dwell(instance: InstanceSpec, operation: OperationSpec) -> float:
    assert operation.deformation_rate is not None
    return -math.log1p(-instance.physics.minimum_deformation_ratio) / operation.deformation_rate


def schedule_violations(
    instance: InstanceSpec, entries: list[ScheduleEntry]
) -> list[str]:
    """Return resource, process-readiness, deformation, and precedence violations."""

    violations: list[str] = []
    jobs = {job.job_id: job for job in instance.jobs}
    operations = {
        job.job_id: {operation.operation_id: operation for operation in job.operations}
        for job in instance.jobs
    }
    by_machine: dict[str, list[ScheduleEntry]] = defaultdict(list)
    by_job: dict[str, list[ScheduleEntry]] = defaultdict(list)

    for entry in entries:
        by_machine[entry.machine_id].append(entry)
        by_job[entry.job_id].append(entry)
        if entry.start < -TOLERANCE:
            violations.append(f"negative start time: {entry}")
        if entry.end <= entry.start:
            violations.append(f"non-positive interval: {entry}")
        if entry.machine_id not in instance.machines:
            violations.append(f"unknown machine {entry.machine_id}")
        if entry.job_id not in jobs:
            violations.append(f"unknown job {entry.job_id}")

    for machine_id, intervals in by_machine.items():
        ordered = sorted(intervals, key=lambda item: (item.start, item.end))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.end - TOLERANCE:
                violations.append(
                    f"machine conflict on {machine_id}: "
                    f"{previous.operation_id} / {current.operation_id}"
                )

    for job_id, intervals in by_job.items():
        ordered = sorted(intervals, key=lambda item: (item.start, item.end))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.end - TOLERANCE:
                violations.append(
                    f"job conflict on {job_id}: "
                    f"{previous.operation_id} / {current.operation_id}"
                )

    for entry in entries:
        job = jobs.get(entry.job_id)
        if job is None:
            continue
        if entry.kind is ActionType.NORMAL:
            operation = operations[entry.job_id].get(entry.operation_id)
            if operation is None:
                violations.append(f"unknown operation {entry.job_id}/{entry.operation_id}")
                continue
            if entry.machine_id not in operation.machines:
                violations.append(
                    f"unqualified machine {entry.machine_id} for "
                    f"{entry.job_id}/{entry.operation_id}"
                )
            else:
                expected = operation.machines[entry.machine_id]
                if abs(_duration(entry) - expected) > TOLERANCE:
                    violations.append(
                        f"incorrect duration for {entry.job_id}/{entry.operation_id} "
                        f"on {entry.machine_id}: {_duration(entry):.6g}, expected {expected:.6g}"
                    )
        elif entry.kind is ActionType.STRESS_RELIEF:
            expected = instance.stress_relief.machines.get(entry.machine_id)
            if expected is None:
                violations.append(f"unqualified stress-relief machine {entry.machine_id}")
            elif abs(_duration(entry) - expected) > TOLERANCE:
                violations.append(
                    f"incorrect stress-relief duration on {entry.machine_id}: "
                    f"{_duration(entry):.6g}, expected {expected:.6g}"
                )
        elif entry.kind is ActionType.COMPENSATION:
            previous_normals = [
                candidate
                for candidate in by_job[entry.job_id]
                if candidate.kind is ActionType.NORMAL
                and candidate.end <= entry.start + TOLERANCE
            ]
            original_machine = (
                max(previous_normals, key=lambda item: item.end).machine_id
                if previous_normals
                else None
            )
            expected = instance.compensation.machines.get(entry.machine_id)
            if expected is None and (
                instance.compensation.allow_original_machine
                and entry.machine_id == original_machine
                and instance.compensation.machines
            ):
                expected = (
                    min(instance.compensation.machines.values())
                    * instance.compensation.original_machine_duration_factor
                )
            if expected is None:
                violations.append(f"unqualified compensation machine {entry.machine_id}")
            elif abs(_duration(entry) - expected) > TOLERANCE:
                violations.append(
                    f"incorrect compensation duration on {entry.machine_id}: "
                    f"{_duration(entry):.6g}, expected {expected:.6g}"
                )

    threshold = instance.compensation.trigger_deformation
    for job_id, job in jobs.items():
        timeline = sorted(by_job[job_id], key=lambda item: (item.start, item.end))
        normal_entries = [entry for entry in timeline if entry.kind is ActionType.NORMAL]
        expected_ids = [operation.operation_id for operation in job.operations]
        actual_ids = [entry.operation_id for entry in normal_entries]
        if actual_ids != expected_ids:
            violations.append(
                f"operation sequence for {job_id}: {actual_ids}, expected {expected_ids}"
            )

        relief_entries = [
            entry for entry in timeline if entry.kind is ActionType.STRESS_RELIEF
        ]
        compensation_entries = [
            entry for entry in timeline if entry.kind is ActionType.COMPENSATION
        ]
        if len(relief_entries) > instance.stress_relief.max_per_job:
            violations.append(f"too many stress-relief operations for {job_id}")
        if len(compensation_entries) > instance.compensation.max_per_job:
            violations.append(f"too many compensation operations for {job_id}")

        operation_index = {
            operation.operation_id: index
            for index, operation in enumerate(job.operations)
        }
        for auxiliary in relief_entries + compensation_entries:
            previous = [
                entry
                for entry in normal_entries
                if entry.end <= auxiliary.start + TOLERANCE
            ]
            if not previous:
                violations.append(
                    f"auxiliary operation before first normal operation for {job_id}"
                )
                continue
            previous_normal = max(previous, key=lambda item: item.end)
            previous_index = operation_index.get(previous_normal.operation_id)
            if previous_index == len(job.operations) - 1:
                violations.append(
                    f"auxiliary operation after final operation for {job_id}"
                )
            spec = (
                instance.stress_relief
                if auxiliary.kind is ActionType.STRESS_RELIEF
                else instance.compensation
            )
            if not spec.allowed_after(previous_normal.operation_id):
                violations.append(
                    f"{auxiliary.kind.value} not allowed after "
                    f"{job_id}/{previous_normal.operation_id}"
                )

        for previous_normal, next_normal in zip(normal_entries, normal_entries[1:]):
            previous_index = operation_index.get(previous_normal.operation_id)
            next_index = operation_index.get(next_normal.operation_id)
            if (
                previous_index is None
                or next_index is None
                or next_index != previous_index + 1
            ):
                continue
            previous_operation = job.operations[previous_index]
            ready_at = previous_normal.end + _required_dwell(
                instance, previous_operation
            )
            release_completed = False
            stable_deformation = 0.0
            between = [
                entry
                for entry in timeline
                if entry.kind is not ActionType.NORMAL
                and entry.start >= previous_normal.end - TOLERANCE
                and entry.end <= next_normal.start + TOLERANCE
            ]
            for auxiliary in between:
                if auxiliary.kind is ActionType.STRESS_RELIEF:
                    if release_completed:
                        violations.append(
                            f"stress relief repeated after stabilization for {job_id}"
                        )
                    if auxiliary.start >= ready_at - TOLERANCE:
                        violations.append(
                            f"stress relief after natural readiness for {job_id}"
                        )
                    assert previous_operation.deformation_scale is not None
                    stable_deformation = previous_operation.deformation_scale
                    release_completed = True
                elif auxiliary.kind is ActionType.COMPENSATION:
                    if not release_completed and auxiliary.start < ready_at - TOLERANCE:
                        violations.append(
                            f"process readiness: compensation for {job_id} starts at "
                            f"{auxiliary.start:.6g} before readiness {ready_at:.6g}"
                        )
                    deformation = (
                        stable_deformation
                        if release_completed
                        else _natural_deformation(
                            previous_operation,
                            auxiliary.start - previous_normal.end,
                        )
                    )
                    if threshold is None or deformation <= threshold + TOLERANCE:
                        violations.append(
                            f"compensation below mandatory threshold for {job_id}: "
                            f"D={deformation:.6g}"
                        )
                    stable_deformation = (
                        deformation
                        * instance.physics.compensation_deformation_factor
                    )
                    release_completed = True

            if not release_completed and next_normal.start < ready_at - TOLERANCE:
                violations.append(
                    f"process readiness: {job_id}/{next_normal.operation_id} starts at "
                    f"{next_normal.start:.6g} before readiness {ready_at:.6g}"
                )
            deformation = (
                stable_deformation
                if release_completed
                else _natural_deformation(
                    previous_operation,
                    next_normal.start - previous_normal.end,
                )
            )
            if threshold is not None and deformation > threshold + TOLERANCE:
                violations.append(
                    f"deformation threshold: {job_id}/{next_normal.operation_id} "
                    f"starts with D={deformation:.6g} > {threshold:.6g}"
                )

    return violations


def assert_valid_schedule(instance: InstanceSpec, entries: list[ScheduleEntry]) -> None:
    violations = schedule_violations(instance, entries)
    if violations:
        raise ValueError("invalid schedule: " + "; ".join(violations))
