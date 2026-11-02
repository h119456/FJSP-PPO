"""Event-driven deformation-aware flexible job-shop environment."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ppo_rs.config import InstanceSpec, OperationSpec
from ppo_rs.environment.scheduler import MachineLedger
from ppo_rs.physics import (
    CompensationModel,
    DeformationModel,
    StressReliefModel,
    SyntheticCompensationModel,
    SyntheticIncreasingDeformation,
    SyntheticStressReliefModel,
)
from ppo_rs.types import Action, ActionType, ScheduleEntry


@dataclass
class _JobRuntime:
    next_operation: int = 0
    busy: bool = False
    anchor_time: float = 0.0
    deformation_anchor: float = 0.0
    release_completed: bool = False
    after_operation: str | None = None
    last_normal_machine_id: str | None = None
    history: list[str] = field(default_factory=list)
    relief_count: int = 0
    compensation_count: int = 0
    stress_release_wait: float = 0.0
    machine_queue_wait: float = 0.0
    post_release_queue_wait: float = 0.0
    completed_at: float | None = None


@dataclass
class _RunningEvent:
    action: Action
    end: float
    start_deformation: float


class DeformationFJSPEnv:
    """Gymnasium-style environment with a variable legal-action set.

    The environment does not depend on Gymnasium itself. ``reset`` and ``step``
    use its current return convention so the core remains usable in minimal
    production and test installations.
    """

    ENVIRONMENT_SEMANTICS_VERSION = 7

    OBSERVATION_NAMES = (
        "normalized_time",
        "completed_operation_ratio",
        "busy_job_ratio",
        "busy_machine_ratio",
        "mean_machine_availability_gap",
        "ready_job_ratio",
        "mean_dwell_progress_ratio",
        "max_remaining_dwell_ratio",
        "release_completed_ratio",
        "mean_deformation_to_compensation_threshold",
        "max_deformation_to_compensation_threshold",
        "relief_usage_ratio",
        "compensation_usage_ratio",
        "mandatory_compensation_job_ratio",
    )
    ACTION_FEATURE_NAMES = (
        "is_normal",
        "is_stress_relief",
        "is_compensation",
        "is_wait",
        "normalized_duration",
        "normalized_cost",
        "job_progress",
        "dwell_progress_ratio",
        "remaining_dwell_ratio",
        "release_completed",
        "deformation_to_compensation_threshold",
        "uses_original_machine",
        "job_relief_usage_ratio",
        "job_compensation_usage_ratio",
        "normalized_deformation_scale",
        "normalized_deformation_rate",
        "post_action_deformation_to_compensation_threshold",
        "natural_wait_saved_ratio",
        "normalized_mandatory_compensation_duration",
        "normalized_mandatory_compensation_cost",
        "machine_utilization",
    )
    OBSERVATION_DIM = len(OBSERVATION_NAMES)
    ACTION_FEATURE_DIM = len(ACTION_FEATURE_NAMES)

    GRAPH_NODE_TYPES = ("job", "operation", "machine")
    GRAPH_EDGE_TYPES = (
        "job_to_operation",
        "operation_to_job",
        "precedes",
        "succeeds",
        "operation_to_machine",
        "machine_to_operation",
        "operation_to_stress_relief",
        "stress_relief_to_operation",
        "operation_to_compensation",
        "compensation_to_operation",
    )
    GRAPH_NODE_FEATURE_DIM = 11
    GRAPH_EDGE_FEATURE_DIM = 2
    GRAPH_NODE_TYPE_COUNT = len(GRAPH_NODE_TYPES)
    GRAPH_EDGE_TYPE_COUNT = len(GRAPH_EDGE_TYPES)

    def __init__(
        self,
        instance: InstanceSpec,
        *,
        deformation_model: DeformationModel | None = None,
        stress_relief_model: StressReliefModel | None = None,
        compensation_model: CompensationModel | None = None,
    ) -> None:
        self.instance = instance
        p = instance.physics
        self.deformation_model = deformation_model or SyntheticIncreasingDeformation()
        self.stress_relief_model = stress_relief_model or SyntheticStressReliefModel()
        self.compensation_model = compensation_model or SyntheticCompensationModel(
            p.compensation_deformation_factor
        )
        self._jobs_by_id = {job.job_id: job for job in instance.jobs}
        self._total_operations = sum(len(job.operations) for job in instance.jobs)
        operations = [
            operation
            for job in instance.jobs
            for operation in job.operations
        ]
        self._deformation_scale = max(
            (
                operation.deformation_scale
                if operation.deformation_scale is not None
                else instance.physics.default_deformation_scale
            )
            for operation in operations
        )
        self._deformation_rate_scale = max(
            (
                operation.deformation_rate
                if operation.deformation_rate is not None
                else instance.physics.default_deformation_rate
            )
            for operation in operations
        )
        maximum_auxiliary_cost = max(
            instance.stress_relief.cost,
            instance.compensation.cost,
        )
        self._cost_scale = maximum_auxiliary_cost if maximum_auxiliary_cost > 0 else 1.0
        durations = [
            duration
            for job in instance.jobs
            for operation in job.operations
            for duration in operation.machines.values()
        ]
        durations += list(instance.stress_relief.machines.values())
        durations += list(instance.compensation.machines.values())
        if instance.compensation.allow_original_machine and instance.compensation.machines:
            durations.append(
                min(instance.compensation.machines.values())
                * instance.compensation.original_machine_duration_factor
            )
        ratio = instance.physics.minimum_deformation_ratio
        durations += [
            -math.log1p(-ratio)
            / (
                operation.deformation_rate
                if operation.deformation_rate is not None
                else instance.physics.default_deformation_rate
            )
            for job in instance.jobs
            for operation in job.operations[:-1]
        ]
        self._duration_scale = max(durations, default=1.0)
        self._initialize_graph_schema()
        self._rng = random.Random()
        self.reset()

    def _initialize_graph_schema(self) -> None:
        """Build the immutable topology shared by every state of this instance."""

        self._graph_job_nodes: dict[str, int] = {}
        self._graph_operation_nodes: dict[tuple[str, str], int] = {}
        self._graph_machine_nodes: dict[str, int] = {}
        node_types: list[int] = []

        for job in self.instance.jobs:
            self._graph_job_nodes[job.job_id] = len(node_types)
            node_types.append(0)
        for job in self.instance.jobs:
            for operation in job.operations:
                self._graph_operation_nodes[(job.job_id, operation.operation_id)] = len(
                    node_types
                )
                node_types.append(1)
        for machine in self.instance.machines:
            self._graph_machine_nodes[machine] = len(node_types)
            node_types.append(2)

        sources: list[int] = []
        destinations: list[int] = []
        edge_types: list[int] = []
        edge_features: list[list[float]] = []

        def add_edge(
            source: int,
            destination: int,
            relation: int,
            duration: float = 0.0,
            cost: float = 0.0,
        ) -> None:
            sources.append(source)
            destinations.append(destination)
            edge_types.append(relation)
            edge_features.append(
                [
                    duration / max(1.0, self._duration_scale),
                    cost / self._cost_scale,
                ]
            )

        for job in self.instance.jobs:
            job_node = self._graph_job_nodes[job.job_id]
            for operation_index, operation in enumerate(job.operations):
                operation_node = self._graph_operation_nodes[
                    (job.job_id, operation.operation_id)
                ]
                add_edge(job_node, operation_node, 0)
                add_edge(operation_node, job_node, 1)
                if operation_index + 1 < len(job.operations):
                    next_operation = job.operations[operation_index + 1]
                    next_node = self._graph_operation_nodes[
                        (job.job_id, next_operation.operation_id)
                    ]
                    add_edge(operation_node, next_node, 2)
                    add_edge(next_node, operation_node, 3)
                for machine, duration in operation.machines.items():
                    machine_node = self._graph_machine_nodes[machine]
                    add_edge(operation_node, machine_node, 4, duration)
                    add_edge(machine_node, operation_node, 5, duration)

                if operation_index + 1 >= len(job.operations):
                    continue
                if self.instance.stress_relief.enabled and self.instance.stress_relief.allowed_after(
                    operation.operation_id
                ):
                    for machine, duration in self.instance.stress_relief.machines.items():
                        machine_node = self._graph_machine_nodes[machine]
                        add_edge(
                            operation_node,
                            machine_node,
                            6,
                            duration,
                            self.instance.stress_relief.cost,
                        )
                        add_edge(
                            machine_node,
                            operation_node,
                            7,
                            duration,
                            self.instance.stress_relief.cost,
                        )
                if self.instance.compensation.enabled and self.instance.compensation.allowed_after(
                    operation.operation_id
                ):
                    for machine, duration in self.instance.compensation.machines.items():
                        machine_node = self._graph_machine_nodes[machine]
                        add_edge(
                            operation_node,
                            machine_node,
                            8,
                            duration,
                            self.instance.compensation.cost,
                        )
                        add_edge(
                            machine_node,
                            operation_node,
                            9,
                            duration,
                            self.instance.compensation.cost,
                        )

        self._graph_node_types = np.asarray(node_types, dtype=np.int64)
        self._graph_edge_index = np.asarray(
            [sources, destinations], dtype=np.int64
        )
        self._graph_edge_types = np.asarray(edge_types, dtype=np.int64)
        self._graph_edge_features = np.asarray(edge_features, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[list[float], dict[str, Any]]:
        del options
        if seed is not None:
            self._rng.seed(seed)
        self.current_time = 0.0
        self.decision_count = 0
        self.ledger = MachineLedger(self.instance.machines)
        self.jobs = {job.job_id: _JobRuntime() for job in self.instance.jobs}
        self.running: list[_RunningEvent] = []
        self.relief_cost = 0.0
        self.compensation_cost = 0.0
        self._terminated = False
        self._truncated = False
        return self.get_observation(), self._info()

    def _operation(self, job_id: str) -> OperationSpec | None:
        runtime = self.jobs[job_id]
        job = self._jobs_by_id[job_id]
        if runtime.next_operation >= len(job.operations):
            return None
        return job.operations[runtime.next_operation]

    def _completed_operation(self, job_id: str) -> OperationSpec | None:
        runtime = self.jobs[job_id]
        if runtime.next_operation == 0:
            return None
        return self._jobs_by_id[job_id].operations[runtime.next_operation - 1]

    def current_dwell(self, job_id: str, at: float | None = None) -> float:
        """Elapsed placement time since the latest process-state anchor."""

        runtime = self.jobs[job_id]
        instant = self.current_time if at is None else at
        return max(0.0, instant - runtime.anchor_time)

    def compute_deformation(self, job_id: str, at: float | None = None) -> float:
        runtime = self.jobs[job_id]
        completed = self._completed_operation(job_id)
        if completed is None:
            return runtime.deformation_anchor
        if runtime.release_completed:
            return runtime.deformation_anchor
        dwell = self.current_dwell(job_id, at)
        scale = (
            completed.deformation_scale
            if completed.deformation_scale is not None
            else self.instance.physics.default_deformation_scale
        )
        rate = (
            completed.deformation_rate
            if completed.deformation_rate is not None
            else self.instance.physics.default_deformation_rate
        )
        return self.deformation_model.compute(
            job_id=job_id,
            after_operation=runtime.after_operation,
            anchor_deformation=runtime.deformation_anchor,
            dwell_time=dwell,
            deformation_scale=scale,
            deformation_rate=rate,
            history=runtime.history,
        )

    def check_process_readiness(self, job_id: str, at: float | None = None) -> bool:
        """Check natural dwell or its completed active stress-relief substitute."""

        operation = self._operation(job_id)
        runtime = self.jobs[job_id]
        if operation is None or runtime.next_operation == 0:
            return True
        if runtime.release_completed:
            return True
        return self.current_dwell(job_id, at) + 1e-9 >= self.required_dwell(job_id)

    def required_dwell(self, job_id: str) -> float:
        """Time for the completed operation to reach the configured D/A ratio."""

        completed = self._completed_operation(job_id)
        if completed is None or self._operation(job_id) is None:
            return 0.0
        rate = (
            completed.deformation_rate
            if completed.deformation_rate is not None
            else self.instance.physics.default_deformation_rate
        )
        ratio = self.instance.physics.minimum_deformation_ratio
        return -math.log1p(-ratio) / rate

    def remaining_dwell(self, job_id: str, at: float | None = None) -> float:
        operation = self._operation(job_id)
        runtime = self.jobs[job_id]
        if operation is None or runtime.next_operation == 0 or runtime.release_completed:
            return 0.0
        return max(0.0, self.required_dwell(job_id) - self.current_dwell(job_id, at))

    def _needs_compensation(self, job_id: str, at: float | None = None) -> bool:
        operation = self._operation(job_id)
        if operation is None or runtime_next_is_first(self.jobs[job_id]):
            return False
        threshold = self.instance.compensation.trigger_deformation
        if not self.instance.compensation.enabled or threshold is None:
            return False
        return self.compute_deformation(job_id, at) > threshold + 1e-9

    def _normal_actions(self) -> list[Action]:
        actions: list[Action] = []
        for job_id, runtime in self.jobs.items():
            operation = self._operation(job_id)
            if runtime.busy or operation is None:
                continue
            if not self.check_process_readiness(job_id) or self._needs_compensation(job_id):
                continue
            for machine_id, duration in operation.machines.items():
                if self.ledger.is_available(machine_id, self.current_time):
                    actions.append(
                        Action(
                            ActionType.NORMAL,
                            job_id,
                            operation.operation_id,
                            machine_id,
                            duration,
                            0.0,
                        )
                    )
        return actions

    def _stress_relief_actions(self) -> list[Action]:
        spec = self.instance.stress_relief
        if not spec.enabled:
            return []
        actions: list[Action] = []
        for job_id, runtime in self.jobs.items():
            operation = self._operation(job_id)
            if runtime.busy or operation is None or runtime.next_operation == 0:
                continue
            if runtime.relief_count >= spec.max_per_job or runtime.after_operation is None:
                continue
            if not spec.allowed_after(runtime.after_operation):
                continue
            # Active stress relief replaces only the still-unfinished natural
            # dwell. Once the job is naturally ready or already stabilized,
            # driving it to saturation has no scheduling benefit.
            if runtime.release_completed or self.check_process_readiness(job_id):
                continue
            for machine_id, duration in spec.machines.items():
                if self.ledger.is_available(machine_id, self.current_time):
                    actions.append(
                        Action(
                            ActionType.STRESS_RELIEF,
                            job_id,
                            f"relief_{runtime.relief_count + 1}",
                            machine_id,
                            duration,
                            spec.cost,
                        )
                    )
        return actions

    def _compensation_actions(self) -> list[Action]:
        spec = self.instance.compensation
        if not spec.enabled:
            return []
        actions: list[Action] = []
        for job_id, runtime in self.jobs.items():
            operation = self._operation(job_id)
            if runtime.busy or operation is None or runtime.next_operation == 0:
                continue
            if runtime.compensation_count >= spec.max_per_job or runtime.after_operation is None:
                continue
            if not spec.allowed_after(runtime.after_operation):
                continue
            # Compensation corrects deformation only after natural release is
            # sufficiently complete or active stress relief has stabilized it.
            if not self.check_process_readiness(job_id):
                continue
            threshold = spec.trigger_deformation
            if threshold is None or self.compute_deformation(job_id) <= threshold + 1e-9:
                continue
            for machine_id, duration in spec.machines.items():
                if self.ledger.is_available(machine_id, self.current_time):
                    actions.append(
                        Action(
                            ActionType.COMPENSATION,
                            job_id,
                            f"compensation_{runtime.compensation_count + 1}",
                            machine_id,
                            duration,
                            spec.cost,
                        )
                    )
            original_machine = runtime.last_normal_machine_id
            if (
                spec.allow_original_machine
                and original_machine is not None
                and original_machine not in spec.machines
                and self.ledger.is_available(original_machine, self.current_time)
            ):
                dedicated_duration = min(spec.machines.values())
                actions.append(
                    Action(
                        ActionType.COMPENSATION,
                        job_id,
                        f"compensation_{runtime.compensation_count + 1}",
                        original_machine,
                        dedicated_duration * spec.original_machine_duration_factor,
                        spec.cost,
                    )
                )
        return actions

    def _next_relevant_time(self) -> float | None:
        candidates = [event.end for event in self.running if event.end > self.current_time + 1e-9]
        for job_id, runtime in self.jobs.items():
            operation = self._operation(job_id)
            if runtime.busy or operation is None or runtime.next_operation == 0:
                continue
            if not self.check_process_readiness(job_id):
                candidates.append(runtime.anchor_time + self.required_dwell(job_id))
        future = [value for value in candidates if value > self.current_time + 1e-9]
        return min(future) if future else None

    def get_valid_actions(self) -> list[Action]:
        if self._terminated or self._truncated:
            return []
        normal = self._normal_actions()
        relief = self._stress_relief_actions()
        compensation = self._compensation_actions()
        actions = normal + relief + compensation
        future = self._next_relevant_time()
        # WAIT advances to the next real event. It remains available even when
        # normal work is legal, allowing the policy to choose intentional idle
        # time without introducing arbitrary fixed time steps.
        if future is not None:
            actions.append(Action(ActionType.WAIT, duration=future - self.current_time))
        return actions

    def step(
        self, action: int | Action
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode end; call reset()")
        valid = self.get_valid_actions()
        if not valid:
            self._resolve_dead_end()
            return self.get_observation(), 0.0, self._terminated, self._truncated, self._info()
        selected = valid[action] if isinstance(action, int) else action
        if selected not in valid:
            raise ValueError(f"illegal action at t={self.current_time}: {selected}")

        objective_before = self._partial_objective()
        self.decision_count += 1
        if selected.kind is ActionType.WAIT:
            next_time = self._next_relevant_time()
            if next_time is None:
                self._truncated = True
            else:
                self._advance_time(next_time)
        else:
            self._schedule(selected)
            self._advance_while_forced()

        self._update_end_flags()
        reward = -(self._partial_objective() - objective_before)
        info = self._info()
        info["selected_action"] = selected
        return self.get_observation(), reward, self._terminated, self._truncated, info

    def _schedule(self, action: Action) -> None:
        assert action.job_id is not None and action.machine_id is not None
        assert action.operation_id is not None
        runtime = self.jobs[action.job_id]
        if runtime.busy:
            raise RuntimeError(f"job {action.job_id} is already processing")
        start_deformation = self.compute_deformation(action.job_id)
        end = self.current_time + action.duration
        entry = ScheduleEntry(
            action.kind,
            action.job_id,
            action.operation_id,
            action.machine_id,
            self.current_time,
            end,
            action.cost,
        )
        self.ledger.reserve(entry)
        runtime.busy = True
        if action.kind is ActionType.STRESS_RELIEF:
            runtime.relief_count += 1
            self.relief_cost += action.cost
        elif action.kind is ActionType.COMPENSATION:
            runtime.compensation_count += 1
            self.compensation_cost += action.cost
        self.running.append(_RunningEvent(action, end, start_deformation))

    def _advance_while_forced(self) -> None:
        while not self._all_complete():
            actions = self._normal_actions() + self._stress_relief_actions() + self._compensation_actions()
            if actions:
                return
            next_time = self._next_relevant_time()
            if next_time is None:
                return
            self._advance_time(next_time)

    def _advance_time(self, new_time: float) -> None:
        if new_time <= self.current_time + 1e-9:
            raise RuntimeError("event clock did not advance")
        if new_time > self.instance.environment.max_time + 1e-9:
            self._truncated = True
            return
        delta = new_time - self.current_time
        for job_id, runtime in self.jobs.items():
            operation = self._operation(job_id)
            if not runtime.busy and operation is not None and runtime.next_operation > 0:
                if runtime.release_completed:
                    runtime.post_release_queue_wait += delta
                    continue
                ready_at = runtime.anchor_time + self.required_dwell(job_id)
                release_wait = max(
                    0.0,
                    min(new_time, ready_at) - min(max(self.current_time, runtime.anchor_time), ready_at),
                )
                runtime.stress_release_wait += release_wait
                runtime.machine_queue_wait += delta - release_wait
        self.current_time = new_time
        finished = [event for event in self.running if event.end <= new_time + 1e-9]
        self.running = [event for event in self.running if event.end > new_time + 1e-9]
        for event in sorted(finished, key=lambda item: item.end):
            self._complete_event(event)

    def _complete_event(self, event: _RunningEvent) -> None:
        action = event.action
        assert action.job_id is not None
        runtime = self.jobs[action.job_id]
        runtime.busy = False
        if action.kind is ActionType.NORMAL:
            operation = self._operation(action.job_id)
            if operation is None or operation.operation_id != action.operation_id:
                raise RuntimeError("normal-operation state changed while event was running")
            runtime.next_operation += 1
            runtime.history.append(f"NORMAL:{action.operation_id}")
            # Each normal operation defines a new deformation reference:
            # immediately after completion t=0 and D=0 for that operation's
            # independent A(1-exp(-Bt)) curve.
            runtime.deformation_anchor = 0.0
            runtime.release_completed = False
            runtime.anchor_time = event.end
            runtime.after_operation = operation.operation_id
            runtime.last_normal_machine_id = action.machine_id
            if runtime.next_operation == len(self._jobs_by_id[action.job_id].operations):
                runtime.completed_at = event.end
        elif action.kind is ActionType.STRESS_RELIEF:
            runtime.deformation_anchor = self.apply_stress_relief(
                action.job_id, event.start_deformation
            )
            runtime.release_completed = True
            runtime.anchor_time = event.end
            runtime.history.append(f"STRESS_RELIEF:{action.operation_id}")
        elif action.kind is ActionType.COMPENSATION:
            runtime.deformation_anchor = self.apply_compensation(
                action.job_id, event.start_deformation
            )
            # Compensation is legal only after natural readiness or active
            # stress relief, so its corrected shape remains stable.
            runtime.release_completed = True
            runtime.anchor_time = event.end
            runtime.history.append(f"COMPENSATION:{action.operation_id}")

    def apply_stress_relief(
        self, job_id: str, deformation: float | None = None
    ) -> float:
        completed = self._completed_operation(job_id)
        if completed is None:
            raise RuntimeError("stress relief requires a completed normal operation")
        maximum = (
            completed.deformation_scale
            if completed.deformation_scale is not None
            else self.instance.physics.default_deformation_scale
        )
        return self.stress_relief_model.apply(
            deformation=(
                deformation if deformation is not None else self.compute_deformation(job_id)
            ),
            maximum_deformation=maximum,
        )

    def apply_compensation(
        self, job_id: str, deformation: float | None = None
    ) -> float:
        return self.compensation_model.apply(
            deformation=(
                deformation if deformation is not None else self.compute_deformation(job_id)
            ),
        )

    def _all_complete(self) -> bool:
        return all(
            runtime.next_operation == len(self._jobs_by_id[job_id].operations)
            for job_id, runtime in self.jobs.items()
        ) and not self.running

    def _resolve_dead_end(self) -> None:
        if self._all_complete():
            self._terminated = True
        else:
            self._truncated = True

    def _update_end_flags(self) -> None:
        if self._all_complete():
            self._terminated = True
        if self.decision_count >= self.instance.environment.max_decisions:
            self._truncated = True
        if self.current_time >= self.instance.environment.max_time:
            self._truncated = True
        if not self._terminated and not self._truncated and not self.get_valid_actions():
            # A non-terminal state with neither a legal action nor a future
            # event is an unrecoverable process/configuration dead end.
            self._truncated = True

    def _partial_objective(self) -> float:
        objective = self.instance.objective
        waiting = sum(
            runtime.stress_release_wait
            + runtime.machine_queue_wait
            + runtime.post_release_queue_wait
            for runtime in self.jobs.values()
        )
        waiting_cost = waiting * objective.waiting_unit_cost
        return (
            objective.makespan_weight * self.current_time
            + objective.relief_cost_weight * self.relief_cost
            + objective.compensation_cost_weight * self.compensation_cost
            + objective.waiting_cost_weight * waiting_cost
        )

    def get_observation(self) -> list[float]:
        job_count = max(1, len(self.jobs))
        machine_count = max(1, len(self.instance.machines))
        completed = sum(runtime.next_operation for runtime in self.jobs.values())
        busy_jobs = sum(runtime.busy for runtime in self.jobs.values())
        busy_machines = sum(
            available > self.current_time + 1e-9 for available in self.ledger.available_at.values()
        )
        inter_operation_jobs = [
            job_id
            for job_id, runtime in self.jobs.items()
            if runtime.next_operation > 0
            and self._operation(job_id) is not None
            and not runtime.busy
        ]
        dwell_progress_ratios: list[float] = []
        remaining_dwell_ratios: list[float] = []
        deformation_ratios: list[float] = []
        threshold = self.instance.compensation.trigger_deformation
        for job_id in inter_operation_jobs:
            runtime = self.jobs[job_id]
            required = self.required_dwell(job_id)
            if required > 0:
                dwell_progress_ratios.append(
                    1.0
                    if runtime.release_completed
                    else min(1.0, self.current_dwell(job_id) / required)
                )
                remaining_dwell_ratios.append(
                    self.remaining_dwell(job_id) / required
                )
            else:
                dwell_progress_ratios.append(1.0)
                remaining_dwell_ratios.append(0.0)
            deformation_ratios.append(
                self.compute_deformation(job_id) / max(1e-9, threshold or 1.0)
            )
        active_count = max(1, len(inter_operation_jobs))
        release_completed = sum(
            self.jobs[job_id].release_completed
            for job_id in inter_operation_jobs
        )
        mandatory_compensation = sum(
            self._needs_compensation(job_id)
            for job_id in inter_operation_jobs
        )
        ready = sum(
            self._operation(job_id) is not None
            and not runtime.busy
            and self.check_process_readiness(job_id)
            and not self._needs_compensation(job_id)
            for job_id, runtime in self.jobs.items()
        )
        availability_gaps = [
            max(0.0, value - self.current_time) for value in self.ledger.available_at.values()
        ]
        max_relief = max(1, self.instance.stress_relief.max_per_job * job_count)
        max_comp = max(1, self.instance.compensation.max_per_job * job_count)
        return [
            self.current_time / max(1.0, self.instance.environment.max_time),
            completed / max(1, self._total_operations),
            busy_jobs / job_count,
            busy_machines / machine_count,
            sum(availability_gaps) / machine_count / max(1.0, self._duration_scale),
            ready / job_count,
            sum(dwell_progress_ratios) / active_count,
            max(remaining_dwell_ratios, default=0.0),
            release_completed / active_count,
            sum(deformation_ratios) / active_count,
            max(deformation_ratios, default=0.0),
            sum(runtime.relief_count for runtime in self.jobs.values()) / max_relief,
            sum(runtime.compensation_count for runtime in self.jobs.values()) / max_comp,
            mandatory_compensation / active_count,
        ]

    def _minimum_compensation_duration(self, job_id: str) -> float:
        spec = self.instance.compensation
        durations = list(spec.machines.values())
        runtime = self.jobs[job_id]
        if (
            spec.allow_original_machine
            and runtime.last_normal_machine_id is not None
            and spec.machines
        ):
            durations.append(
                min(spec.machines.values()) * spec.original_machine_duration_factor
            )
        return min(durations, default=0.0)

    def _machine_utilization(self, machine_id: str | None) -> float:
        if machine_id is None:
            return 0.0
        occupied = sum(
            entry.end - entry.start
            for entry in self.ledger.entries
            if entry.machine_id == machine_id
        )
        return min(
            1.0,
            occupied / max(1.0, self.current_time, self._duration_scale),
        )

    def get_graph_state(self, actions: list[Action] | None = None) -> dict[str, Any]:
        """Return dynamic heterogeneous graph features for the GNN policy."""

        selected = self.get_valid_actions() if actions is None else actions
        features = np.zeros(
            (len(self._graph_node_types), self.GRAPH_NODE_FEATURE_DIM),
            dtype=np.float32,
        )
        job_count = max(1, len(self.instance.jobs))
        machine_count = max(1, len(self.instance.machines))
        total_operations = max(1, self._total_operations)
        threshold = self.instance.compensation.trigger_deformation

        for job in self.instance.jobs:
            runtime = self.jobs[job.job_id]
            node = self._graph_job_nodes[job.job_id]
            operation = self._operation(job.job_id)
            complete = operation is None
            between_operations = runtime.next_operation > 0 and not complete
            required = self.required_dwell(job.job_id) if between_operations else 0.0
            dwell_progress = (
                1.0
                if between_operations and runtime.release_completed
                else (
                    min(1.0, self.current_dwell(job.job_id) / required)
                    if required > 0
                    else 0.0
                )
            )
            remaining_dwell = (
                self.remaining_dwell(job.job_id) / required if required > 0 else 0.0
            )
            deformation_ratio = (
                self.compute_deformation(job.job_id) / max(1e-9, threshold or 1.0)
                if between_operations
                else 0.0
            )
            features[node, :11] = [
                runtime.next_operation / len(job.operations),
                (len(job.operations) - runtime.next_operation) / len(job.operations),
                float(runtime.busy),
                float(complete),
                dwell_progress,
                remaining_dwell,
                float(runtime.release_completed),
                deformation_ratio,
                float(self._needs_compensation(job.job_id)),
                runtime.relief_count
                / max(1, self.instance.stress_relief.max_per_job),
                runtime.compensation_count
                / max(1, self.instance.compensation.max_per_job),
            ]

        running_normal = {
            (event.action.job_id, event.action.operation_id)
            for event in self.running
            if event.action.kind is ActionType.NORMAL
        }
        for job in self.instance.jobs:
            runtime = self.jobs[job.job_id]
            for operation_index, operation in enumerate(job.operations):
                node = self._graph_operation_nodes[
                    (job.job_id, operation.operation_id)
                ]
                durations = list(operation.machines.values())
                scale = operation.deformation_scale or self.instance.physics.default_deformation_scale
                rate = operation.deformation_rate or self.instance.physics.default_deformation_rate
                features[node, :11] = [
                    (operation_index + 1) / len(job.operations),
                    float(operation_index < runtime.next_operation),
                    float(operation_index == runtime.next_operation),
                    float((job.job_id, operation.operation_id) in running_normal),
                    min(durations) / max(1.0, self._duration_scale),
                    (sum(durations) / len(durations))
                    / max(1.0, self._duration_scale),
                    len(operation.machines) / machine_count,
                    scale / max(1e-9, self._deformation_scale),
                    rate / max(1e-9, self._deformation_rate_scale),
                    float(operation_index == 0),
                    float(operation_index + 1 == len(job.operations)),
                ]

        action_loads = {machine: 0 for machine in self.instance.machines}
        for action in selected:
            if action.machine_id is not None:
                action_loads[action.machine_id] += 1
        compatibility_counts = {machine: 0 for machine in self.instance.machines}
        for job in self.instance.jobs:
            for operation in job.operations:
                for machine in operation.machines:
                    compatibility_counts[machine] += 1
        for machine in self.instance.machines:
            node = self._graph_machine_nodes[machine]
            is_relief = machine in self.instance.stress_relief.machines
            is_compensation = machine in self.instance.compensation.machines
            gap = max(0.0, self.ledger.available_at[machine] - self.current_time)
            auxiliary_duration = (
                self.instance.stress_relief.machines.get(machine)
                or self.instance.compensation.machines.get(machine)
                or 0.0
            )
            auxiliary_cost = (
                self.instance.stress_relief.cost
                if is_relief
                else self.instance.compensation.cost
                if is_compensation
                else 0.0
            )
            features[node, :10] = [
                float(not is_relief and not is_compensation),
                float(is_relief),
                float(is_compensation),
                float(gap > 1e-9),
                gap / max(1.0, self._duration_scale),
                self._machine_utilization(machine),
                action_loads[machine] / job_count,
                compatibility_counts[machine] / total_operations,
                auxiliary_duration / max(1.0, self._duration_scale),
                auxiliary_cost / self._cost_scale,
            ]

        action_nodes = np.full((len(selected), 3), -1, dtype=np.int64)
        for index, action in enumerate(selected):
            if action.job_id is None:
                continue
            action_nodes[index, 0] = self._graph_job_nodes[action.job_id]
            operation_id = (
                action.operation_id
                if action.kind is ActionType.NORMAL
                else self.jobs[action.job_id].after_operation
            )
            if operation_id is not None:
                action_nodes[index, 1] = self._graph_operation_nodes[
                    (action.job_id, operation_id)
                ]
            if action.machine_id is not None:
                action_nodes[index, 2] = self._graph_machine_nodes[action.machine_id]

        return {
            "node_features": features,
            "node_types": self._graph_node_types,
            "edge_index": self._graph_edge_index,
            "edge_types": self._graph_edge_types,
            "edge_features": self._graph_edge_features,
            "action_nodes": action_nodes,
        }

    def get_action_features(self, actions: list[Action] | None = None) -> list[list[float]]:
        selected = self.get_valid_actions() if actions is None else actions
        features: list[list[float]] = []
        for action in selected:
            kind = [float(action.kind is value) for value in ActionType]
            if action.job_id is None:
                features.append(
                    kind
                    + [action.duration / self._duration_scale, 0.0]
                    + [0.0] * (self.ACTION_FEATURE_DIM - len(kind) - 2)
                )
                continue
            runtime = self.jobs[action.job_id]
            job = self._jobs_by_id[action.job_id]
            deformation = self.compute_deformation(action.job_id)
            operation = self._operation(action.job_id)
            threshold = self.instance.compensation.trigger_deformation
            required_dwell = self.required_dwell(action.job_id)
            dwell_progress_ratio = (
                1.0
                if runtime.release_completed and runtime.next_operation > 0
                else (
                    min(1.0, self.current_dwell(action.job_id) / required_dwell)
                    if required_dwell > 0
                    else 0.0
                )
            )
            remaining_dwell_ratio = (
                self.remaining_dwell(action.job_id) / required_dwell
                if required_dwell > 0
                else 0.0
            )
            completed = self._completed_operation(action.job_id)
            relevant_operation = operation if action.kind is ActionType.NORMAL else completed
            scale = (
                relevant_operation.deformation_scale
                if relevant_operation is not None
                and relevant_operation.deformation_scale is not None
                else self.instance.physics.default_deformation_scale
            )
            rate = (
                relevant_operation.deformation_rate
                if relevant_operation is not None
                and relevant_operation.deformation_rate is not None
                else self.instance.physics.default_deformation_rate
            )

            if action.kind is ActionType.NORMAL:
                post_deformation = 0.0
            elif action.kind is ActionType.STRESS_RELIEF:
                post_deformation = scale
            elif action.kind is ActionType.COMPENSATION:
                post_deformation = (
                    deformation * self.instance.physics.compensation_deformation_factor
                )
            else:
                post_deformation = deformation

            requires_compensation = bool(
                threshold is not None
                and post_deformation > threshold + 1e-9
                and action.kind is not ActionType.NORMAL
            )
            mandatory_compensation_duration = (
                self._minimum_compensation_duration(action.job_id)
                if requires_compensation
                else 0.0
            )
            mandatory_compensation_cost = (
                self.instance.compensation.cost
                if requires_compensation
                else 0.0
            )
            wait_saved = (
                max(
                    0.0,
                    self.remaining_dwell(action.job_id)
                    - action.duration,
                )
                if action.kind is ActionType.STRESS_RELIEF
                else 0.0
            )
            features.append(
                kind
                + [
                    action.duration / self._duration_scale,
                    action.cost / self._cost_scale,
                    runtime.next_operation / len(job.operations),
                    dwell_progress_ratio,
                    remaining_dwell_ratio,
                    float(runtime.release_completed),
                    deformation / max(1e-9, threshold or 1.0),
                    float(
                        action.kind is ActionType.COMPENSATION
                        and action.machine_id == runtime.last_normal_machine_id
                        and action.machine_id not in self.instance.compensation.machines
                    ),
                    runtime.relief_count
                    / max(1, self.instance.stress_relief.max_per_job),
                    runtime.compensation_count
                    / max(1, self.instance.compensation.max_per_job),
                    scale / max(1e-9, self._deformation_scale),
                    rate / max(1e-9, self._deformation_rate_scale),
                    post_deformation / max(1e-9, threshold or 1.0),
                    wait_saved / required_dwell if required_dwell > 0 else 0.0,
                    mandatory_compensation_duration / max(1.0, self._duration_scale),
                    mandatory_compensation_cost / self._cost_scale,
                    self._machine_utilization(action.machine_id),
                ]
            )
        return features

    def get_schedule(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.ledger.entries]

    def get_metrics(
        self, inference_time: float = 0.0, *, validate: bool = True
    ) -> dict[str, Any]:
        from ppo_rs.environment.constraints import schedule_violations

        makespan = max(
            (runtime.completed_at or 0.0 for runtime in self.jobs.values()), default=0.0
        )
        normal_time = sum(
            entry.end - entry.start
            for entry in self.ledger.entries
            if entry.kind is ActionType.NORMAL
        )
        stress_release_wait = sum(
            runtime.stress_release_wait for runtime in self.jobs.values()
        )
        machine_queue_wait = sum(
            runtime.machine_queue_wait for runtime in self.jobs.values()
        )
        post_release_queue_wait = sum(
            runtime.post_release_queue_wait for runtime in self.jobs.values()
        )
        inter_operation_idle = (
            stress_release_wait + machine_queue_wait + post_release_queue_wait
        )
        utilization = {
            machine: (
                sum(
                    entry.end - entry.start
                    for entry in self.ledger.entries
                    if entry.machine_id == machine
                )
                / makespan
                if makespan > 0
                else 0.0
            )
            for machine in self.instance.machines
        }
        violations = (
            schedule_violations(self.instance, self.ledger.entries)
            if validate
            else []
        )
        readiness_violations = [
            violation
            for violation in violations
            if violation.startswith("process readiness:")
        ]
        compensation_entries = [
            entry
            for entry in self.ledger.entries
            if entry.kind is ActionType.COMPENSATION
        ]
        dedicated_compensation_count = sum(
            entry.machine_id in self.instance.compensation.machines
            for entry in compensation_entries
        )
        return {
            "makespan": makespan,
            "normal_processing_time": normal_time,
            "stress_release_wait_time": stress_release_wait,
            "machine_queue_wait_time": machine_queue_wait,
            "post_release_queue_wait_time": post_release_queue_wait,
            "inter_operation_idle_time": inter_operation_idle,
            "stress_relief_count": sum(r.relief_count for r in self.jobs.values()),
            "stress_relief_cost": self.relief_cost,
            "compensation_count": sum(r.compensation_count for r in self.jobs.values()),
            "dedicated_compensation_count": dedicated_compensation_count,
            "original_machine_compensation_count": (
                len(compensation_entries) - dedicated_compensation_count
            ),
            "compensation_cost": self.compensation_cost,
            "total_independent_cost": self.relief_cost + self.compensation_cost,
            "objective": self._partial_objective(),
            "process_readiness_violations": len(readiness_violations),
            "schedule_violation_count": len(violations),
            "schedule_violations": violations,
            "machine_utilization": utilization,
            "policy_inference_seconds": inference_time,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }

    def _info(self) -> dict[str, Any]:
        actions = self.get_valid_actions() if not (self._terminated or self._truncated) else []
        return {
            "time": self.current_time,
            "valid_actions": actions,
            "action_features": self.get_action_features(actions),
            # Full independent schedule validation is performed once when an
            # episode finishes. Re-running it at every decision would make PPO
            # collection unnecessarily expensive.
            "metrics": self.get_metrics(validate=False),
        }


def runtime_next_is_first(runtime: _JobRuntime) -> bool:
    """Small named predicate that keeps quality checks readable."""

    return runtime.next_operation == 0


# Compatibility name retained for existing training scripts and checkpoints.
ResidualStressFJSPEnv = DeformationFJSPEnv
