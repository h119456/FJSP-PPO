from __future__ import annotations

from dataclasses import replace
import importlib
import math
import time
from types import SimpleNamespace

import pytest
import torch

from ppo_rs.algorithms import RolloutBuffer, collect_episode
from ppo_rs.baselines import policies
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.evaluation.runner import evaluate_model
from ppo_rs.models import build_actor_critic
from ppo_rs.types import Action, ActionType

from tests.helpers import tiny_instance


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def perf_counter(self) -> float:
        return self.now


class _TimedEnvironment:
    def __init__(self, clock: _Clock, *, truncated: bool = False) -> None:
        self.clock = clock
        self.truncated = truncated
        self.steps = 0

    def _info(self) -> dict:
        return {
            "valid_actions": [Action(ActionType.NORMAL)],
            "action_features": [[0.0]],
        }

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict]:
        self.clock.advance(2.0)
        self.steps = 0
        return [0.0], self._info()

    def get_graph_state(self, actions: list[Action]) -> dict:
        self.clock.advance(3.0)
        return {"step": self.steps}

    def step(self, choice: int) -> tuple:
        assert choice == 0
        self.clock.advance(7.0)
        self.steps += 1
        done = self.steps == 2
        return [float(self.steps)], -1.0, done and not self.truncated, done and self.truncated, self._info()

    def get_metrics(self, inference_time: float = 0.0) -> dict:
        # Final schedule validation is part of a complete episode measurement.
        self.clock.advance(11.0)
        return {
            "objective": 2.0,
            "terminated": not self.truncated,
            "truncated": self.truncated,
            "policy_inference_seconds": inference_time,
        }

    def get_schedule(self) -> list[dict]:
        # Schedule extraction/serialization is outside episode timing.
        self.clock.advance(13.0)
        return [{"steps": self.steps}]


class _TimedModel:
    def __init__(self, clock: _Clock, *, graph: bool, device: str = "cpu") -> None:
        self.clock = clock
        self.requires_graph = graph
        self.device = torch.device(device)

    def act(self, observation: list, features: list, *, graph_state: dict | None, deterministic: bool) -> tuple:
        assert (graph_state is not None) is self.requires_graph
        self.clock.advance(5.0)
        return 0, -0.5, 0.0


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)

    def unexpected_cuda_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU and rule episodes must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda_sync)
    return clock


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_collection_times_reset_decisions_steps_buffer_and_final_metrics(
    fake_clock: _Clock, monkeypatch: pytest.MonkeyPatch, graph: bool, truncated: bool,
) -> None:
    original_add = RolloutBuffer.add

    def timed_add(buffer: RolloutBuffer, *args: object, **kwargs: object) -> None:
        fake_clock.advance(1.0)
        original_add(buffer, *args, **kwargs)

    monkeypatch.setattr(RolloutBuffer, "add", timed_add)
    buffer, metrics = collect_episode(
        _TimedEnvironment(fake_clock, truncated=truncated),
        _TimedModel(fake_clock, graph=graph),
        seed=7,
    )

    assert metrics["episode_wall_seconds"] == 39.0 + (6.0 if graph else 0.0)
    assert metrics["policy_inference_seconds"] == 10.0
    assert metrics["graph_construction_seconds"] == (6.0 if graph else 0.0)
    assert metrics["terminated"] is not truncated
    assert metrics["truncated"] is truncated
    assert buffer.rewards == [-1.0, -1.0]
    assert buffer.dones == [False, True]
    assert all((state is not None) is graph for state in buffer.graph_states)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_evaluation_times_complete_episode_before_schedule_extraction(
    fake_clock: _Clock, graph: bool, truncated: bool,
) -> None:
    metrics, schedule = evaluate_model(
        _TimedEnvironment(fake_clock, truncated=truncated),
        _TimedModel(fake_clock, graph=graph),
    )

    assert metrics["episode_wall_seconds"] == 37.0 + (6.0 if graph else 0.0)
    assert metrics["policy_inference_seconds"] == 10.0
    assert metrics["graph_construction_seconds"] == (6.0 if graph else 0.0)
    assert metrics["terminated"] is not truncated
    assert metrics["truncated"] is truncated
    assert metrics["policy"] == "ppo"
    assert schedule == [{"steps": 2}]
    assert fake_clock.now == metrics["episode_wall_seconds"] + 13.0


def test_rule_baseline_reports_the_same_episode_scope_without_graph_or_cuda(
    fake_clock: _Clock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(actions: list[Action]) -> int:
        fake_clock.advance(5.0)
        return 0

    monkeypatch.setitem(policies.BASELINES, "ordinary_fjsp", decide)
    metrics, schedule = policies.run_baseline(_TimedEnvironment(fake_clock), "ordinary_fjsp")

    assert metrics["episode_wall_seconds"] == 37.0
    assert metrics["policy_inference_seconds"] == 10.0
    assert metrics["graph_construction_seconds"] == 0.0
    assert metrics["policy"] == "ordinary_fjsp"
    assert schedule == [{"steps": 2}]
    assert fake_clock.now == 50.0


def test_validation_reports_mean_episode_time_and_whole_set_wall_time(
    fake_clock: _Clock, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = importlib.import_module("ppo_rs.train")
    aggregate_metrics = training._mean_validation_metrics

    def construct_env(instance: object, **kwargs: object) -> _TimedEnvironment:
        fake_clock.advance(4.0)
        return _TimedEnvironment(fake_clock)

    def aggregate(results: list[dict]) -> dict:
        fake_clock.advance(17.0)
        return aggregate_metrics(results)

    monkeypatch.setattr(training, "ResidualStressFJSPEnv", construct_env)
    monkeypatch.setattr(training, "_mean_validation_metrics", aggregate)
    metrics, schedules = training._evaluate_validation_set(
        (SimpleNamespace(name="first"), SimpleNamespace(name="second")),
        _TimedModel(fake_clock, graph=False),
        seed=7, observation_profile="unified23", action_feature_profile="compact21",
    )

    assert metrics["instance_count"] == 2
    assert metrics["episode_wall_seconds"] == 37.0
    assert metrics["policy_inference_seconds"] == 10.0
    assert metrics["graph_construction_seconds"] == 0.0
    # Two constructor/episode/schedule passes, followed by aggregation.
    assert metrics["validation_wall_seconds"] == 2 * (4.0 + 37.0 + 13.0) + 17.0
    assert set(schedules) == {"first", "second"}


@pytest.mark.parametrize("runner", [collect_episode, evaluate_model])
def test_cuda_synchronizes_only_at_episode_boundaries(
    fake_clock: _Clock, monkeypatch: pytest.MonkeyPatch, runner: object,
) -> None:
    calls: list[tuple[torch.device, float]] = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append((device, fake_clock.now))
    )
    result = runner(
        _TimedEnvironment(fake_clock), _TimedModel(fake_clock, graph=True, device="cuda:0")
    )
    metrics = result[1] if runner is collect_episode else result[0]

    assert calls == [(torch.device("cuda:0"), 0.0), (torch.device("cuda:0"), 43.0)]
    assert metrics["episode_wall_seconds"] == 43.0


@pytest.mark.parametrize("model_type", ["mlp", "gnn_attention"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_real_policy_timings_preserve_schedule_and_objective(
    monkeypatch: pytest.MonkeyPatch, model_type: str, dynamic: bool,
) -> None:
    def unexpected_cuda_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU models must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda_sync)
    instance = tiny_instance(jobs=2)
    if dynamic:
        instance = replace(instance, dynamic_events={
            "mode": "fixed",
            "future_events_visible_to_agent": False,
            "unreleased_jobs_visible_to_agent": False,
            "events": [{
                "time": 0.5, "type": "machine_breakdown", "machine_id": "M1",
                "repair_duration": 1.0, "interruption_policy": "pause_resume",
                "known_to_agent_before_occurrence": False,
            }],
        })
    env = ResidualStressFJSPEnv(instance)
    model = build_actor_critic(
        model_type, env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8,
        node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
        node_type_count=env.GRAPH_NODE_TYPE_COUNT,
        edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    buffer, training_metrics = collect_episode(env, model, seed=7, deterministic=True)
    training_schedule = env.get_schedule()
    validation_metrics, validation_schedule = evaluate_model(env, model, seed=7)

    assert len(buffer) > 0
    assert training_metrics["terminated"] and validation_metrics["terminated"]
    assert not training_metrics["truncated"] and not validation_metrics["truncated"]
    assert training_schedule == validation_schedule
    assert training_metrics["objective"] == validation_metrics["objective"]
    for metrics in (training_metrics, validation_metrics):
        assert metrics["schedule_violation_count"] == 0
        for key in ("episode_wall_seconds", "policy_inference_seconds", "graph_construction_seconds"):
            assert math.isfinite(metrics[key]) and metrics[key] >= 0.0
        assert metrics["episode_wall_seconds"] >= (
            metrics["policy_inference_seconds"] + metrics["graph_construction_seconds"]
        )
        assert metrics["policy_inference_seconds"] > 0.0
        assert (metrics["graph_construction_seconds"] > 0.0) is (model_type == "gnn_attention")
