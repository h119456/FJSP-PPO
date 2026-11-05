from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import importlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic
from ppo_rs.train import _evaluate_validation_set
from tests.helpers import tiny_instance


def _instances() -> tuple:
    base = tiny_instance(jobs=2)
    return (
        replace(base, name="static"),
        replace(base, name="release", dynamic_events={
            "mode": "fixed", "future_events_visible_to_agent": False,
            "unreleased_jobs_visible_to_agent": False,
            "events": [{"time": 0.5, "type": "job_release", "job_id": "J2"}],
        }),
        replace(base, name="breakdown", dynamic_events={
            "mode": "fixed", "future_events_visible_to_agent": False,
            "unreleased_jobs_visible_to_agent": False,
            "events": [{"time": 0.5, "type": "machine_breakdown", "machine_id": "M1",
                        "repair_duration": 1.0, "interruption_policy": "pause_resume"}],
        }),
    )


def _model(model_type: str):
    env = ResidualStressFJSPEnv(_instances()[0], observation_profile="unified23")
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
    return model


def _without_timings(value):
    if isinstance(value, dict):
        return {key: _without_timings(item) for key, item in value.items()
                if not key.endswith("_seconds")}
    if isinstance(value, (list, tuple)):
        return [_without_timings(item) for item in value]
    return value


def _evaluate(instances, model):
    return _evaluate_validation_set(
        instances, model, seed=107,
        observation_profile="unified23", action_feature_profile="compact21",
    )


def _validation_worker(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2,
        init_method=f"file://{directory}/rendezvous",
        timeout=timedelta(seconds=45),
    )
    training = importlib.import_module("ppo_rs.train")
    evaluate_model = training.evaluate_model
    expected = json.loads((Path(directory) / "expected.json").read_text())
    try:
        for model_type in ("mlp", "gnn_attention"):
            model = _model(model_type)
            for count in (1, 3):
                calls = []

                def record(env, policy, *, seed):
                    calls.append((env.instance.name, seed))
                    return evaluate_model(env, policy, seed=seed)

                rng = torch.get_rng_state().clone()
                with patch.object(training, "evaluate_model", record):
                    metrics, schedules = _evaluate(_instances()[:count], model)
                assert torch.equal(torch.get_rng_state(), rng)
                assert calls == [(_instances()[index].name, 107 + index)
                                 for index in range(rank, count, 2)]
                assert [row["name"] for row in metrics["instances"]] == [
                    instance.name for instance in _instances()[:count]
                ]
                assert list(schedules) == [instance.name for instance in _instances()[:count]]
                assert _without_timings((metrics, schedules)) == expected[f"{model_type}_{count}"]
                assert metrics["terminated"] and not metrics["truncated"]
                durations = [None, None]
                dist.all_gather_object(durations, metrics["validation_wall_seconds"])
                assert durations[0] == durations[1] and durations[0] > 0

        def fail_on_rank_one(env, policy, *, seed):
            if rank == 1:
                raise ValueError("intentional validation failure")
            return evaluate_model(env, policy, seed=seed)

        with patch.object(training, "evaluate_model", fail_on_rank_one):
            with pytest.raises(RuntimeError, match="rank1: ValueError: intentional validation failure"):
                _evaluate(_instances(), _model("mlp"))
        with pytest.raises(RuntimeError, match="validation requires at least one instance"):
            _evaluate((), _model("mlp"))
    finally:
        dist.destroy_process_group()


def test_parallel_validation_matches_serial_and_propagates_failures(tmp_path: Path) -> None:
    expected = {}
    for model_type in ("mlp", "gnn_attention"):
        model = _model(model_type)
        for count in (1, 3):
            expected[f"{model_type}_{count}"] = _without_timings(_evaluate(_instances()[:count], model))
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    context = mp.spawn(_validation_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("parallel validation exceeded test deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_empty_serial_validation_fails_clearly() -> None:
    with pytest.raises(ValueError, match="at least one instance"):
        _evaluate((), _model("mlp"))
