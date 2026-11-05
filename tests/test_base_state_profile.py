from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.evaluation import evaluate_model
from ppo_rs.models import build_actor_critic
from ppo_rs.models.factory import build_model_from_checkpoint
from ppo_rs.train import _validate_transfer_schema

from tests.helpers import choose, tiny_instance


def _dynamic(instance, events):
    return replace(instance, dynamic_events={
        "mode": "fixed", "future_events_visible_to_agent": False,
        "unreleased_jobs_visible_to_agent": False, "events": events,
    })


def _events():
    return [
        {"type": "job_release", "time": 2.0, "job_id": "J2"},
        {"type": "machine_breakdown", "time": 0.5, "machine_id": "M1",
         "repair_duration": 10.0, "interruption_policy": "pause_resume"},
    ]


@pytest.mark.parametrize("dynamic", [False, True])
def test_compact18_contains_exact_base14_and_shared_public_dynamic4(dynamic):
    instance = tiny_instance(jobs=2)
    if dynamic:
        instance = _dynamic(instance, _events())
    envs = [ResidualStressFJSPEnv(instance, observation_profile=profile)
            for profile in ("compact14", "compact18", "unified23")]
    compact, reserved, full = envs
    assert reserved.OBSERVATION_NAMES == (
        compact.COMPACT_OBSERVATION_NAMES + full.DYNAMIC_OBSERVATION_NAMES
    )
    assert reserved.OBSERVATION_DIM == 18 and reserved.ACTION_FEATURE_DIM == 21
    assert reserved.GRAPH_NODE_FEATURE_DIM == 13
    seen_event_features = False
    for step in range(instance.environment.max_decisions):
        np.testing.assert_array_equal(reserved.get_observation()[:14], compact.get_observation())
        np.testing.assert_array_equal(reserved.get_observation()[-4:], full.get_observation()[-4:])
        assert reserved.get_valid_actions() == compact.get_valid_actions() == full.get_valid_actions()
        np.testing.assert_array_equal(reserved.get_action_features(), compact.get_action_features())
        graphs = [env.get_graph_state() for env in envs]
        np.testing.assert_array_equal(graphs[1]["node_features"], graphs[2]["node_features"])
        np.testing.assert_array_equal(graphs[1]["node_features"][:, :11], graphs[0]["node_features"])
        if not dynamic:
            assert reserved.get_observation()[-4:] == [0.0] * 4
            assert not graphs[1]["node_features"][:, -2:].any()
        seen_event_features |= any(reserved.get_observation()[-4:])
        if step == 0:
            choices = [choose(env, "NORMAL", job_id="J1", machine="M1") for env in envs]
        elif step == 1:
            choices = [choose(env, "WAIT") for env in envs]
        else:
            choices = [0] * len(envs)
        results = [env.step(choice) for env, choice in zip(envs, choices)]
        assert [result[2:4] for result in results].count(results[0][2:4]) == 3
        if results[0][2] or results[0][3]:
            assert results[0][2] and not results[0][3]
            break
    else:
        pytest.fail("fixture did not terminate")
    assert seen_event_features is dynamic
    assert reserved.get_schedule() == compact.get_schedule() == full.get_schedule()


def test_compact18_does_not_reveal_hidden_jobs_or_future_event_times():
    base = tiny_instance(jobs=2, compensation_threshold=10.0)
    hidden = replace(base.jobs[1], operations=tuple(
        replace(op, machines={"M2": 100.0}, deformation_scale=100.0,
                deformation_rate=100.0) for op in base.jobs[1].operations
    ))
    left = _dynamic(base, [{"type": "job_release", "time": 5.0, "job_id": "J2"}])
    right = _dynamic(replace(base, jobs=(base.jobs[0], hidden)), [
        {"type": "job_release", "time": 9.0, "job_id": "J2"},
        {"type": "machine_breakdown", "time": 8.0, "machine_id": "M2",
         "repair_duration": 7.0, "interruption_policy": "pause_resume"},
    ])
    envs = [ResidualStressFJSPEnv(instance, observation_profile="compact18")
            for instance in (left, right)]
    for stage in range(3):
        np.testing.assert_array_equal(envs[0].get_observation(), envs[1].get_observation())
        assert envs[0].get_valid_actions() == envs[1].get_valid_actions()
        np.testing.assert_array_equal(envs[0].get_action_features(), envs[1].get_action_features())
        graphs = [env.get_graph_state() for env in envs]
        for key in graphs[0]:
            np.testing.assert_array_equal(graphs[0][key], graphs[1][key], err_msg=key)
        if stage == 0:
            for env in envs:
                env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
        elif stage == 1:
            for env in envs:
                env.step(choose(env, "WAIT"))


@pytest.mark.parametrize("model_type", ["mlp", "gnn_attention"])
def test_static_compact18_checkpoint_loads_directly_into_dynamic_model(model_type):
    static_env = ResidualStressFJSPEnv(tiny_instance(jobs=2), observation_profile="compact18")
    dynamic_env = ResidualStressFJSPEnv(
        _dynamic(tiny_instance(jobs=2), _events()), observation_profile="compact18",
    )
    model = build_actor_critic(
        model_type, static_env.OBSERVATION_DIM, static_env.ACTION_FEATURE_DIM, 8,
        node_feature_dim=static_env.GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=static_env.GRAPH_EDGE_FEATURE_DIM,
        node_type_count=static_env.GRAPH_NODE_TYPE_COUNT,
        edge_type_count=static_env.GRAPH_EDGE_TYPE_COUNT,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = {
        "model_type": model_type, "model_state": model.state_dict(),
        "model_kwargs": dict(getattr(model, "checkpoint_kwargs", {})),
        "environment_semantics_version": static_env.ENVIRONMENT_SEMANTICS_VERSION,
        "dynamic_input_semantics_version": static_env.DYNAMIC_INPUT_SEMANTICS_VERSION,
        "observation_dim": static_env.OBSERVATION_DIM,
        "action_feature_dim": static_env.ACTION_FEATURE_DIM,
        "observation_profile": "compact18", "action_feature_profile": "compact21",
        "hidden_dim": 8,
    }
    transferred = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
    _validate_transfer_schema(checkpoint, transferred, "compact18", "compact21")
    metrics, _ = evaluate_model(dynamic_env, transferred, seed=7)
    assert metrics["terminated"] and not metrics["truncated"]
    assert metrics["schedule_violation_count"] == 0
    assert metrics["machine_breakdown_event_count"] == 1
    assert metrics["job_release_event_count"] == 1
