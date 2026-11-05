"""Policy inputs must agree whenever the observable history agrees."""

from dataclasses import replace
import warnings

import numpy as np
import pytest
import torch

from ppo_rs.baselines import BASELINES, run_baseline
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.environment.checkpoint_semantics import check_dynamic_input_semantics
from ppo_rs.train import train
from tests.helpers import choose, tiny_instance


def dynamic(instance, events):
    return replace(instance, dynamic_events={"mode": "fixed", "events": events})


def release(time=5.0, job_id="J2"):
    return {"type": "job_release", "time": time, "job_id": job_id}


def failure(time=4.0, machine="M1", duration=2.0):
    return {"type": "machine_breakdown", "time": time, "machine_id": machine,
            "repair_duration": duration, "interruption_policy": "pause_resume"}


def assert_same_inputs(left, right):
    np.testing.assert_array_equal(left.get_observation(), right.get_observation())
    actions = left.get_valid_actions()
    assert actions == right.get_valid_actions()  # includes candidate set/order/duration
    np.testing.assert_array_equal(left.get_action_features(), right.get_action_features())
    left_graph, right_graph = left.get_graph_state(), right.get_graph_state()
    assert left_graph.keys() == right_graph.keys()
    for key in left_graph:
        np.testing.assert_array_equal(left_graph[key], right_graph[key], err_msg=key)


@pytest.mark.parametrize("state_profile", ["compact14", "compact18", "enhanced19", "unified23"])
@pytest.mark.parametrize("action_profile", ["compact21", "enhanced25"])
def test_hidden_job_parameters_routing_and_count_do_not_change_inputs(state_profile, action_profile):
    base = tiny_instance(jobs=2, compensation_threshold=10.0)
    hidden = base.jobs[1]
    changed = replace(hidden, operations=tuple(
        replace(op, machines={"M2": 100.0}, deformation_scale=100.0,
                deformation_rate=100.0) for op in hidden.operations
    ) + (replace(hidden.operations[0], operation_id="O3", machines={"C": 200.0}),))
    extra = replace(changed, job_id="J3")
    first = dynamic(base, [release()])
    second = dynamic(replace(base, jobs=(base.jobs[0], changed, extra)),
                     [release(9.0), release(12.0, "J3"), failure(8.0, "M2", 7.0)])
    envs = [ResidualStressFJSPEnv(i, observation_profile=state_profile,
                                action_feature_profile=action_profile) for i in (first, second)]
    assert_same_inputs(*envs)
    for env in envs:
        env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
    assert_same_inputs(*envs)  # dwell state
    for env in envs:
        env.step(choose(env, "WAIT"))
    assert envs[0].current_time == envs[1].current_time == 4.0
    assert_same_inputs(*envs)  # immediately before hidden releases


@pytest.mark.parametrize("events", [[failure(0.5)], [failure(0.8, "M2", 8.0)], []])
def test_wait_presence_and_horizon_do_not_reveal_private_queue(events):
    base = tiny_instance(jobs=2, compensation_threshold=10.0)
    left = ResidualStressFJSPEnv(dynamic(base, []))
    right = ResidualStressFJSPEnv(dynamic(base, events))
    assert_same_inputs(left, right)
    for env in (left, right):
        env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
    assert left.current_time == right.current_time == 0.0
    assert_same_inputs(left, right)  # public completion at 1; failure may be earlier
    if events:
        _, _, _, _, info = right.step(choose(right, "WAIT"))
        assert right.current_time == events[0]["time"]
        assert info["selected_action"].duration == 1.0


def test_hidden_routes_do_not_change_failure_ratio_or_graph_during_repair():
    base = tiny_instance(jobs=3, compensation_threshold=10.0)
    hidden = replace(base.jobs[2], operations=tuple(
        replace(op, machines={"C": 100.0}, deformation_scale=100.0,
                deformation_rate=100.0) for op in base.jobs[2].operations
    ))
    envs = [ResidualStressFJSPEnv(dynamic(i, [release(job_id="J3"), failure(0.5)])) for i in
            (base, replace(base, jobs=(*base.jobs[:2], hidden)))]
    for env in envs:
        env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
        env.step(choose(env, "WAIT"))
    assert envs[0].current_time == envs[1].current_time == 0.5
    assert_same_inputs(*envs)
    state = dict(zip(envs[0].OBSERVATION_NAMES, envs[0].get_observation()))
    assert state["failed_production_machine_ratio"] == 0.5


def test_release_updates_scales_and_graph_and_reset_hides_them_again():
    base = tiny_instance(jobs=2, compensation_threshold=10.0)
    hidden = replace(base.jobs[1], operations=tuple(
        replace(op, machines={"M2": 100.0}, deformation_scale=100.0,
                deformation_rate=100.0) for op in base.jobs[1].operations
    ))
    instance = dynamic(replace(base, jobs=(base.jobs[0], hidden)), [release(2.0)])
    env = ResidualStressFJSPEnv(instance)
    before = env.get_action_features()
    initial_scale = env._duration_scale
    while "J2" not in env.jobs:
        env.step(choose(env, "WAIT"))
    assert env.current_time == 2.0
    assert "J2" in env._graph_job_nodes
    assert env._duration_scale == 100.0
    assert any(a.job_id == "J2" for a in env.get_valid_actions())
    env.reset(seed=7)
    assert "J2" not in env._graph_job_nodes
    assert env._duration_scale == initial_scale
    np.testing.assert_array_equal(before, env.get_action_features())


def test_wait_without_events_advances_and_compensation_does_not_starve():
    env = ResidualStressFJSPEnv(dynamic(tiny_instance(compensation_threshold=0.4), []))
    duration = env.get_valid_actions()[choose(env, "WAIT")].duration
    assert duration > 0
    env.step(choose(env, "WAIT"))
    assert env.current_time == duration
    env.step(choose(env, "NORMAL", machine="M1"))
    env.step(choose(env, "WAIT"))  # natural release, mandatory compensation
    assert any(a.kind.value == "COMPENSATION" for a in env.get_valid_actions())
    assert all(a.kind.value != "WAIT" for a in env.get_valid_actions())


def test_hidden_event_does_not_add_wait_when_only_compensation_is_useful():
    base = tiny_instance(jobs=2, compensation_threshold=0.4)
    envs = [ResidualStressFJSPEnv(dynamic(base, events)) for events in
            ([], [failure(10.0)])]
    for env in envs:
        # Dispatch both jobs, wait for completion and then natural readiness.
        env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
        env.step(choose(env, "NORMAL", job_id="J2", machine="M2"))
        while any(a.kind.value == "WAIT" for a in env.get_valid_actions()):
            env.step(choose(env, "WAIT"))
    assert_same_inputs(*envs)
    assert all(a.kind.value == "COMPENSATION" for a in envs[0].get_valid_actions())


@pytest.mark.parametrize("policy", list(BASELINES))
@pytest.mark.parametrize("events", [[], [release(2.0)], [failure(0.5)],
                                    [release(2.0), failure(0.5)]])
def test_rule_policies_complete_all_dynamic_scenarios(policy, events):
    metrics, _ = run_baseline(ResidualStressFJSPEnv(dynamic(tiny_instance(jobs=2), events)), policy, seed=7)
    assert metrics["terminated"] and not metrics["truncated"]
    assert metrics["schedule_violation_count"] == 0


def test_checkpoint_contract_allows_static_use_warns_transfer_and_rejects_old_dynamic_resume():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        check_dynamic_input_semantics({}, dynamic=False, resume=True)
        assert not recorded
    with pytest.warns(UserWarning, match="historical dynamic scores"):
        check_dynamic_input_semantics({}, dynamic=True)
    with pytest.raises(ValueError, match="initialize_from"):
        check_dynamic_input_semantics({}, dynamic=True, resume=True)
    check_dynamic_input_semantics({"dynamic_input_semantics_version": 2}, dynamic=True, resume=True)


@pytest.mark.parametrize("model_type", ["mlp", "gnn_attention"])
def test_dynamic_training_and_new_checkpoint_resume_on_cpu(tmp_path, model_type):
    import json
    from dataclasses import asdict

    source = tmp_path / "instance.json"
    # The serialized fixture must use the public loader's id schema.
    raw = asdict(dynamic(tiny_instance(jobs=2), [release(2.0), failure(0.5)]))
    raw["jobs"] = [{"id": j.job_id, "operations": [
        {"id": op.operation_id, "machines": op.machines,
         "deformation_scale": op.deformation_scale, "deformation_rate": op.deformation_rate}
        for op in j.operations]} for j in tiny_instance(jobs=2).jobs]
    source.write_text(json.dumps(raw))
    config = {"train_instance": str(source), "eval_instance": str(source),
              "device": "cpu", "output_dir": str(tmp_path / "run"),
              "episodes": 2, "episodes_per_update": 1, "validation_interval": 1,
              "latest_checkpoint_interval": 1, "checkpoint_interval": 2,
              "mini_batch_size": 64, "ppo_epochs": 1, "hidden_dim": 8,
              "model_type": model_type, "enable_plots": False,
              "observation_profile": "unified23"}
    train(config)
    path = tmp_path / "run/checkpoints/latest.pt"
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["environment_semantics_version"] == 7
    assert ckpt["dynamic_input_semantics_version"] == 2
    config["episodes"] = 3
    _, history, _ = train(config, resume_from=path)
    assert len(history) == 3
    # Legacy dynamic checkpoints cannot silently append a different protocol.
    ckpt.pop("dynamic_input_semantics_version")
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="initialize_from"):
        train(config, resume_from=path)
    config["output_dir"] = str(tmp_path / "fresh_transfer")
    config["episodes"] = 1
    with pytest.warns(UserWarning, match="historical dynamic scores"):
        train(config, initialize_from=path)
