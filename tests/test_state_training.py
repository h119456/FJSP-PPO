from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.evaluation import evaluate_model
from ppo_rs.models import build_model_from_checkpoint
from ppo_rs.train import _validate_transfer_schema, train


def _small_config(group: str, model: str, phase: str, output: Path) -> dict:
    config = json.loads(Path(f"configs/state_ablation/{group}_{model}_{phase}.json").read_text())
    config.update(
        train_instances="data/examples/synthetic_train.json",
        eval_instances="data/examples/synthetic_test.json",
        test_instances="data/examples/synthetic_test.json",
        output_dir=str(output), device="cpu", distributed_world_size=1,
        hidden_dim=8, episodes=2, episodes_per_update=2, ppo_epochs=1,
        validation_interval=2, latest_checkpoint_interval=2, checkpoint_interval=2,
    )
    return config


@pytest.mark.parametrize("group,observation_dim,action_dim", [("base", 18, 21), ("full", 23, 25)])
@pytest.mark.parametrize("model", ["mlp", "gnn"])
def test_state_group_static_checkpoint_finetunes_dynamic_without_shape_change(
    tmp_path: Path, group: str, observation_dim: int, action_dim: int, model: str,
) -> None:
    static = _small_config(group, model, "static", tmp_path / "static")
    _, static_history, static_evaluation = train(static)
    assert len(static_history) == 2
    assert static_evaluation["terminated"] and not static_evaluation["truncated"]
    static_run, = (tmp_path / "static").iterdir()
    source = static_run / "checkpoints" / "best.pt"
    checkpoint = torch.load(source, weights_only=False)
    assert checkpoint["observation_dim"] == observation_dim
    assert checkpoint["action_feature_dim"] == action_dim
    assert checkpoint["observation_profile"] == static["observation_profile"]
    if model == "gnn":
        assert checkpoint["model_kwargs"]["node_feature_dim"] == 13

    # Two optimization episodes do not guarantee a competent dispatch policy.
    # Use a neutral uniform policy for the transfer smoke so this checks schema,
    # checkpoint loading and dynamic execution rather than training convergence.
    checkpoint["model_state"] = {
        key: torch.zeros_like(value) for key, value in checkpoint["model_state"].items()
    }
    source = tmp_path / "neutral_static_policy.pt"
    torch.save(checkpoint, source)

    dynamic_content = deepcopy(json.loads(Path("data/examples/synthetic_train.json").read_text()))
    dynamic_content["name"] = "static_to_dynamic_shape_smoke"
    dynamic_content["dynamic_events"] = {
        "mode": "fixed", "scenario": "machine_breakdown",
        "future_events_visible_to_agent": False, "unreleased_jobs_visible_to_agent": False,
        "events": [{
            "time": .5, "type": "machine_breakdown", "machine_id": "M1",
            "repair_duration": 1., "interruption_policy": "pause_resume",
            "known_to_agent_before_occurrence": False,
        }],
    }
    dynamic_path = tmp_path / "dynamic.json"
    dynamic_path.write_text(json.dumps(dynamic_content))
    dynamic = _small_config(group, model, "dynamic", tmp_path / "dynamic")
    dynamic.update(train_instances=str(dynamic_path), eval_instances=str(dynamic_path), test_instances=str(dynamic_path))

    env = ResidualStressFJSPEnv(
        load_instance(dynamic_path), observation_profile=dynamic["observation_profile"],
        action_feature_profile=dynamic["action_feature_profile"],
    )
    transferred = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
    _validate_transfer_schema(
        checkpoint, transferred, dynamic["observation_profile"], dynamic["action_feature_profile"],
    )
    before, _ = evaluate_model(env, transferred, seed=1007)
    assert before["terminated"] and not before["truncated"]
    assert before["schedule_violation_count"] == 0
    assert before["machine_breakdown_event_count"] == 1

    _, history, evaluation = train(dynamic, initialize_from=source)
    assert len(history) == 2
    assert [row["episode"] for row in history] == [1., 2.]  # Fine-tuning starts a fresh run.
    assert evaluation["terminated"] and not evaluation["truncated"]
    assert evaluation["schedule_violation_count"] == 0
    assert evaluation["machine_breakdown_event_count"] == 1
    dynamic_run, = (tmp_path / "dynamic").iterdir()
    saved = torch.load(dynamic_run / "checkpoints" / "final.pt", weights_only=False)
    assert saved["observation_dim"] == checkpoint["observation_dim"]
    assert saved["action_feature_dim"] == checkpoint["action_feature_dim"]
    assert saved["model_kwargs"] == checkpoint["model_kwargs"]
    assert saved["config"]["initialize_from"] == str(source.resolve())
    assert source.exists()
