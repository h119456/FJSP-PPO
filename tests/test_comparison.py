from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.check_comparison import audit_comparison, effective_config, main
from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic


def _paired_configs() -> tuple[dict, dict]:
    return tuple(
        json.loads(Path(f"configs/dataset_compare_{model}.json").read_text())
        for model in ("mlp", "gnn")
    )


def test_old_configs_fail_cli_with_optimizer_differences(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "check_comparison.py", "--mlp-config", "configs/dataset.json",
        "--gnn-config", "configs/dataset_gnn_pilot.json",
    ])
    assert main() == 1
    result = json.loads(capsys.readouterr().out)
    assert {"mini_batch_size", "episodes_per_update"} <= result["setting_differences"].keys()


def test_matched_pair_reports_actual_input_and_capacity():
    result = audit_comparison(*_paired_configs())
    assert result["passed"], result["errors"]
    assert result["models"]["mlp"]["observation_dim"] == 23
    assert result["models"]["gnn"]["action_feature_dim"] == 21
    assert result["models"]["gnn"]["graph_node_feature_dim"] == 13
    assert result["models"]["gnn"]["global_mini_batch_size"] == 64
    assert result["models"]["gnn"]["local_mini_batch_size"] == 8
    assert result["models"]["gnn"]["distributed_world_size"] == 8
    assert result["models"]["gnn"]["parameter_count"] > result["models"]["mlp"]["parameter_count"]
    assert result["datasets"]["mlp"]["train"]["instance_count"] == 100


def test_renamed_duplicate_across_splits_is_detected(tmp_path):
    mlp, gnn = _paired_configs()
    original = json.loads(Path("data/examples/synthetic_train.json").read_text())
    for index, split in enumerate(("train", "validation", "test")):
        directory = tmp_path / split
        directory.mkdir()
        content = deepcopy(original)
        content["name"] = f"renamed_{index}"
        (directory / "example.json").write_text(json.dumps(content))
        key = {"train": "train_instances", "validation": "eval_instances", "test": "test_instances"}[split]
        mlp[key] = gnn[key] = str(directory)
    result = audit_comparison(mlp, gnn)
    assert not result["passed"]
    assert any("dataset leakage" in error for error in result["errors"])


def test_dependent_defaults_match_training_entrypoint():
    config = effective_config({"seed": 11, "learning_rate": 0.001,
                               "entropy_coef": 0.03, "episodes_per_update": 10})
    assert config["eval_seed"] == 1011
    assert config["final_learning_rate"] == 0.001
    assert config["final_entropy_coef"] == 0.03
    assert config["validation_interval"] == config["latest_checkpoint_interval"] == 10


def test_intervals_must_land_on_update_boundaries():
    mlp, gnn = _paired_configs()
    mlp["validation_interval"] = gnn["validation_interval"] = 13
    result = audit_comparison(mlp, gnn)
    assert not result["passed"]
    assert any("validation_interval must be divisible" in error for error in result["errors"])


def test_matched_but_invalid_budget_and_clip_are_rejected():
    mlp, gnn = _paired_configs()
    for config in (mlp, gnn):
        config.update(episodes=5, episodes_per_update=10, clip_ratio=1.0)
    result = audit_comparison(mlp, gnn)
    assert not result["passed"]
    assert any("episodes must be at least" in error for error in result["errors"])
    assert any("clip_ratio must be less than 1" in error for error in result["errors"])


def _transfer_configs(tmp_path):
    configs = _paired_configs()
    env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
    for index, config in enumerate(configs):
        config["hidden_dim"] = 8
        model = build_actor_critic(
            config["model_type"], env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8,
            node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
            edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
            node_type_count=env.GRAPH_NODE_TYPE_COUNT,
            edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
        )
        path = tmp_path / f"source_{index}.pt"
        torch.save({
            "environment_semantics_version": env.ENVIRONMENT_SEMANTICS_VERSION,
            "observation_profile": config["observation_profile"],
            "action_feature_profile": config["action_feature_profile"],
            "observation_dim": env.OBSERVATION_DIM,
            "action_feature_dim": env.ACTION_FEATURE_DIM,
            "hidden_dim": 8,
            "model_type": config["model_type"],
            "model_kwargs": getattr(model, "checkpoint_kwargs", {}),
            "model_state": model.state_dict(),
            "episode": 1900 - index * 100,
        }, path)
        config["initialize_from"] = str(path)
    return configs


def test_distinct_compatible_source_checkpoints_are_allowed(tmp_path):
    result = audit_comparison(*_transfer_configs(tmp_path))
    assert result["passed"], result["errors"]
    assert result["setting_differences"] == {}
    for label, episode in (("mlp", 1900), ("gnn", 1800)):
        source = result["models"][label]["initialization"]
        assert source["compatible"]
        assert source["source_episode"] == episode
        assert len(source["sha256"]) == 64


@pytest.mark.parametrize("fault", ["wrong_model", "semantics", "profile", "graph", "missing", "weights"])
def test_transfer_preflight_rejects_unusable_sources(tmp_path, fault):
    mlp, gnn = _transfer_configs(tmp_path)
    source = Path(gnn["initialize_from"])
    if fault == "missing":
        source.unlink()
    else:
        checkpoint = torch.load(source, weights_only=True)
        if fault == "wrong_model":
            checkpoint["model_type"] = "mlp"
        elif fault == "semantics":
            checkpoint["environment_semantics_version"] = -1
        elif fault == "profile":
            checkpoint["observation_profile"] = "compact14"
        elif fault == "graph":
            checkpoint["model_kwargs"]["attention_heads"] = 2
        elif fault == "weights":
            checkpoint["model_state"].pop(next(iter(checkpoint["model_state"])))
        torch.save(checkpoint, source)
    result = audit_comparison(mlp, gnn)
    assert not result["passed"]
    assert any(error.startswith("gnn:") for error in result["errors"])


def test_transfer_pair_cannot_mix_pretraining_and_finetuning(tmp_path):
    mlp, gnn = _transfer_configs(tmp_path)
    gnn.pop("initialize_from")
    result = audit_comparison(mlp, gnn)
    assert not result["passed"]
    assert any("both models must initialize" in error for error in result["errors"])
