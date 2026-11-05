from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_comparison import _constraints, effective_config


CONFIG_ROOT = Path("configs/state_ablation")


def _config(group: str, model: str, phase: str) -> dict:
    return json.loads((CONFIG_ROOT / f"{group}_{model}_{phase}.json").read_text())


@pytest.mark.parametrize("phase", ["static", "dynamic"])
def test_state_ablation_templates_share_training_protocol(phase: str) -> None:
    configs = {
        (group, model): effective_config(_config(group, model, phase))
        for group in ("base", "full")
        for model in ("mlp", "gnn")
    }
    allowed_differences = {
        "model_type", "observation_profile", "action_feature_profile", "output_dir",
    }
    reference = configs[("base", "mlp")]
    for (group, model), config in configs.items():
        assert _constraints(config) == []
        assert {
            key: value for key, value in config.items() if key not in allowed_differences
        } == {
            key: value for key, value in reference.items() if key not in allowed_differences
        }
        assert config["observation_profile"] == ("compact18" if group == "base" else "unified23")
        assert config["action_feature_profile"] == ("compact21" if group == "base" else "enhanced25")
        assert config["model_type"] == ("mlp" if model == "mlp" else "gnn_attention")
        assert config["episodes"] == (20000 if phase == "static" else 10000)
        assert config["distributed_world_size"] == 8
        assert config["mini_batch_size"] == 64
        assert config["episodes_per_update"] == 10
        assert config["ppo_epochs"] == 4
        assert config["output_dir_mode"] == "unique"
        assert config["output_dir"] == f"outputs/state_ablation/dataset2/{group}/{phase}/{model}/seed7"
        assert "initialize_from" not in config  # The launcher resolves an explicit or matching static run.
        for split, key in (("train", "train_instances"), ("validation", "eval_instances"), ("test", "test_instances")):
            assert config[key] == f"data/dataset2/{phase}/{split}"
            assert Path(config[key]).is_dir()


@pytest.mark.parametrize("group", ["base", "full"])
@pytest.mark.parametrize("model", ["mlp", "gnn"])
def test_static_and_dynamic_templates_keep_transfer_schema(group: str, model: str) -> None:
    static = _config(group, model, "static")
    dynamic = _config(group, model, "dynamic")
    for key in ("model_type", "observation_profile", "action_feature_profile", "hidden_dim", "gnn_layers", "attention_heads"):
        assert static.get(key) == dynamic.get(key)
    assert dynamic["learning_rate"] == 1e-4
    assert dynamic["final_learning_rate"] == 2e-5
    assert dynamic["entropy_coef"] == .01
    assert dynamic["final_entropy_coef"] == .003
