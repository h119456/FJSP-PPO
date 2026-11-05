from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic
from scripts import train_state_comparison as launcher


def _templates(tmp_path: Path) -> Path:
    target = tmp_path / "templates"
    target.mkdir()
    config = json.loads(Path("configs/dataset2.json").read_text())
    for group in ("base", "full"):
        for model in launcher.MODELS:
            for phase in ("static", "dynamic"):
                row = deepcopy(config)
                row.update(
                    model_type="mlp" if model == "mlp" else "gnn_attention",
                    observation_profile="compact18" if group == "base" else "unified23",
                    action_feature_profile="compact21" if group == "base" else "enhanced25",
                )
                (target / f"{group}_{model}_{phase}.json").write_text(json.dumps(row))
    return target


def _configs(tmp_path: Path, **kwargs) -> dict:
    return launcher.build_run_configs(
        group="base", phase="static", config_root=_templates(tmp_path),
        output_root=tmp_path / "outputs", **kwargs,
    )


def test_build_configs_keeps_templates_and_pairs_settings(tmp_path: Path) -> None:
    templates = _templates(tmp_path)
    before = {path.name: path.read_bytes() for path in templates.iterdir()}
    configs = launcher.build_run_configs(
        group="base", phase="static", dataset="dataset3_cost0", seed=17,
        world_size=2, device="cpu", output_root=tmp_path / "out", config_root=templates,
    )
    assert set(configs) == {"mlp", "gnn"}
    for label, config in configs.items():
        assert config["train_instances"].endswith("data/dataset3_cost0/static/train")
        assert config["output_dir"].endswith(f"dataset3_cost0/base/static/{label}/seed17")
        assert config["output_dir_mode"] == "unique"
        assert config["seed"] == 17
        assert config["distributed_world_size"] == 2
        assert config["device"] == "cpu"
        assert "initialize_from" not in config
        assert config["mini_batch_size"] == 64
    assert before == {path.name: path.read_bytes() for path in templates.iterdir()}


def test_dynamic_requires_matching_static_source_and_never_silently_selects(tmp_path: Path) -> None:
    templates = _templates(tmp_path)
    options = dict(group="base", phase="dynamic", model="mlp", config_root=templates,
                   output_root=tmp_path / "outputs")
    with pytest.raises(FileNotFoundError, match="train the matching static group first"):
        launcher.build_run_configs(**options)
    parent = tmp_path / "outputs/dataset2/base/static/mlp/seed7"
    checkpoint = parent / "first/checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = launcher.build_run_configs(**options)["mlp"]
    assert config["initialize_from"] == str(checkpoint)
    second = parent / "second/checkpoints/best.pt"
    second.parent.mkdir(parents=True)
    second.touch()
    with pytest.raises(ValueError, match="choose explicitly"):
        launcher.build_run_configs(**options)
    config = launcher.build_run_configs(**options, initialize_from={"mlp": second})["mlp"]
    assert config["initialize_from"] == str(second)
    with pytest.raises(ValueError, match="selected models"):
        launcher.build_run_configs(**options, initialize_from={"gnn": second})


def test_static_run_directory_and_argument_validation(tmp_path: Path) -> None:
    templates = _templates(tmp_path)
    run = tmp_path / "static_run"
    checkpoint = run / "checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = launcher.build_run_configs(
        group="full", phase="dynamic", model="gnn", config_root=templates,
        output_root=tmp_path / "out", static_runs={"gnn": run},
    )["gnn"]
    assert config["initialize_from"] == str(checkpoint)
    with pytest.raises(ValueError, match="dynamic-only"):
        launcher.build_run_configs(group="base", phase="static", model="mlp",
                                   config_root=templates, initialize_from={"mlp": checkpoint})
    with pytest.raises(ValueError, match="positive integer"):
        launcher.build_run_configs(group="base", phase="static", world_size=0)
    for raw in (["invalid"], ["bad=a"], ["mlp=a", "mlp=b"]):
        with pytest.raises(ValueError):
            launcher._model_paths(raw, "--initialize-from")


def test_single_model_preflight_validates_own_checkpoint_without_peer(tmp_path: Path) -> None:
    # A real shape/weight check is necessary: a path can exist but refer to the
    # other state group, which must be rejected before launching any process.
    instance = "data/examples/synthetic_train.json"
    env = ResidualStressFJSPEnv(load_instance(instance), observation_profile="unified23",
                              action_feature_profile="enhanced25")
    model = build_actor_critic("mlp", env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8)
    source = tmp_path / "source.pt"
    checkpoint = {
        "environment_semantics_version": env.ENVIRONMENT_SEMANTICS_VERSION,
        "observation_profile": "unified23", "action_feature_profile": "enhanced25",
        "observation_dim": 23, "action_feature_dim": 25, "hidden_dim": 8,
        "model_type": "mlp", "model_kwargs": {}, "model_state": model.state_dict(), "episode": 10,
    }
    torch.save(checkpoint, source)
    config = {
        "model_type": "mlp", "observation_profile": "unified23", "action_feature_profile": "enhanced25",
        "train_instance": instance, "eval_instance": "data/examples/synthetic_test.json",
        "test_instances": "data/examples/synthetic_validation.json", "hidden_dim": 8,
        "episodes": 10, "episodes_per_update": 10, "validation_interval": 10,
        "checkpoint_interval": 10, "distributed_world_size": 1,
        "initialize_from": str(source), "output_dir_mode": "unique",
    }
    # Use independent copies with real processing differences for split checks.
    original = json.loads(Path(instance).read_text())
    for index, key in enumerate(("train_instance", "eval_instance", "test_instances")):
        row = deepcopy(original)
        row["jobs"][0]["operations"][0]["machines"]["M1"] += index * 0.1
        path = tmp_path / f"split{index}.json"
        path.write_text(json.dumps(row))
        config[key] = str(path)
    result = launcher.audit_run_configs({"mlp": config})
    assert result["passed"], result["errors"]
    assert set(result["models"]) == {"mlp"}
    checkpoint["observation_profile"] = "compact18"
    torch.save(checkpoint, source)
    result = launcher.audit_run_configs({"mlp": config})
    assert not result["passed"]


def test_check_mode_never_launches_or_requires_cuda(tmp_path: Path, monkeypatch, capsys) -> None:
    config = {"mlp": {"output_dir": str(tmp_path / "unused")}}
    monkeypatch.setattr(launcher, "build_run_configs", lambda **kwargs: config)
    monkeypatch.setattr(launcher, "audit_run_configs", lambda values: {"passed": True, "errors": []})

    def unexpected(*args, **kwargs):
        raise AssertionError("check-only must not inspect CUDA or launch a subprocess")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    assert launcher.main(["--group", "base", "--phase", "static", "--check"]) == 0
    assert json.loads(capsys.readouterr().out)["check_only"]
    assert not list(tmp_path.iterdir())
    command = launcher.training_command(tmp_path / "config.json", 8)
    assert "--nproc_per_node=8" in command
    assert command[-2:] == ["--config", str(tmp_path / "config.json")]
