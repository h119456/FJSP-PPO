from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic
from scripts.evaluate_dataset import main


@pytest.mark.parametrize(
    "model_type,observation_profile,action_profile,profile_location",
    [
        ("mlp", "compact14", "enhanced25", "checkpoint"),
        ("gnn_attention", "enhanced19", "enhanced25", "config"),
        ("mlp", "compact14", "compact21", "legacy"),
        ("mlp", "compact18", "compact21", "checkpoint"),
        ("gnn_attention", "compact18", "compact21", "checkpoint"),
        ("gnn_attention", "unified23", "enhanced25", "checkpoint"),
    ],
)
def test_dataset_cli_evaluates_checkpoint_input_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    observation_profile: str,
    action_profile: str,
    profile_location: str,
) -> None:
    source = "data/examples/synthetic_test.json"
    env = ResidualStressFJSPEnv(
        load_instance(source),
        observation_profile=observation_profile,
        action_feature_profile=action_profile,
    )
    model = build_actor_critic(
        model_type,
        env.OBSERVATION_DIM,
        env.ACTION_FEATURE_DIM,
        8,
        node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
        node_type_count=env.GRAPH_NODE_TYPE_COUNT,
        edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
    )
    # Uniform logits provide a deterministic, finite dispatch policy without training.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = {
        "environment_semantics_version": env.ENVIRONMENT_SEMANTICS_VERSION,
        "episode": 1,
        "total_environment_steps": 0,
        "model_type": model_type,
        "model_state": model.state_dict(),
        "model_kwargs": dict(getattr(model, "checkpoint_kwargs", {})),
        "hidden_dim": 8,
        "observation_dim": env.OBSERVATION_DIM,
        "action_feature_dim": env.ACTION_FEATURE_DIM,
    }
    profiles = {
        "observation_profile": observation_profile,
        "action_feature_profile": action_profile,
    }
    if profile_location == "checkpoint":
        checkpoint.update(profiles)
        checkpoint["config"] = {
            "observation_profile": "unified23",
            "action_feature_profile": "compact21",
        }
    elif profile_location == "config":
        checkpoint["config"] = profiles
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_dataset.py",
            "--instances", source,
            "--checkpoint", f"candidate={checkpoint_path}",
            "--output", str(output),
            "--device", "cpu",
        ],
    )

    main()

    result = json.loads(output.read_text())
    metadata = result["checkpoint_metadata"]["candidate"]
    assert metadata["model_type"] == model_type
    assert metadata["observation_profile"] == observation_profile
    assert metadata["action_feature_profile"] == action_profile
    assert metadata["observation_dim"] == env.OBSERVATION_DIM
    assert metadata["action_feature_dim"] == env.ACTION_FEATURE_DIM
    assert metadata["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert result["aggregates"]["candidate"]["valid_instance_count"] == 1
    assert result["aggregates"]["candidate"]["total_schedule_violations"] == 0
    # The CLI must persist timings for both learned policies and rule baselines.
    for name, aggregate in result["aggregates"].items():
        metrics = result["instances"][0]["policies"][name]
        assert aggregate["mean_episode_wall_seconds"] > 0
        assert aggregate["total_episode_wall_seconds"] == metrics["episode_wall_seconds"]
        for key in ("policy_inference_seconds", "graph_construction_seconds"):
            assert aggregate[f"mean_{key}"] == metrics[key]
            assert 0 <= metrics[key] <= metrics["episode_wall_seconds"]
        if name != "candidate" or model_type == "mlp":
            assert metrics["graph_construction_seconds"] == 0
