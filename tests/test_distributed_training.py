from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import build_actor_critic
from ppo_rs.train import _episode_order, _rank_zero_phase, _run_directory, _validate_transfer_schema, train


def _training_worker(rank: int, rendezvous: str, output: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        config = {
            "train_instance": "data/examples/synthetic_train.json",
            "eval_instance": "data/examples/synthetic_test.json",
            "device": "cpu", "distributed_world_size": 2,
            "output_dir": output, "output_dir_mode": "unique",
            "episodes": 4, "episodes_per_update": 2,
            "validation_interval": 2, "latest_checkpoint_interval": 2,
            "checkpoint_interval": 2, "mini_batch_size": 64,
            "learning_rate": 3e-4, "final_learning_rate": 5e-5,
            "ppo_epochs": 1, "hidden_dim": 8,
            "observation_profile": "unified23", "enable_plots": False,
        }
        model, history, evaluation = train(config)
        vectors = [None, None]
        dist.all_gather_object(vectors, torch.cat([p.detach().flatten() for p in model.parameters()]))
        assert torch.equal(vectors[0], vectors[1])
        assert len(history) == 4
        assert [int(row["episode"]) for row in history] == [1, 2, 3, 4]
        assert [row["collection_rank"] for row in history] == [0, 1, 0, 1]
        assert evaluation["terminated"] and not evaluation["truncated"]
        runs = [path for path in Path(output).iterdir() if path.is_dir()]
        assert len(runs) == 1
        run = runs[0]
        checkpoint = torch.load(run / "checkpoints/latest.pt", weights_only=True)
        assert checkpoint["episode"] == 4
        assert checkpoint["observation_dim"] == 23
        assert checkpoint["action_feature_dim"] == 21
        assert len(checkpoint["distributed_random_states"]) == 2
        assert checkpoint["distributed_world_size"] == 2
        assert checkpoint["config"]["global_mini_batch_size"] == 64
        assert checkpoint["config"]["per_rank_mini_batch_size"] == 32
        assert checkpoint["total_environment_steps"] == sum(row["episode_steps"] for row in history)
        updates = [row for row in history if row["policy_loss"] is not None]
        assert len(updates) == 2
        assert updates[0]["learning_rate"] == pytest.approx(3e-4)
        assert updates[-1]["learning_rate"] == pytest.approx(5e-5)

        # Resuming reuses the existing run and restores every rank's RNG state.
        config["episodes"] = 6
        _, resumed_history, _ = train(config, resume_from=run / "checkpoints/latest.pt")
        assert len(resumed_history) == 6
        assert len([path for path in Path(output).iterdir() if path.is_dir()]) == 1
        with pytest.raises(RuntimeError, match="intentional root failure"):
            with _rank_zero_phase():
                if rank == 0:
                    raise ValueError("intentional root failure")
        if rank == 0:
            (Path(output) / "distributed_test_result.json").write_text(json.dumps({"passed": True}))
    finally:
        dist.destroy_process_group()


def test_two_rank_training_collects_global_budget_and_saves_one_run(tmp_path: Path) -> None:
    context = mp.spawn(
        _training_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "runs")),
        nprocs=2, join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("distributed training exceeded test deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert json.loads((tmp_path / "runs/distributed_test_result.json").read_text())["passed"]


def test_unique_run_directory_and_resume_location(tmp_path: Path) -> None:
    config = {"output_dir": str(tmp_path), "output_dir_mode": "unique", "seed": 7}
    first = _run_directory(config, None)
    second = _run_directory(config, None)
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert _run_directory(config, first / "checkpoints/latest.pt") == first
    assert _run_directory(config, first / "model.pt") == first


def test_static_dynamic_schema_stays_zero_during_complete_rollout() -> None:
    env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
    observation, info = env.reset(seed=7)
    done = False
    while not done:
        assert len(observation) == 23
        assert observation[-4:] == [0.0] * 4
        graph = env.get_graph_state(info["valid_actions"])
        assert graph["node_features"].shape[1] == 13
        assert (graph["node_features"][:, -2:] == 0).all()
        observation, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
    assert terminated and not truncated


def test_episode_plan_is_prefix_stable_for_resume() -> None:
    assert _episode_order(5, 17, 7) == _episode_order(5, 29, 7)[:17]


@pytest.mark.parametrize("invalid_field", ["environment_semantics_version", "observation_profile", "model_kwargs"])
def test_transfer_rejects_incompatible_checkpoint_schema(invalid_field: str) -> None:
    env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
    model = build_actor_critic(
        "gnn_attention", env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8,
        node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
        edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
        node_type_count=env.GRAPH_NODE_TYPE_COUNT,
        edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
    )
    checkpoint = {
        "environment_semantics_version": env.ENVIRONMENT_SEMANTICS_VERSION,
        "observation_profile": "unified23", "action_feature_profile": "compact21",
        "model_kwargs": model.checkpoint_kwargs,
    }
    _validate_transfer_schema(checkpoint, model, "unified23", "compact21")
    invalid_values = {
        "environment_semantics_version": 1,
        "observation_profile": "enhanced19",
        "model_kwargs": {**model.checkpoint_kwargs, "attention_heads": 2},
    }
    checkpoint[invalid_field] = invalid_values[invalid_field]
    with pytest.raises(ValueError):
        _validate_transfer_schema(checkpoint, model, "unified23", "compact21")
