from __future__ import annotations

from datetime import timedelta
import math
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ppo_rs.algorithms import PPO, PPOConfig, RolloutBuffer
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import ActorCritic, build_actor_critic
from tests.helpers import tiny_instance


def _model_and_buffer(model_type: str, sample_count: int = 7):
    torch.manual_seed(401)
    if model_type == "mlp":
        model = ActorCritic(3, 2, hidden_dim=8)
        graph = None
        observations = [[index / 10, 0.2, -0.1] for index in range(sample_count)]
        candidates = [
            [[0.1 * index, 0.2], [0.3, -0.1]] + ([[0.5, 0.7]] if index % 2 else [])
            for index in range(sample_count)
        ]
    else:
        env = ResidualStressFJSPEnv(tiny_instance(jobs=2))
        state, info = env.reset()
        gated = model_type == "gnn_physical"
        if gated:
            from tests.helpers import choose
            env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
            while env.jobs["J1"].busy:
                env.step(choose(env, "WAIT"))
            state, info = env.get_observation(), {"valid_actions":env.get_valid_actions(), "action_features":env.get_action_features()}
        graph = env.get_graph_state(info["valid_actions"], include_physics=gated)
        # Leave some relation modules unused on purpose: valid FJSP graphs
        # can lack individual process/auxiliary relations on either rank.
        keep = graph["edge_types"] < (8 if gated else 2)
        graph["edge_index"] = graph["edge_index"][:, keep]
        graph["edge_types"] = graph["edge_types"][keep]
        graph["edge_features"] = graph["edge_features"][keep]
        if gated:
            graph["physical_edge_mask"] = graph["physical_edge_mask"][keep]
        model = build_actor_critic(
            "gnn_attention", env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, 8,
            node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
            edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
            node_type_count=env.GRAPH_NODE_TYPE_COUNT,
            edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
            physical_gating=gated,
            shared_kv_film=model_type == "gnn_film",
        )
        observations = [state[:] for _ in range(sample_count)]
        candidates = [info["action_features"] for _ in range(sample_count)]
    # A shared additive logit bias cancels out of Categorical probabilities.
    # Its analytically zero gradient suffers different float32 cancellation
    # before/after all-reduce, which Adam can amplify despite identical
    # policies. Freeze this unidentifiable scalar for the parameter comparison.
    scorer = model.actor.scorer if model_type == "mlp" else model.actor_scorer
    scorer[-1].bias.requires_grad_(False)
    buffer = RolloutBuffer()
    for index, (observation, features) in enumerate(zip(observations, candidates)):
        choice, log_prob, value = model.act(
            observation, features, graph_state=graph, deterministic=True
        )
        buffer.add(
            observation, features, choice, log_prob, value,
            reward=(index - 3) * 0.05, done=index in {2, sample_count - 1}, graph_state=graph,
        )
    return model, buffer


def _update_config(batch_size: int = 3) -> PPOConfig:
    # Three global samples cannot be divided equally across two ranks, and
    # seven samples leave a final batch smaller than world_size.
    return PPOConfig(ppo_epochs=2, mini_batch_size=batch_size, learning_rate=1e-3)


def _distributed_worker(rank: int, directory: str, model_type: str,
                        world_size: int, batch_size: int, sample_count: int) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size,
        init_method=f"file://{directory}/rendezvous",
        timeout=timedelta(seconds=30),
    )
    try:
        model, buffer = _model_and_buffer(model_type, sample_count)
        algorithm = PPO(model, _update_config(batch_size))
        # Different non-root RNGs ensure the update really broadcasts rank0's
        # sample permutation rather than relying on identical worker seeds.
        torch.manual_seed(913 + rank)
        losses = algorithm.update(buffer)
        torch.save(
            {"state": model.state_dict(), "losses": losses},
            Path(directory) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_type", ["mlp", "gnn_attention", "gnn_physical", "gnn_film"])
@pytest.mark.parametrize("world_size,batch_size,sample_count", [(2, 3, 7), (3, 64, 67)])
def test_ddp_update_matches_global_batch_with_masked_padding(
    tmp_path, model_type, world_size, batch_size, sample_count,
):
    torch.set_num_threads(1)
    model, buffer = _model_and_buffer(model_type, sample_count)
    algorithm = PPO(model, _update_config(batch_size))
    torch.manual_seed(913)
    expected_losses = algorithm.update(buffer)
    context = mp.spawn(
        _distributed_worker, args=(str(tmp_path), model_type, world_size, batch_size, sample_count),
        nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("distributed PPO workers exceeded the 60 second timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    results = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(world_size)]
    for name, expected in model.state_dict().items():
        assert all(torch.equal(results[0]["state"][name], row["state"][name]) for row in results), name
        torch.testing.assert_close(
            results[0]["state"][name], expected, atol=3e-5, rtol=3e-4, msg=name,
        )
    assert all(results[0]["losses"] == row["losses"] for row in results)
    for key, expected in expected_losses.items():
        assert math.isfinite(results[0]["losses"][key])
        assert results[0]["losses"][key] == pytest.approx(expected, rel=1e-4, abs=1e-5)
