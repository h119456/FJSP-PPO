from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ppo_rs.algorithms import PPO, PPOConfig, RolloutBuffer, generalized_advantage_estimate
from ppo_rs.models import ActorCritic


def _buffer() -> tuple[ActorCritic, RolloutBuffer]:
    torch.manual_seed(17)
    model = ActorCritic(3, 2, hidden_dim=8)
    buffer = RolloutBuffer()
    for index in range(7):
        state = [index / 7, -0.3, 0.2]
        features = [[0.2, index / 8], [0.4, -0.1]]
        if index % 2:
            features.append([-0.2, 0.1])
        choice, log_prob, value = model.act(state, features, deterministic=True)
        buffer.add(state, features, choice, log_prob, value, index / 5 - 1, index in (2, 6))
    return model, buffer


def test_rollout_host_padding_preserves_exact_values_and_episode_gae():
    _, buffer = _buffer()
    graph_states = [None] * len(buffer)
    buffer.graph_states = graph_states
    data = buffer.tensors(device=torch.device("cpu"), gamma=0.97, gae_lambda=0.91)
    expected_candidates = torch.zeros((7, 3, 2), dtype=torch.float32)
    expected_masks = torch.zeros((7, 3), dtype=torch.bool)
    for index, candidates in enumerate(buffer.action_features):
        expected_candidates[index, :len(candidates)] = torch.tensor(candidates)
        expected_masks[index, :len(candidates)] = True
    advantages, returns = generalized_advantage_estimate(
        buffer.rewards, buffer.values, buffer.dones,
        last_value=0.0, gamma=0.97, gae_lambda=0.91,
    )
    expected = {
        "observations": torch.tensor(buffer.observations, dtype=torch.float32),
        "action_features": expected_candidates,
        "action_masks": expected_masks,
        "choices": torch.tensor(buffer.choices, dtype=torch.long),
        "old_log_probs": torch.tensor(buffer.old_log_probs, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
    }
    assert set(data) == set(expected) | {"graph_states"}
    for key, value in expected.items():
        assert torch.equal(data[key], value), key
        assert data[key].dtype == value.dtype, key
    assert data["graph_states"] is graph_states
    # Tensorization owns its storage; later trajectory mutation cannot alter it.
    buffer.action_features[0][0][0] = 999.0
    assert data["action_features"][0, 0, 0] == pytest.approx(0.2)


def test_ppo_report_averages_minibatches_including_uneven_last_batch(monkeypatch):
    model, buffer = _buffer()
    algorithm = PPO(model, PPOConfig(ppo_epochs=2, mini_batch_size=3))
    records = []

    def before_step(*args, **kwargs):
        # Capture the actual scalar losses at every optimizer step independently
        # of the new on-device reporting accumulator.
        records.append(tuple(float(value) for value in latest_metrics))

    data = buffer.tensors(device=model.device, gamma=1.0, gae_lambda=0.95)
    advantages = data["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    original_evaluate = model.evaluate_actions
    original_randperm = torch.randperm
    selected = []
    latest_metrics = []

    def permutation(*args, **kwargs):
        result = original_randperm(*args, **kwargs)
        selected[:] = result.tolist()
        return result

    def evaluate(observations, features, masks, choices, **kwargs):
        nonlocal latest_metrics
        indices = selected[:len(choices)]
        del selected[:len(choices)]
        log_probs, entropy, values = original_evaluate(observations, features, masks, choices, **kwargs)
        with torch.no_grad():
            ratio = (log_probs - data["old_log_probs"][indices]).exp()
            policy = -torch.minimum(
                ratio * advantages[indices],
                ratio.clamp(0.8, 1.2) * advantages[indices],
            ).mean()
            value = (values - data["returns"][indices]).square().mean()
            latest_metrics = [policy, value, entropy.mean()]
        return log_probs, entropy, values

    monkeypatch.setattr(torch, "randperm", permutation)
    monkeypatch.setattr(model, "evaluate_actions", evaluate)
    hook = algorithm.optimizer.register_step_pre_hook(before_step)
    try:
        actual = algorithm.update(buffer)
    finally:
        hook.remove()
    assert len(records) == 6  # 3, 3, 1 samples per epoch, for two epochs.
    expected = [sum(values) / len(records) for values in zip(*records)]
    for key, value in zip(("policy_loss", "value_loss", "entropy"), expected):
        assert actual[key] == pytest.approx(value, rel=1e-6, abs=1e-8)


def _reduction_worker(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2,
        init_method=f"file://{directory}/rendezvous",
        timeout=timedelta(seconds=30),
    )
    original_reduce = dist.all_reduce
    try:
        model, buffer = _buffer()
        algorithm = PPO(model, PPOConfig(ppo_epochs=2, mini_batch_size=3))
        reductions = []

        def reduce(tensor, *args, **kwargs):
            reductions.append(tensor.numel())
            return original_reduce(tensor, *args, **kwargs)

        dist.all_reduce = reduce
        result = algorithm.update(buffer)
        assert reductions == [3]  # DDP gradient collectives are handled internally.
        torch.save(result, Path(directory) / f"metrics{rank}.pt")
    finally:
        dist.all_reduce = original_reduce
        dist.destroy_process_group()


def test_ddp_aggregates_log_metrics_once_per_rollout(tmp_path):
    context = mp.spawn(_reduction_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("metric reduction workers exceeded test deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    first, second = [torch.load(tmp_path / f"metrics{rank}.pt", weights_only=True) for rank in (0, 1)]
    assert first == second
