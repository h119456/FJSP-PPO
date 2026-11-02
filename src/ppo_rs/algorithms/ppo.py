"""Clipped PPO implementation for variable-size scheduling decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ppo_rs.environment.fjsp_env import ResidualStressFJSPEnv
from ppo_rs.models import batch_graph_states

from .rollout_buffer import RolloutBuffer


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5


class PPO:
    def __init__(self, model: nn.Module, config: PPOConfig) -> None:
        self.model = model
        self.config = config
        self.entropy_coef = config.entropy_coef
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_learning_rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError("learning rate must be positive")
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def set_entropy_coef(self, value: float) -> None:
        if value < 0:
            raise ValueError("entropy coefficient must be non-negative")
        self.entropy_coef = float(value)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run multi-epoch clipped policy/value updates over one rollout."""

        data = buffer.tensors(
            device=self.model.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        advantages = data["advantages"]
        if len(buffer) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        sample_count = len(buffer)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0.0}
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(sample_count, device=self.model.device)
            for start in range(0, sample_count, self.config.mini_batch_size):
                indices = permutation[start : start + self.config.mini_batch_size]
                graph_batch = None
                if bool(getattr(self.model, "requires_graph", False)):
                    selected_graphs = [
                        data["graph_states"][index]
                        for index in indices.detach().cpu().tolist()
                    ]
                    if any(graph is None for graph in selected_graphs):
                        raise ValueError("GNN rollout is missing graph states")
                    graph_batch = batch_graph_states(
                        selected_graphs,
                        action_slots=data["action_features"].shape[1],
                        device=self.model.device,
                    )
                log_probs, entropy, values = self.model.evaluate_actions(
                    data["observations"][indices],
                    data["action_features"][indices],
                    data["action_masks"][indices],
                    data["choices"][indices],
                    graph_batch=graph_batch,
                )
                ratio = torch.exp(log_probs - data["old_log_probs"][indices])
                raw = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * advantages[indices]
                policy_loss = -torch.minimum(raw, clipped).mean()
                value_loss = nn.functional.mse_loss(values, data["returns"][indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach().item())
                totals["value_loss"] += float(value_loss.detach().item())
                totals["entropy"] += float(entropy_mean.detach().item())
                totals["updates"] += 1.0
        count = max(1.0, totals.pop("updates"))
        return {key: value / count for key, value in totals.items()}


def collect_episode(
    env: ResidualStressFJSPEnv,
    model: nn.Module,
    *,
    seed: int | None = None,
    deterministic: bool = False,
) -> tuple[RolloutBuffer, dict[str, Any]]:
    """Collect one complete scheduling trajectory using the current policy."""

    observation, info = env.reset(seed=seed)
    buffer = RolloutBuffer()
    terminated = truncated = False
    while not (terminated or truncated):
        actions = info["valid_actions"]
        features = info["action_features"]
        if not actions:
            observation, _, terminated, truncated, info = env.step(0)
            continue
        graph_state = (
            env.get_graph_state(actions)
            if bool(getattr(model, "requires_graph", False))
            else None
        )
        choice, log_prob, value = model.act(
            observation,
            features,
            graph_state=graph_state,
            deterministic=deterministic,
        )
        next_observation, reward, terminated, truncated, next_info = env.step(choice)
        buffer.add(
            observation,
            features,
            choice,
            log_prob,
            value,
            reward,
            terminated or truncated,
            graph_state,
        )
        observation, info = next_observation, next_info
    return buffer, env.get_metrics()
