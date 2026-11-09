"""Clipped PPO implementation for variable-size scheduling decisions."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ppo_rs.environment.fjsp_env import ResidualStressFJSPEnv
from ppo_rs.models import batch_graph_states
from ppo_rs.timing import EpisodeTimer

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


class _ActionEvaluation(nn.Module):
    """Expose the shared policy evaluation through DDP's forward entry point."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        observations: torch.Tensor,
        action_features: torch.Tensor,
        action_masks: torch.Tensor,
        choices: torch.Tensor,
        *,
        graph_batch=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.evaluate_actions(
            observations, action_features, action_masks, choices,
            graph_batch=graph_batch,
        )


class PPO:
    def __init__(self, model: nn.Module, config: PPOConfig) -> None:
        self.model = model
        self.config = config
        self.entropy_coef = config.entropy_coef
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._distributed_evaluator = None
        if self.world_size > 1:
            # The original model remains available for local rollout collection
            # and unprefixed checkpoints. Only differentiable PPO evaluation
            # goes through DDP so its gradient synchronization hooks are used.
            device = model.device
            device_ids = [device.index] if device.type == "cuda" else None
            self._distributed_evaluator = DistributedDataParallel(
                _ActionEvaluation(model),
                device_ids=device_ids,
                find_unused_parameters=bool(getattr(model, "requires_graph", False)),
            )
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
        """Update using a global rollout replicated identically on every rank.

        ``mini_batch_size`` is the global sample count, rather than a per-rank
        size. Padding on uneven or final batches contributes zero loss; its
        gradients therefore equal the single-process global batch mean.
        """

        data = buffer.tensors(
            device=self.model.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        advantages = data["advantages"]
        if len(buffer) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        sample_count = len(buffer)
        # Metrics do not affect gradients. Keep their sums on device and reduce
        # once per rollout instead of synchronizing every mini-batch for logs.
        totals = torch.zeros(3, dtype=torch.float64, device=self.model.device)
        update_count = 0
        requires_graph = bool(getattr(self.model, "requires_graph", False))
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                permutation = torch.randperm(sample_count, device=self.model.device)
            else:
                permutation = torch.empty(sample_count, dtype=torch.long, device=self.model.device)
            if self.world_size > 1:
                dist.broadcast(permutation, src=0)
            # Python graph objects need host indices. Copy the permutation
            # once per epoch rather than waiting on CUDA for each mini-batch.
            graph_permutation = permutation.detach().cpu().tolist() if requires_graph else None
            for start in range(0, sample_count, self.config.mini_batch_size):
                indices = permutation[start : start + self.config.mini_batch_size]
                global_count = indices.numel()
                valid_samples = None
                if self.world_size > 1:
                    local_count = (global_count + self.world_size - 1) // self.world_size
                    positions = torch.arange(
                        self.rank * local_count, (self.rank + 1) * local_count,
                        device=self.model.device,
                    )
                    valid_samples = positions < global_count
                    # Every rank executes a real forward/backward, even if a
                    # tiny final batch has fewer samples than there are GPUs.
                    indices = indices[positions.remainder(global_count)]
                graph_batch = None
                if graph_permutation is not None:
                    graph_indices = graph_permutation[start : start + global_count]
                    if self.world_size > 1:
                        graph_indices = [
                            graph_indices[position % global_count]
                            for position in range(self.rank * local_count, (self.rank + 1) * local_count)
                        ]
                    selected_graphs = [
                        data["graph_states"][index]
                        for index in graph_indices
                    ]
                    if any(graph is None for graph in selected_graphs):
                        raise ValueError("GNN rollout is missing graph states")
                    graph_batch = batch_graph_states(
                        selected_graphs,
                        action_slots=data["action_features"].shape[1],
                        device=self.model.device,
                    )
                evaluator = self._distributed_evaluator or self.model.evaluate_actions
                log_probs, entropy, values = evaluator(
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
                policy_samples = -torch.minimum(raw, clipped)
                value_samples = nn.functional.mse_loss(
                    values, data["returns"][indices], reduction="none"
                )
                if valid_samples is None:
                    policy_loss = policy_samples.mean()
                    value_loss = value_samples.mean()
                    entropy_mean = entropy.mean()
                else:
                    weight = valid_samples.to(policy_samples.dtype)
                    # DDP averages gradients over world_size ranks. Scaling
                    # each masked local sum compensates for that averaging.
                    scale = self.world_size / global_count
                    policy_loss = (policy_samples * weight).sum() * scale
                    value_loss = (value_samples * weight).sum() * scale
                    entropy_mean = (entropy * weight).sum() * scale
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                metrics = torch.stack([
                    policy_loss.detach(), value_loss.detach(), entropy_mean.detach()
                ])
                totals.add_(metrics.to(dtype=totals.dtype))
                update_count += 1
        if self.world_size > 1:
            dist.all_reduce(totals)
            totals /= self.world_size
        values = (totals / max(1, update_count)).tolist()
        return dict(zip(("policy_loss", "value_loss", "entropy"), values))


def collect_episode(
    env: ResidualStressFJSPEnv,
    model: nn.Module,
    *,
    seed: int | None = None,
    deterministic: bool = False,
) -> tuple[RolloutBuffer, dict[str, Any]]:
    """Collect one complete scheduling trajectory using the current policy."""

    timer = EpisodeTimer(getattr(model, "device", None))
    observation, info = env.reset(seed=seed)
    buffer = RolloutBuffer()
    terminated = truncated = False
    while not (terminated or truncated):
        actions = info["valid_actions"]
        features = info["action_features"]
        if not actions:
            observation, _, terminated, truncated, info = env.step(0)
            continue
        graph_state = None
        if bool(getattr(model, "requires_graph", False)):
            graph_started = time.perf_counter()
            graph_state = env.get_graph_state(
                actions, **({"include_physics": True} if getattr(model, "physical_gating", False) else {}),
            )
            timer.graph_construction_seconds += time.perf_counter() - graph_started
        inference_started = time.perf_counter()
        choice, log_prob, value = model.act(
            observation,
            features,
            graph_state=graph_state,
            deterministic=deterministic,
        )
        timer.policy_inference_seconds += time.perf_counter() - inference_started
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
    metrics = env.get_metrics(timer.policy_inference_seconds)
    return buffer, timer.finish(metrics)
