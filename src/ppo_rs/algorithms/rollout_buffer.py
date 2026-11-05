"""On-policy rollout storage supporting variable candidate counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .gae import generalized_advantage_estimate


@dataclass
class RolloutBuffer:
    observations: list[list[float]] = field(default_factory=list)
    action_features: list[list[list[float]]] = field(default_factory=list)
    choices: list[int] = field(default_factory=list)
    old_log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    graph_states: list[dict[str, Any] | None] = field(default_factory=list)

    def add(
        self,
        observation: list[float],
        action_features: list[list[float]],
        choice: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        graph_state: dict[str, Any] | None = None,
    ) -> None:
        self.observations.append(list(observation))
        self.action_features.append([list(row) for row in action_features])
        self.choices.append(int(choice))
        self.old_log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.graph_states.append(graph_state)

    def extend(self, other: "RolloutBuffer") -> None:
        """Append complete trajectories collected with the same policy version."""

        self.observations.extend(other.observations)
        self.action_features.extend(other.action_features)
        self.choices.extend(other.choices)
        self.old_log_probs.extend(other.old_log_probs)
        self.values.extend(other.values)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.graph_states.extend(other.graph_states)

    def __len__(self) -> int:
        return len(self.rewards)

    def tensors(
        self,
        *,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if not self.observations:
            raise ValueError("cannot tensorize an empty rollout")
        advantages, returns = generalized_advantage_estimate(
            self.rewards,
            self.values,
            self.dones,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        max_actions = max(len(rows) for rows in self.action_features)
        action_dim = len(self.action_features[0][0])
        # Build variable-length candidates once on the host. Writing one small
        # device tensor per decision otherwise launches thousands of transfers
        # for every rollout, even though all inputs already live on the CPU.
        padded = np.zeros((len(self), max_actions, action_dim), dtype=np.float32)
        masks = np.zeros((len(self), max_actions), dtype=np.bool_)
        for index, rows in enumerate(self.action_features):
            count = len(rows)
            if count:
                padded[index, :count] = rows
            masks[index, :count] = True
        return {
            "observations": torch.tensor(self.observations, dtype=torch.float32, device=device),
            "action_features": torch.from_numpy(padded).to(device),
            "action_masks": torch.from_numpy(masks).to(device),
            "choices": torch.tensor(self.choices, dtype=torch.long, device=device),
            "old_log_probs": torch.tensor(self.old_log_probs, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "graph_states": self.graph_states,
        }

