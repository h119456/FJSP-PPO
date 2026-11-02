"""State-value network."""

from __future__ import annotations

from torch import nn


class StateCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        return self.network(states).squeeze(-1)

