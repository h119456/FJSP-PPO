"""Generalized advantage estimation."""

from __future__ import annotations

from collections.abc import Sequence


def generalized_advantage_estimate(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute detached scalar advantages and value targets."""

    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have equal lengths")
    advantages = [0.0] * len(rewards)
    running = 0.0
    next_value = float(last_value)
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
        next_value = values[index]
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns

