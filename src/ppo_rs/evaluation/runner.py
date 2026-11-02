"""Learned-policy evaluation."""

from __future__ import annotations

import time
from typing import Any

from ppo_rs.environment import ResidualStressFJSPEnv
from torch import nn


def evaluate_model(
    env: ResidualStressFJSPEnv,
    model: nn.Module,
    *,
    seed: int = 0,
    deterministic: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation, info = env.reset(seed=seed)
    terminated = truncated = False
    inference = 0.0
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
        started = time.perf_counter()
        choice, _, _ = model.act(
            observation,
            features,
            graph_state=graph_state,
            deterministic=deterministic,
        )
        inference += time.perf_counter() - started
        observation, _, terminated, truncated, info = env.step(choice)
    metrics = env.get_metrics(inference)
    metrics["policy"] = "ppo"
    return metrics, env.get_schedule()

