"""Learned-policy evaluation."""

from __future__ import annotations

import time
from typing import Any

from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.timing import EpisodeTimer
from torch import nn


def evaluate_model(
    env: ResidualStressFJSPEnv,
    model: nn.Module,
    *,
    seed: int = 0,
    deterministic: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timer = EpisodeTimer(getattr(model, "device", None))
    observation, info = env.reset(seed=seed)
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
        started = time.perf_counter()
        choice, _, _ = model.act(
            observation,
            features,
            graph_state=graph_state,
            deterministic=deterministic,
        )
        timer.policy_inference_seconds += time.perf_counter() - started
        observation, _, terminated, truncated, info = env.step(choice)
    metrics = timer.finish(env.get_metrics(timer.policy_inference_seconds))
    metrics["policy"] = "ppo"
    return metrics, env.get_schedule()

