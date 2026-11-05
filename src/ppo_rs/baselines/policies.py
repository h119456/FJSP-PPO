"""Quality-safe rule baselines evaluated on the same physics environment."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.timing import EpisodeTimer
from ppo_rs.types import Action, ActionType


Policy = Callable[[list[Action]], int]


def _index(actions: list[Action], ordered_kinds: tuple[ActionType, ...]) -> int:
    for kind in ordered_kinds:
        choices = [(index, action) for index, action in enumerate(actions) if action.kind is kind]
        if choices:
            if kind is ActionType.NORMAL:
                return min(choices, key=lambda item: item[1].duration)[0]
            return choices[0][0]
    return 0


def ordinary_fjsp(actions: list[Action]) -> int:
    """Conventional job/machine order, wrapped by the common safety filter."""

    for index, action in enumerate(actions):
        if action.kind is ActionType.NORMAL:
            return index
    return _index(
        actions, (ActionType.WAIT, ActionType.COMPENSATION, ActionType.STRESS_RELIEF)
    )


def natural_wait(actions: list[Action]) -> int:
    """Prefer natural stress evolution whenever no normal operation is legal."""

    return _index(
        actions,
        (ActionType.NORMAL, ActionType.WAIT, ActionType.COMPENSATION, ActionType.STRESS_RELIEF),
    )


def relief_rule(actions: list[Action]) -> int:
    """Trigger stress relief whenever processing is blocked and relief is legal."""

    return _index(
        actions,
        (ActionType.NORMAL, ActionType.STRESS_RELIEF, ActionType.COMPENSATION, ActionType.WAIT),
    )


def compensation_rule(actions: list[Action]) -> int:
    """Prioritize triggered deformation compensation, then normal processing."""

    return _index(
        actions,
        (ActionType.COMPENSATION, ActionType.NORMAL, ActionType.WAIT, ActionType.STRESS_RELIEF),
    )


BASELINES: dict[str, Policy] = {
    "ordinary_fjsp": ordinary_fjsp,
    "natural_wait": natural_wait,
    "stress_relief_rule": relief_rule,
    "compensation_rule": compensation_rule,
}


def run_baseline(
    env: ResidualStressFJSPEnv, name: str, *, seed: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    policy = BASELINES[name]
    timer = EpisodeTimer()
    _, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        actions = info["valid_actions"]
        if not actions:
            _, _, terminated, truncated, info = env.step(0)
            continue
        started = time.perf_counter()
        choice = policy(actions)
        timer.policy_inference_seconds += time.perf_counter() - started
        _, _, terminated, truncated, info = env.step(choice)
    metrics = timer.finish(env.get_metrics(timer.policy_inference_seconds))
    metrics["policy"] = name
    return metrics, env.get_schedule()

