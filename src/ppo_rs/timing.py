"""Wall-clock measurements for one scheduling episode, in seconds."""

from __future__ import annotations

import time
from typing import Any

import torch


class EpisodeTimer:
    """Time reset through final metrics, with CUDA work bounded at episode edges.

    The model's ``act`` API returns Python scalars, so its device work already
    completes before returning. Avoid extra synchronization on every decision.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else None
        self.policy_inference_seconds = 0.0
        self.graph_construction_seconds = 0.0
        self._synchronize()
        self._started = time.perf_counter()

    def _synchronize(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def finish(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Attach timings after final metric computation has completed."""

        self._synchronize()
        metrics.update(
            episode_wall_seconds=time.perf_counter() - self._started,
            policy_inference_seconds=self.policy_inference_seconds,
            graph_construction_seconds=self.graph_construction_seconds,
        )
        return metrics
