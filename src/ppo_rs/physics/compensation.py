"""Deformation-compensation state-transition contract."""

from __future__ import annotations

from typing import Protocol

class CompensationModel(Protocol):
    def apply(self, *, deformation: float) -> float:
        """Return post-compensation residual deformation."""


class SyntheticCompensationModel:
    """Synthetic partial compensation used only for tests and software demos."""

    def __init__(self, deformation_factor: float) -> None:
        self.deformation_factor = float(deformation_factor)

    def apply(self, *, deformation: float) -> float:
        return max(0.0, deformation * self.deformation_factor)
