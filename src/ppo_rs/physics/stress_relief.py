"""Active stress-relief state-transition contract."""

from __future__ import annotations

from typing import Protocol

class StressReliefModel(Protocol):
    def apply(self, *, deformation: float, maximum_deformation: float) -> float:
        """Return the stable deformation reached after active stress relief."""


class SyntheticStressReliefModel:
    """Drive deformation directly to the completed operation's saturation A."""

    def apply(self, *, deformation: float, maximum_deformation: float) -> float:
        del deformation
        return max(0.0, float(maximum_deformation))

