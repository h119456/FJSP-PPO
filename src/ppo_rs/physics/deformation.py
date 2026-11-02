"""Deformation model contract.

The included implementation is deterministic synthetic test behaviour. It is
not a calibrated law for any real material.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence


class DeformationModel(Protocol):
    def compute(
        self,
        *,
        job_id: str,
        after_operation: str | None,
        anchor_deformation: float,
        dwell_time: float,
        deformation_scale: float,
        deformation_rate: float,
        history: Sequence[str],
    ) -> float:
        """Return predicted deformation at a non-negative dwell time."""


class ExponentialDeformationModel:
    """Per-operation saturating exponential growth reference model.

    ``D(t) = D_anchor + A * (1 - exp(-B * t))``. ``scale`` is ``A`` and
    ``rate`` is ``B``. With a zero anchor this is exactly ``A(1-exp(-Bt))``.
    Parameters are supplied by the completed operation; residual stress is not
    calculated.
    """

    def compute(
        self,
        *,
        job_id: str,
        after_operation: str | None,
        anchor_deformation: float,
        dwell_time: float,
        deformation_scale: float,
        deformation_rate: float,
        history: Sequence[str],
    ) -> float:
        del job_id, after_operation, history
        if dwell_time < 0:
            raise ValueError("dwell_time cannot be negative")
        if deformation_scale <= 0 or deformation_rate <= 0:
            raise ValueError("exponential deformation parameters must be positive")
        growth = -math.expm1(-deformation_rate * dwell_time)
        return max(0.0, anchor_deformation + deformation_scale * growth)


# Backward-compatible import name; behaviour is now exponential and receives
# per-operation parameters in ``compute``.
SyntheticIncreasingDeformation = ExponentialDeformationModel

