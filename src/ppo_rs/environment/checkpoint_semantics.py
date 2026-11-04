"""Track the dynamic policy-input contract independently of static semantics."""

from __future__ import annotations

import warnings
from typing import Any

from .fjsp_env import ResidualStressFJSPEnv


def check_dynamic_input_semantics(
    checkpoint: dict[str, Any], *, dynamic: bool, resume: bool = False
) -> None:
    if not dynamic:
        return
    version = checkpoint.get("dynamic_input_semantics_version")
    current = ResidualStressFJSPEnv.DYNAMIC_INPUT_SEMANTICS_VERSION
    if version == current:
        return
    message = (
        f"checkpoint dynamic input semantics={version!r}, current={current}; "
        "dynamic inputs now use visible-job scales and public WAIT horizons. "
    )
    if resume:
        raise ValueError(message + "Start a new run with initialize_from instead of resuming.")
    warnings.warn(
        message + "Weight transfer/evaluation is allowed, but historical dynamic scores "
        "require re-evaluation and dynamic training should start a fresh run.",
        UserWarning,
        stacklevel=2,
    )
