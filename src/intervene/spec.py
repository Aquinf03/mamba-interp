"""Intervention specification: what to patch, where, and how.

Each ``Intervention`` describes a single causal condition:

- ``target``: which object to modify - ``h`` | ``B`` | ``C`` | ``delta`` | ``residual`` | ``ssm``
- ``op``:     how to modify it    - ``zero`` | ``clamp`` | ``swap`` | ``noise`` | ``local`` | ``restore``
- ``window``: which token positions to touch - ``write`` | ``query`` | ``all`` | ``junk`` | ``pad`` | ``filler``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class Intervention:
    name: str
    target: str
    op: str
    window: str
    layers: List[int]
    clamp: float = 0.0
    # If set, zero only these N-slot indices of h (used for channel ablations).
    channels: List[int] | None = None
    # For restore ops: which position to paste the donor state back.
    restore_window: str = "last_write"  # last_write | value | query

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def late_core(layers: Sequence[int]) -> List[Intervention]:
    """Standard AR causal suite: state vs residual vs gate, write and query windows."""
    L = list(layers)
    return [
        Intervention("zero_h_write",        "h",        "zero",  "write", L),
        Intervention("zero_h_query",         "h",        "zero",  "query", L),
        Intervention("swap_h_write_end",     "h",        "swap",  "write", L),
        Intervention("zero_B_write",         "B",        "zero",  "write", L),
        Intervention("clamp_delta_write",    "delta",    "clamp", "write", L, clamp=0.0),
        Intervention("zero_C_query",         "C",        "zero",  "query", L),
        Intervention("local_only",           "ssm",      "local", "all",   L),
        Intervention("residual_noise_write", "residual", "noise", "write", L),
        Intervention("residual_noise_query", "residual", "noise", "query", L),
    ]


def atr_core(layers: Sequence[int]) -> List[Intervention]:
    """Shorter ATR suite: state vs residual vs conv/D-skip."""
    L = list(layers)
    return [
        Intervention("zero_h_write",        "h",        "zero",  "write", L),
        Intervention("swap_h_write_end",     "h",        "swap",  "write", L),
        Intervention("local_only",           "ssm",      "local", "all",   L),
        Intervention("residual_noise_write", "residual", "noise", "write", L),
        Intervention("residual_noise_query", "residual", "noise", "query", L),
    ]
