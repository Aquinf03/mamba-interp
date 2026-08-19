"""Causal patches on h / B / C / Δ / residual for Mamba interpretability."""

from __future__ import annotations

from .hooks import PatchState, apply_patches
from .run import ConditionResult, run_condition, summarize_condition
from .spec import Intervention, atr_core, late_core
from .windows import ar_windows, layer_group, queried_value_index, token_roles

__all__ = [
    "Intervention",
    "PatchState",
    "ConditionResult",
    "apply_patches",
    "ar_windows",
    "layer_group",
    "late_core",
    "atr_core",
    "queried_value_index",
    "token_roles",
    "run_condition",
    "summarize_condition",
]
