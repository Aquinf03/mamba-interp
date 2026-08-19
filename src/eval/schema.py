"""Shared schema for frozen synthetic eval examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class EvalExample:
    """One next-token prediction item.

    ``input_ids`` ends at the last context token; the gold next token is ``target_id``.
    Positions are 0-based indices into ``input_ids``.
    """

    example_id: str
    task: str
    split: str
    text: str
    input_ids: List[int]
    target_id: int
    target_text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalExample":
        return cls(**d)


@dataclass
class SplitManifest:
    task: str
    split: str
    seed: int
    n: int
    tokenizer_name: str
    model_id: str
    path: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
