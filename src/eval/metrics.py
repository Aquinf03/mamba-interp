"""Next-token accuracy helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class SplitResult:
    name: str
    task: str
    n: int
    correct: int
    accuracy: float
    mean_seq_len: float
    mean_target_logprob: float
    mean_top1_prob: float
    config_note: str = ""
    ablation: str = "none"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def accuracy(correct: int, n: int) -> float:
    return float(correct) / float(n) if n else 0.0
