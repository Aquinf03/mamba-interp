"""Generate fresh AR training data (not the frozen v1 eval splits)."""

from __future__ import annotations

import random
from typing import List

from src.eval.ar import build_ar_example
from src.eval.schema import EvalExample
from src.eval.vocab import TokenPool


def build_ar_train_set(
    seed: int,
    pool: TokenPool,
    *,
    n: int,
    n_pairs_choices: tuple[int, ...] = (2, 3, 4, 5),
    split: str = "train",
) -> List[EvalExample]:
    """Sample ``n`` AR examples with random pair counts from ``n_pairs_choices``."""
    rng = random.Random(seed)
    out: List[EvalExample] = []
    for i in range(n):
        n_pairs = rng.choice(list(n_pairs_choices))
        out.append(
            build_ar_example(
                rng, pool,
                n_pairs=n_pairs,
                n_queries=1,
                example_id=f"ar_train_{i:06d}",
                split=split,
            )
        )
    return out
