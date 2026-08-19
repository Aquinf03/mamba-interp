"""ATR (associative treecall / induction): ... A B ... A → predict B.

Task format::

    [prefix noise] A0 B0 [gap] A1 B1 ... [gap/2] A_last → predict B_last

With ``n_marks=1`` this is classic bigram induction after a random prefix.
With ``n_marks>1`` the model must chain through a hierarchy of mark pairs.
"""

from __future__ import annotations

import random
from typing import List

from .schema import EvalExample
from .vocab import TokenPool, format_with_sep


def build_induction_example(
    rng: random.Random,
    pool: TokenPool,
    *,
    n_marks: int,
    prefix_len: int,
    gap_len: int,
    example_id: str,
    split: str,
) -> EvalExample:
    """Build one ATR example. Each mark pair (A, B) is drawn without replacement."""
    if n_marks < 1:
        raise ValueError("n_marks >= 1")

    pieces: List[str] = []
    pieces.extend(pool.sample_with_replacement(rng, prefix_len))

    marks = []
    for m in range(n_marks):
        a, b = pool.sample_unique(rng, 2)
        marks.append((a, b))
        pieces.extend([a, b])
        if m + 1 < n_marks:
            pieces.extend(pool.sample_with_replacement(rng, gap_len))

    a_last, b_last = marks[-1]
    pieces.extend(pool.sample_with_replacement(rng, max(1, gap_len // 2)))
    pieces.append(a_last)

    text = format_with_sep(pieces, pool.sep)
    ids = pool.encode(text)
    target_id = pool.encode(b_last)[0]

    meta = {
        "n_marks": n_marks,
        "prefix_len": prefix_len,
        "gap_len": gap_len,
        "marks": [{"a": a, "b": b} for a, b in marks],
        "query_a": a_last,
        "target_b": b_last,
        "seq_len": len(ids),
        "family": "induction",
    }
    return EvalExample(
        example_id=example_id,
        task="atr",
        split=split,
        text=text,
        input_ids=ids,
        target_id=target_id,
        target_text=b_last,
        meta=meta,
    )


def build_atr_split(
    seed: int,
    pool: TokenPool,
    *,
    split: str,
    n: int,
    n_marks: int,
    prefix_len: int,
    gap_len: int,
) -> List[EvalExample]:
    rng = random.Random(seed)
    return [
        build_induction_example(
            rng, pool,
            n_marks=n_marks,
            prefix_len=prefix_len,
            gap_len=gap_len,
            example_id=f"atr_{split}_{i:05d}",
            split=split,
        )
        for i in range(n)
    ]
