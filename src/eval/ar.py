"""Associative recall (AR): store key→value bindings, query a key, predict value.

Task format::

    K0 V0 K1 V1 ... KQ → predict VQ

Keys and values are drawn without replacement from the token pool.
Optional ``filler_every`` inserts noise tokens between pairs (capacity pressure).
Optional ``ood_pad`` appends random tokens before the final query (length OOD).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from .schema import EvalExample
from .vocab import TokenPool, format_with_sep


def build_ar_example(
    rng: random.Random,
    pool: TokenPool,
    *,
    n_pairs: int,
    n_queries: int,
    example_id: str,
    split: str,
    filler_every: int = 0,
    filler_n: int = 0,
    ood_pad: int = 0,
) -> EvalExample:
    """Build one AR example; the last query key is the scored prediction position.

    For multi-query (``n_queries > 1``), earlier queries are teacher-forced into the
    context so the model sees the full transcript. Only the final query is scored.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs >= 1")
    if n_queries < 1 or n_queries > n_pairs:
        raise ValueError("1 <= n_queries <= n_pairs")

    keys = pool.sample_unique(rng, n_pairs)
    values = pool.sample_unique(rng, n_pairs)
    pairs = list(zip(keys, values))

    pieces: List[str] = []
    for i, (k, v) in enumerate(pairs):
        pieces.extend([k, v])
        if filler_every > 0 and filler_n > 0 and (i + 1) % filler_every == 0 and i + 1 < n_pairs:
            pieces.extend(pool.sample_with_replacement(rng, filler_n))

    query_order = rng.sample(range(n_pairs), k=n_queries)
    query_keys = [keys[i] for i in query_order]

    kv_map = dict(pairs)
    for qk in query_keys[:-1]:
        pieces.append(qk)
        pieces.append(kv_map[qk])

    if ood_pad > 0:
        pieces.extend(pool.sample_with_replacement(rng, ood_pad))

    final_k = query_keys[-1]
    final_v = kv_map[final_k]
    pieces.append(final_k)

    text = format_with_sep(pieces, pool.sep)
    ids = pool.encode(text)
    target_id = pool.encode(final_v)[0]

    meta: Dict[str, Any] = {
        "n_pairs": n_pairs,
        "n_queries": n_queries,
        "pairs": [{"key": k, "value": v} for k, v in pairs],
        "query_keys": query_keys,
        "final_key": final_k,
        "final_value": final_v,
        "filler_every": filler_every,
        "filler_n": filler_n,
        "ood_pad": ood_pad,
        "seq_len": len(ids),
    }
    return EvalExample(
        example_id=example_id,
        task="ar",
        split=split,
        text=text,
        input_ids=ids,
        target_id=target_id,
        target_text=final_v,
        meta=meta,
    )


def build_ar_split(
    seed: int,
    pool: TokenPool,
    *,
    split: str,
    n: int,
    n_pairs: int,
    n_queries: int = 1,
    filler_every: int = 0,
    filler_n: int = 0,
    ood_pad: int = 0,
) -> List[EvalExample]:
    rng = random.Random(seed)
    return [
        build_ar_example(
            rng, pool,
            n_pairs=n_pairs,
            n_queries=n_queries,
            example_id=f"ar_{split}_{i:05d}",
            split=split,
            filler_every=filler_every,
            filler_n=filler_n,
            ood_pad=ood_pad,
        )
        for i in range(n)
    ]
