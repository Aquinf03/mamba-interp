"""Minimal factual recall split (subject phrase → object token).

This split is included as a diagnostic to confirm that AR finetune does not
accidentally transfer to real-world factual knowledge. Expected accuracy: 0%.
"""

from __future__ import annotations

from typing import List, Tuple

from .schema import EvalExample
from .vocab import TokenPool


FACTS: List[Tuple[str, str]] = [
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Spain is", "Madrid"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for oxygen is", "O"),
    ("Water freezes at zero degrees", "Celsius"),
]


def build_factual_split(pool: TokenPool, *, split: str = "eval") -> List[EvalExample]:
    out: List[EvalExample] = []
    for i, (prompt, answer) in enumerate(FACTS):
        full_context = prompt if prompt.endswith(" ") else prompt + " "
        ids = pool.encode(full_context)
        tid = pool.encode(answer)
        out.append(
            EvalExample(
                example_id=f"fact_{split}_{i:05d}",
                task="factual",
                split=split,
                text=full_context,
                input_ids=ids,
                target_id=tid[0],
                target_text=answer,
                meta={"prompt": prompt, "answer": answer, "answer_n_tokens": len(tid)},
            )
        )
    return out
