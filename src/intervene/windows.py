"""Token-position windows and role labels for AR / ATR / length sequences.

Each ``EvalExample`` stores the full input_ids; the functions here decode that
sequence into named windows that the intervention hooks use to know *which*
positions to touch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.eval.schema import EvalExample


def ar_windows(ex: EvalExample) -> Dict[str, List[int]]:
    """Return named position windows for ``ex``.

    The last token is always the query key; everything before it is the write
    / store window. ``junk`` = filler + OOD pad positions (empty on plain AR).
    """
    t = len(ex.input_ids)
    if t < 1:
        raise ValueError(f"{ex.example_id}: empty input_ids")
    roles = token_roles(ex)
    query      = [t - 1]
    write      = list(range(t - 1))
    junk       = [i for i, r in enumerate(roles) if r in ("filler", "pad")]
    pad        = [i for i, r in enumerate(roles) if r == "pad"]
    filler     = [i for i, r in enumerate(roles) if r == "filler"]
    last_write = [write[-1]] if write else []
    value: List[int] = []
    vi = queried_value_index(ex)
    if vi is not None:
        value = [vi]
    return {
        "write":      write,
        "query":      query,
        "all":        list(range(t)),
        "junk":       junk,
        "pad":        pad,
        "filler":     filler,
        "value":      value,
        "last_write": last_write,
    }


def token_roles(ex: EvalExample) -> List[str]:
    """Assign a semantic role to each input token. Falls back to write/query on mismatch."""
    t = len(ex.input_ids)
    meta = ex.meta or {}
    family = meta.get("family")
    roles = _atr_roles(meta) if (ex.task == "atr" or family == "induction") else _ar_roles(meta)
    if len(roles) != t:
        roles = ["write"] * (t - 1) + ["query"]
    return roles


def _ar_roles(meta: dict) -> List[str]:
    n_pairs = int(meta.get("n_pairs") or 0)
    if n_pairs < 1:
        return []
    filler_every = int(meta.get("filler_every") or 0)
    filler_n     = int(meta.get("filler_n") or 0)
    ood_pad      = int(meta.get("ood_pad") or 0)
    n_queries    = int(meta.get("n_queries") or 1)
    roles: List[str] = []
    for i in range(n_pairs):
        roles.extend(["key", "value"])
        if filler_every > 0 and filler_n > 0 and (i + 1) % filler_every == 0 and i + 1 < n_pairs:
            roles.extend(["filler"] * filler_n)
    for _ in range(max(0, n_queries - 1)):
        roles.extend(["key", "value"])
    if ood_pad > 0:
        roles.extend(["pad"] * ood_pad)
    roles.append("query")
    return roles


def _atr_roles(meta: dict) -> List[str]:
    n_marks    = int(meta.get("n_marks") or 0)
    prefix_len = int(meta.get("prefix_len") or 0)
    gap_len    = int(meta.get("gap_len") or 0)
    if n_marks < 1:
        return []
    roles: List[str] = ["prefix"] * prefix_len
    for m in range(n_marks):
        roles.extend(["mark_a", "mark_b"])
        if m + 1 < n_marks:
            roles.extend(["gap"] * gap_len)
    roles.extend(["gap"] * max(1, gap_len // 2))
    roles.append("query")
    return roles


def queried_value_index(ex: EvalExample) -> Optional[int]:
    """Return the token index where the answer value was first written."""
    roles = token_roles(ex)
    meta = ex.meta or {}
    if ex.task == "atr" or meta.get("family") == "induction":
        idxs = [i for i, r in enumerate(roles) if r == "mark_b"]
        return idxs[-1] if idxs else None
    final_key = meta.get("final_key")
    pairs = meta.get("pairs") or []
    if not pairs or final_key is None:
        return None
    pair_i = 0
    for i, r in enumerate(roles):
        if r != "key":
            continue
        if pair_i >= len(pairs):
            break
        if pairs[pair_i].get("key") == final_key and i + 1 < len(roles) and roles[i + 1] == "value":
            return i + 1
        pair_i += 1
    return None


def layer_group(n_layers: int, name: str) -> List[int]:
    """Return a contiguous third of the layer range by name (early / mid / late / all)."""
    third = max(1, n_layers // 3)
    if name == "early":
        return list(range(0, third))
    if name == "mid":
        return list(range(third, 2 * third))
    if name == "late":
        return list(range(2 * third, n_layers))
    if name == "all":
        return list(range(n_layers))
    raise ValueError(f"unknown layer group '{name}'")
