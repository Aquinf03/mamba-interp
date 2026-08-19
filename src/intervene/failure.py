"""Length/stuffing failure analysis: state geometry + Δ-clamp on junk tokens.

The two failure modes are distinguished mechanistically:
- OOD pad tokens overwrite the stored bind (Δ-clamp helps).
- Filler/stuffing tokens are load-bearing context; clamping their Δ hurts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from src.eval.schema import EvalExample
from src.hooks import capture_mamba_states
from src.intervene.run import run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import ar_windows, queried_value_index


def effective_rank(mat: torch.Tensor) -> float:
    """Effective rank via participation ratio of squared singular values. mat: [E, N]."""
    s = torch.linalg.svdvals(mat.float())
    p = s.square()
    z = p.sum().clamp_min(1e-12)
    p = p / z
    ent = -(p * p.clamp_min(1e-12).log()).sum()
    return float(ent.exp())


@torch.no_grad()
def example_geometry(
    model: Any,
    ex: EvalExample,
    *,
    layers: Sequence[int],
    device: str,
) -> Dict[str, float]:
    """Compute per-example state diagnostics (norm, effective rank, mean Δ by token role).

    These diagnostics show that geometry does *not* separate correct from incorrect
    examples on the length/stuffing splits - the causal evidence comes from the Δ-clamp.
    """
    ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
    with capture_mamba_states(model, layers=list(layers), keep_on_cpu=True) as cap:
        out = model(input_ids=ids, use_cache=False)
    t_q = len(ex.input_ids) - 1
    t_w = queried_value_index(ex)
    if t_w is None:
        t_w = max(0, t_q - 1)
    logits = out.logits[0, t_q]
    pred   = int(logits.argmax().item())
    logp   = torch.log_softmax(logits.float(), dim=-1)
    wins   = ar_windows(ex)
    rec: Dict[str, float] = {
        "correct":   1.0 if pred == int(ex.target_id) else 0.0,
        "target_lp": float(logp[int(ex.target_id)].item()),
        "seq_len":   float(len(ex.input_ids)),
        "n_junk":    float(len(wins.get("junk") or [])),
    }
    for L in layers:
        tr = cap.layers[L]
        h_w = tr.h[0, t_w].float()
        h_q = tr.h[0, t_q].float()
        rec[f"L{L}_h_write_norm"]  = float(h_w.norm())
        rec[f"L{L}_h_query_norm"]  = float(h_q.norm())
        rec[f"L{L}_h_write_erank"] = effective_rank(h_w)
        rec[f"L{L}_h_query_erank"] = effective_rank(h_q)
        rec[f"L{L}_delta_query"]   = float(tr.delta[0, t_q].float().mean())
        junk = wins.get("junk") or []
        rec[f"L{L}_delta_junk"] = float(tr.delta[0, junk].float().mean()) if junk else float("nan")
    return rec


def mean_by_correct(rows: Sequence[Dict[str, float]], key: str) -> Dict[str, Optional[float]]:
    """Split a metric by correctness and return means for each group."""
    ok  = [r[key] for r in rows if r.get("correct", 0) >= 0.5 and key in r and r[key] == r[key]]
    bad = [r[key] for r in rows if r.get("correct", 0) < 0.5  and key in r and r[key] == r[key]]
    def _m(xs): return sum(xs) / len(xs) if xs else None
    return {"correct": _m(ok), "wrong": _m(bad), "n_ok": len(ok), "n_bad": len(bad)}


def run_delta_clamp_junk(
    model: Any,
    examples: Sequence[EvalExample],
    *,
    layers: Sequence[int],
    device: str,
    seed: int = 0,
):
    """Clamp Δ → 0 on all filler/pad positions so junk cannot overwrite stored binds."""
    spec = Intervention("clamp_delta_junk", "delta", "clamp", "junk", list(layers), clamp=0.0)
    print(f"condition {spec.name} n={len(examples)} ...", flush=True)
    scores, _, _ = run_condition(
        model, examples, spec, device=device, layers=layers, capture=False, seed=seed,
    )
    return scores
