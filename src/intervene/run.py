"""Run one intervention condition over an eval split (batch size 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.eval.schema import EvalExample

from .hooks import PatchState, apply_patches
from .spec import Intervention
from .windows import ar_windows


@dataclass
class ExampleScore:
    example_id: str
    correct: bool
    target_lp: float
    pred_id: int


@dataclass
class ConditionResult:
    name: str
    n: int
    acc: float
    mean_lp: float
    acc_on_clean_ok: Optional[float]
    d_lp_on_clean_ok: Optional[float]
    n_clean_ok: int
    layers: List[int]
    window: str
    target: str
    op: str
    meta: Dict[str, Any] = field(default_factory=dict)
    acc_ci95: Optional[List[float]] = None
    donor_acc: Optional[float] = None
    donor_n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "acc": self.acc,
            "acc_ci95": self.acc_ci95,
            "mean_lp": self.mean_lp,
            "acc_on_clean_ok": self.acc_on_clean_ok,
            "d_lp_on_clean_ok": self.d_lp_on_clean_ok,
            "n_clean_ok": self.n_clean_ok,
            "donor_acc": self.donor_acc,
            "donor_n": self.donor_n,
            "layers": self.layers,
            "window": self.window,
            "target": self.target,
            "op": self.op,
            "meta": self.meta,
        }


@torch.no_grad()
def _forward_logits(model: Any, ex: EvalExample, device: str) -> torch.Tensor:
    ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=False)
    return out.logits[0, len(ex.input_ids) - 1]


def _score(logits: torch.Tensor, ex: EvalExample) -> ExampleScore:
    logp = torch.log_softmax(logits.float(), dim=-1)
    pred = int(logits.argmax().item())
    tgt  = int(ex.target_id)
    return ExampleScore(
        example_id=ex.example_id,
        correct=pred == tgt,
        target_lp=float(logp[tgt].item()),
        pred_id=pred,
    )


def bootstrap_acc(correct: Sequence[bool], *, n_boot: int = 1000, seed: int = 0) -> List[float]:
    """Bootstrap 95% CI on accuracy. Returns [lo, hi]."""
    n = len(correct)
    if n == 0:
        return [0.0, 0.0]
    flags = torch.tensor([1.0 if c else 0.0 for c in correct])
    g = torch.Generator().manual_seed(seed)
    accs = [float(flags[torch.randint(0, n, (n,), generator=g)].mean()) for _ in range(n_boot)]
    t = torch.tensor(accs)
    return [float(t.quantile(0.025)), float(t.quantile(0.975))]


def donor_accuracy(
    scores: Sequence[ExampleScore],
    examples: Sequence[EvalExample],
) -> tuple[Optional[float], int]:
    """Fraction of examples where the model predicted the *next* example's target.

    Used for the swap condition: if donor_acc >> chance, the intervention is
    transplanting content - not just breaking the recipient's bind.
    Excludes pairs where the two target ids collide.
    """
    n = len(examples)
    if n < 2:
        return None, 0
    hits = valid = 0
    for i, s in enumerate(scores):
        dt = int(examples[(i + 1) % n].target_id)
        if dt == int(examples[i].target_id):
            continue
        valid += 1
        if int(s.pred_id) == dt:
            hits += 1
    return (hits / valid if valid else None), valid


def summarize_condition(
    spec: Optional[Intervention],
    scores: Sequence[ExampleScore],
    clean: Optional[Sequence[ExampleScore]],
    *,
    layers: Sequence[int],
    examples: Optional[Sequence[EvalExample]] = None,
    donor: bool = False,
) -> ConditionResult:
    n = len(scores)
    acc    = sum(1 for s in scores if s.correct) / n if n else 0.0
    mean_lp = sum(s.target_lp for s in scores) / n if n else 0.0
    acc_ok = d_lp = None
    n_ok = 0
    if clean is not None:
        ok_idx = [i for i, c in enumerate(clean) if c.correct]
        n_ok = len(ok_idx)
        if n_ok:
            acc_ok = sum(1 for i in ok_idx if scores[i].correct) / n_ok
            d_lp   = sum(scores[i].target_lp - clean[i].target_lp for i in ok_idx) / n_ok
    d_acc, d_n = (None, 0)
    if donor and examples is not None:
        d_acc, d_n = donor_accuracy(scores, examples)
    return ConditionResult(
        name=spec.name if spec is not None else "clean",
        n=n,
        acc=acc,
        acc_ci95=bootstrap_acc([s.correct for s in scores]),
        mean_lp=mean_lp,
        acc_on_clean_ok=acc_ok,
        d_lp_on_clean_ok=d_lp,
        n_clean_ok=n_ok,
        layers=list(layers),
        window=spec.window if spec is not None else "none",
        target=spec.target if spec is not None else "none",
        op=spec.op if spec is not None else "none",
        donor_acc=d_acc,
        donor_n=d_n,
    )


def _restore_index(ex: EvalExample, spec: Optional[Intervention]) -> Optional[int]:
    """Token index at which to paste the donor h back (from ``restore_window``)."""
    if spec is None or spec.op != "restore":
        return None
    wins = ar_windows(ex)
    rw = spec.restore_window or "last_write"
    if rw == "last_write":
        w = wins.get("last_write") or wins.get("write") or []
        return int(w[-1]) if w else None
    slot = wins.get(rw) or []
    return int(slot[0]) if slot else None


@torch.no_grad()
def run_condition(
    model: Any,
    examples: Sequence[EvalExample],
    spec: Optional[Intervention],
    *,
    device: str,
    layers: Sequence[int],
    capture: bool = False,
    donors: Optional[List[Dict[int, torch.Tensor]]] = None,
    donor_residuals: Optional[List[Dict[int, torch.Tensor]]] = None,
    donor_C: Optional[List[Dict[int, torch.Tensor]]] = None,
    delta_mags: Optional[List[Dict[int, torch.Tensor]]] = None,
    channel_sets: Optional[List[List[int]]] = None,
    seed: int = 0,
    captures: Optional[Dict[str, Any]] = None,
) -> tuple[List[ExampleScore], List[Dict[int, torch.Tensor]], List[Dict[int, torch.Tensor]]]:
    """Run one condition over all examples. Returns scores, mixer traces, h_write_end traces."""
    scores: List[ExampleScore] = []
    mixers: List[Dict[int, torch.Tensor]] = []
    hs: List[Dict[int, torch.Tensor]] = []
    h_seqs: List[Dict[int, torch.Tensor]] = []
    res_seqs: List[Dict[int, torch.Tensor]] = []
    state = PatchState(spec=spec, capture=capture)
    with apply_patches(model, state, layers):
        for i, ex in enumerate(examples):
            wins = ar_windows(ex)
            state.times          = list(wins.get(spec.window if spec else "write") or [])
            state.restore_at     = _restore_index(ex, spec)
            state.mixer_out      = {}
            state.h_write_end    = {}
            state.h_seq          = {}
            state.residual_seq   = {}
            state.donor_h        = donors[i]        if donors is not None        else {}
            state.donor_residual = donor_residuals[i] if donor_residuals is not None else {}
            state.donor_C        = donor_C[i]       if donor_C is not None       else {}
            state.channels_override = channel_sets[i] if channel_sets is not None else None
            state.delta_mag      = delta_mags[i]    if delta_mags is not None    else {}
            state.noise_seed     = seed + i
            logits = _forward_logits(model, ex, device)
            scores.append(_score(logits, ex))
            mixers.append({k: v.clone() for k, v in state.mixer_out.items()})
            hs.append({k: v.clone() for k, v in state.h_write_end.items()})
            if capture:
                h_seqs.append({k: v.clone() for k, v in state.h_seq.items()})
                res_seqs.append({k: v.clone() for k, v in state.residual_seq.items()})
            if (i + 1) % 16 == 0:
                print(f"    {i + 1}/{len(examples)}", flush=True)
    if captures is not None:
        captures["h_seq"]       = h_seqs
        captures["residual_seq"] = res_seqs
    return scores, mixers, hs


def mixer_delta_mags(
    clean: Sequence[Dict[int, torch.Tensor]],
    intervened: Sequence[Dict[int, torch.Tensor]],
) -> List[Dict[int, torch.Tensor]]:
    """Per-example, per-layer L2 of (mixer_intervened − mixer_clean) at each position: [T]."""
    out: List[Dict[int, torch.Tensor]] = []
    for c, iv in zip(clean, intervened):
        row: Dict[int, torch.Tensor] = {}
        for L, mix_c in c.items():
            mix_i = iv.get(L)
            if mix_i is not None and mix_i.shape == mix_c.shape:
                row[L] = (mix_i.float() - mix_c.float()).norm(dim=-1)
        out.append(row)
    return out


def mixer_budget_summary(
    mags: Sequence[Dict[int, torch.Tensor]],
    examples: Sequence[EvalExample],
    window: str,
) -> Dict[str, float]:
    """Aggregate per-position L2 magnitudes across examples and layers for a window."""
    vals: List[float] = []
    for mag_row, ex in zip(mags, examples):
        times = ar_windows(ex).get(window) or []
        for vec in mag_row.values():
            for t in times:
                if t < int(vec.numel()):
                    vals.append(float(vec[t].item()))
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "n": 0}
    t = torch.tensor(vals)
    return {"mean": float(t.mean()), "median": float(t.median()), "p90": float(t.quantile(0.9)), "n": len(vals)}


def results_to_markdown(
    results: Sequence[ConditionResult],
    *,
    title: str = "Causal interventions",
) -> str:
    lines = [
        f"# {title}", "",
        "| name | layers | window | n | acc | ci95 | mean_lp | acc\\|clean_ok | Δlp\\|clean_ok | donor_acc |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in results:
        acc_ok = "" if r.acc_on_clean_ok is None else f"{r.acc_on_clean_ok:.3f}"
        dlp    = "" if r.d_lp_on_clean_ok is None else f"{r.d_lp_on_clean_ok:.3f}"
        donor  = "" if r.donor_acc is None else f"{r.donor_acc:.3f}"
        ci     = f"{r.acc_ci95[0]:.3f}–{r.acc_ci95[1]:.3f}" if r.acc_ci95 else ""
        lo, hi = (r.layers[0], r.layers[-1]) if r.layers else (-1, -1)
        layer_s = "-" if not r.layers else f"{lo}-{hi}"
        lines.append(
            f"| {r.name} | {layer_s} | {r.window} | {r.n} | {r.acc:.3f} | {ci} | "
            f"{r.mean_lp:.3f} | {acc_ok} | {dlp} | {donor} |"
        )
    lines.append("")
    return "\n".join(lines)
