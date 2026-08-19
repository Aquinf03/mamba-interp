"""Shared causal-suite runner for AR and ATR conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from src.eval.schema import EvalExample

from .run import (
    ConditionResult,
    mixer_budget_summary,
    mixer_delta_mags,
    run_condition,
    summarize_condition,
)
from .spec import Intervention


def run_causal_suite(
    model: Any,
    examples: Sequence[EvalExample],
    specs: Sequence[Intervention],
    *,
    device: str,
    core_layers: Sequence[int],
    seed: int = 0,
    extra_zero_h_groups: Optional[Dict[str, List[int]]] = None,
    layer_sweep: bool = False,
) -> Dict[str, Any]:
    """Run clean + all listed specs. Captures mixer outputs for residual-noise budgeting.

    The residual-noise conditions (``op == "noise"``) must appear *after* the
    ``zero_h_write`` spec in ``specs`` because the noise budget is derived from the
    mixer-output change produced by that wipe.

    Returns a dict with ``results`` (list of ``ConditionResult``) and ``budgets``.
    """
    print("condition clean ...", flush=True)
    clean_scores, clean_mix, clean_h = run_condition(
        model, examples, None, device=device, layers=core_layers, capture=True, seed=seed,
    )
    results: List[ConditionResult] = [
        summarize_condition(None, clean_scores, None, layers=core_layers, examples=examples)
    ]
    print(f"  acc={results[-1].acc:.3f} lp={results[-1].mean_lp:.3f}", flush=True)

    # Shift-by-one donor: example i uses example i+1's clean h for the swap condition.
    donors = clean_h[1:] + clean_h[:1]
    zero_h_mix = None
    budgets: Dict[str, Any] = {}

    for spec in specs:
        print(f"condition {spec.name} ...", flush=True)
        kwargs = dict(device=device, layers=spec.layers, seed=seed)

        if spec.op == "swap":
            scores, _, _ = run_condition(model, examples, spec, capture=False, donors=donors, **kwargs)
            row = summarize_condition(spec, scores, clean_scores, layers=spec.layers,
                                      examples=examples, donor=True)

        elif spec.op == "noise":
            if zero_h_mix is None:
                print("  skip (need zero_h_write mixer traces first)", flush=True)
                continue
            mags = mixer_delta_mags(clean_mix, zero_h_mix)
            scores, _, _ = run_condition(model, examples, spec, capture=False, delta_mags=mags, **kwargs)
            row = summarize_condition(spec, scores, clean_scores, layers=spec.layers)

        else:
            need_cap = spec.name == "zero_h_write"
            scores, mix, _ = run_condition(model, examples, spec, capture=need_cap, **kwargs)
            if need_cap:
                zero_h_mix = mix
                mags = mixer_delta_mags(clean_mix, zero_h_mix)
                budgets["zero_h_write_mixer_L2_write"] = mixer_budget_summary(mags, examples, "write")
                budgets["zero_h_write_mixer_L2_query"] = mixer_budget_summary(mags, examples, "query")
                print(
                    f"  mixer L2 write mean={budgets['zero_h_write_mixer_L2_write']['mean']:.4f} "
                    f"query mean={budgets['zero_h_write_mixer_L2_query']['mean']:.4f}",
                    flush=True,
                )
            row = summarize_condition(spec, scores, clean_scores, layers=spec.layers)

        results.append(row)
        extra = f" donor_acc={row.donor_acc:.3f}" if row.donor_acc is not None else ""
        print(
            f"  acc={row.acc:.3f} lp={row.mean_lp:.3f} "
            f"acc|ok={row.acc_on_clean_ok} dlp={row.d_lp_on_clean_ok}{extra}",
            flush=True,
        )

    # Optional named layer groups (early / mid / etc.) beyond the core suite.
    if extra_zero_h_groups:
        for gname, glayers in extra_zero_h_groups.items():
            spec = Intervention(f"zero_h_write_{gname}", "h", "zero", "write", list(glayers))
            print(f"condition {spec.name} ...", flush=True)
            scores, _, _ = run_condition(model, examples, spec, device=device, layers=glayers,
                                         capture=False, seed=seed)
            row = summarize_condition(spec, scores, clean_scores, layers=glayers)
            results.append(row)
            print(f"  acc={row.acc:.3f} lp={row.mean_lp:.3f}", flush=True)

    # Full per-layer sweep: zero h at each layer individually.
    if layer_sweep:
        n_layers = len(model.backbone.layers)
        for L in range(n_layers):
            spec = Intervention(f"zero_h_write_L{L}", "h", "zero", "write", [L])
            print(f"condition {spec.name} ...", flush=True)
            scores, _, _ = run_condition(model, examples, spec, device=device, layers=[L],
                                         capture=False, seed=seed)
            row = summarize_condition(spec, scores, clean_scores, layers=[L])
            results.append(row)
            print(f"  acc={row.acc:.3f}", flush=True)

    return {"results": results, "budgets": budgets, "clean_acc": results[0].acc}
