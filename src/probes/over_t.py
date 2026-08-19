"""Probe AR value identity at multiple sequence positions (store vs readout).

Probes are fitted at four sites:
  first      - t=0, nothing stored yet (expected: chance)
  value_write - token where the answer value was written into the sequence
  last_write - last token before the query (expected: chance - no linear letter after the list)
  query      - the query key position (residual wins here because the model assembles the answer)

The ``last_write`` null (both linear and MLP) is the key negative result: the bind
is in L17 h causally, but is not a linearly decodable letter sitting in residual or h.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from src.eval.ar_train import build_ar_train_set
from src.eval.schema import EvalExample
from src.eval.vocab import TokenPool
from src.hooks import capture_mamba_states
from src.intervene.windows import queried_value_index
from src.probes.linear import FEATURE_FNS, ProbeResult, fit_linear_probe


SITES = ("first", "value_write", "last_write", "query")


def site_index(ex: EvalExample, site: str) -> int:
    t = len(ex.input_ids)
    if site == "first":
        return 0
    if site == "query":
        return t - 1
    if site == "last_write":
        return max(0, t - 2)
    if site == "value_write":
        vi = queried_value_index(ex)
        return int(vi) if vi is not None else max(0, t - 2)
    raise ValueError(f"unknown site '{site}'")


def collect_features(
    model,
    examples: Sequence[EvalExample],
    *,
    layers: Sequence[int],
    feat_names: Sequence[str],
    device: str,
) -> Tuple[Dict[Tuple[int, str, str], List[Tensor]], List[int]]:
    """Capture features at all (layer, site, feature) combinations in one pass."""
    bucket: Dict[Tuple[int, str, str], List[Tensor]] = {}
    y: List[int] = []
    n = len(examples)
    for i, ex in enumerate(examples):
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            with capture_mamba_states(model, layers=list(layers), keep_on_cpu=True) as cap:
                _ = model(input_ids=ids, use_cache=False)
        y.append(int(ex.target_id))
        for L in layers:
            tr = cap.layers[L]
            for site in SITES:
                t = site_index(ex, site)
                for fname in feat_names:
                    key = (int(L), site, fname)
                    bucket.setdefault(key, []).append(FEATURE_FNS[fname](tr, t=t))
        if (i + 1) % 16 == 0:
            print(f"  captured {i + 1}/{n}", flush=True)
    return bucket, y


def fit_all(
    bucket: Dict[Tuple[int, str, str], List[Tensor]],
    y: List[int],
    *,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    probe_steps: int,
    seed: int,
) -> List[ProbeResult]:
    """Fit one probe per (layer, site, feature) and return all results."""
    y_t = torch.tensor(y, dtype=torch.long)
    results: List[ProbeResult] = []
    for (L, site, fname), xs in sorted(bucket.items()):
        X = torch.stack(xs, dim=0)
        tr_i = list(train_idx)
        te_i = list(test_idx)
        X_tr, y_tr = X[tr_i], y_t[tr_i]
        X_te, y_te = X[te_i], y_t[te_i]
        mu = X_tr.mean(0, keepdim=True)
        sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        tr_acc, te_acc, n_cls = fit_linear_probe(
            X_tr, y_tr, X_te, y_te, steps=probe_steps, seed=seed,
        )
        results.append(ProbeResult(
            feature=fname, layer=L, when=site,
            train_acc=tr_acc, test_acc=te_acc,
            n_train=len(tr_i), n_test=len(te_i),
            n_classes=n_cls, feat_dim=int(X.shape[1]),
        ))
        print(f"L{L} {site} {fname}: train={tr_acc:.3f} test={te_acc:.3f}", flush=True)
    return results


def make_probe_split(tokenizer_name: str, *, n: int, seed: int) -> List[EvalExample]:
    """Fresh AR examples for probing - not the frozen v1 eval split."""
    pool = TokenPool(tokenizer_name=tokenizer_name)
    return build_ar_train_set(seed, pool, n=n, n_pairs_choices=(4,))
