"""Linear and MLP probes for decoding AR value identity from internal representations.

Key finding: at the ``last_write`` position (end of the key-value list, before the
query), neither residual nor h carries a linearly or MLP-decodable value letter.
Residual wins at the query only because the model has already assembled the answer -
that is readout, not storage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ProbeResult:
    feature: str
    layer: int
    when: str
    train_acc: float
    test_acc: float
    n_train: int
    n_test: int
    n_classes: int
    feat_dim: int

    def to_dict(self) -> dict:
        return asdict(self)


def _map_labels(y: Tensor) -> Tuple[Tensor, Dict[int, int]]:
    """Map token-id labels to contiguous 0-based class indices."""
    uniq = sorted(int(v) for v in y.unique().tolist())
    m = {tok: i for i, tok in enumerate(uniq)}
    mapped = torch.tensor([m[int(v)] for v in y.tolist()], dtype=torch.long)
    return mapped, m


def fit_linear_probe(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    *,
    steps: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> Tuple[float, float, int]:
    """Multiclass logistic probe (softmax loss). Inputs on CPU float32 [N, F], [N]."""
    torch.manual_seed(seed)
    y_tr, label_map = _map_labels(y_train)
    y_te_list, keep = [], []
    for i, v in enumerate(y_test.tolist()):
        v = int(v)
        if v in label_map:
            y_te_list.append(label_map[v])
            keep.append(i)
    if not keep:
        return 0.0, 0.0, len(label_map)
    X_te = X_test[keep].float()
    y_te = torch.tensor(y_te_list, dtype=torch.long)
    n_classes = len(label_map)
    W = torch.zeros(X_train.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    X_tr = X_train.float()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(X_tr @ W + b, y_tr).backward()
        opt.step()
    with torch.no_grad():
        tr_acc = float((X_tr @ W + b).argmax(-1).eq(y_tr).float().mean())
        te_acc = float((X_te @ W + b).argmax(-1).eq(y_te).float().mean())
    return tr_acc, te_acc, n_classes


def fit_mlp_probe(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    *,
    hidden: int = 256,
    steps: int = 800,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> Tuple[float, float, int]:
    """Two-layer ReLU MLP probe. Used to check whether superposition is the cause
    of the linear probe null at ``last_write``."""
    torch.manual_seed(seed)
    y_tr, label_map = _map_labels(y_train)
    y_te_list, keep = [], []
    for i, v in enumerate(y_test.tolist()):
        v = int(v)
        if v in label_map:
            y_te_list.append(label_map[v])
            keep.append(i)
    if not keep:
        return 0.0, 0.0, len(label_map)
    X_te = X_test[keep].float()
    y_te = torch.tensor(y_te_list, dtype=torch.long)
    n_classes = len(label_map)
    feat = X_train.shape[1]
    W1 = torch.zeros(feat, hidden, requires_grad=True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = torch.zeros(hidden, n_classes, requires_grad=True)
    b2 = torch.zeros(n_classes, requires_grad=True)
    torch.nn.init.kaiming_uniform_(W1, a=5 ** 0.5)
    torch.nn.init.kaiming_uniform_(W2, a=5 ** 0.5)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr, weight_decay=weight_decay)
    X_tr = X_train.float()

    def logits(X):
        return torch.relu(X @ W1 + b1) @ W2 + b2

    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(logits(X_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        tr_acc = float(logits(X_tr).argmax(-1).eq(y_tr).float().mean())
        te_acc = float(logits(X_te).argmax(-1).eq(y_te).float().mean())
    return tr_acc, te_acc, n_classes


# ----- Feature extractors -----

def feature_residual(trace, t: int = -1) -> Tensor:
    """Skip stream (residual input to the mixer). [D]"""
    return trace.residual[0, t].float().reshape(-1)


def feature_residual_out(trace, t: int = -1) -> Tensor:
    """Skip + mixer output (full residual stream). [D]"""
    return trace.residual_out[0, t].float().reshape(-1)


def feature_h_flat(trace, t: int = -1) -> Tensor:
    """Flattened SSM state h[E, N]. [E*N]"""
    return trace.h[0, t].float().reshape(-1)


def feature_h_mean_e(trace, t: int = -1) -> Tensor:
    """h averaged over intermediate dim E. [N]"""
    return trace.h[0, t].float().mean(dim=0).reshape(-1)


def feature_h_mean_n(trace, t: int = -1) -> Tensor:
    """h averaged over state-slot dim N. [E]"""
    return trace.h[0, t].float().mean(dim=-1).reshape(-1)


def feature_delta_mean(trace, t: int = -1) -> Tensor:
    """Mean Δ across E at position t. [E]"""
    return trace.delta[0, t].float().reshape(-1)


FEATURE_FNS = {
    "residual":     feature_residual,
    "residual_out": feature_residual_out,
    "h_flat":       feature_h_flat,
    "h_mean_e":     feature_h_mean_e,
    "h_mean_n":     feature_h_mean_n,
    "delta":        feature_delta_mean,
}
