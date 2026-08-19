"""Behavioral baseline: teacher-forced next-token accuracy on frozen splits."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Sequence, Set

import torch
import torch.nn as nn

from .metrics import SplitResult, accuracy
from .schema import EvalExample


@contextmanager
def ablate_mixer_layers(model: nn.Module, layer_idxs: Sequence[int]):
    """Replace selected Mamba blocks with identity (residual passes through unchanged).

    Used to measure whether late mixers are required for the head - they are (late ablation → 0%).
    """
    backbone = model.backbone if hasattr(model, "backbone") else model
    want: Set[int] = {int(i) for i in layer_idxs}
    originals = {}

    for i, block in enumerate(backbone.layers):
        if i not in want:
            continue
        originals[i] = block.forward

        def make_fwd(b=block):
            def forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
                return hidden_states
            return forward

        block.forward = make_fwd()

    try:
        yield
    finally:
        for i, block in enumerate(backbone.layers):
            if i in originals:
                block.forward = originals[i]


@torch.no_grad()
def predict_next_tokens(
    model: nn.Module,
    examples: Sequence[EvalExample],
    *,
    device: str = "cuda",
    batch_size: int = 1,
    pad_id: int = 0,
) -> dict:
    """Return per-example predictions, correctness flags, and target log-probs."""
    model.eval()
    preds: List[int] = []
    correct_flags: List[bool] = []
    target_lps: List[float] = []
    top1_probs: List[float] = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        lengths = [len(ex.input_ids) for ex in batch]
        max_len = max(lengths)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, ex in enumerate(batch):
            L = len(ex.input_ids)
            ids[i, :L] = torch.tensor(ex.input_ids, dtype=torch.long, device=device)
            mask[i, :L] = 1

        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)

        for i, ex in enumerate(batch):
            t = lengths[i] - 1
            pred = int(logits[i, t].argmax().item())
            tgt = int(ex.target_id)
            preds.append(pred)
            correct_flags.append(pred == tgt)
            target_lps.append(float(log_probs[i, t, tgt].item()))
            top1_probs.append(float(probs[i, t].max().item()))

    return {
        "preds": preds,
        "correct": correct_flags,
        "target_logprob": target_lps,
        "top1_prob": top1_probs,
    }


def evaluate_split(
    model: nn.Module,
    examples: Sequence[EvalExample],
    *,
    name: str,
    device: str = "cuda",
    batch_size: int = 1,
    ablation: str = "none",
    config_note: str = "",
) -> SplitResult:
    if not examples:
        return SplitResult(
            name=name, task="empty", n=0, correct=0, accuracy=0.0,
            mean_seq_len=0.0, mean_target_logprob=0.0, mean_top1_prob=0.0,
            config_note=config_note, ablation=ablation,
        )

    pack = predict_next_tokens(model, examples, device=device, batch_size=batch_size)
    n = len(examples)
    n_correct = sum(1 for c in pack["correct"] if c)
    mean_len = sum(len(e.input_ids) for e in examples) / n
    mean_lp = sum(pack["target_logprob"]) / n
    mean_p = sum(pack["top1_prob"]) / n
    return SplitResult(
        name=name,
        task=examples[0].task,
        n=n,
        correct=n_correct,
        accuracy=accuracy(n_correct, n),
        mean_seq_len=mean_len,
        mean_target_logprob=mean_lp,
        mean_top1_prob=mean_p,
        config_note=config_note,
        ablation=ablation,
        meta={
            "nll": -mean_lp,
            "ppl_target": math.exp(-mean_lp) if mean_lp > -50 else float("inf"),
        },
    )


def results_to_markdown(results: Sequence[SplitResult], title: str = "Behavioral baseline") -> str:
    lines = [
        f"# {title}", "",
        "| name | task | ablation | n | mean_len | acc | mean_lp(target) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.task} | {r.ablation} | {r.n} | {r.mean_seq_len:.1f} | "
            f"{r.accuracy:.3f} | {r.mean_target_logprob:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
