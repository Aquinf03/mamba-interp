"""Load / save frozen eval JSONL splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from .schema import EvalExample, SplitManifest


def save_jsonl(path: Path, examples: Sequence[EvalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[EvalExample]:
    out: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(EvalExample.from_dict(json.loads(line)))
    return out


def save_manifest(path: Path, manifest: SplitManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collate_cpu(examples: Sequence[EvalExample], pad_id: int = 0):
    """Right-pad input_ids to the same length. Returns plain lists (no torch dependency)."""
    lengths = [len(ex.input_ids) for ex in examples]
    max_len = max(lengths) if lengths else 0
    batch_ids, attn, targets = [], [], []
    for ex in examples:
        pad = max_len - len(ex.input_ids)
        batch_ids.append(ex.input_ids + [pad_id] * pad)
        attn.append([1] * len(ex.input_ids) + [0] * pad)
        targets.append(ex.target_id)
    return {
        "input_ids": batch_ids,
        "attention_mask": attn,
        "target_ids": targets,
        "example_ids": [ex.example_id for ex in examples],
        "tasks": [ex.task for ex in examples],
    }
