"""CPU-only preview of frozen splits (no model)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import collate_cpu, load_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl",
    )
    p.add_argument("--k", type=int, default=3, help="print k examples")
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    exs = load_jsonl(args.split)
    print(f"loaded {len(exs)} from {args.split}")
    lens = [len(e.input_ids) for e in exs]
    print(f"seq_len min/mean/max = {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
    print(f"tasks: {Counter(e.task for e in exs)}")

    for e in exs[: args.k]:
        print("---")
        print(e.example_id, "target=", e.target_text, e.target_id)
        print(e.text)
        print("ids_len", len(e.input_ids), "meta_keys", sorted(e.meta.keys()))

    batch = collate_cpu(exs[: args.batch])
    print("--- batch", args.batch)
    print("input_ids rows", len(batch["input_ids"]), "width", len(batch["input_ids"][0]))
    print("targets", batch["target_ids"])


if __name__ == "__main__":
    main()
