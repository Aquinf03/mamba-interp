"""One command: run the full paper suite (AR + ATR interventions, probes-over-t, failure analysis)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-layer-sweep", action="store_true")
    p.add_argument("--max-examples", type=int, default=0, help="0=full; >0 caps every split (debug)")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print("\n=== " + " ".join(cmd) + " ===\n", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    args = parse_args()
    py = sys.executable
    mid = args.model_id
    extra = ["--max-examples", str(args.max_examples)] if args.max_examples else []
    steps = []

    ar_cmd = [
        py, str(ROOT / "scripts" / "run_interventions.py"),
        "--model-id", mid,
        "--device", args.device,
        "--split", str(ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl"),
        "--out-json", str(ROOT / "logs" / "intervene_ar.json"),
        "--out-md", str(ROOT / "logs" / "intervene_ar.md"),
        *extra,
    ]
    if not args.skip_layer_sweep:
        ar_cmd.append("--layer-sweep")
    steps.append(ar_cmd)

    steps.append([
        py, str(ROOT / "scripts" / "run_interventions.py"),
        "--model-id", mid, "--device", args.device, "--table", "atr",
        "--split", str(ROOT / "data" / "splits" / "v1" / "atr_short_eval.jsonl"),
        "--out-json", str(ROOT / "logs" / "intervene_atr_short.json"),
        "--out-md", str(ROOT / "logs" / "intervene_atr_short.md"),
        *extra,
    ])
    steps.append([
        py, str(ROOT / "scripts" / "run_interventions.py"),
        "--model-id", mid, "--device", args.device, "--table", "atr",
        "--split", str(ROOT / "data" / "splits" / "v1" / "atr_mid_eval.jsonl"),
        "--out-json", str(ROOT / "logs" / "intervene_atr_mid.json"),
        "--out-md", str(ROOT / "logs" / "intervene_atr_mid.md"),
        *extra,
    ])
    probe_cmd = [
        py, str(ROOT / "scripts" / "run_probes_over_t.py"),
        "--model-id", mid, "--device", args.device,
    ]
    if args.max_examples:
        probe_cmd.extend(["--n", str(max(32, args.max_examples))])
    steps.append(probe_cmd)
    fail_cmd = [
        py, str(ROOT / "scripts" / "run_failure.py"),
        "--model-id", mid, "--device", args.device,
    ]
    if args.max_examples:
        fail_cmd.extend(["--max-examples", str(args.max_examples)])
    steps.append(fail_cmd)

    failed = []
    for cmd in steps:
        code = run(cmd)
        if code != 0:
            failed.append(cmd)
            print(f"FAILED exit={code}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} step(s) failed")
    print("\nOK strong suite. Read logs/intervene_ar.md, intervene_atr_*.md, probes_over_t.md, failure.md", flush=True)


if __name__ == "__main__":
    main()
