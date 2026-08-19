"""AR acc + L17 write-wipe vs finetune step (needs runs/ar_ft_trace/step_*)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.io import load_jsonl
from src.intervene.run import run_condition, summarize_condition
from src.intervene.spec import Intervention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "runs" / "ar_ft_trace")
    p.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--l17", type=int, default=17)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--out-md", type=Path, default=ROOT / "logs" / "l17_emergence.md")
    p.add_argument("--out-json", type=Path, default=ROOT / "logs" / "l17_emergence.json")
    return p.parse_args()


def list_ckpts(root: Path) -> list[tuple[int, Path]]:
    found = []
    for p in sorted(root.glob("step_*")):
        if p.is_dir():
            try:
                found.append((int(p.name.split("_", 1)[1]), p))
            except ValueError:
                continue
    final = root / "checkpoint"
    if final.is_dir():
        found.append((10**9, final))
    found.sort()
    return found


def main() -> None:
    args = parse_args()
    ckpts = list_ckpts(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(
            f"no step_* under {args.ckpt_dir}. Run:\n"
            f"  python scripts\\finetune_ar.py --out-dir {args.ckpt_dir} --save-every 200"
        )
    examples = load_jsonl(args.split)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    dtype = getattr(torch, args.dtype)
    spec = Intervention(f"zero_h_L{args.l17}", "h", "zero", "write", [args.l17])
    rows = []
    t0 = time.time()
    for step, path in ckpts:
        label = "final" if step >= 10**9 else str(step)
        print(f"loading {path} ({label}) ...", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = MambaForCausalLM.from_pretrained(str(path), torch_dtype=dtype)
        model = model.to(args.device)
        model.eval()
        clean_s, _, _ = run_condition(
            model, examples, None, device=args.device, layers=[args.l17], capture=False
        )
        wipe_s, _, _ = run_condition(
            model, examples, spec, device=args.device, layers=[args.l17], capture=False
        )
        cr = summarize_condition(None, clean_s, None, layers=[args.l17])
        wr = summarize_condition(spec, wipe_s, clean_s, layers=[args.l17])
        rec = {
            "step": None if label == "final" else step,
            "path": str(path),
            "clean_acc": cr.acc,
            "l17_wipe_acc": wr.acc,
            "drop": cr.acc - wr.acc,
        }
        rows.append(rec)
        print(f"  clean={cr.acc:.3f} L{args.l17}_wipe={wr.acc:.3f} drop={rec['drop']:.3f}", flush=True)
        del model
    payload = {"ckpt_dir": str(args.ckpt_dir), "l17": args.l17, "n": len(examples), "rows": rows,
               "wall_s": round(time.time() - t0, 2)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# L17 store vs finetune step",
        "",
        "| step | clean_acc | L17_wipe_acc | drop |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        st = "final" if r["step"] is None else r["step"]
        lines.append(f"| {st} | {r['clean_acc']:.3f} | {r['l17_wipe_acc']:.3f} | {r['drop']:.3f} |")
    lines.append("")
    lines.append(
        "Drop rising while clean acc rises => the L17 store is learned during FT, not inherited."
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print("\n".join(lines).encode("ascii", "replace").decode("ascii"))
    print(f"wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
