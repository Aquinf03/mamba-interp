"""L17 control experiments: h vs residual restore (A1), restore-over-time (A2), neighbor layers (A3).

Writes logs/l17_restore_residual.md, l17_restore_time.md, l17_restore_neighbors.md.
Does not overwrite l17_restore.md (the primary sufficiency result).

  A1: compare restoring L17 h vs L17 residual skip at the same position.
      h recovers; residual does not - the bind is recurrent, not in the skip bus.
  A2: restore L17 h at three positions (value token, last write, query)
      after a late wipe. Localizes *when* the bind is fully formed in L17.
  A3: restore L16 / L17 / L18 h at last write after a late wipe.
      Only L17 recovers - the result is layer-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.io import load_jsonl
from src.eval.schema import EvalExample
from src.intervene.run import results_to_markdown, run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import ar_windows, layer_group


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--l17", type=int, default=17)
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "logs")
    p.add_argument(
        "--only",
        default="all",
        choices=("all", "residual"),
        help="residual = A1 only (h vs residual restore)",
    )
    p.add_argument(
        "--tag",
        default="",
        help="filename suffix, e.g. atr -> l17_restore_residual_atr.md",
    )
    return p.parse_args()


def site_index(ex: EvalExample, site: str) -> int:
    wins = ar_windows(ex)
    if site == "last_write":
        w = wins.get("last_write") or wins.get("write") or []
        if not w:
            raise SystemExit(f"{ex.example_id}: no last_write")
        return int(w[-1])
    slot = wins.get(site) or []
    if not slot:
        raise SystemExit(f"{ex.example_id}: no window {site}")
    return int(slot[0])


def slice_h(
    h_seqs: Sequence[Dict[int, torch.Tensor]],
    examples: Sequence[EvalExample],
    layer: int,
    site: str,
) -> List[Dict[int, torch.Tensor]]:
    out: List[Dict[int, torch.Tensor]] = []
    for seq, ex in zip(h_seqs, examples):
        if layer not in seq:
            raise SystemExit(f"missing h_seq layer {layer}")
        t = site_index(ex, site)
        ten = seq[layer]
        if t >= int(ten.shape[0]):
            raise SystemExit(f"{ex.example_id}: t={t} vs T={ten.shape[0]}")
        out.append({layer: ten[t].contiguous()})
    return out


def slice_res(
    res_seqs: Sequence[Dict[int, torch.Tensor]],
    examples: Sequence[EvalExample],
    layer: int,
    site: str,
) -> List[Dict[int, torch.Tensor]]:
    out: List[Dict[int, torch.Tensor]] = []
    for seq, ex in zip(res_seqs, examples):
        if layer not in seq:
            raise SystemExit(f"missing residual_seq layer {layer}")
        t = site_index(ex, site)
        ten = seq[layer]
        if t >= int(ten.shape[0]):
            raise SystemExit(f"{ex.example_id}: residual t={t} vs T={ten.shape[0]}")
        out.append({layer: ten[t].contiguous()})
    return out


def go(
    model,
    examples,
    spec: Optional[Intervention],
    *,
    device: str,
    layers: Sequence[int],
    clean_scores,
    seed: int,
    donors=None,
    donor_residuals=None,
):
    print(f"== {spec.name if spec else 'clean'} ==", flush=True)
    scores, _, _ = run_condition(
        model,
        examples,
        spec,
        device=device,
        layers=layers,
        capture=False,
        donors=donors,
        donor_residuals=donor_residuals,
        seed=seed,
    )
    row = summarize_condition(spec, scores, clean_scores, layers=layers)
    print(f"  acc={row.acc:.3f}", flush=True)
    return row


def write_md(path: Path, title: str, rows, note: str) -> None:
    md = results_to_markdown(rows, title=title)
    md += "\n" + note.strip() + "\n"
    path.write_text(md, encoding="utf-8")
    print(md.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    examples = load_jsonl(args.split)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    n_layers = len(model.backbone.layers)
    late = layer_group(n_layers, "late")
    L = args.l17
    L16, L18 = L - 1, L + 1
    if L16 not in late or L18 not in late:
        raise SystemExit(f"neighbors L{L16}/L{L18} not in late {late[0]}-{late[-1]}")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cap: dict = {}
    print("== clean capture (h_seq + residual_seq, late layers) ==", flush=True)
    clean_scores, _, _ = run_condition(
        model,
        examples,
        None,
        device=args.device,
        layers=late,
        capture=True,
        seed=args.seed,
        captures=cap,
    )
    clean_row = summarize_condition(None, clean_scores, None, layers=late)
    print(f"  clean acc={clean_row.acc:.3f}", flush=True)
    h_seqs = cap.get("h_seq") or []
    res_seqs = cap.get("residual_seq") or []
    if len(h_seqs) != len(examples) or len(res_seqs) != len(examples):
        raise SystemExit(f"capture mismatch h={len(h_seqs)} res={len(res_seqs)} n={len(examples)}")

    wipe = Intervention("zero_h_write_late", "h", "zero", "write", late)
    wipe_row = go(model, examples, wipe, device=args.device, layers=late, clean_scores=clean_scores, seed=args.seed)

    h_lw = slice_h(h_seqs, examples, L, "last_write")
    r_lw = slice_res(res_seqs, examples, L, "last_write")
    r_q = slice_res(res_seqs, examples, L, "query")
    tag = f"_{args.tag}" if args.tag else ""

    rest_h = Intervention(
        "restore_L17_h_last_write", "h", "restore", "write", late, restore_window="last_write"
    )
    rest_h_row = go(
        model, examples, rest_h, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donors=h_lw,
    )

    rest_r_lw = Intervention(
        "restore_L17_residual_last_write", "residual", "restore", "write", late,
        restore_window="last_write",
    )
    rest_r_lw_row = go(
        model, examples, rest_r_lw, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donor_residuals=r_lw,
    )

    rest_r_q = Intervention(
        "restore_L17_residual_query", "residual", "restore", "write", late,
        restore_window="query",
    )
    rest_r_q_row = go(
        model, examples, rest_r_q, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donor_residuals=r_q,
    )

    write_md(
        out / f"l17_restore_residual{tag}.md",
        f"A1 L17 restore h vs residual{f' ({args.tag})' if args.tag else ''}",
        [clean_row, wipe_row, rest_h_row, rest_r_lw_row, rest_r_q_row],
        (
            f"split: `{args.split}`\n\n"
            "Late h wipe on write, then copy **this example's** clean L17 vector at one site.\n"
            "- restore h at last write: sufficiency (must ~clean).\n"
            "- restore residual skip at last write: same site, residual is not recurrent (must stay near wipe).\n"
            "- restore residual skip at query: fair readout control. If this recovers, residual at query is sufficient "
            "and C3 is only about the store during write."
        ),
    )
    (out / f"l17_restore_residual{tag}.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "split": str(args.split),
                "n": len(examples),
                "results": [r.to_dict() for r in [clean_row, wipe_row, rest_h_row, rest_r_lw_row, rest_r_q_row]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.only == "residual":
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        print(f"peak_vram_gb={peak:.3f} wall_s={time.time() - t0:.1f}", flush=True)
        return

    h_val = slice_h(h_seqs, examples, L, "value")
    h_q = slice_h(h_seqs, examples, L, "query")
    h16 = slice_h(h_seqs, examples, L16, "last_write")
    h18 = slice_h(h_seqs, examples, L18, "last_write")

    rest_val = Intervention(
        "restore_L17_h_value", "h", "restore", "write", late, restore_window="value"
    )
    rest_val_row = go(
        model, examples, rest_val, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donors=h_val,
    )
    rest_q = Intervention(
        "restore_L17_h_query", "h", "restore", "write", late, restore_window="query"
    )
    rest_q_row = go(
        model, examples, rest_q, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donors=h_q,
    )
    write_md(
        out / f"l17_restore_time{tag}.md",
        "A2 L17 h restore over time",
        [clean_row, wipe_row, rest_val_row, rest_h_row, rest_q_row],
        (
            "After late h wipe, paste this example's clean L17 h at one token. "
            "Tokens before the site stay wiped; after the site, recurrence runs. "
            "Value vs last-write vs query localizes *when* the bind is in L17 h."
        ),
    )
    (out / f"l17_restore_time{tag}.json").write_text(
        json.dumps(
            {"results": [r.to_dict() for r in [clean_row, wipe_row, rest_val_row, rest_h_row, rest_q_row]]},
            indent=2,
        ),
        encoding="utf-8",
    )

    rest_16 = Intervention(
        "restore_L16_h_last_write", "h", "restore", "write", late, restore_window="last_write"
    )
    rest_16_row = go(
        model, examples, rest_16, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donors=h16,
    )
    rest_18 = Intervention(
        "restore_L18_h_last_write", "h", "restore", "write", late, restore_window="last_write"
    )
    rest_18_row = go(
        model, examples, rest_18, device=args.device, layers=late,
        clean_scores=clean_scores, seed=args.seed, donors=h18,
    )
    write_md(
        out / f"l17_restore_neighbors{tag}.md",
        "A3 restore L16 / L17 / L18 h",
        [clean_row, wipe_row, rest_16_row, rest_h_row, rest_18_row],
        (
            "Same late wipe + last-write restore, but paste neighbor-layer h instead of L17. "
            "Neighbors must not recover if sufficiency is L17-specific."
        ),
    )
    (out / f"l17_restore_neighbors{tag}.json").write_text(
        json.dumps(
            {"results": [r.to_dict() for r in [clean_row, wipe_row, rest_16_row, rest_h_row, rest_18_row]]},
            indent=2,
        ),
        encoding="utf-8",
    )

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"peak_vram_gb={peak:.3f} wall_s={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
