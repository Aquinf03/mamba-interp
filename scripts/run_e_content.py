"""Extended L17 analysis: channel wipes, SVD editing, C-patch, MLP probe, multi-query, and implicit maps.

  E1: wipe k N-channels by random / top-energy selection → how spread is the bind?
  E2: replace h with SVD reconstruction at last write → rank structure of the bind.
  E3: patch donor C at the query (same keys, different values) → C alone is not enough.
  E4: MLP probe at last_write → even a nonlinear probe cannot decode the value letter.
  E5: multi-query AR (2 queries) + L17 wipe → both binds live in L17 h.
  E6: hidden-attention-style score (h · C_query) as a control map.

Writes new logs only. Multi-query split goes to data/splits/v2/ (v1 is unchanged).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import MambaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = str(ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from src.eval.ar import build_ar_split
from src.eval.io import load_jsonl, save_jsonl, save_manifest
from src.eval.schema import SplitManifest
from src.eval.vocab import TokenPool
from src.hooks import capture_mamba_states
from src.intervene.run import run_condition, summarize_condition
from src.intervene.spec import Intervention
from src.intervene.windows import queried_value_index, token_roles
from src.probes.linear import FEATURE_FNS, fit_linear_probe, fit_mlp_probe
from src.probes.over_t import make_probe_split, site_index

import run_l17 as l17


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=str(ROOT / "runs" / "ar_ft" / "checkpoint"))
    p.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "v1" / "ar_eval.jsonl")
    p.add_argument("--l17", type=int, default=17)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--n-mlp", type=int, default=384)
    p.add_argument("--n-keypatch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "logs")
    return p.parse_args()


def ascii_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def go(model, examples, spec, *, device, layers, clean=None, seed=0, **kw):
    print(f"== {spec.name if spec else 'clean'} ==", flush=True)
    scores, _, hs = run_condition(
        model, examples, spec, device=device, layers=layers, seed=seed, **kw
    )
    row = summarize_condition(spec, scores, clean, layers=layers)
    print(f"  acc={row.acc:.3f}", flush=True)
    return row, scores, hs


def svd_edit(h: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """h [E, N] -> same shape, float then back."""
    mat = h.float()
    if mat.dim() == 3:
        mat = mat[0]
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    S2 = S.clone()
    k = max(0, min(int(k), int(S2.numel())))
    if mode == "zero_top":
        S2[:k] = 0
    elif mode == "keep_top":
        S2[k:] = 0
    else:
        raise ValueError(mode)
    rec = (U * S2) @ Vh
    return rec.to(dtype=h.dtype)


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ascii_print("\n".join(lines))
    print(f"wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    examples = load_jsonl(args.split)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    dtype = getattr(torch, args.dtype)
    L = args.l17
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model_id} ...", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = MambaForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()
    n_state = int(model.backbone.layers[L].mixer.ssm_state_size)

    # ---- clean capture L17 h ----
    print("== clean L17 capture ==", flush=True)
    clean_scores, _, clean_h = run_condition(
        model, examples, None, device=args.device, layers=[L], capture=True, seed=args.seed
    )
    clean_row = summarize_condition(None, clean_scores, None, layers=[L])
    print(f"  clean acc={clean_row.acc:.3f}", flush=True)
    h_lw = []
    for row in clean_h:
        if L not in row:
            raise SystemExit("missing L17 h_write_end")
        h_lw.append(row[L])

    # energy per N-channel, mean over examples
    energy = torch.stack([h.float().pow(2).mean(dim=0) for h in h_lw], dim=0).mean(0)  # [N]
    order = torch.argsort(energy, descending=True).tolist()
    rng = random.Random(args.seed + 3)

    e1_rows = [clean_row]
    e1_md = [
        "# E1 L17 multi-channel wipes",
        "",
        f"N={n_state}. Single-N already ~clean. Random = one fixed subset. "
        "top = highest mean energy over examples at last write.",
        "",
        "| name | k | how | acc |",
        "|---|---:|---|---:|",
        f"| clean | 0 | - | {clean_row.acc:.3f} |",
    ]
    for k in (2, 4, 8):
        rand_idx = sorted(rng.sample(range(n_state), k))
        top_idx = sorted(order[:k])
        for how, idx in (("random", rand_idx), ("top_energy", top_idx)):
            spec = Intervention(f"zero_h_L17_k{k}_{how}", "h", "zero", "write", [L], channels=idx)
            row, _, _ = go(model, examples, spec, device=args.device, layers=[L], clean=clean_scores, seed=args.seed)
            e1_rows.append(row)
            e1_md.append(f"| {spec.name} | {k} | {how} {idx} | {row.acc:.3f} |")
        # per-example top-k
        sets = []
        for h in h_lw:
            en = h.float().pow(2).mean(dim=0)
            sets.append(torch.argsort(en, descending=True)[:k].tolist())
        spec = Intervention(f"zero_h_L17_k{k}_top_per_ex", "h", "zero", "write", [L], channels=list(range(k)))
        row, _, _ = go(
            model, examples, spec, device=args.device, layers=[L], clean=clean_scores,
            seed=args.seed, channel_sets=sets,
        )
        e1_rows.append(row)
        e1_md.append(f"| {spec.name} | {k} | per-example top energy | {row.acc:.3f} |")
    e1_md.append("")
    e1_md.append("If k=8 still ~clean, the bind is spread across N. If top-energy k=2 dies, a few slots carry it.")
    write(out / "l17_channels_k.md", e1_md)
    (out / "l17_channels_k.json").write_text(
        json.dumps({"energy": energy.tolist(), "order": order, "results": [r.to_dict() for r in e1_rows]}, indent=2),
        encoding="utf-8",
    )

    # ---- E2 SVD ----
    e2_md = [
        "# E2 L17 h SVD at last write",
        "",
        "Replace last-write L17 h with SVD reconstruction (no late wipe). "
        "zero_top k = drop largest k singular values. keep_top k = keep only those.",
        "",
        "| name | acc |",
        "|---|---:|",
        f"| clean | {clean_row.acc:.3f} |",
    ]
    e2_rows = [clean_row]
    for mode, k in (("zero_top", 1), ("zero_top", 2), ("zero_top", 4), ("zero_top", 8),
                    ("keep_top", 1), ("keep_top", 4), ("keep_top", 8)):
        donors = [{L: svd_edit(h, k, mode)} for h in h_lw]
        spec = Intervention(f"svd_{mode}_{k}", "h", "swap", "write", [L], restore_window="last_write")
        row, _, _ = go(
            model, examples, spec, device=args.device, layers=[L], clean=clean_scores,
            seed=args.seed, donors=donors,
        )
        e2_rows.append(row)
        e2_md.append(f"| {spec.name} | {row.acc:.3f} |")
    e2_md.append("")
    e2_md.append("If zero_top 1–2 does nothing and keep_top 1 fails, the code is distributed in rank, not one component.")
    write(out / "l17_svd.md", e2_md)
    (out / "l17_svd.json").write_text(
        json.dumps({"results": [r.to_dict() for r in e2_rows]}, indent=2), encoding="utf-8"
    )

    # ---- E3 C patch (key-matched) ----
    pool = TokenPool(tokenizer_name=args.model_id)
    nkp = args.n_keypatch if args.max_examples <= 0 else min(args.n_keypatch, args.max_examples)
    cleans, donors = l17.keypatch_pairs(pool, nkp, args.seed + 7)
    print(f"== E3 keypatch C at query n={nkp} ==", flush=True)
    donor_C = []
    d_scores, _, _ = run_condition(
        model, donors, None, device=args.device, layers=[L], capture=False, seed=args.seed
    )
    c_scores, _, _ = run_condition(
        model, cleans, None, device=args.device, layers=[L], capture=False, seed=args.seed
    )
    for ex in donors:
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=args.device)
        with capture_mamba_states(model, layers=[L], keep_on_cpu=True) as cap:
            model(input_ids=ids, use_cache=False)
        tq = len(ex.input_ids) - 1
        donor_C.append({L: cap.layers[L].C[0, tq].contiguous()})
    spec = Intervention("keypatch_L17_C_query", "C", "swap", "query", [L])
    p_scores, _, _ = run_condition(
        model, cleans, spec, device=args.device, layers=[L], capture=False,
        donor_C=donor_C, seed=args.seed,
    )
    kp = {
        "n": nkp,
        "clean_acc": sum(1 for s in c_scores if s.correct) / nkp,
        "donor_seq_acc": sum(1 for s in d_scores if s.correct) / nkp,
        "patch_acc_original_value": sum(1 for s in p_scores if s.correct) / nkp,
        "patch_acc_donor_value": sum(
            1 for s, d in zip(p_scores, donors) if int(s.pred_id) == int(d.target_id)
        )
        / nkp,
    }
    e3 = [
        "# E3 L17 C patch at query (key-matched)",
        "",
        "Same keys; donor has a different value. Patch donor L17 **C at query** onto the clean sequence (h untouched).",
        "",
        "| n | clean | donor-seq | still original V | **now donor V** |",
        "|---:|---:|---:|---:|---:|",
        f"| {nkp} | {kp['clean_acc']:.3f} | {kp['donor_seq_acc']:.3f} | "
        f"{kp['patch_acc_original_value']:.3f} | {kp['patch_acc_donor_value']:.3f} |",
        "",
        "If donor V stays ~0, readout C is not a portable address for that key; store stays in h.",
        "",
    ]
    write(out / "l17_C_patch.md", e3)
    (out / "l17_C_patch.json").write_text(json.dumps(kp, indent=2), encoding="utf-8")

    # ---- E6 implicit C-h map on frozen AR (uses captures) ----
    print("== E6 implicit maps ==", flush=True)
    masses = {"queried_value": [], "other_value": [], "key": [], "query": [], "other": []}
    for i, ex in enumerate(examples):
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=args.device)
        with capture_mamba_states(model, layers=[L], keep_on_cpu=True) as cap:
            model(input_ids=ids, use_cache=False)
        tr = cap.layers[L]
        tq = len(ex.input_ids) - 1
        h = tr.h[0].float()  # [T,E,N]
        Cq = tr.C[0, tq].float()  # [N]
        score = (h * Cq).sum(dim=-1).mean(dim=-1)  # [T]
        attn = torch.softmax(score, dim=0)
        roles = token_roles(ex)
        vi = queried_value_index(ex)
        acc = {"queried_value": 0.0, "other_value": 0.0, "key": 0.0, "query": 0.0, "other": 0.0}
        for t, a in enumerate(attn.tolist()):
            r = roles[t] if t < len(roles) else "other"
            if vi is not None and t == vi:
                acc["queried_value"] += a
            elif r == "value":
                acc["other_value"] += a
            elif r == "key":
                acc["key"] += a
            elif r == "query":
                acc["query"] += a
            else:
                acc["other"] += a
        for k, v in acc.items():
            masses[k].append(v)
        if (i + 1) % 16 == 0:
            print(f"    {i + 1}/{len(examples)}", flush=True)

    def _m(xs):
        return sum(xs) / len(xs) if xs else 0.0

    e6 = [
        "# E6 L17 Hidden-Attention-style map (control)",
        "",
        f"score_t = mean_E (h_t · C_query). Softmax over t. Frozen AR n={len(examples)}. Not a reimplementation of Hidden Attention / LaTIM.",
        "",
        "| mass on | mean softmax |",
        "|---|---:|",
    ]
    for k in ("queried_value", "other_value", "key", "query", "other"):
        e6.append(f"| {k} | {_m(masses[k]):.3f} |")
    e6.append("")
    e6.append(
        "If queried-value mass is high, token maps *highlight* the bind; they still do not prove the store. "
        "Our causal result is L17 h wipe/restore, not this heatmap."
    )
    write(out / "control_maps.md", e6)
    (out / "control_maps.json").write_text(
        json.dumps({k: _m(v) for k, v in masses.items()}, indent=2), encoding="utf-8"
    )

    # ---- E5 multi-query v2 ----
    v2 = ROOT / "data" / "splits" / "v2"
    v2.mkdir(parents=True, exist_ok=True)
    mq = build_ar_split(
        2026, pool, split="mq_eval", n=len(examples), n_pairs=4, n_queries=2
    )
    save_jsonl(v2 / "ar_mq_eval.jsonl", mq)
    save_manifest(
        v2 / "ar_mq_eval.manifest.json",
        SplitManifest(
            task="ar",
            split="mq_eval",
            seed=2026,
            n=len(mq),
            tokenizer_name=args.model_id,
            model_id=args.model_id,
            path="data/splits/v2/ar_mq_eval.jsonl",
            config={"n_pairs": 4, "n_queries": 2},
        ),
    )
    print(f"== E5 multi-query n={len(mq)} ==", flush=True)
    mq_clean, mq_cs, _ = go(model, mq, None, device=args.device, layers=[L], seed=args.seed)
    wipe = Intervention("zero_h_write_L17", "h", "zero", "write", [L])
    mq_w, mq_ws, _ = go(model, mq, wipe, device=args.device, layers=[L], clean=mq_cs, seed=args.seed)

    # score first query (teacher-forced): token after first extra key = 2*n_pairs
    first_hits = 0
    first_n = 0
    for ex in mq:
        n_pairs = int(ex.meta.get("n_pairs") or 4)
        t_q1 = 2 * n_pairs  # extra query key
        if t_q1 >= len(ex.input_ids) - 1:
            continue
        q1_key = (ex.meta.get("query_keys") or [None])[0]
        pairs = {p["key"]: p["value"] for p in (ex.meta.get("pairs") or [])}
        if q1_key not in pairs:
            continue
        tgt = pool.encode(pairs[q1_key])[0]
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, t_q1]
        first_n += 1
        if int(logits.argmax().item()) == int(tgt):
            first_hits += 1
    first_acc = first_hits / first_n if first_n else 0.0
    e5 = [
        "# E5 multi-query AR + L17 wipe",
        "",
        "v2 split (`data/splits/v2/ar_mq_eval.jsonl`): 4 pairs, 2 queries. Intermediate query is answered in-transcript; final query is scored as usual. v1 untouched.",
        "",
        f"| condition | n | final-query acc | first-query acc (teacher-forced) |",
        f"|---|---:|---:|---:|",
        f"| clean | {len(mq)} | {mq_clean.acc:.3f} | {first_acc:.3f} |",
        f"| L17 write-wipe | {len(mq)} | {mq_w.acc:.3f} |  |",
        "",
        "If final-query acc dies under L17 wipe, the second bind also lives in L17 h.",
        "",
    ]
    write(out / "l17_multiquery.md", e5)
    (out / "l17_multiquery.json").write_text(
        json.dumps(
            {
                "n": len(mq),
                "clean_final": mq_clean.acc,
                "wipe_final": mq_w.acc,
                "clean_first_tf": first_acc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- E4 MLP last_write ----
    print(f"== E4 MLP last_write n={args.n_mlp} ==", flush=True)
    probe_ex = make_probe_split(args.model_id, n=args.n_mlp, seed=args.seed + 99)
    feats = ("residual", "h_mean_n", "h_flat")
    bucket = {f: [] for f in feats}
    y = []
    for i, ex in enumerate(probe_ex):
        ids = torch.tensor([ex.input_ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            with capture_mamba_states(model, layers=[L], keep_on_cpu=True) as cap:
                model(input_ids=ids, use_cache=False)
        tr = cap.layers[L]
        t = site_index(ex, "last_write")
        y.append(int(ex.target_id))
        for f in feats:
            bucket[f].append(FEATURE_FNS[f](tr, t=t))
        if (i + 1) % 16 == 0:
            print(f"    {i + 1}/{len(probe_ex)}", flush=True)
    n = len(probe_ex)
    g = torch.Generator().manual_seed(args.seed + 99)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = max(8, int(0.75 * n))
    tr_i, te_i = perm[:n_train], perm[n_train:]
    y_t = torch.tensor(y)
    e4_lines = [
        "# E4 MLP vs linear at last_write L17",
        "",
        f"fresh AR n={n} seed={args.seed + 99}, not v1. Chance ~ 1/n_classes. Linear last_write was ~0.10 in probes_over_t.",
        "",
        "| feature | dim | linear train | linear test | MLP train | MLP test | n_classes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    e4_json = []
    for f in feats:
        X = torch.stack(bucket[f], dim=0)
        X_tr, y_tr = X[tr_i], y_t[tr_i]
        X_te, y_te = X[te_i], y_t[te_i]
        mu = X_tr.mean(0, keepdim=True)
        sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        lin_tr, lin_te, n_cls = fit_linear_probe(X_tr, y_tr, X_te, y_te, steps=400, seed=args.seed)
        mlp_tr, mlp_te, _ = fit_mlp_probe(X_tr, y_tr, X_te, y_te, steps=800, seed=args.seed)
        e4_lines.append(
            f"| {f} | {X.shape[1]} | {lin_tr:.3f} | {lin_te:.3f} | {mlp_tr:.3f} | {mlp_te:.3f} | {n_cls} |"
        )
        e4_json.append(
            {"feature": f, "dim": int(X.shape[1]), "linear_train": lin_tr, "linear_test": lin_te,
             "mlp_train": mlp_tr, "mlp_test": mlp_te, "n_classes": n_cls}
        )
        print(f"  {f} lin_te={lin_te:.3f} mlp_te={mlp_te:.3f}", flush=True)
    e4_lines.append("")
    e4_lines.append("If MLP test stays near chance, last_write has no easily decoded value code (superposed / key-addressed).")
    write(out / "probes_mlp_lastwrite.md", e4_lines)
    (out / "probes_mlp_lastwrite.json").write_text(json.dumps(e4_json, indent=2), encoding="utf-8")

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"peak_vram_gb={peak:.3f} wall_s={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
