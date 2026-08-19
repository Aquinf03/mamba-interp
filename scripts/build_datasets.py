"""Freeze eval splits to data/splits/ (CPU + tokenizer only, no GPU required).

Do NOT regenerate v1 under new seeds. Bump ``dataset_version`` in configs/datasets.yaml
first, then run with a new ``--out-dir`` to avoid overwriting the locked splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import (
    STUFF_CONFIGS,
    TokenPool,
    build_ar_split,
    build_atr_split,
    build_factual_split,
    build_length_suite,
    save_jsonl,
    save_manifest,
)
from src.eval.schema import SplitManifest


def _load_cfg(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML required: pip install pyyaml") from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  type=Path, default=ROOT / "configs" / "datasets.yaml")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    p.add_argument("--smoke",   action="store_true", help="tiny n for a quick sanity check")
    args = p.parse_args()

    cfg     = _load_cfg(args.config)
    seed    = int(cfg["seed"])
    n       = int(cfg["n_smoke"] if args.smoke else cfg["n_eval"])
    tok_name = cfg["tokenizer_name"]
    model_id = cfg["model_id"]
    version  = cfg["dataset_version"]

    print(f"building dataset {version} seed={seed} n={n} tokenizer={tok_name}", flush=True)
    pool = TokenPool(tokenizer_name=tok_name)
    out  = args.out_dir / version
    out.mkdir(parents=True, exist_ok=True)
    manifests = []

    # AR
    ar_cfg = cfg["ar"]
    ar     = build_ar_split(seed + 1, pool, split="eval", n=n,
                             n_pairs=int(ar_cfg["n_pairs"]),
                             n_queries=int(ar_cfg["n_queries"]))
    ar_path = out / "ar_eval.jsonl"
    save_jsonl(ar_path, ar)
    m = SplitManifest(task="ar", split="eval", seed=seed + 1, n=len(ar),
                      tokenizer_name=tok_name, model_id=model_id,
                      path=str(ar_path.relative_to(ROOT)), config=dict(ar_cfg))
    save_manifest(out / "ar_eval.manifest.json", m)
    manifests.append(m.to_dict())
    print(f"  ar: {len(ar)}  mean_len={sum(e.meta['seq_len'] for e in ar)/len(ar):.1f}", flush=True)

    # ATR short / mid
    for name, key, seed_off in (("atr_short", "atr_short", 2), ("atr_mid", "atr_mid", 3)):
        c = cfg[key]
        examples = build_atr_split(seed + seed_off, pool, split="eval", n=n,
                                   n_marks=int(c["n_marks"]),
                                   prefix_len=int(c["prefix_len"]),
                                   gap_len=int(c["gap_len"]))
        path = out / f"{name}_eval.jsonl"
        save_jsonl(path, examples)
        m = SplitManifest(task="atr", split=name, seed=seed + seed_off, n=len(examples),
                          tokenizer_name=tok_name, model_id=model_id,
                          path=str(path.relative_to(ROOT)), config=dict(c))
        save_manifest(out / f"{name}_eval.manifest.json", m)
        manifests.append(m.to_dict())
        print(f"  {name}: {len(examples)}  "
              f"mean_len={sum(e.meta['seq_len'] for e in examples)/len(examples):.1f}", flush=True)

    # Length / stuffing
    n_len = int(cfg["length"]["n_per_config"])
    if args.smoke:
        n_len = min(n_len, 8)
    length_map = build_length_suite(seed + 4, pool, split="eval",
                                    n_per_config=n_len, configs=STUFF_CONFIGS)
    for name, examples in length_map.items():
        path = out / f"length_{name}_eval.jsonl"
        save_jsonl(path, examples)
        m = SplitManifest(task="length", split=name, seed=seed + 4, n=len(examples),
                          tokenizer_name=tok_name, model_id=model_id,
                          path=str(path.relative_to(ROOT)),
                          config=dict(STUFF_CONFIGS[name]))
        save_manifest(out / f"length_{name}_eval.manifest.json", m)
        manifests.append(m.to_dict())
        mean_len = sum(e.meta["seq_len"] for e in examples) / len(examples)
        print(f"  length/{name}: {len(examples)}  mean_len={mean_len:.1f}", flush=True)

    # Factual (diagnostic - expected 0% after AR FT)
    if cfg.get("factual", True):
        facts = build_factual_split(pool, split="eval")
        path  = out / "factual_eval.jsonl"
        save_jsonl(path, facts)
        m = SplitManifest(task="factual", split="eval", seed=seed, n=len(facts),
                          tokenizer_name=tok_name, model_id=model_id,
                          path=str(path.relative_to(ROOT)),
                          config={"source": "fixed_fact_list"})
        save_manifest(out / "factual_eval.manifest.json", m)
        manifests.append(m.to_dict())
        print(f"  factual: {len(facts)}", flush=True)

    index = {
        "dataset_version": version,
        "seed": seed,
        "model_id": model_id,
        "tokenizer_name": tok_name,
        "n_symbols": len(pool.symbols),
        "manifests": manifests,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print("DO NOT regenerate without bumping dataset_version.", flush=True)


if __name__ == "__main__":
    main()
