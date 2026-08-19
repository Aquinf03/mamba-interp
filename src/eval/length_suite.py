"""Length / stuffed-capacity eval suite built on the AR backbone.

``STUFF_CONFIGS`` is frozen - do not renumber without bumping the dataset version.
Each config produces an independent random stream so editing one ``n`` does not
reshuffle any other split.
"""

from __future__ import annotations

from typing import Dict, List

from .ar import build_ar_split
from .schema import EvalExample
from .vocab import TokenPool


STUFF_CONFIGS: Dict[str, dict] = {
    "stuff_light": {"n_pairs": 4, "filler_every": 1, "filler_n": 2, "ood_pad": 0, "n_queries": 1},
    "stuff_heavy": {"n_pairs": 8, "filler_every": 1, "filler_n": 4, "ood_pad": 0, "n_queries": 1},
    "len_short":   {"n_pairs": 2, "filler_every": 0, "filler_n": 0, "ood_pad": 0,  "n_queries": 1},
    "len_mid":     {"n_pairs": 8, "filler_every": 0, "filler_n": 0, "ood_pad": 0,  "n_queries": 1},
    "len_ood_pad": {"n_pairs": 4, "filler_every": 0, "filler_n": 0, "ood_pad": 32, "n_queries": 1},
}


def build_length_suite(
    seed: int,
    pool: TokenPool,
    *,
    split: str,
    n_per_config: int,
    configs: Dict[str, dict] | None = None,
) -> Dict[str, List[EvalExample]]:
    """Build all length/stuffing splits. Each config gets an independent RNG stream."""
    configs = configs or STUFF_CONFIGS
    out: Dict[str, List[EvalExample]] = {}
    for i, (name, cfg) in enumerate(sorted(configs.items())):
        examples = build_ar_split(
            seed + 1000 * (i + 1),
            pool,
            split=f"{split}/{name}",
            n=n_per_config,
            n_pairs=cfg["n_pairs"],
            n_queries=cfg.get("n_queries", 1),
            filler_every=cfg.get("filler_every", 0),
            filler_n=cfg.get("filler_n", 0),
            ood_pad=cfg.get("ood_pad", 0),
        )
        for ex in examples:
            ex.task = "length"
            ex.meta["suite_config"] = name
            ex.meta["suite_params"] = dict(cfg)
            ex.example_id = ex.example_id.replace("ar_", f"len_{name}_", 1)
        out[name] = examples
    return out
