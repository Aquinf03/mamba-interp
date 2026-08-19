"""Token pool for synthetic AR/ATR tasks.

Symbols are single-token strings under the Mamba-130M tokenizer (GPT-NeoX BPE).
The pool is reused across train and eval splits; the tokenizer is bound at construction.
"""

from __future__ import annotations

import random
from typing import List, Sequence

from transformers import AutoTokenizer


# Single-character symbols that encode to exactly one token in the GPT-NeoX vocabulary.
# The full set is filtered down at init time; a subset ≥16 symbols is required.
DEFAULT_ALPHABET = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]


class TokenPool:
    """Alphabet of symbols bound to a tokenizer.

    By default only symbols that encode to a single token are kept, so that
    ``target_id = encode(value)[0]`` is unambiguous and teacher-forcing is clean.
    """

    def __init__(
        self,
        tokenizer_name: str = "state-spaces/mamba-130m-hf",
        alphabet: Sequence[str] | None = None,
        sep: str = " ",
        prefer_single_token: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sep = sep
        self.tokenizer_name = tokenizer_name
        alphabet = list(alphabet or DEFAULT_ALPHABET)
        if prefer_single_token:
            keep = [s for s in alphabet
                    if len(self.tokenizer.encode(s, add_special_tokens=False)) == 1]
            if len(keep) < 16:
                # Fall back: allow multi-token; eval still uses the first target token.
                keep = list(alphabet)
            self.symbols = keep
        else:
            self.symbols = list(alphabet)
        if len(self.symbols) < 8:
            raise RuntimeError(f"Token pool too small: {len(self.symbols)}")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def sample_unique(self, rng: random.Random, k: int) -> List[str]:
        if k > len(self.symbols):
            raise ValueError(f"need {k} unique symbols, pool has {len(self.symbols)}")
        return rng.sample(self.symbols, k)

    def sample_with_replacement(self, rng: random.Random, k: int) -> List[str]:
        return [rng.choice(self.symbols) for _ in range(k)]


def format_with_sep(pieces: Sequence[str], sep: str) -> str:
    return sep.join(pieces)
