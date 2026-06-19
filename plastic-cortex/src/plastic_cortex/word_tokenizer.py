"""Word-level tokenizer for faster CPU-native training.

Splits on whitespace and punctuation, keeps a fixed small vocabulary of the
most frequent words, and maps everything else to ``<UNK>``. This reduces
sequence length 5-10x compared to character-level models while staying
pure Python + NumPy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from collections import Counter


DEFAULT_SPECIALS = ["<PAD>", "<UNK>", "<EOS>"]


class WordTokenizer:
    """Map strings to integer word-token sequences and back.

    Args:
        vocab_size: Maximum number of word tokens (excluding specials).
            Lower is faster; default 1000.
        specials: Ordered list of special tokens to prepend.
    """

    def __init__(self, vocab_size: int = 1000, specials: list[str] | None = None) -> None:
        self._vocab_size = vocab_size
        self._specials = list(specials if specials is not None else DEFAULT_SPECIALS)
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self, words: list[str] | None = None) -> None:
        self._token_to_id.clear()
        self._id_to_token.clear()
        for idx, tok in enumerate(self._specials):
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok
        if words:
            for idx, word in enumerate(words, start=len(self._specials)):
                if word not in self._token_to_id:
                    self._token_to_id[word] = idx
                    self._id_to_token[idx] = word

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<UNK>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def max_vocab_size(self) -> int:
        return self._vocab_size + len(self._specials)

    def _words(self, text: str) -> list[str]:
        # Keep contractions and hyphenated words; split on spaces and punctuation.
        tokens = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*|[^a-zA-Z0-9\s]", text)
        return [t.lower() for t in tokens]

    def fit(self, texts: Iterable[str]) -> "WordTokenizer":
        """Build vocabulary from the most frequent words in *texts*."""
        counts = Counter[str]()
        for text in texts:
            counts.update(self._words(text))
        most_common = [w for w, _ in counts.most_common(self._vocab_size)]
        self._build_vocab(most_common)
        return self

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of integer ids."""
        tokens = self._words(text)
        return [self._token_to_id.get(tok, self.unk_id) for tok in tokens] + [self.eos_id]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of integer ids back to a string."""
        out: list[str] = []
        last_needs_space = False
        for tok in tokens:
            if tok == self.eos_id:
                break
            word = self._id_to_token.get(tok, "<UNK>")
            if word == "<PAD>":
                continue
            is_punct = len(word) == 1 and not word.isalnum()
            if is_punct:
                if out and out[-1] == " ":
                    out.pop()
                out.append(word)
                last_needs_space = True
            else:
                if last_needs_space and out:
                    out.append(" ")
                out.append("[?]" if word == "<UNK>" else word)
                last_needs_space = True
        return "".join(out)

    def save(self, path: str | Path) -> None:
        """Serialize the tokenizer to JSON."""
        payload = {
            "type": "WordTokenizer",
            "vocab_size": self._vocab_size,
            "specials": self._specials,
            "words": [self._id_to_token[i] for i in range(len(self._specials), len(self._token_to_id))],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=payload["vocab_size"], specials=payload["specials"])
        tok._build_vocab(payload.get("words", []))
        return tok
