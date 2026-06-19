"""Byte-pair encoding (BPE) tokenizer for out-of-vocabulary handling.

Starts with bytes 0-255 as the initial vocabulary and greedily merges the
most frequent adjacent token pairs until the requested vocabulary size is
reached. Because every UTF-8 character decomposes into in-vocabulary bytes,
there are no out-of-vocabulary gaps for text.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    """Map strings to integer BPE-token sequences and back."""

    def __init__(self, vocab_size: int = 500) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (bytes 0-255)")
        self._vocab_size = vocab_size
        self.merges: list[tuple[int, int]] = []
        self._merge_to_id: dict[tuple[int, int], int] = {}
        self._id_to_token: dict[int, tuple[int, ...]] = {}
        self._eos_id: int | None = None
        self._build_init_vocab()

    def _build_init_vocab(self) -> None:
        self._id_to_token.clear()
        self._merge_to_id.clear()
        for i in range(256):
            self._id_to_token[i] = (i,)

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)
    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token id."""
        if self._eos_id is None:
            raise RuntimeError("Tokenizer has not been fitted yet")
        return self._eos_id

    def _text_to_bytes(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def fit(self, texts: str | Iterable[str]) -> "BPETokenizer":
        """Learn BPE merges from *texts*.

        *texts* may be a single string or an iterable of strings (e.g. lines).
        """
        if isinstance(texts, str):
            corpus = texts
        else:
            corpus = "\n".join(texts)

        ids = self._text_to_bytes(corpus)
        vocab = {i: (i,) for i in range(256)}  # id -> byte expansion
        merges: list[tuple[int, int]] = []
        merge_to_id: dict[tuple[int, int], int] = {}

        while len(vocab) < self._vocab_size - 1 and len(ids) > 1:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += 1

            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < 1:
                break

            new_id = len(vocab)
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merge_to_id[best_pair] = new_id

            # Replace all non-overlapping occurrences of the pair.
            new_ids: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            merges.append(best_pair)

        self.merges = merges
        self._merge_to_id = merge_to_id
        self._id_to_token = vocab
        self._eos_id = len(vocab)
        self._id_to_token[self._eos_id] = ()
        return self

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of integer ids."""
        ids = self._text_to_bytes(text)
        for pair in self.merges:
            new_id = self._merge_to_id[pair]
            a, b = pair
            new_ids: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of integer ids back to a string."""
        bytes_out = bytearray()
        # Process tokens left-to-right using a depth-first stack expansion.
        stack = list(reversed(tokens))
        while stack:
            idx = stack.pop()
            if idx == self._eos_id:
                continue
            expansion = self._id_to_token.get(idx)
            if expansion is None:
                raise ValueError(f"Token id {idx} is not in the vocabulary")
            if len(expansion) == 1 and expansion[0] < 256:
                bytes_out.append(expansion[0])
            elif expansion == ():
                continue
            else:
                # Push children so the leftmost byte is expanded first.
                for part in reversed(expansion):
                    stack.append(part)
        try:
            return bytes_out.decode("utf-8")
        except UnicodeDecodeError:
            return bytes_out.decode("utf-8", errors="replace")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(vocab_size=payload["vocab_size"])
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]

        for a, b in tokenizer.merges:
            new_id = len(tokenizer._id_to_token)
            expansion = tokenizer._id_to_token[a] + tokenizer._id_to_token[b]
            tokenizer._id_to_token[new_id] = expansion
            tokenizer._merge_to_id[(a, b)] = new_id

        tokenizer._eos_id = payload.get("eos_id", tokenizer._vocab_size - 1)
        tokenizer._id_to_token[tokenizer._eos_id] = ()
        return tokenizer

    def save(self, path: str | Path) -> None:
        """Serialize the tokenizer to JSON."""
        payload = {
            "type": "BPETokenizer",
            "vocab_size": self._vocab_size,
            "eos_id": self._eos_id,
            "merges": [list(pair) for pair in self.merges],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

