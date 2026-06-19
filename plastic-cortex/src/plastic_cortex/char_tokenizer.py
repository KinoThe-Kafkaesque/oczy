"""Character-level tokenizer for the tiny NumPy language model backend.

The tokenizer keeps a fixed ordered vocabulary:

    <PAD> <UNK> <EOS> followed by the printable character set.

It can grow its vocabulary via ``fit()`` so a corpus that contains non-ASCII
code points (or just unusual characters) can still be encoded losslessly.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Iterable


DEFAULT_SPECIALS = ["<PAD>", "<UNK>", "<EOS>"]


class CharTokenizer:
    """Map strings to integer token sequences and back.

    Args:
        chars: Ordered string of regular characters to include after the
            special tokens.  When ``None``, ``string.printable`` is used.
        specials: Ordered list of special tokens to prepend.  Defaults to
            ``["<PAD>", "<UNK>", "<EOS>"]``.
    """

    def __init__(
        self,
        chars: str | None = None,
        specials: list[str] | None = None,
    ) -> None:
        self._specials = list(specials if specials is not None else DEFAULT_SPECIALS)
        chars = string.printable if chars is None else chars
        self._chars = list(dict.fromkeys(chars))  # preserve order, remove dupes
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        self._token_to_id.clear()
        self._id_to_token.clear()
        for idx, tok in enumerate(self._specials + self._chars):
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok

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

    def fit(self, texts: Iterable[str]) -> "CharTokenizer":
        """Grow the vocabulary with every character observed in *texts*."""
        merged = dict.fromkeys(self._chars)
        for text in texts:
            for ch in text:
                if ch not in self._token_to_id:
                    merged[ch] = None
        self._chars = list(merged.keys())
        self._build_vocab()
        return self

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of integer ids."""
        return [self._token_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of integer ids back to a string."""
        out: list[str] = []
        for tok in tokens:
            out.append(self._id_to_token.get(tok, "<UNK>"))
        return "".join(out)

    def save(self, path: str | Path) -> None:
        """Serialize the tokenizer to JSON."""
        payload = {
            "specials": self._specials,
            "chars": self._chars,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars="".join(payload["chars"]), specials=payload["specials"])
