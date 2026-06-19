"""Tests for the character-level tokenizer used by the NumPy LM backend."""

from __future__ import annotations

import tempfile
from pathlib import Path

from plastic_cortex.char_tokenizer import CharTokenizer


def test_default_vocab_encodes_printable():
    tok = CharTokenizer()
    text = "Hello, world!"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_special_token_ids_are_reserved():
    tok = CharTokenizer()
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.eos_id == 2
    assert tok.vocab_size == 3 + len(tok._chars)

def test_fit_grows_vocab():
    tok = CharTokenizer(chars="abc")
    before = tok.vocab_size
    tok.fit(["café", "π"])
    assert tok.vocab_size > before
    assert tok.decode(tok.encode("caféπ")) == "caféπ"


def test_save_load_roundtrip():
    tok = CharTokenizer().fit(["café"])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tok.json"
        tok.save(path)
        restored = CharTokenizer.load(path)
        assert restored.vocab_size == tok.vocab_size
        assert restored.decode(restored.encode("café")) == "café"
