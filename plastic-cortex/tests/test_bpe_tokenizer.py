from pathlib import Path

import pytest

from plastic_cortex.bpe_tokenizer import BPETokenizer


def test_bpe_simple_roundtrip() -> None:
    tok = BPETokenizer(vocab_size=300)
    tok.fit(["hello world", "world hello"])
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_oov_roundtrip_after_fitting_different_corpus() -> None:
    tok = BPETokenizer(vocab_size=300)
    tok.fit(["the quick brown fox jumps over the lazy dog"])
    # These characters may appear individually but the full words are unlikely
    # to have been learned as merged tokens with a 300-size vocabulary.
    text = "a quizzical zebra appeared"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_vocab_growth_up_to_limit() -> None:
    tok = BPETokenizer(vocab_size=300)
    assert tok.vocab_size == 256  # initial single-byte tokens
    tok.fit(["hello world "] * 100)
    # Repeated whitespace + word bytes should yield some merges.
    assert 256 < tok.vocab_size <= 300


def test_bpe_save_load_roundtrip(tmp_path: Path) -> None:
    tok = BPETokenizer(vocab_size=300)
    tok.fit(["hello world", "world hello"])
    path = tmp_path / "bpe_tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    text = "hello world oov"
    assert loaded.decode(loaded.encode(text)) == text


def test_bpe_empty_encode_decode() -> None:
    tok = BPETokenizer(vocab_size=300)
    tok.fit(["hello world"])
    assert tok.encode("") == []
    assert tok.decode([]) == ""
