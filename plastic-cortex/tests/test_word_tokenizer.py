from pathlib import Path

import pytest

from plastic_cortex.word_tokenizer import WordTokenizer


def test_word_tokenizer_fits_and_vocab_is_bounded() -> None:
    texts = ["hello world hello", "world test foo bar baz"]
    tok = WordTokenizer(vocab_size=3)
    tok.fit(texts)
    # 3 words + 3 specials = 6
    assert tok.vocab_size == 6
    assert tok.max_vocab_size == 6


def test_word_tokenizer_encode_decode_roundtrip() -> None:
    texts = ["hello world", "world hello"]
    tok = WordTokenizer(vocab_size=10)
    tok.fit(texts)
    ids = tok.encode("hello world")
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "hello world"


def test_word_tokenizer_unk_for_unknown_words() -> None:
    tok = WordTokenizer(vocab_size=2)
    tok.fit(["hello world"])
    ids = tok.encode("hello world unknown")
    decoded = tok.decode(ids)
    assert "[?]" in decoded


def test_word_tokenizer_punctuation_handling() -> None:
    tok = WordTokenizer(vocab_size=10)
    tok.fit(["hello, world!"])
    ids = tok.encode("hello, world!")
    decoded = tok.decode(ids)
    assert "," in decoded
    assert "!" in decoded


def test_word_tokenizer_save_load_roundtrip(tmp_path: Path) -> None:
    tok = WordTokenizer(vocab_size=5)
    tok.fit(["hello world this is a test"])
    path = tmp_path / "word_tokenizer.json"
    tok.save(path)

    loaded = WordTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.decode(tok.encode("hello world")) == tok.decode(tok.encode("hello world"))


def test_word_tokenizer_special_ids_are_consistent() -> None:
    tok = WordTokenizer(vocab_size=10)
    tok.fit(["hello world"])
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.eos_id == 2


def test_word_tokenizer_case_folding() -> None:
    tok = WordTokenizer(vocab_size=10)
    tok.fit(["Hello WORLD"])
    ids = tok.encode("hello world")
    decoded = tok.decode(ids)
    assert decoded == "hello world"

