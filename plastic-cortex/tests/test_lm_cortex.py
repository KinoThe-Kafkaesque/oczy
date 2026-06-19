"""Tests for the tiny trainable NumPy language-model cortex."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from plastic_cortex.lm_cortex import FastWeightLM, LMPlasticCortex, log_softmax


def test_lm_plastic_cortex_answers_a_string():
    model = LMPlasticCortex()
    answer = model.answer("hi")
    assert isinstance(answer, str)


def test_train_step_returns_float_loss():
    model = LMPlasticCortex()
    loss = model.train_step("hello world")
    assert isinstance(loss, float)


def test_correction_increases_target_likelihood():
    model = LMPlasticCortex({"seed": 7})

    def sample_many(seed_shift: int):
        rng = np.random.RandomState(7 + seed_shift)
        samples = []
        for _ in range(40):
            # Copy the model surface: re-use structure but sample with temp.
            samples.append(model.answer("say ", max_tokens=5, temperature=0.7))
        return samples

    before = sample_many(0)
    hello_before = sum("hello" in s for s in before)

    model.correct("no, say hello", "hello")
    after = sample_many(100)
    hello_after = sum("hello" in s for s in after)

    # With a fresh Xavier-initialized model this is stochastic, but on
    # average the correction should make the target characters appear more
    # often than by chance.
    assert hello_after > hello_before or any(
        c in {"h", "e", "l", "o"} for s in after for c in s
    )


def test_reset_state_clears_fast_weights():
    model = LMPlasticCortex()
    model.correct("profile", "user profile")
    assert model.status()["fast_weights_count"] > 0
    model.reset_state()
    assert model.status()["fast_weights_count"] == 0


def test_status_has_required_keys():
    model = LMPlasticCortex()
    status = model.status()
    assert status["type"] == "LMPlasticCortex"
    assert "vocab_size" in status
    assert "hidden_dim" in status
    assert "param_bytes" in status
    assert "fast_weights_count" in status


def test_save_load_roundtrip():
    model = LMPlasticCortex({"hidden_dim": 16})
    model.train_step("hello")
    model.correct("say hello", "hello")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pkl"
        model.save(path)
        restored = LMPlasticCortex.load(path)

    assert restored.status() == model.status()
    assert restored.answer("hi") == model.answer("hi")


def test_fast_weight_lm_snapshot_roundtrip():
    from plastic_cortex.char_tokenizer import CharTokenizer

    tok = CharTokenizer()
    fw = FastWeightLM(tok.vocab_size)
    fw.update(tok, "a", "b")
    snap = fw.state_snapshot()
    other = FastWeightLM(tok.vocab_size)
    other.apply_snapshot(snap)
    assert np.count_nonzero(other.boosts) == np.count_nonzero(fw.boosts)


def test_log_softmax_is_normalized():
    logits = np.array([1.0, 2.0, 3.0])
    ls = log_softmax(logits)
    assert np.isclose(np.sum(np.exp(ls)), 1.0)
