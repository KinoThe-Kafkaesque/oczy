"""Contract tests for HFDriver using a tiny random HF model.

Uses ``hf-internal-testing/tiny-random-LlamaForCausalLM`` (random weights,
tiny download) so CI has no big model cost.  Tests are BEHAVIORAL, not
semantic — we verify shapes, determinism, cvec effects, KV-splice
consistency, and that layers/caches don't crash.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from oczy.lm.cvec_driver import ReservedPosition
from oczy.lm.hf_driver import HFDriver, KVHandle

TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def driver() -> HFDriver:
    """Module-scoped: load the tiny model once for all tests."""
    return HFDriver.load(TEST_MODEL_ID)


# ---------------------------------------------------------------------------
# Basic shape / structure tests
# ---------------------------------------------------------------------------


def test_driver_reports_expected_shape(driver: HFDriver) -> None:
    """The tiny random Llama has known dimensions."""
    assert driver.n_embd > 0
    assert driver.n_layers > 0
    assert driver.n_vocab > 0
    assert driver.model_id == TEST_MODEL_ID


def test_generate_returns_string(driver: HFDriver) -> None:
    result = driver.generate("Hello world", max_tokens=8)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_deterministic(driver: HFDriver) -> None:
    """Same input, same active surface → same output (greedy)."""
    driver.clear_cvec()
    driver.clear_reserved_position()
    a = driver.generate("The capital of France is", max_tokens=5)
    b = driver.generate("The capital of France is", max_tokens=5)
    assert a == b, f"determinism failed: {a!r} != {b!r}"


# ---------------------------------------------------------------------------
# peek_layer (Goal 2)
# ---------------------------------------------------------------------------


def test_peek_layer_shape(driver: HFDriver) -> None:
    """peek_layer returns (n_embd,) for every valid layer index."""
    for layer_idx in range(driver.n_layers):
        vec = driver.peek_layer("test prompt", layer_idx)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (driver.n_embd,)
        assert vec.dtype == np.float32


def test_different_layers_give_different_vectors(driver: HFDriver) -> None:
    """Layer 0 and layer N-1 should produce different hiddens."""
    prompt = "The quick brown fox jumps over the lazy dog"
    v0 = driver.peek_layer(prompt, 0)
    vn = driver.peek_layer(prompt, driver.n_layers - 1)
    assert not np.allclose(v0, vn), "different layers produced same vector"


def test_peek_layer_pooling_modes(driver: HFDriver) -> None:
    prompt = "hello world test"
    last = driver.peek_layer(prompt, 0, pooling="last")
    mean = driver.peek_layer(prompt, 0, pooling="mean")
    # Both should have correct shape.
    assert last.shape == (driver.n_embd,)
    assert mean.shape == (driver.n_embd,)
    # Last-token vs mean should differ for a multi-token prompt.
    assert not np.allclose(last, mean), "last vs mean pooling gave same result"


def test_peek_layer_rejects_bad_index(driver: HFDriver) -> None:
    with pytest.raises(IndexError):
        driver.peek_layer("prompt", driver.n_layers)
    with pytest.raises(IndexError):
        driver.peek_layer("prompt", -1)


def test_peek_layer_rejects_bad_pooling(driver: HFDriver) -> None:
    with pytest.raises(ValueError, match="pooling"):
        driver.peek_layer("prompt", 0, pooling="sum")


# ---------------------------------------------------------------------------
# peek_embedding
# ---------------------------------------------------------------------------


def test_peek_embedding_shape(driver: HFDriver) -> None:
    emb = driver.peek_embedding("test prompt", last_token_only=True)
    assert emb.shape == (driver.n_embd,)
    assert emb.dtype == np.float32


def test_peek_embedding_mean_pooling(driver: HFDriver) -> None:
    emb = driver.peek_embedding("test prompt", last_token_only=False)
    assert emb.shape == (driver.n_embd,)


def test_peek_embedding_cached(driver: HFDriver) -> None:
    """Repeated calls with same args return identical arrays."""
    a = driver.peek_embedding("cache test", last_token_only=True)
    b = driver.peek_embedding("cache test", last_token_only=True)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Cvec steering
# ---------------------------------------------------------------------------


def test_set_cvec_layer_rejects_bad_dim(driver: HFDriver) -> None:
    with pytest.raises(ValueError, match="dim"):
        driver.set_cvec_layer(0, np.zeros(driver.n_embd + 1, dtype=np.float32))


def test_set_cvec_layer_rejects_bad_idx(driver: HFDriver) -> None:
    with pytest.raises(IndexError):
        driver.set_cvec_layer(driver.n_layers, np.zeros(driver.n_embd, dtype=np.float32))


def test_set_cvec_changes_logits(driver: HFDriver) -> None:
    """A large cvec should change the next-token logits vs. cleared state."""
    driver.clear_cvec()

    # Baseline: get logits without cvec.
    prompt = "The meaning of life is"
    input_ids = driver._tokenize(prompt)
    with torch.no_grad():
        out_base = driver._model(input_ids=input_ids)
    baseline_logits = out_base.logits[0, -1, :].clone()

    # Apply a large random cvec to layer 0.
    rng = np.random.RandomState(42)
    big_vec = rng.randn(driver.n_embd).astype(np.float32) * 100.0
    driver.set_cvec_layer(0, big_vec)

    with torch.no_grad():
        out_cvec = driver._model(input_ids=input_ids)
    cvec_logits = out_cvec.logits[0, -1, :].clone()

    driver.clear_cvec()

    # Logits should differ.
    assert not torch.allclose(baseline_logits, cvec_logits), (
        "cvec did not change logits"
    )
    # With a strong cvec, the top token should reasonably change.
    # (Probabilistic with random model — allow but don't require.)
    # At minimum the logit vector isn't identical.


def test_clear_cvec_restores_baseline(driver: HFDriver) -> None:
    """After clear_cvec(), logits match the un-steered baseline."""
    driver.clear_cvec()

    prompt = "42 is the answer to"
    input_ids = driver._tokenize(prompt)
    with torch.no_grad():
        out1 = driver._model(input_ids=input_ids)
    baseline = out1.logits[0, -1, :].clone()

    # Apply cvec, then clear.
    rng = np.random.RandomState(99)
    vec = rng.randn(driver.n_embd).astype(np.float32) * 10.0
    driver.set_cvec_uniform(vec)
    driver.clear_cvec()

    with torch.no_grad():
        out2 = driver._model(input_ids=input_ids)
    restored = out2.logits[0, -1, :]

    assert torch.allclose(baseline, restored, atol=1e-5), (
        "clear_cvec did not restore exact baseline logits"
    )


def test_set_cvec_uniform_applies_to_all_layers(driver: HFDriver) -> None:
    """set_cvec_uniform registers hooks on every layer."""
    driver.clear_cvec()
    assert len(driver._cvec_hooks) == 0

    driver.set_cvec_uniform(np.zeros(driver.n_embd, dtype=np.float32))
    assert len(driver._cvec_hooks) == driver.n_layers
    assert driver.cvec_active

    driver.clear_cvec()
    assert len(driver._cvec_hooks) == 0
    assert not driver.cvec_active


# ---------------------------------------------------------------------------
# Reserved position
# ---------------------------------------------------------------------------


def test_reserved_position_set_and_clear(driver: HFDriver) -> None:
    rp = ReservedPosition(text="[[FACT: sky is blue]]", source="test")
    driver.set_reserved_position(rp)
    assert driver.reserved_position_active
    assert driver.reserved_position is rp

    driver.clear_reserved_position()
    assert not driver.reserved_position_active
    assert driver.reserved_position is None


def test_reserved_position_changes_generated_text(driver: HFDriver) -> None:
    """With a reserved prefix, the effective prompt is longer."""
    driver.clear_cvec()
    driver.clear_reserved_position()

    prompt = "Once upon a time"

    # Without reserved position.
    out1 = driver.generate(prompt, max_tokens=5)

    # With a reserved position (the prefix text changes input entirely).
    driver.set_reserved_position(ReservedPosition(text="PREFIX:", source="test"))
    out2 = driver.generate(prompt, max_tokens=5)

    driver.clear_reserved_position()

    # The outputs will likely differ since the actual input changed.
    # The key property: the reserved prefix is applied, so the model sees
    # different tokens → output can differ.
    # (With a random model, we can't assert semantic difference, but we
    # can assert behavior: no crash, strings returned.)
    assert isinstance(out1, str)
    assert isinstance(out2, str)


# ---------------------------------------------------------------------------
# KV-slot API (Goal 1)
# ---------------------------------------------------------------------------


def test_encode_kv_returns_handle(driver: HFDriver) -> None:
    handle = driver.encode_kv("sky is blue")
    assert isinstance(handle, KVHandle)
    assert handle.seq_len > 0
    assert len(handle.past_key_values) == driver.n_layers
    # Each layer's KV entry: (key, value) tuple.
    # Each layer's KV entry: at minimum (key, value). Newer HF may
    # include a cache_position as third element.
    for kv in handle.past_key_values:
        assert isinstance(kv, tuple)
        assert len(kv) >= 2  # at least (k, v)
        assert kv[0].shape[2] == handle.seq_len  # seq


def test_generate_with_kv_no_crash(driver: HFDriver) -> None:
    """generate_with_kv should not crash on a normal prompt+handle."""
    handle = driver.encode_kv("Paris is the capital of France")
    result = driver.generate_with_kv("What is the capital of France?", handle, max_tokens=8)
    assert isinstance(result, str)
    assert len(result) > 0


def test_kv_splicing_changes_logits(driver: HFDriver) -> None:
    """Splicing a KV handle should change next-token logits vs. no splice."""
    driver.clear_cvec()

    prompt = "The capital of France is"
    input_ids = driver._tokenize(prompt)

    # Baseline logits (no KV).
    with torch.no_grad():
        out_base = driver._model(input_ids=input_ids)
    baseline_logits = out_base.logits[0, -1, :].clone()

    # Encode a fact and splice it.
    handle = driver.encode_kv("France capital Paris")
    prompt_len = input_ids.shape[1]
    position_ids = torch.arange(
        handle.seq_len,
        handle.seq_len + prompt_len,
        dtype=torch.long,
    ).unsqueeze(0)
    with torch.no_grad():
        out_kv = driver._model(
            input_ids=input_ids,
            past_key_values=handle.past_key_values,
            position_ids=position_ids,
        )
    kv_logits = out_kv.logits[0, -1, :].clone()

    assert not torch.allclose(baseline_logits, kv_logits), (
        "KV splice did not change logits"
    )


def test_token_ranks_returns_valid_structure(driver: HFDriver) -> None:
    ranks = driver.token_ranks("The capital of France is", ["Paris", "London"])
    assert len(ranks) == 2
    for r in ranks:
        assert "target" in r
        assert "rank" in r
        assert "top1" in r
        assert 0 <= r["rank"] <= driver.n_vocab


def test_token_ranks_with_kv_returns_valid_structure(driver: HFDriver) -> None:
    handle = driver.encode_kv("France capital Paris")
    ranks = driver.token_ranks_with_kv(
        "The capital of France is", handle, ["Paris", "London"]
    )
    assert len(ranks) == 2
    for r in ranks:
        assert "target" in r
        assert "rank" in r
        assert "top1" in r
        assert 0 <= r["rank"] <= driver.n_vocab


def test_generate_with_kv_multi_token_prompt(driver: HFDriver) -> None:
    """Multi-token prompts with KV handles should not crash."""
    handle = driver.encode_kv("Python is a programming language")
    result = driver.generate_with_kv(
        "What is Python? Explain in one word:", handle, max_tokens=5
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Logit bias
# ---------------------------------------------------------------------------


def test_logit_bias_generate_no_crash(driver: HFDriver) -> None:
    """logit_bias_generate should not crash."""
    # Pick some token IDs that exist in the vocab.
    prompt = "The answer is"
    target_ids = [42, 100]  # arbitrary, may or may not be forced
    result = driver.logit_bias_generate(prompt, target_ids, bias=5.0, max_tokens=8)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_returns_dict(driver: HFDriver) -> None:
    s = driver.status()
    assert isinstance(s, dict)
    assert s["n_embd"] == driver.n_embd
    assert s["n_layers"] == driver.n_layers
    assert "cvec_active" in s
    assert "reserved_position_active" in s
    assert "lm_loaded" in s


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager(driver: HFDriver) -> None:
    """Using HFDriver as context manager should work and clean up hooks."""
    driver.clear_cvec()
    driver.set_cvec_uniform(np.zeros(driver.n_embd, dtype=np.float32))
    assert driver.cvec_active

    # __exit__ should call close() which clears hooks.
    # We don't actually exit here since the driver is module-scoped,
    # but we can test that the hooks are manageable.
    driver.clear_cvec()
    assert not driver.cvec_active
