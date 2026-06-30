"""Tests for DifferentiableFactIndex."""

import numpy as np
import pytest

from oczy.experiments.differentiable_fact_index import DifferentiableFactIndex


def test_store_and_retrieve_basic() -> None:
    """Store a fact and retrieve it with the same query."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16, lora_rank=2)
    q = np.random.randn(16).astype(np.float32)
    idx.store(q, "hello world")
    results = idx.retrieve(q, k=1)
    assert len(results) == 1
    assert results[0][0] == "hello world"
    assert results[0][1] > 0


def test_retrieve_returns_top_k() -> None:
    """Multiple facts; retrieve returns k closest."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16, lora_rank=2)
    rng = np.random.RandomState(42)
    q1 = rng.randn(16).astype(np.float32)
    q2 = rng.randn(16).astype(np.float32)
    q3 = rng.randn(16).astype(np.float32)

    idx.store(q1, "fact_one")
    idx.store(q2, "fact_two")
    idx.store(q3, "fact_three")

    results = idx.retrieve(q1, k=2)
    assert len(results) == 2
    assert results[0][0] == "fact_one"  # closest to q1
    assert results[0][1] > results[1][1]


def test_lora_modulates_scores() -> None:
    """LoRA adapter changes retrieval scores after correction."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16, lora_rank=2, lr_lora=1.0)
    q = np.random.randn(16).astype(np.float32)
    idx.store(q, "base_fact")

    # Query something different; baseline should give low scores.
    q2 = np.random.randn(16).astype(np.float32)
    baseline = idx.retrieve_baseline(q2, k=1)
    # Store a correction: query q2 should map to the same fact.
    idx.store(q2, "base_fact", is_correction=True)
    lora_result = idx.retrieve(q2, k=1)
    # After LoRA update, the score for the fact should change.
    # (Direction depends on random init, but the score should differ.)
    assert lora_result[0][0] == "base_fact"
    assert abs(lora_result[0][1] - baseline[0][1]) > 1e-6


def test_state_dict_roundtrip() -> None:
    """Save and load preserves state."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16, lora_rank=2)
    q = np.random.randn(16).astype(np.float32)
    idx.store(q, "test_fact")

    state = idx.state_dict()
    idx2 = DifferentiableFactIndex(n_facts=4, d_model=8)  # wrong size
    idx2.load_state_dict(state)  # overrides size

    results = idx2.retrieve(q, k=1)
    assert len(results) == 1
    assert results[0][0] == "test_fact"


def test_invalid_query_shape_raises() -> None:
    """Wrong dimension query raises ValueError."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16)
    with pytest.raises(ValueError):
        idx.store(np.random.randn(8).astype(np.float32), "bad")
    with pytest.raises(ValueError):
        idx.retrieve(np.random.randn(8).astype(np.float32))


def test_multiple_updates_dont_explode() -> None:
    """Repeated Hebbian updates keep embeddings bounded."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16, lora_rank=2, lr_fact=0.01)
    q = np.random.randn(16).astype(np.float32)
    for _ in range(100):
        idx.store(q, "stable", is_correction=True)
    # Embeddings should not explode.
    assert np.all(np.abs(idx.F) < 100)
    assert np.all(np.abs(idx.A) < 100)
    assert np.all(np.abs(idx.B) < 100)


def test_retrieve_baseline_ignores_lora() -> None:
    """retrieve_baseline always uses only F, even after LoRA updates."""
    idx = DifferentiableFactIndex(n_facts=8, d_model=16, lora_rank=2, lr_lora=1.0)
    q = np.random.randn(16).astype(np.float32)
    idx.store(q, "fact")

    q2 = np.random.randn(16).astype(np.float32)
    # Store as correction WITHOUT allocating a new row in F by using
    # _update_lora directly.  This tests that retrieve_baseline ignores
    # LoRA changes.
    idx._update_lora(q2 / (np.linalg.norm(q2) + 1e-8), 0)
    lora_result = idx.retrieve(q2, k=1, use_lora=True)
    baseline_result = idx.retrieve_baseline(q2, k=1)
    # LoRA-modulated retrieval should differ from baseline.
    assert abs(lora_result[0][1] - baseline_result[0][1]) > 1e-6
