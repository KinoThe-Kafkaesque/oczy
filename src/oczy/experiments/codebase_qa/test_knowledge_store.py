"""Unit tests for oczy.experiments.codebase_qa.knowledge_store."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from oczy.experiments.codebase_qa.knowledge_store import KnowledgeStore
from oczy.lm import ReservedPosition


def test_adding_facts_increases_record_count() -> None:
    store = KnowledgeStore()
    assert store.status(include_size=True)["record_count"] == 0
    store.add_fact("python version", "Requires Python 3.10 or newer.")
    assert store.status(include_size=True)["record_count"] == 1
    store.add_fact("license", "MIT license.")
    assert store.status(include_size=True)["record_count"] == 2


def test_keyword_recall_top_k_ordered_by_score() -> None:
    store = KnowledgeStore()
    store.add_fact("python version", "Requires Python 3.10 or newer.")
    store.add_fact("license", "MIT license.")
    store.add_fact("dependencies", "numpy, pytest, and uv are required.")

    results = store.recall("python dependencies", k=2)
    assert len(results) == 2
    # The top result should contain the strongest keyword overlap.
    # TF-IDF ranks the 'python version' fact higher because both the key and
    # value contain the rare token 'python', making it the most relevant match.
    assert results[0]["key"] == "python version"
    assert results[0]["score"] >= results[1]["score"]



def test_embedding_recall_perfect_match_query_equals_key() -> None:
    def embed_fn(text: str) -> np.ndarray:
        # Deterministic embedding keyed on input text makes equality trivial.
        np.random.seed(abs(hash(text)) % (2**31))
        return np.random.rand(16)

    store = KnowledgeStore(embed_fn=embed_fn)
    store.add_fact("python version", "Requires Python 3.10 or newer.")

    results = store.recall("python version", k=1)
    assert len(results) == 1
    assert results[0]["key"] == "python version"
    assert results[0]["score"] == pytest.approx(1.0)


def test_format_context_contains_expected_keys_and_values() -> None:
    store = KnowledgeStore()
    store.add_fact("python version", "Requires Python 3.10 or newer.")
    store.add_fact("license", "MIT license.")

    context = store.format_context("python", k=1)
    assert "Retrieved repository facts:" in context
    assert "Key: python version" in context
    assert "Value: Requires Python 3.10 or newer." in context
    assert context.endswith("\n")


def test_pickle_roundtrip_preserves_facts_and_uses_keyword_fallback() -> None:
    def embed_fn(text: str) -> np.ndarray:
        return np.ones(8) * (hash(text) % 10)

    store = KnowledgeStore(embed_fn=embed_fn)
    store.add_fact(
        "python version",
        "Requires Python 3.10 or newer.",
        metadata={"src": "pyproject.toml"},
    )
    store.add_fact("license", "MIT license.")

    serialized = pickle.dumps(store)
    loaded = pickle.loads(serialized)

    assert loaded.embed_fn is None
    assert loaded.status()["record_count"] == 2

    # After load, recall uses keyword fallback.
    results = loaded.recall("python version")
    keys = {r["key"] for r in results}
    assert "python version" in keys

    # Metadata is preserved.
    python_fact = next(r for r in results if r["key"] == "python version")
    assert python_fact["metadata"] == {"src": "pyproject.toml"}


def test_status_returns_expected_keys() -> None:
    store = KnowledgeStore()
    status = store.status(include_size=True)
    assert status["project"] == "oczy.experiments.codebase_qa.knowledge_store"
    assert "serialized_bytes" in status
    assert "record_count" in status
    assert "dim" in status
    assert status["dim"] is None


def test_status_reports_embedding_dim() -> None:
    def embed_fn(text: str) -> np.ndarray:
        return np.zeros(32)

    store = KnowledgeStore(embed_fn=embed_fn)
    store.add_fact("key", "value")
    assert store.status(include_size=True)["dim"] == 32


def test_empty_store_returns_empty_recall() -> None:
    store = KnowledgeStore()
    assert store.recall("anything") == []
    assert store.format_context("anything", k=3) == "Retrieved repository facts:\n"

def test_format_context_min_score_filters_low_relevance_facts() -> None:
    store = KnowledgeStore()
    store.add_fact("relevant", "Python is the primary language used in this repository.")
    context = store.format_context("what language does the repo use", k=3, min_score=0.1)
    assert "Python" in context
    assert "moon" not in context


def test_get_reserved_position_returns_token_for_matching_fact() -> None:
    store = KnowledgeStore()
    store.add_fact(
        "business vertical",
        "The term 'Profile' in this repository refers to a business vertical.",
        metadata={"reserved_token": "vertical"},
    )

    pos = store.get_reserved_position("business vertical", k=1, min_score=0.0)

    assert isinstance(pos, ReservedPosition)
    assert pos.text == "vertical"
    assert pos.source == "knowledge_store"
    assert pos.exact_uptake_score is not None and pos.exact_uptake_score > 0



def test_get_reserved_position_returns_none_without_reserved_token_metadata() -> None:
    store = KnowledgeStore()
    store.add_fact("plain fact", "This fact has no reserved token.")

    assert store.get_reserved_position("plain fact", k=1, min_score=0.0) is None


def test_get_reserved_position_respects_min_score() -> None:
    store = KnowledgeStore()
    store.add_fact(
        "gamma fact",
        "Qwerty placeholder detail.",
        metadata={"reserved_token": "vertical"},
    )

    assert store.get_reserved_position("alpha beta query", k=1, min_score=0.18) is None
