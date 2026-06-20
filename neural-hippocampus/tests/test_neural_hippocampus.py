"""Tests for the Neural Hippocampus minimal prototype.

These tests exercise the surprise gate, replay retrieval, and the
compress-and-decay consolidation loop.
"""

import pytest

from neural_hippocampus import NeuralHippocampus


def test_high_surprise_stored_and_low_surprise_rejected():
    """Only episodes with prediction error high enough to push surprise above
    the gate should live in fast memory."""
    hippo = NeuralHippocampus(config={"surprise_threshold": 0.5})

    low_id = hippo.store(
        query="what is profiling?",
        answer="a tool",
        correction="not what I meant",
        prediction_error=0.1,
    )
    high_id = hippo.store(
        query="what is a profile?",
        answer="user profile page",
        correction="I mean business vertical",
        prediction_error=0.9,
    )

    assert low_id is None
    assert high_id is not None
    assert hippo.status()["episode_count"] == 1


def test_replay_returns_relevant_episodes():
    """Replaying for a query should return the closest stored episode first."""
    hippo = NeuralHippocampus(config={"surprise_threshold": 0.0})

    queries = ["sql injection", "business profile", "api timeout"]
    for q in queries:
        hippo.store(
            query=q,
            answer="placeholder",
            correction=f"correction for {q}",
            prediction_error=0.8,
        )

    results = hippo.reinforce("business profile", k=3)
    assert len(results) == 3
    # The exact same query must win.
    assert results[0]["query"] == "business profile"
    # All returned traces were marked as replayed.
    assert all(r["replay_count"] > 0 for r in results)


def test_consolidation_reduces_storage_and_preserves_behavior():
    """Clustering frequently replayed episodes should create slow-update
    summaries and, with decay enabled, shrink the raw trace buffer."""
    hippo = NeuralHippocampus(
        config={
            "surprise_threshold": 0.0,
            "cluster_similarity": 0.5,
            "replay_threshold": 1,
        }
    )

    # Create several related episodes we can group.
    related = ["profile means vertical", "profile is industry", "profile category"]
    unrelated = ["api auth token", "timeout retry logic"]
    for q in related:
        hippo.store(q, "placeholder", "business vertical", 0.8)
    for q in unrelated:
        hippo.store(q, "placeholder", "fix network", 0.8)

    before_bytes = hippo.status()["trace_bytes"]
    before_count = hippo.status()["episode_count"]

    # Replay the related cluster a few times so it becomes eligible.
    for _ in range(3):
        hippo.reinforce("profile means vertical")
        hippo.reinforce("api auth token")

    summaries = hippo.consolidate()

    after_bytes = hippo.status()["trace_bytes"]
    after_count = hippo.status()["episode_count"]

    assert len(summaries) > 0
    assert after_bytes < before_bytes
    assert after_count < before_count

    # Each summary should retain enough behavioural signal to be useful.
    for summary in summaries:
        assert summary["n_episodes"] >= 1
        assert summary["avg_surprise"] > 0
        assert len(summary["trace_ids"]) == summary["n_episodes"]
        assert isinstance(summary["embedding"], list)

    # No id collision / returned trace ids were removed from fast memory.
    all_remaining = {t["id"] for t in hippo.memory.traces.values()}
    for summary in summaries:
        for removed_id in summary["trace_ids"]:
            assert removed_id not in all_remaining


def test_status_is_ready_and_reports_counts():
    hippo = NeuralHippocampus()
    status = hippo.status()
    assert status["ready"] is True
    assert status["episode_count"] == 0
    assert status["slow_update_count"] == 0
    assert status["trace_bytes"] >= 0


def test_forward_still_raises_not_implemented():
    hippo = NeuralHippocampus()
    with pytest.raises(NotImplementedError):
        hippo.forward(None)


def test_corrected_answer_round_trips_through_replay():
    """T1: store(query, answer, correction, prediction_error, corrected_answer=X) must return X via reinforce."""
    h = NeuralHippocampus(config={"surprise_threshold": 0.0})
    h.store("what is a profile?", "user profile page",
            correction="I mean business vertical", prediction_error=0.9,
            corrected_answer="business vertical")
    r = h.reinforce("what is a profile?", k=1)
    assert r and r[0].get("corrected_answer") == "business vertical"
