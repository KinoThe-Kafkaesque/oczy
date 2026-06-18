"""Smoke and correctness tests for the Correction-to-Competence Benchmark."""

import math

from correction_benchmark import (
    AlwaysWrongAgent,
    EpisodeResult,
    OracleAgent,
    ProbeResult,
    Scorer,
    build_dataset,
    run_benchmark,
)


_MIN_EPISODES = 10
_REQUIRED_CATEGORIES = {"transfer", "scope", "forgetting"}


def test_dataset_has_enough_episodes_with_all_probe_categories():
    dataset = build_dataset()
    assert len(dataset) >= _MIN_EPISODES
    for ep in dataset:
        assert ep.request.strip()
        assert ep.initial_wrong_answer.strip()
        assert ep.correction.strip()
        assert ep.corrected_answer.strip()
        categories = {probe.category for probe in ep.probes}
        assert categories == _REQUIRED_CATEGORIES, ep.request


def test_oracle_agent_perfect_scores():
    scores = run_benchmark(OracleAgent())
    assert scores["correction_uptake_latency"] == 0.0
    assert scores["transfer_score"] == 1.0
    assert scores["scope_score"] == 1.0
    assert scores["forgetting_score"] == 1.0
    assert 0 < scores["memory_bytes_per_delta"] < math.inf


def test_always_wrong_agent_scores_near_zero():
    scores = run_benchmark(AlwaysWrongAgent())
    assert scores["correction_uptake_latency"] == 1.0
    assert scores["transfer_score"] == 0.0
    assert scores["scope_score"] == 0.0
    assert scores["forgetting_score"] == 0.0
    assert scores["memory_bytes_per_delta"] == 0.0


def test_scorer_manual_perfect_result():
    ep = build_dataset()[0]
    result = EpisodeResult(
        episode=ep,
        initial_answer=ep.initial_wrong_answer,
        post_correction_answer=ep.corrected_answer,
        probe_results=tuple(
            ProbeResult(probe=probe, answer=probe.expected, correct=True)
            for probe in ep.probes
        ),
    )
    scores = Scorer.score((result,), agent=None)
    assert scores["correction_uptake_latency"] == 0.0
    assert scores["transfer_score"] == 1.0
    assert scores["scope_score"] == 1.0
    assert scores["forgetting_score"] == 1.0


def test_scorer_manual_unlearned_result():
    ep = build_dataset()[0]
    result = EpisodeResult(
        episode=ep,
        initial_answer=ep.initial_wrong_answer,
        post_correction_answer=ep.initial_wrong_answer,
        probe_results=tuple(
            ProbeResult(probe=probe, answer="wrong", correct=False)
            for probe in ep.probes
        ),
    )
    scores = Scorer.score((result,), agent=None)
    assert scores["correction_uptake_latency"] == 1.0
    assert scores["transfer_score"] == 0.0
    assert scores["scope_score"] == 0.0
    assert scores["forgetting_score"] == 0.0
