"""Tests for semantic fallback matching in the curriculum scoring module."""

from __future__ import annotations

from oczy.experiments.organism_curriculum.dataset import Episode, Probe
from oczy.experiments.organism_curriculum.scoring import (
    matches,
    probe_matches,
)


def _file_episode(expected: str) -> Episode:
    """Build an episode whose ambiguous token is ``file``."""
    return Episode(
        id="test-file",
        initial_request="What does file mean here?",
        default_response="",
        correction_utterance="No, 'file' means submit it officially.",
        corrected_label="submit it officially",
        corrected_response="submit it officially",
        domain="general",
        probes=(
            Probe(
                request="What should I do with it?",
                expected=expected,
                category="retention",
                match_mode="sense",
            ),
        ),
    )


def test_semantic_sense_match() -> None:
    """An exact paraphrase of the expected label matches with semantic on."""
    assert matches(
        "submit it officially",
        "submit it officially",
        ambiguous_token="file",
        match_mode="sense",
        semantic=True,
    )


def test_semantic_neighbor_match() -> None:
    """A free-form answer in the same neighbourhood as the expected label
    matches via the semantic fallback even when token overlap is weaker."""
    assert matches(
        "I'll submit the report officially",
        "submit it officially",
        ambiguous_token="file",
        match_mode="sense",
        semantic=True,
    )


def test_semantic_rejection() -> None:
    """An unrelated answer outside the ambiguous word's neighbourhood fails."""
    assert not matches(
        "the weather is sunny today",
        "submit it officially",
        ambiguous_token="file",
        match_mode="sense",
        semantic=True,
    )

def test_semantic_disabled_preserves_old_behavior() -> None:
    """With ``semantic=False`` (default), the neighbour fallback never runs."""
    # Answer is in the neighbourhood but shares no token with the expected
    # label -> fails without the semantic fallback.
    assert not matches(
        "just paperwork and documents",
        "submit it officially",
        ambiguous_token="file",
        match_mode="sense",
        semantic=False,
    )
    # The same answer is accepted once the semantic fallback is enabled.
    assert matches(
        "just paperwork and documents",
        "submit it officially",
        ambiguous_token="file",
        match_mode="sense",
        semantic=True,
    )


def test_semantic_neighbor_only_on_answer_not_expected() -> None:
    """If the expected label is not in the neighbourhood, reject even when the
    answer is."""
    assert not matches(
        "submit the paperwork",
        "something unrelated",
        ambiguous_token="file",
        match_mode="sense",
        semantic=True,
    )


def test_semantic_unknown_ambiguous_token() -> None:
    """An ambiguous token absent from the neighbour map falls back to base
    matching only."""
    assert not matches(
        "totally different answer",
        "submit it officially",
        ambiguous_token="zzz",
        match_mode="sense",
        semantic=True,
    )


def test_probe_matches_semantic_flag() -> None:
    """``probe_matches`` threads the ``semantic`` flag through to ``matches``."""
    episode = _file_episode("submit it officially")
    probe = episode.probes[0]
    assert probe_matches(
        "I'll submit the report officially",
        probe,
        episode,
        semantic=True,
    )
    # Default semantic=False rejects the paraphrase-only path when there is no
    # direct token overlap with the expected label.
    assert not probe_matches(
        "paperwork paperwork paperwork",
        probe,
        episode,
    )


def test_existing_match_modes_unchanged() -> None:
    """Exact, contains, and sense modes behave as before without semantic."""
    assert matches("Hello", "hello", match_mode="exact")
    assert matches("the big cat", "big cat", match_mode="contains")
    assert matches("a red car", "the car is red", match_mode="sense")
    assert not matches("dogs", "cats", match_mode="sense")
