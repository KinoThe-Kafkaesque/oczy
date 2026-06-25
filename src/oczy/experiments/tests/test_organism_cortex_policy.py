"""Tests for OrganismAgent cortex policy-head scoring in _rank_answer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from oczy.experiments.organism import OrganismAgent


@dataclass
class _MockCortexConfig:
    use_policy_head: bool = True


class _MockCortexAgent:
    """Minimal CortexAgent stand-in for policy-head scoring tests."""

    def __init__(self, policy_scores: np.ndarray | None = None) -> None:
        self._last_utterance: str | None = None
        self.config = _MockCortexConfig()
        self._policy_scores = policy_scores

    def perceive(self, request: str) -> None:
        self._last_utterance = request
        self.warm_state = True

    def policy_score(self, candidates: list[str]) -> np.ndarray:
        if self._policy_scores is not None:
            return self._policy_scores
        return np.array([10.0] + [0.0] * (len(candidates) - 1))


def test_cortex_policy_default_off_uses_legacy_ranking() -> None:
    organism = OrganismAgent({})
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"
    assert organism.answer("x") == "a"


def test_cortex_policy_boosts_preferred_candidate() -> None:
    """Policy head favours 'b' even though the fast organ returned 'a'."""
    # Matching candidate order ["a", "b"]: low for "a", high for "b".
    mock_cortex = _MockCortexAgent(policy_scores=np.array([0.0, 10.0]))
    organism = OrganismAgent(
        {"use_cortex_policy": True, "cortex_agent": mock_cortex}
    )
    organism.plastic_cortex.labels = ["a", "b"]
    organism.plastic_cortex.answer = lambda request: "a"

    assert organism.answer("x") == "b"
    assert mock_cortex._last_utterance == "x"


def test_cortex_policy_warning_without_cortex_agent() -> None:
    with pytest.warns(UserWarning, match="cortex_agent"):
        organism = OrganismAgent({"use_cortex_policy": True})
    assert organism.cortex_agent is None
